"""M28 — GRPO+KL Distill with VERIFIED EXEC REWARD (cross-vocab).

Standard GRPO+ recipe (DeepCoder-style) with the auxiliary
distribution-matching KL term from M6b layered on. Reward is
**task-correctness** (1.0 if the sampled completion passes the unit
test, 0.0 otherwise), not teacher-likelihood. This makes the policy
gradient style-agnostic: student is rewarded for code that PASSES,
not for code that LOOKS like teacher.

This is the variant M27 should have been. M27 used teacher-LL as
reward proxy because it's free (same forward as the KL term, no
sandbox per rollout); that broke style-shift immunity by literally
making "be teacher-like" the optimization target. See
docs/STYLE_SHIFT_ISSUE.md "GRPO+KL Distill with TEACHER-LL reward".

Per prompt p:
  1. Sample K student completions y_1..y_K (T=gen-temperature, top_p=0.95)
  2. Reward r_k = exec_reward(prompt_text + decode(y_k))
       = 1.0 if exec(prompt_definition + completion + assert_test) passes
       = 0.0 otherwise (timeout, assertion failure, syntax error, etc.)
     Style-AGNOSTIC by construction — the verifier doesn't care what
     style the code is in, only whether it passes the assert.
  3. Group-relative advantage (binary reward → mean is empirical pass-rate
     in the group; std handles all-pass and all-fail edge cases):
        a_k = (r_k - mean(r)) / (std(r) + eps)
        if std(r) == 0 (all K samples got same reward): a_k = 0 (no PG signal)
  4. Loss = - sum_k a_k * mean_j log pi_student(token_j | y_<j)         # policy gradient
           + lambda_kl_ref * KL(pi_student || pi_base_frozen)            # ref-anchor (GRPO+ standard)
           + lambda_distill * KL(pi_student || pi_teacher_proj)          # distill aux (M6b-style)

Test extraction:
  - MBPP-train prompts contain ONE assert in the docstring (e.g.,
    "assert max_chain_length([Pair(5,24)...], 4) == 3"). We extract
    that assert and use it as the verifier. If extraction fails for
    a sample, reward defaults to 0.0 (treat as failure to be
    conservative).

KL terms unchanged from M27 (still distribution-matching at aligned
sampled positions; same projection, same anchor).
"""
from __future__ import annotations
import argparse, json, math, time, sys, os, re, signal
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# ================== Sandboxed exec for verified reward ==================

EXEC_TIMEOUT_S = 5  # per-rollout exec budget; small for training-time speed

class _ExecTimeout(Exception): ...

def _alarm_handler(signum, frame):
    raise _ExecTimeout()

def _exec_pass(code: str) -> bool:
    """True if the code runs to completion without raising. Sandboxed via SIGALRM
    + suppressed stdout/stderr; used per-rollout to compute verified reward."""
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(EXEC_TIMEOUT_S)
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            exec(code, {})
        return True
    except Exception:
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

# Match assert lines from MBPP-train docstring prompts
_ASSERT_RE = re.compile(r"^(assert\s.+?)$", re.MULTILINE)

def extract_test_from_prompt(prompt: str) -> str | None:
    """Pull the first assert line out of the MBPP-train docstring prompt."""
    m = _ASSERT_RE.search(prompt)
    return m.group(1).strip() if m else None

def verified_reward(prompt: str, completion: str, test: str | None) -> float:
    """Reward = 1.0 if exec(prompt + completion + test) passes, else 0.0."""
    if test is None:
        return 0.0
    code = prompt + "\n" + completion + "\n" + test + "\n"
    return 1.0 if _exec_pass(code) else 0.0

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

sys.path.insert(0, "/workspace/cross-tokenizer-distill")
from ctd.mapper import VocabMapper
from ctd.alignment import build_alignment
from ctd.util import make_teacher_token_blacklist


class PromptDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_prompt_len=384):
        self.records = []
        with open(jsonl_path) as f:
            for line in f:
                self.records.append(json.loads(line))
        self.tok = tokenizer
        self.max_prompt_len = max_prompt_len

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        ids = self.tok(r["prompt"], add_special_tokens=False, truncation=True,
                       max_length=self.max_prompt_len, return_tensors=None)["input_ids"]
        return {"prompt_ids": ids, "prompt_text": r["prompt"]}


def collate(batch, pad_id, side="left"):
    max_len = max(len(b["prompt_ids"]) for b in batch)
    ids, am = [], []
    for b in batch:
        L = len(b["prompt_ids"]); pad = max_len - L
        if side == "left":
            ids.append([pad_id]*pad + b["prompt_ids"])
            am.append([0]*pad + [1]*L)
        else:
            ids.append(b["prompt_ids"] + [pad_id]*pad)
            am.append([1]*L + [0]*pad)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(am, dtype=torch.long),
        "prompt_texts": [b["prompt_text"] for b in batch],
    }


@torch.no_grad()
def sample_K_continuations(student, tok, input_ids, attention_mask, max_new_tokens, K, temperature):
    """Return [B*K, L_full] by repeating each prompt K times then generating once."""
    student.eval()
    B = input_ids.shape[0]
    rep_ids = input_ids.repeat_interleave(K, dim=0)
    rep_attn = attention_mask.repeat_interleave(K, dim=0)
    out = student.generate(
        input_ids=rep_ids, attention_mask=rep_attn,
        max_new_tokens=max_new_tokens,
        do_sample=True, temperature=temperature, top_p=0.95,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
    )
    student.train()
    new_attn = (out != tok.pad_token_id).long()
    new_attn[:, :input_ids.shape[1]] = rep_attn  # restore prompt mask
    return out, new_attn, rep_attn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", default="deepseek-ai/deepseek-coder-1.3b-instruct")
    p.add_argument("--teacher", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--corpus", default="data/mbpp_train_val_prompt.jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Softmax temperature for KL terms (NOT generation)")
    p.add_argument("--gen-temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--max-prompt-len", type=int, default=384)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--prompts-per-batch", type=int, default=1,
                   help="P prompts per micro-batch; effective batch = P*K")
    p.add_argument("--K", type=int, default=4, help="Samples per prompt (group size)")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--warmup-steps", type=int, default=8)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mapper-cache", default="cache_mapper/qwen25c7b_to_dscoder1.3b_first_token.pt")
    p.add_argument("--multi-token", default="first_token", choices=["strict","first_token","distribute"])
    p.add_argument("--top-k-teacher", type=int, default=64)
    p.add_argument("--top-k-student", type=int, default=32)
    p.add_argument("--lambda-kl-ref", type=float, default=0.1,
                   help="GRPO standard: KL(pi || pi_ref) anchor weight")
    p.add_argument("--lambda-distill", type=float, default=0.5,
                   help="Teacher-distill KL weight at aligned sampled positions")
    p.add_argument("--exec-timeout", type=int, default=5,
                   help="Per-rollout exec timeout (seconds) for verified reward")
    p.add_argument("--adv-eps", type=float, default=1e-4)
    p.add_argument("--wandb-project", default=None,
                   help="If set, log to wandb under this project (e.g. ctd-validation)")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-name", default=None,
                   help="Run name; defaults to last component of --output-dir")
    p.add_argument("--mask-teacher-tokens", default="",
                   help="Comma-separated TEACHER-vocab token strings to mask before KL. "
                        "Recommended for thinking-mode teachers: '<think>,</think>'. "
                        "Each is encoded with the teacher tokenizer; all matching token IDs "
                        "get -inf logit so they're excluded from softmax + projection.")
    p.add_argument("--mask-teacher-token-ids", default="",
                   help="Comma-separated TEACHER-vocab token IDs to mask (alternative to "
                        "--mask-teacher-tokens; both can be set).")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    global EXEC_TIMEOUT_S
    EXEC_TIMEOUT_S = args.exec_timeout
    print(f"[grpo-kl-v] M28 GRPO+KL Distill (verified exec): student={args.student} teacher={args.teacher}", flush=True)
    print(f"[grpo-kl-v] K={args.K} lambda_kl_ref={args.lambda_kl_ref} lambda_distill={args.lambda_distill} exec_timeout={EXEC_TIMEOUT_S}s", flush=True)

    use_wandb = False
    if args.wandb_project:
        if not WANDB_AVAILABLE:
            print("[grpo-kl-v] WARNING: --wandb-project set but wandb not installed; skipping", flush=True)
        elif not os.environ.get("WANDB_API_KEY"):
            print("[grpo-kl-v] WARNING: --wandb-project set but WANDB_API_KEY env var empty; skipping", flush=True)
        else:
            run_name = args.wandb_name or Path(args.output_dir).name
            wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       name=run_name, config=vars(args))
            use_wandb = True
            print(f"[grpo-kl-v] wandb: project={args.wandb_project} run={run_name}", flush=True)

    s_tok = AutoTokenizer.from_pretrained(args.student)
    t_tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if s_tok.pad_token is None: s_tok.pad_token = s_tok.eos_token
    if t_tok.pad_token is None: t_tok.pad_token = t_tok.eos_token
    s_tok.padding_side = "left"

    Path(args.mapper_cache).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.mapper_cache).exists():
        print(f"[grpo-kl-v] loading mapper from {args.mapper_cache}", flush=True)
        cached = torch.load(args.mapper_cache, weights_only=False)
        mapper = cached["mapper"]; cov = cached["cov"]
    else:
        print(f"[grpo-kl-v] building VocabMapper(multi_token={args.multi_token})...", flush=True)
        mapper = VocabMapper.from_tokenizers(t_tok, s_tok, multi_token=args.multi_token)
        cov = mapper.coverage_report()
        torch.save({"mapper": mapper, "cov": cov}, args.mapper_cache)
    print(f"[grpo-kl-v] mapper coverage: single={cov.single_token_rate:.1%} multi={cov.multi_token_rate:.1%}", flush=True)

    # Resolve teacher-vocab token-id blacklist via shared CTD helper.
    # Same IDs are reusable for teacher-generation paths via bad_words_ids
    # (see ctd.util.bad_words_ids_for_generate).
    teacher_mask_ids = make_teacher_token_blacklist(
        t_tok, args.mask_teacher_tokens, args.mask_teacher_token_ids)
    if teacher_mask_ids:
        print(f"[grpo-kl-v] masking {len(teacher_mask_ids)} teacher token IDs from KL: "
              f"{teacher_mask_ids[:10]}{'...' if len(teacher_mask_ids) > 10 else ''}",
              flush=True)
        teacher_mask_id_tensor = torch.tensor(teacher_mask_ids, dtype=torch.long, device="cuda")
    else:
        teacher_mask_id_tensor = None

    print("[grpo-kl-v] loading student (bf16)...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.bfloat16, device_map={"": "cuda"})
    print("[grpo-kl-v] loading teacher (bf16, frozen)...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16, device_map={"": "cuda"}, trust_remote_code=True)
    teacher.eval()
    for p_ in teacher.parameters(): p_.requires_grad_(False)
    print(f"[grpo-kl-v] loading FROZEN BASE student (ref policy, lambda_kl_ref={args.lambda_kl_ref})...", flush=True)
    base_frozen = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.bfloat16, device_map={"": "cuda"})
    base_frozen.eval()
    for p_ in base_frozen.parameters(): p_.requires_grad_(False)

    lora = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank*2, target_modules="all-linear",
                      bias="none", task_type="CAUSAL_LM")
    student = get_peft_model(student, lora)
    student.print_trainable_parameters()

    ds = PromptDataset(args.corpus, s_tok, max_prompt_len=args.max_prompt_len)
    dl = DataLoader(ds, batch_size=args.prompts_per_batch, shuffle=True,
                    collate_fn=lambda b: collate(b, s_tok.pad_token_id, side="left"),
                    num_workers=0, pin_memory=False, drop_last=True)
    total_steps = len(dl) * args.epochs // args.grad_accum
    print(f"[grpo-kl-v] dataset={len(ds)} prompts/micro-batch={args.prompts_per_batch} K={args.K} "
          f"epochs={args.epochs} optim_steps={total_steps}", flush=True)

    trainable = [p_ for p_ in student.parameters() if p_.requires_grad]
    opt = AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    def lr_lambda(s):
        if s < args.warmup_steps: return s / max(1, args.warmup_steps)
        prog = (s - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    sched = LambdaLR(opt, lr_lambda)

    student.train()
    step = 0; opt_step = 0; t_start = time.time()
    accum_pg = 0.0; accum_kl_ref = 0.0; accum_distill = 0.0
    accum_reward_mean = 0.0; accum_reward_std = 0.0; accum_aligned = 0
    accum_pass = 0; accum_seen = 0  # M28: track pass-rate over rollouts

    for epoch in range(args.epochs):
        for batch in dl:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            prompt_texts = batch["prompt_texts"]
            P = input_ids.shape[0]
            K = args.K

            # 1. Sample K continuations per prompt -> [P*K, L_full]
            full_s, full_attn, rep_attn = sample_K_continuations(
                student, s_tok, input_ids, attention_mask,
                args.max_new_tokens, K, args.gen_temperature)
            BK = full_s.shape[0]  # = P*K

            # 2. Per sample (b in 0..BK-1): re-tokenize for teacher, build alignment
            kept = []  # list of dicts
            for b in range(BK):
                row = full_s[b]
                pad_count_b = int((rep_attn[b] == 0).sum().item())
                row_t = row[pad_count_b:].tolist()
                while row_t and row_t[-1] == s_tok.pad_token_id:
                    row_t.pop()
                if not row_t: continue
                p_text = prompt_texts[b // K]
                p_ids_s = s_tok(p_text, add_special_tokens=False, truncation=True,
                                max_length=args.max_prompt_len)["input_ids"]
                p_len_b = len(p_ids_s)
                if p_len_b >= len(row_t): continue
                text = s_tok.decode(row_t, skip_special_tokens=False)
                teacher_ids = t_tok.encode(text, add_special_tokens=False)
                if not teacher_ids: continue
                align = build_alignment(text, teacher_ids, row_t, t_tok, s_tok,
                                        mode="student_offset", suffix_reencode=True)
                kept.append({
                    "b": b, "prompt_idx": b // K, "row": row_t,
                    "teacher_ids": teacher_ids, "align": align,
                    "p_len": p_len_b, "len_s": len(row_t),
                    "pad_count": pad_count_b,
                })

            if not kept:
                continue

            # 3. Live student forward (gradient) on full_s
            s_out = student(input_ids=full_s, attention_mask=full_attn)
            s_logits = s_out.logits  # [BK, L_full, V_s]
            s_log_probs = F.log_softmax(s_logits / args.temperature, dim=-1)

            # Frozen ref forward (no grad)
            with torch.no_grad():
                b_out = base_frozen(input_ids=full_s, attention_mask=full_attn)
                b_log_probs = F.log_softmax(b_out.logits / args.temperature, dim=-1)

            # Teacher forward (no grad) on its own re-tokenization
            max_t = max(len(k["teacher_ids"]) for k in kept)
            t_pad = t_tok.pad_token_id
            t_input = torch.full((len(kept), max_t), t_pad, dtype=torch.long, device="cuda")
            t_attn = torch.zeros_like(t_input)
            for i, k in enumerate(kept):
                t_input[i, :len(k["teacher_ids"])] = torch.tensor(k["teacher_ids"], device="cuda")
                t_attn[i, :len(k["teacher_ids"])] = 1
            with torch.no_grad():
                t_out = teacher(input_ids=t_input, attention_mask=t_attn)
            t_logits = t_out.logits  # [n_kept, L_t, V_t]
            if teacher_mask_id_tensor is not None:
                # Zero out masked teacher tokens before softmax → they get 0 probability
                # and the rest renormalizes naturally. Used to suppress thinking-mode
                # markers (<think>, </think>) for reasoning teachers.
                t_logits = t_logits.index_fill(-1, teacher_mask_id_tensor, float("-inf"))
            t_log_probs = F.log_softmax(t_logits / args.temperature, dim=-1)

            # 4. Per kept sample: compute VERIFIED reward (exec on prompt+gen+test),
            #    student_logp_of_sampled, and distill KL contributions at aligned positions.
            sample_pg_logp = []  # list of scalar tensors (mean log pi_s for each sample)
            sample_reward = []   # list of float (1.0 if exec passes else 0.0)
            sample_idx = []      # parallel: original kept-index
            sample_passed = 0    # count of samples that passed exec (for logging)
            distill_sum = s_logits.new_zeros(())
            n_distill_pos = 0

            for i, k in enumerate(kept):
                b_orig = k["b"]; p_len = k["p_len"]; len_s = k["len_s"]
                pad_count = k["pad_count"]; row_t = k["row"]
                valid_s_full = []; valid_t = []; valid_target_tok = []

                for entry in k["align"].entries:
                    j = entry.student_pos
                    if not entry.valid or entry.suffix_token_ids is not None: continue
                    if j < p_len - 1 or j > len_s - 2: continue
                    valid_s_full.append(pad_count + j)
                    valid_t.append(entry.teacher_pos)
                    valid_target_tok.append(row_t[j + 1])

                if not valid_s_full: continue
                v_s = torch.tensor(valid_s_full, device="cuda")
                v_t = torch.tensor(valid_t, device="cuda")
                v_tgt = torch.tensor(valid_target_tok, device="cuda")

                s_lp_pos = s_log_probs[b_orig, v_s, :]                 # [N, V_s]
                t_lp_pos = t_log_probs[i, v_t, :]                      # [N, V_t]

                # === Verified reward via sandboxed exec ===
                # Decode JUST the continuation (skip prompt tokens) for the verifier.
                completion_ids = row_t[p_len:]
                completion_text = s_tok.decode(completion_ids, skip_special_tokens=True)
                p_text = prompt_texts[k["prompt_idx"]]
                test_line = extract_test_from_prompt(p_text)
                reward_val = verified_reward(p_text, completion_text, test_line)
                sample_reward.append(reward_val)
                if reward_val > 0.5: sample_passed += 1

                # === Policy-gradient log-pi: mean log pi_s(target) over continuation ===
                pg_lp = s_lp_pos.gather(-1, v_tgt.unsqueeze(-1)).squeeze(-1)  # [N]
                sample_pg_logp.append(pg_lp.mean())  # length-normalize
                sample_idx.append(i)

                # === Distill KL aux: KL(pi_s || projected pi_t) at aligned positions ===
                # (Same as M27 — distribution-matching aux loss, M6b-style.)
                t_topk_log, t_topk_idx = t_lp_pos.topk(args.top_k_teacher, dim=-1)
                proj_log, proj_idx = mapper.project_topk(
                    t_topk_log, t_topk_idx,
                    out_topk=args.top_k_student, already_softmaxed=False)
                proj_p = proj_log.exp()
                s_lp_at = s_lp_pos.gather(-1, proj_idx)
                per_pos_distill = (proj_p * (proj_log - s_lp_at)).sum(-1)
                per_pos_distill = per_pos_distill * (args.temperature ** 2)
                distill_sum = distill_sum + per_pos_distill.sum()
                n_distill_pos += per_pos_distill.shape[0]

            if not sample_pg_logp:
                continue

            # 5. Group-relative advantages: group by prompt_idx
            #    For each group of K samples from the same prompt, normalize.
            rewards_t = torch.tensor(sample_reward, device="cuda", dtype=torch.float32)
            adv = torch.zeros_like(rewards_t)
            # Build group ids
            group_ids = torch.tensor([kept[i]["prompt_idx"] for i in sample_idx], device="cuda")
            for g in group_ids.unique().tolist():
                mask = (group_ids == g)
                if mask.sum().item() < 2:
                    adv[mask] = 0.0
                    continue
                r_g = rewards_t[mask]
                adv[mask] = (r_g - r_g.mean()) / (r_g.std(unbiased=False) + args.adv_eps)

            # 6. Same-vocab KL ref anchor: KL(pi_s || pi_base) at sampled continuation positions
            #    (compute over the same v_s positions used per sample)
            kl_ref_sum = s_logits.new_zeros(())
            n_kl_ref_pos = 0
            for idx_local, k_idx in enumerate(sample_idx):
                k = kept[k_idx]
                p_len = k["p_len"]; len_s = k["len_s"]; pad_count = k["pad_count"]
                if p_len - 1 > len_s - 2: continue
                positions = list(range(pad_count + p_len - 1, pad_count + len_s - 1))
                if not positions: continue
                pos_t = torch.tensor(positions, device="cuda")
                s_lp = s_log_probs[k["b"], pos_t, :]   # [M, V_s]
                b_lp = b_log_probs[k["b"], pos_t, :]
                # Forward KL: KL(pi_s || pi_base) = sum_v pi_s * (log pi_s - log pi_base)
                # But for ref-anchor in PPO/GRPO we use REVERSE form to penalize drift FROM ref.
                # Standard GRPO uses k1 estimator: KL(pi_s || pi_ref) ≈ exp(diff) - 1 - diff
                # where diff = log pi_ref - log pi_s. We'll use the FKL closed form:
                # KL(pi_s || pi_base) = sum pi_s * (log pi_s - log pi_base) — penalizes
                # student putting mass where base doesn't.
                s_p = s_lp.exp()
                kl_per_pos = (s_p * (s_lp - b_lp)).sum(-1)
                kl_ref_sum = kl_ref_sum + kl_per_pos.sum()
                n_kl_ref_pos += kl_per_pos.shape[0]

            # 7. Assemble loss
            # Policy gradient: -E[adv * log_pi(sampled)]
            pg_terms = torch.stack(sample_pg_logp)
            pg_loss = -(adv.detach() * pg_terms).mean()
            kl_ref_loss = kl_ref_sum / max(1, n_kl_ref_pos)
            distill_loss = distill_sum / max(1, n_distill_pos)
            loss = pg_loss + args.lambda_kl_ref * kl_ref_loss + args.lambda_distill * distill_loss
            (loss / args.grad_accum).backward()

            accum_pg += pg_loss.item()
            accum_kl_ref += kl_ref_loss.item()
            accum_distill += distill_loss.item()
            accum_reward_mean += float(rewards_t.mean().item())
            accum_reward_std += float(rewards_t.std(unbiased=False).item())
            accum_aligned += n_distill_pos
            accum_pass += sample_passed
            accum_seen += len(sample_reward)
            step += 1

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % args.logging_steps == 0:
                    g = args.grad_accum
                    elapsed = time.time() - t_start
                    avg_pg = accum_pg / g
                    avg_kl_ref = accum_kl_ref / g
                    avg_distill = accum_distill / g
                    avg_r_mean = accum_reward_mean / g
                    avg_r_std = accum_reward_std / g
                    cur_lr = sched.get_last_lr()[0]
                    total_loss = avg_pg + args.lambda_kl_ref * avg_kl_ref + args.lambda_distill * avg_distill
                    pass_rate = accum_pass / max(1, accum_seen)
                    print(f"[grpo-kl-v] step={opt_step}/{total_steps} "
                          f"pg={avg_pg:+.4f} kl_ref={avg_kl_ref:.4f} "
                          f"distill={avg_distill:.4f} "
                          f"pass_rate={pass_rate:.2%} ({accum_pass}/{accum_seen}) "
                          f"reward(mean/std)={avg_r_mean:.3f}/{avg_r_std:.3f} "
                          f"lr={cur_lr:.2e} aligned={accum_aligned} {elapsed:.0f}s", flush=True)
                    if use_wandb:
                        wandb.log({
                            "train/step": opt_step,
                            "train/loss_total": total_loss,
                            "train/loss_pg": avg_pg,
                            "train/loss_kl_ref": avg_kl_ref,
                            "train/loss_distill": avg_distill,
                            "train/reward_mean": avg_r_mean,
                            "train/reward_std": avg_r_std,
                            "train/pass_rate": pass_rate,
                            "train/passed": accum_pass,
                            "train/seen": accum_seen,
                            "train/lr": cur_lr,
                            "train/aligned_pos": accum_aligned,
                            "train/elapsed_s": elapsed,
                            "train/epoch": epoch + (step / max(1, len(dl))),
                        }, step=opt_step)
                accum_pg = 0.0; accum_kl_ref = 0.0; accum_distill = 0.0
                accum_reward_mean = 0.0; accum_reward_std = 0.0; accum_aligned = 0
                accum_pass = 0; accum_seen = 0

        ck = out / f"epoch-{epoch+1}"
        student.save_pretrained(ck)
        print(f"[grpo-kl-v] epoch {epoch+1} saved -> {ck}", flush=True)

    student.save_pretrained(out)
    print(f"[grpo-kl-v] DONE -> {out}", flush=True)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
