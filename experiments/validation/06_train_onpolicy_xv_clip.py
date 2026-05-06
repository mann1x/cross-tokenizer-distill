"""Cross-vocab on-policy distillation — M6b (bugfix re-run).

Differences vs 06_train_onpolicy_xv.py:
  - Forward STUDENT on the same left-padded `full_s` used for sampling
    (mirrors M5's structure exactly). Avoids RoPE position drift between
    sampling and training.
  - Index `prompt_texts` via the ORIGINAL batch position (b_orig), not
    the position in the per-batch filtered list. Removes the
    sample-drop indexing bug.
  - Loss aggregation: global mean over all valid (batch, position)
    pairs (matches M5's `(per_pos * mask).sum() / mask.sum()`).
  - Mapper rebuild defaults to `multi_token=first_token` (puts teacher
    mass on a single coherent next-token target instead of smearing).

Teacher is still forwarded on its own re-tokenized right-padded tensor
(cross-vocab requires that — vocab differs).
"""
from __future__ import annotations
import argparse, json, math, time, sys, os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, "/workspace/cross-tokenizer-distill")
from ctd.mapper import VocabMapper
from ctd.alignment import build_alignment


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
    """Left-pad to align right edges (uniform generation start)."""
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
def sample_continuation(student, tok, input_ids, attention_mask, max_new_tokens, temperature=1.0):
    student.eval()
    out = student.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True, temperature=temperature, top_p=0.95,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
    )
    student.train()
    prompt_len = input_ids.shape[1]  # left-padded width
    new_attn = (out != tok.pad_token_id).long()
    new_attn[:, :prompt_len] = attention_mask  # restore prompt mask exactly
    return out, new_attn, prompt_len


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", default="deepseek-ai/deepseek-coder-1.3b-instruct")
    p.add_argument("--teacher", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--corpus", default="data/mbpp_train_prompts.jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--gen-temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-prompt-len", type=int, default=384)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--warmup-steps", type=int, default=8)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mapper-cache", default="cache_mapper/qwen25c7b_to_dscoder1.3b_first_token.pt")
    p.add_argument("--multi-token", default="first_token", choices=["strict","first_token","distribute"])
    p.add_argument("--top-k-teacher", type=int, default=64)
    p.add_argument("--teacher-logit-clip", type=float, default=0.0,
                   help="Mobius trick: clip teacher logits to [-c,+c] before softmax. "
                        "0.0 = disabled (M6b default). 15.0 = recommended starting point.")
    p.add_argument("--top-k-student", type=int, default=32)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"[xv-onpolicy-b] student={args.student} teacher={args.teacher} multi_token={args.multi_token}", flush=True)

    s_tok = AutoTokenizer.from_pretrained(args.student)
    t_tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if s_tok.pad_token is None: s_tok.pad_token = s_tok.eos_token
    if t_tok.pad_token is None: t_tok.pad_token = t_tok.eos_token
    s_tok.padding_side = "left"

    Path(args.mapper_cache).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.mapper_cache).exists():
        print(f"[xv-onpolicy-b] loading mapper from {args.mapper_cache}", flush=True)
        cached = torch.load(args.mapper_cache, weights_only=False)
        mapper = cached["mapper"]; cov = cached["cov"]
    else:
        print(f"[xv-onpolicy-b] building VocabMapper(multi_token={args.multi_token})...", flush=True)
        mapper = VocabMapper.from_tokenizers(t_tok, s_tok, multi_token=args.multi_token)
        cov = mapper.coverage_report()
        torch.save({"mapper": mapper, "cov": cov}, args.mapper_cache)
    print(f"[xv-onpolicy-b] mapper coverage: single={cov.single_token_rate:.1%} multi={cov.multi_token_rate:.1%}", flush=True)

    print("[xv-onpolicy-b] loading student (bf16)...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.bfloat16, device_map={"": "cuda"})
    print("[xv-onpolicy-b] loading teacher (bf16, frozen)...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16, device_map={"": "cuda"}, trust_remote_code=True)
    teacher.eval()
    for p_ in teacher.parameters(): p_.requires_grad_(False)

    lora = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank*2, target_modules="all-linear",
                      bias="none", task_type="CAUSAL_LM")
    student = get_peft_model(student, lora)
    student.print_trainable_parameters()

    ds = PromptDataset(args.corpus, s_tok, max_prompt_len=args.max_prompt_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, s_tok.pad_token_id, side="left"),
                    num_workers=0, pin_memory=False, drop_last=True)
    total_steps = len(dl) * args.epochs // args.grad_accum
    print(f"[xv-onpolicy-b] dataset={len(ds)} batches={len(dl)} epochs={args.epochs} optim_steps={total_steps}", flush=True)

    trainable = [p_ for p_ in student.parameters() if p_.requires_grad]
    opt = AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    def lr_lambda(s):
        if s < args.warmup_steps: return s / max(1, args.warmup_steps)
        prog = (s - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    sched = LambdaLR(opt, lr_lambda)

    student.train()
    step = 0; opt_step = 0; t_start = time.time()
    accum_loss = 0.0; accum_aligned = 0; accum_dropped = 0

    for epoch in range(args.epochs):
        for batch in dl:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            prompt_texts = batch["prompt_texts"]
            B0 = input_ids.shape[0]

            # 1. Sample student continuations on LEFT-padded input
            full_s, full_attn, prompt_pad_len = sample_continuation(
                student, s_tok, input_ids, attention_mask,
                args.max_new_tokens, temperature=args.gen_temperature)
            # full_s: [B0, prompt_pad_len + max_new_tokens] (left-padded, RoPE-consistent with sampling)

            # 2. Per-example: decode student tokens (skipping left pads), re-tokenize for teacher,
            #    build alignment. Track which ORIGINAL batch indices survive.
            orig_idx = []         # original batch position
            student_rows = []     # list[list[int]] — full_s row stripped of pads
            teacher_seqs = []     # list[list[int]] — teacher re-tokenization
            alignments = []       # list[AlignmentTable]
            prompt_lens_s = []    # length of prompt in student tokens (excluding pads)
            for b in range(B0):
                row = full_s[b]
                # Identify the prompt's first non-pad position via attention_mask
                # (for left-padded input, mask is 0..0,1..1).
                pad_count_b = int((attention_mask[b] == 0).sum().item())
                row_t = row[pad_count_b:].tolist()
                while row_t and row_t[-1] == s_tok.pad_token_id:
                    row_t.pop()
                if not row_t: continue
                # Compute prompt_len in student tokens for this row
                p_text = prompt_texts[b]  # original batch position — CORRECT indexing
                p_ids_s = s_tok(p_text, add_special_tokens=False, truncation=True,
                                max_length=args.max_prompt_len)["input_ids"]
                p_len_b = len(p_ids_s)
                if p_len_b >= len(row_t): continue  # nothing to distill
                text = s_tok.decode(row_t, skip_special_tokens=False)
                teacher_ids = t_tok.encode(text, add_special_tokens=False)
                if not teacher_ids: continue
                align = build_alignment(text, teacher_ids, row_t, t_tok, s_tok, mode="student_offset", suffix_reencode=True)
                orig_idx.append(b)
                student_rows.append(row_t)
                teacher_seqs.append(teacher_ids)
                alignments.append(align)
                prompt_lens_s.append(p_len_b)

            if not student_rows:
                continue

            # 3. Forward STUDENT on the original left-padded full_s/full_attn (M5-style).
            #    This keeps RoPE positions identical to sampling.
            s_out = student(input_ids=full_s, attention_mask=full_attn)
            s_logits = s_out.logits  # [B0, L_full, V_s]
            s_log_probs = F.log_softmax(s_logits / args.temperature, dim=-1)

            # 4. Forward TEACHER on its own right-packed tensor (cross-vocab).
            max_t = max(len(t) for t in teacher_seqs)
            t_pad = t_tok.pad_token_id
            t_input = torch.full((len(teacher_seqs), max_t), t_pad, dtype=torch.long, device="cuda")
            t_attn = torch.zeros_like(t_input)
            for i, ts in enumerate(teacher_seqs):
                t_input[i, :len(ts)] = torch.tensor(ts, device="cuda")
                t_attn[i, :len(ts)] = 1
            with torch.no_grad():
                t_out = teacher(input_ids=t_input, attention_mask=t_attn)
            t_logits = t_out.logits  # [B_kept, L_t, V_t]

            # 5. Per-example: build per-position KL contributions, sum into a global running accumulator.
            #    Critical mapping: full_s[b_orig] is left-padded. The student's position in full_s
            #    that holds "student token j of student_rows[i]" = pad_count_b + j.
            sum_loss = s_logits.new_zeros(())
            n_valid = 0
            for i, b_orig in enumerate(orig_idx):
                pad_count_b = int((attention_mask[b_orig] == 0).sum().item())
                p_len = prompt_lens_s[i]
                len_s = len(student_rows[i])
                # Distill at student positions j ∈ [p_len-1, len_s-2] (M5 convention)
                # i.e. logit at student_rows position j predicts token j+1.
                # In full_s coords, that's position pad_count_b + j.
                valid_s_full = []
                valid_t = []
                for entry in alignments[i].entries:
                    j = entry.student_pos
                    if not entry.valid or entry.suffix_token_ids is not None: continue
                    if j < p_len - 1 or j > len_s - 2: continue
                    valid_s_full.append(pad_count_b + j)
                    valid_t.append(entry.teacher_pos)
                accum_dropped += sum(1 for e in alignments[i].entries
                                     if (not e.valid or e.suffix_token_ids is not None)
                                     and (p_len - 1 <= e.student_pos <= len_s - 2))
                if not valid_s_full: continue
                v_s_t = torch.tensor(valid_s_full, device="cuda")
                v_t_t = torch.tensor(valid_t, device="cuda")

                s_lp = s_log_probs[b_orig, v_s_t, :]                # [N, V_s]
                t_lg = t_logits[i, v_t_t, :]                        # [N, V_t]
                if args.teacher_logit_clip > 0:
                    t_lg = t_lg.clamp(-args.teacher_logit_clip, args.teacher_logit_clip)
                t_lp = F.log_softmax(t_lg / args.temperature, dim=-1)
                t_topk_log, t_topk_idx = t_lp.topk(args.top_k_teacher, dim=-1)
                proj_log, proj_idx = mapper.project_topk(
                    t_topk_log, t_topk_idx,
                    out_topk=args.top_k_student, already_softmaxed=False)
                proj_p = proj_log.exp()
                s_lp_at = s_lp.gather(-1, proj_idx)                  # [N, K']
                per_pos = (proj_p * (proj_log - s_lp_at)).sum(-1)    # [N]
                per_pos = per_pos * (args.temperature ** 2)
                sum_loss = sum_loss + per_pos.sum()
                n_valid += per_pos.shape[0]

            if n_valid == 0: continue
            # Global mean over all valid positions across the batch (M5-style).
            loss = sum_loss / n_valid
            (loss / args.grad_accum).backward()
            accum_loss += loss.item()
            accum_aligned += n_valid
            step += 1

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % args.logging_steps == 0:
                    avg = accum_loss / args.grad_accum
                    elapsed = time.time() - t_start
                    print(f"[xv-onpolicy-b] step={opt_step}/{total_steps} loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e} aligned={accum_aligned} dropped={accum_dropped} {elapsed:.0f}s", flush=True)
                accum_loss = 0.0; accum_aligned = 0; accum_dropped = 0

        ck = out / f"epoch-{epoch+1}"
        student.save_pretrained(ck)
        print(f"[xv-onpolicy-b] epoch {epoch+1} saved -> {ck}", flush=True)

    student.save_pretrained(out)
    print(f"[xv-onpolicy-b] DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
