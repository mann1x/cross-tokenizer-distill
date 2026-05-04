"""Cross-vocab on-policy distillation (M6).

Same loop as 06_train_onpolicy.py but teacher uses a DIFFERENT tokenizer.
At each step:
  1. Student samples continuation in student vocab.
  2. Decode full sequence (prompt+continuation) to text.
  3. Teacher re-tokenizes the same text in its own vocab.
  4. Forward both models on their respective tokenizations.
  5. Build byte-anchor alignment between the two tokenizations.
  6. For each student position with a matching teacher position, project
     teacher's distribution to student vocab via VocabMapper, compute
     KL/JSD/RKL at that student position. Skip non-aligned positions.

The mapper is built ONCE at startup (~1 min). Alignment is rebuilt per
batch (~10ms/example).

This validates the v6U direction: live cross-vocab distill from a
strong cross-tokenizer teacher (e.g. Qwen2.5-Coder-7B as proxy for
Qwen3-Coder-7B in the eventual Mythic-RDT pipeline).
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
                rec = json.loads(line)
                self.records.append(rec)
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
    ids = []; am = []
    for b in batch:
        L = len(b["prompt_ids"])
        pad = max_len - L
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
    return out


def kl_full_vocab(student_log_probs, teacher_probs_proj, mask, T=1.0):
    """Forward KL on full student vocab, against projected teacher distribution."""
    teacher_log = teacher_probs_proj.clamp(min=1e-12).log()
    per_pos = (teacher_probs_proj * (teacher_log - student_log_probs)).sum(dim=-1)
    per_pos = per_pos * (T ** 2)
    m = mask.to(per_pos.dtype)
    n = m.sum().clamp(min=1.0)
    return (per_pos * m).sum() / n


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
    p.add_argument("--mapper-cache", default="cache_mapper/qwen25c7b_to_dscoder1.3b.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"[xv-onpolicy] student={args.student} teacher={args.teacher}", flush=True)

    s_tok = AutoTokenizer.from_pretrained(args.student)
    t_tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if s_tok.pad_token is None: s_tok.pad_token = s_tok.eos_token
    if t_tok.pad_token is None: t_tok.pad_token = t_tok.eos_token
    s_tok.padding_side = "left"

    # Build / load VocabMapper
    Path(args.mapper_cache).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.mapper_cache).exists():
        print(f"[xv-onpolicy] loading mapper from {args.mapper_cache}", flush=True)
        cached = torch.load(args.mapper_cache, weights_only=False)
        mapper = cached["mapper"]
        cov = cached["cov"]
    else:
        print(f"[xv-onpolicy] building VocabMapper (teacher->student)...", flush=True)
        mapper = VocabMapper.from_tokenizers(t_tok, s_tok, multi_token="distribute")
        cov = mapper.coverage_report()
        torch.save({"mapper": mapper, "cov": cov}, args.mapper_cache)
        print(f"[xv-onpolicy] mapper cached to {args.mapper_cache}", flush=True)
    print(f"[xv-onpolicy] mapper coverage: single={cov.single_token_rate:.1%} multi={cov.multi_token_rate:.1%}", flush=True)

    print("[xv-onpolicy] loading student (bf16)...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.bfloat16, device_map={"": "cuda"})
    print("[xv-onpolicy] loading teacher (bf16, frozen)...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16, device_map={"": "cuda"}, trust_remote_code=True)
    teacher.eval()
    for p_ in teacher.parameters(): p_.requires_grad_(False)

    lora = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank*2, target_modules="all-linear", bias="none", task_type="CAUSAL_LM")
    student = get_peft_model(student, lora)
    student.print_trainable_parameters()

    ds = PromptDataset(args.corpus, s_tok, max_prompt_len=args.max_prompt_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, s_tok.pad_token_id, side="left"),
                    num_workers=0, pin_memory=False, drop_last=True)
    total_steps = len(dl) * args.epochs // args.grad_accum
    print(f"[xv-onpolicy] dataset={len(ds)} batches={len(dl)} epochs={args.epochs} optim_steps={total_steps}", flush=True)

    trainable = [p_ for p_ in student.parameters() if p_.requires_grad]
    opt = AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    def lr_lambda(step):
        if step < args.warmup_steps: return step / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    sched = LambdaLR(opt, lr_lambda)

    student.train()
    step = 0; opt_step = 0; t_start = time.time()
    accum_loss = 0.0
    accum_aligned = 0; accum_dropped = 0
    for epoch in range(args.epochs):
        for batch in dl:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()

            # 1. Student samples
            full_s = sample_continuation(student, s_tok, input_ids, attention_mask,
                                         args.max_new_tokens, temperature=args.gen_temperature)
            B = full_s.shape[0]

            # 2-3. Decode each sequence, re-tokenize for teacher per-example
            student_seqs = []
            teacher_seqs = []
            alignments = []
            for b in range(B):
                # Strip left-pad
                row = full_s[b]
                non_pad = (row != s_tok.pad_token_id).nonzero(as_tuple=True)[0]
                if len(non_pad) == 0: continue
                row = row[non_pad[0]:].tolist()
                # Trim trailing pad
                while row and row[-1] == s_tok.pad_token_id:
                    row.pop()
                if not row: continue
                text = s_tok.decode(row, skip_special_tokens=False)
                teacher_ids = t_tok.encode(text, add_special_tokens=False)
                if not teacher_ids: continue
                align = build_alignment(text, teacher_ids, row, t_tok, s_tok, mode="byte_anchor")
                student_seqs.append(row)
                teacher_seqs.append(teacher_ids)
                alignments.append(align)
            if not student_seqs: continue

            # 4. Forward teacher on its sequences (pad to max teacher len)
            max_t = max(len(t) for t in teacher_seqs)
            t_pad = t_tok.pad_token_id
            t_input = torch.full((len(teacher_seqs), max_t), t_pad, dtype=torch.long, device="cuda")
            t_attn = torch.zeros_like(t_input)
            for i, ts in enumerate(teacher_seqs):
                t_input[i, :len(ts)] = torch.tensor(ts, device="cuda")
                t_attn[i, :len(ts)] = 1
            with torch.no_grad():
                t_out = teacher(input_ids=t_input, attention_mask=t_attn)
            t_logits = t_out.logits  # [B, L_t, V_t]

            # 5. Forward student on its sequences (pad to max student len)
            max_s = max(len(s) for s in student_seqs)
            s_input = torch.full((len(student_seqs), max_s), s_tok.pad_token_id, dtype=torch.long, device="cuda")
            s_attn = torch.zeros_like(s_input)
            for i, ss in enumerate(student_seqs):
                s_input[i, :len(ss)] = torch.tensor(ss, device="cuda")
                s_attn[i, :len(ss)] = 1
            s_out = student(input_ids=s_input, attention_mask=s_attn)
            s_logits = s_out.logits  # [B, L_s, V_s]
            s_log_probs = F.log_softmax(s_logits / args.temperature, dim=-1)

            # 6. Build per-example loss: gather teacher distributions at aligned positions
            losses = []
            n_valid_total = 0
            n_dropped_total = 0
            # We need to identify continuation positions to mask out prompt + pad.
            # Approach: every position is valid for distill EXCEPT the first prompt_len positions.
            # Since we left-padded student input but here we re-packed, we know the prompt length per example.
            prompt_lens_s = []
            for b_idx, (st, ts) in enumerate(zip(student_seqs, teacher_seqs)):
                # Find prompt length in student tokens — use original prompt encoding
                p_text = batch["prompt_texts"][b_idx]
                p_ids_s = s_tok.encode(p_text, add_special_tokens=False)
                prompt_lens_s.append(len(p_ids_s))

            for b_idx, align in enumerate(alignments):
                p_len = prompt_lens_s[b_idx]
                len_s = len(student_seqs[b_idx])
                # We compute KL at student position j → predicts token j+1.
                # Distill positions: p_len-1 ..  len_s-2 (inclusive)
                valid_idx = []
                t_pos_list = []
                for entry in align.entries:
                    j = entry.student_pos
                    if not entry.valid or entry.suffix_token_ids is not None:
                        n_dropped_total += 1
                        continue
                    if j < p_len - 1 or j > len_s - 2:
                        continue
                    valid_idx.append(j)
                    t_pos_list.append(entry.teacher_pos)
                if not valid_idx:
                    continue
                v_idx_t = torch.tensor(valid_idx, device="cuda")
                t_pos_t = torch.tensor(t_pos_list, device="cuda")
                # Student log-probs at positions valid_idx (predicting next token)
                # Note: build_alignment positions are token positions;
                # the logit that predicts token j+1 lives at position j.
                s_lp = s_log_probs[b_idx, v_idx_t, :]  # [N_aligned, V_s]
                # Teacher logits at corresponding aligned positions
                t_lg = t_logits[b_idx, t_pos_t, :]  # [N_aligned, V_t]
                t_lp = F.log_softmax(t_lg / args.temperature, dim=-1)
                t_topk_log, t_topk_idx = t_lp.topk(64, dim=-1)
                # Project teacher top-K to student vocab (sparse)
                proj_log, proj_idx = mapper.project_topk(t_topk_log, t_topk_idx, out_topk=32, already_softmaxed=False)
                # KL on top-K projected support (Hinton with T^2 rescale)
                proj_p = proj_log.exp()
                # Gather student log-probs at projected indices
                s_lp_at = s_lp.gather(-1, proj_idx)
                per_pos = (proj_p * (proj_log - s_lp_at)).sum(-1)
                per_pos = per_pos * (args.temperature ** 2)
                losses.append(per_pos.mean())
                n_valid_total += len(valid_idx)

            if not losses:
                continue
            loss = sum(losses) / len(losses)
            (loss / args.grad_accum).backward()
            accum_loss += loss.item()
            accum_aligned += n_valid_total
            accum_dropped += n_dropped_total
            step += 1

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % args.logging_steps == 0:
                    avg = accum_loss / args.grad_accum
                    elapsed = time.time() - t_start
                    print(f"[xv-onpolicy] step={opt_step}/{total_steps} loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e} aligned={accum_aligned} dropped={accum_dropped} {elapsed:.0f}s", flush=True)
                accum_loss = 0.0; accum_aligned = 0; accum_dropped = 0

        ck = out / f"epoch-{epoch+1}"
        student.save_pretrained(ck)
        print(f"[xv-onpolicy] epoch {epoch+1} done, saved {ck}", flush=True)

    student.save_pretrained(out)
    print(f"[xv-onpolicy] DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
