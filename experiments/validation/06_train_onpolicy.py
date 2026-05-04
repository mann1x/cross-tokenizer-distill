"""On-policy distillation: student samples its own continuations from
prompts, teacher scores them live, loss applied at continuation positions.

Method ('--method'):
  fkl    forward KL(P_T||P_S) at student-sampled positions
  rkl    reverse KL(P_S||P_T) — MiniLLM
  jsd    generalized JSD with --beta — GKD
  hybrid alpha*FKL + (1-alpha)*RKL with --alpha

Same-vocab teacher only. For cross-vocab use 06_train_onpolicy_xv.py
(not implemented yet).
"""
from __future__ import annotations
import argparse, json, math, time, sys, os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, "/workspace/cross-tokenizer-distill")
from ctd.on_policy_loss import LOSSES


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
        return {"prompt_ids": ids}


def collate(batch, pad_id, side="left"):
    """Left-pad prompts so generation continues from the rightmost token uniformly."""
    max_len = max(len(b["prompt_ids"]) for b in batch)
    ids = []
    am = []
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
    }


@torch.no_grad()
def sample_continuation(student, tok, input_ids, attention_mask, max_new_tokens, temperature=1.0):
    """Sample a continuation from student. Returns (full_ids, full_attn, gen_start_pos).

    full_ids: [B, prompt_len + new_tokens] padded
    gen_start_pos: int — index where generation begins in full_ids
    """
    student.eval()
    out = student.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    student.train()
    prompt_len = input_ids.shape[1]
    new_attn = (out != tok.pad_token_id).long()
    # The prompt portion of attention_mask was correct; force it now in case generation produced eos+pad.
    new_attn[:, :prompt_len] = attention_mask
    return out, new_attn, prompt_len


def build_loss_mask(full_ids, attention_mask, prompt_len, pad_id):
    """Mask is True where loss should apply: positions PREDICTING a continuation token
    (so we mask logits at index i that predicts token at i+1).
    """
    B, L = full_ids.shape
    # Targets at position i+1 (i.e. logits[i] predicts ids[i+1]).
    # Loss positions: i in [prompt_len-1 .. L-2] AND ids[i+1] is not pad.
    mask = torch.zeros((B, L), dtype=torch.bool, device=full_ids.device)
    mask[:, prompt_len-1 : L-1] = True
    # Exclude positions whose target is pad
    target_not_pad = full_ids[:, 1:] != pad_id
    mask[:, :L-1] = mask[:, :L-1] & target_not_pad
    return mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", default="deepseek-ai/deepseek-coder-1.3b-instruct")
    p.add_argument("--teacher", default="deepseek-ai/deepseek-coder-6.7b-instruct")
    p.add_argument("--corpus", default="data/mbpp_train_prompts.jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--method", default="jsd", choices=["fkl","rkl","jsd","hybrid"])
    p.add_argument("--beta", type=float, default=0.5, help="JSD mixing (1=fwd KL, 0=rev KL)")
    p.add_argument("--alpha", type=float, default=0.5, help="hybrid blend weight")
    p.add_argument("--temperature", type=float, default=1.0, help="distill temperature T")
    p.add_argument("--gen-temperature", type=float, default=1.0, help="sampling temperature for student gen")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-prompt-len", type=int, default=384)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--warmup-steps", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-every", type=int, default=0, help="save ckpt every N steps (0=epoch only)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"[on-policy] method={args.method} beta={args.beta} T={args.temperature} gen_T={args.gen_temperature}", flush=True)
    print(f"[on-policy] student={args.student} teacher={args.teacher}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.student)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # so generate is uniform

    print("[on-policy] loading student (bf16)...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.bfloat16, device_map={"": "cuda"})
    print("[on-policy] loading teacher (bf16, frozen)...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16, device_map={"": "cuda"})
    teacher.eval()
    for p_ in teacher.parameters(): p_.requires_grad_(False)

    lora = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank*2, target_modules="all-linear", bias="none", task_type="CAUSAL_LM")
    student = get_peft_model(student, lora)
    student.print_trainable_parameters()

    ds = PromptDataset(args.corpus, tok, max_prompt_len=args.max_prompt_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id, side="left"),
                    num_workers=0, pin_memory=False, drop_last=True)
    total_steps = len(dl) * args.epochs // args.grad_accum
    print(f"[on-policy] dataset={len(ds)} batches/epoch={len(dl)} epochs={args.epochs} total_optim_steps={total_steps}", flush=True)

    trainable_params = [p_ for p_ in student.parameters() if p_.requires_grad]
    opt = AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    def lr_lambda(step):
        if step < args.warmup_steps: return step / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    sched = LambdaLR(opt, lr_lambda)

    loss_fn = LOSSES[args.method]
    loss_kwargs = {"T": args.temperature}
    if args.method == "jsd": loss_kwargs["beta"] = args.beta
    if args.method == "hybrid": loss_kwargs["alpha"] = args.alpha

    student.train()
    step = 0; opt_step = 0; t_start = time.time()
    accum_loss = 0.0
    for epoch in range(args.epochs):
        for batch in dl:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()

            full_ids, full_attn, prompt_len = sample_continuation(
                student, tok, input_ids, attention_mask, args.max_new_tokens,
                temperature=args.gen_temperature)
            loss_mask = build_loss_mask(full_ids, full_attn, prompt_len, tok.pad_token_id)

            # Student forward (with grad)
            s_out = student(input_ids=full_ids, attention_mask=full_attn)
            with torch.no_grad():
                t_out = teacher(input_ids=full_ids, attention_mask=full_attn)

            # Logits at position i predict token at i+1 — slice off the last logit
            s_logits = s_out.logits[:, :-1, :]
            t_logits = t_out.logits[:, :-1, :]
            mask = loss_mask[:, :-1]

            loss = loss_fn(s_logits, t_logits, mask, **loss_kwargs)
            (loss / args.grad_accum).backward()
            accum_loss += loss.item()
            step += 1

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % args.logging_steps == 0:
                    avg = accum_loss / args.grad_accum
                    elapsed = time.time() - t_start
                    n_valid = mask.sum().item()
                    print(f"[on-policy] step={opt_step}/{total_steps} loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e} mask_pos={n_valid} {elapsed:.0f}s", flush=True)
                accum_loss = 0.0

                if args.save_every > 0 and opt_step % args.save_every == 0:
                    ck = out / f"checkpoint-{opt_step}"
                    student.save_pretrained(ck)
                    print(f"[on-policy] saved {ck}", flush=True)

        ck = out / f"epoch-{epoch+1}"
        student.save_pretrained(ck)
        print(f"[on-policy] epoch {epoch+1} done, saved {ck}", flush=True)

    student.save_pretrained(out)
    print(f"[on-policy] DONE. Final adapter -> {out}", flush=True)


if __name__ == "__main__":
    main()
