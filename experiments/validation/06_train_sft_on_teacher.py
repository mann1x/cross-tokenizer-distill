"""M16: SFT on teacher-generated completions.

No teacher forward, no anchor, no KL — pure causal-LM CE on
teacher tokens at continuation positions. Sequence-level imitation
that bypasses token-level KL ceiling.

Input JSONL: {prompt, teacher_completion, prompt_token_len}.
"""
from __future__ import annotations
import argparse, json, math, time, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


class TeacherCompletionDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_prompt_len=384, max_total_len=512,
                 code_only_mask=False):
        self.records = []
        with open(jsonl_path) as f:
            for line in f:
                self.records.append(json.loads(line))
        self.tok = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_total_len = max_total_len
        self.code_only_mask = code_only_mask
        if code_only_mask:
            import re as _re
            self._FENCE = _re.compile(r"```(?:python)?\s*\n(.*?)\n```", _re.DOTALL)

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        prompt_ids = self.tok(r["prompt"], add_special_tokens=False, truncation=True,
                              max_length=self.max_prompt_len, return_tensors=None)["input_ids"]
        comp_text = r["teacher_completion"]
        if self.code_only_mask:
            # Encode with offsets so we can map char ranges -> token positions.
            enc = self.tok(comp_text, add_special_tokens=False,
                           return_offsets_mapping=True, return_tensors=None)
            comp_ids = enc["input_ids"]
            offsets = enc["offset_mapping"]
            # Char ranges INSIDE python code fences (group 1 of FENCE).
            code_ranges = [(m.start(1), m.end(1)) for m in self._FENCE.finditer(comp_text)]
            # If NO fences found, fall back to "all comp tokens count" so the
            # example isn't silently zero-loss (and log will show via mask_pos).
            if not code_ranges:
                comp_mask = [True] * len(comp_ids)
            else:
                comp_mask = [
                    any(rs <= s and e <= re_ for (rs, re_) in code_ranges) and (s != e)
                    for (s, e) in offsets
                ]
        else:
            comp_ids = self.tok(comp_text, add_special_tokens=False,
                                return_tensors=None)["input_ids"]
            comp_mask = [True] * len(comp_ids)
        full = prompt_ids + comp_ids
        full_mask = [False] * len(prompt_ids) + comp_mask
        if len(full) > self.max_total_len:
            full = full[:self.max_total_len]
            full_mask = full_mask[:self.max_total_len]
        return {"input_ids": full, "prompt_len": len(prompt_ids),
                "code_mask": full_mask}


def collate(batch, pad_id, side="right"):
    """Right-pad for SFT — loss mask handles continuation slice."""
    max_len = max(len(b["input_ids"]) for b in batch)
    ids, am, pl, cm = [], [], [], []
    for b in batch:
        L = len(b["input_ids"])
        pad = max_len - L
        ids.append(b["input_ids"] + [pad_id]*pad)
        am.append([1]*L + [0]*pad)
        pl.append(b["prompt_len"])
        # code_mask: present when --code-only-mask is on; pad with False.
        bm = b.get("code_mask", [True]*L)
        cm.append(bm + [False]*pad)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(am, dtype=torch.long),
        "prompt_len": torch.tensor(pl, dtype=torch.long),
        "code_mask": torch.tensor(cm, dtype=torch.bool),
    }


def build_loss_mask(input_ids, attention_mask, prompt_len, pad_id):
    """Mask True at positions predicting a continuation token.
    logits[i] predicts ids[i+1]. Loss applies where i >= prompt_len-1
    AND ids[i+1] is not pad.
    """
    B, L = input_ids.shape
    mask = torch.zeros((B, L), dtype=torch.bool, device=input_ids.device)
    for b in range(B):
        pl = int(prompt_len[b].item())
        mask[b, pl-1:L-1] = True
    target_not_pad = input_ids[:, 1:] != pad_id
    mask[:, :L-1] = mask[:, :L-1] & target_not_pad
    return mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", default="deepseek-ai/deepseek-coder-1.3b-instruct")
    p.add_argument("--corpus", required=True, help="teacher-completion JSONL")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-prompt-len", type=int, default=384)
    p.add_argument("--max-total-len", type=int, default=512)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--warmup-steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--code-only-mask", action="store_true",
                   help="Restrict SFT loss to tokens INSIDE python code fences "
                        "(```python ... ```). For chat-mode teacher completions "
                        "where 55-60%% of the text is prose, this concentrates "
                        "the gradient signal on the actual implementation.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"[sft] student={args.student}", flush=True)
    # trust_remote_code=True: required for DS-Coder-V2-Lite (and other custom-modeling
    # checkpoints). DS-Coder-1.3B / Qwen2.5-Coder ignore the flag, so it's safe to default ON.
    tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    print("[sft] loading student (bf16)...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.bfloat16,
                                                    device_map={"": "cuda"}, trust_remote_code=True)

    lora = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank*2, target_modules="all-linear",
                      bias="none", task_type="CAUSAL_LM")
    student = get_peft_model(student, lora)
    student.print_trainable_parameters()

    ds = TeacherCompletionDataset(args.corpus, tok, max_prompt_len=args.max_prompt_len,
                                   max_total_len=args.max_total_len,
                                   code_only_mask=args.code_only_mask)
    if args.code_only_mask:
        print("[sft] --code-only-mask: SFT loss restricted to python code-fence interiors", flush=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id),
                    num_workers=0, pin_memory=False, drop_last=True)
    total_steps = len(dl) * args.epochs // args.grad_accum
    print(f"[sft] dataset={len(ds)} batches/epoch={len(dl)} epochs={args.epochs} total_optim_steps={total_steps}", flush=True)

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
    for epoch in range(args.epochs):
        for batch in dl:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            prompt_len = batch["prompt_len"]
            mask = build_loss_mask(input_ids, attention_mask, prompt_len, tok.pad_token_id)
            if args.code_only_mask:
                # Only count loss at positions whose TARGET token is inside a
                # code fence. mask is on positions i (predicting i+1), so AND
                # with code_mask shifted by 1.
                code_mask = batch["code_mask"].cuda()
                target_in_code = torch.zeros_like(mask)
                target_in_code[:, :-1] = code_mask[:, 1:]
                mask = mask & target_in_code

            s_out = student(input_ids=input_ids, attention_mask=attention_mask)
            logits = s_out.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            slice_mask = mask[:, :-1]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            tgt_lp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
            loss = -(tgt_lp * slice_mask.float()).sum() / slice_mask.float().sum().clamp(min=1.0)

            (loss / args.grad_accum).backward()
            accum_loss += loss.item()
            step += 1

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % args.logging_steps == 0:
                    avg = accum_loss / args.grad_accum
                    elapsed = time.time() - t_start
                    n_valid = slice_mask.sum().item()
                    print(f"[sft] step={opt_step}/{total_steps} loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e} mask_pos={n_valid} {elapsed:.0f}s", flush=True)
                accum_loss = 0.0

        ck = out / f"epoch-{epoch+1}"
        student.save_pretrained(ck)
        print(f"[sft] epoch {epoch+1} done, saved {ck}", flush=True)

    student.save_pretrained(out)
    print(f"[sft] DONE. Final adapter -> {out}", flush=True)


if __name__ == "__main__":
    main()
