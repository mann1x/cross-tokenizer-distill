Welcome to vast.ai. If authentication fails, try again after a few seconds, and double check your ssh key.
Have fun!
"""Generate teacher completions for SFT-on-teacher distillation.

Loads the teacher (e.g., DS-Coder-6.7B-Instruct or Qwen2.5-Coder-7B),
samples one continuation per corpus prompt, writes a JSONL with
{prompt, teacher_completion, prompt_token_len}.

OPTIONAL: also writes a *student-agnostic* per-token top-K (or top-P
nucleus) logit cache to a torch .pt file. The cache is keyed by record
index in the JSONL; pair them at training time.

The cache schema:
    {
        "meta": {
            "teacher": str, "tokenizer": str, "corpus": str,
            "temperature": float, "top_p": float, "max_new_tokens": int,
            "topk": int | None, "topp": float | None, "seed": int,
            "vocab_size": int, "dtype": "float16",
        },
        "records": [
            {
                "completion_token_ids": LongTensor[T],   # generated tokens (no pad)
                "topk_indices": LongTensor[T, K_t],       # top-K vocab indices per step
                "topk_logprobs": FloatTensor[T, K_t],     # log-softmax probs at those indices
            },
            ...
        ],
    }

K_t is uniform if --cache-logits-topk; variable if --cache-logits-topp
(stored as a list, not a tensor, with a sentinel pad value).

Compatible with `ctd/precompute.py` cache loaders (sparse top-K KL).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/cross-tokenizer-distill")
from ctd.util import make_teacher_token_blacklist, bad_words_ids_for_generate


def topk_from_logits(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """logits: [B, V] -> (indices [B, K] long, logprobs [B, K] fp16 (log-softmax))."""
    logp = F.log_softmax(logits.float(), dim=-1)
    vals, idx = torch.topk(logp, k, dim=-1)
    return idx.to(torch.long), vals.to(torch.float16)


def topp_from_logits(logits: torch.Tensor, p: float) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per-row nucleus selection. Returns list of (indices, logprobs) per row.
    Variable K per row — preserved as Python list of tensors.
    """
    logp = F.log_softmax(logits.float(), dim=-1)
    sorted_logp, sorted_idx = torch.sort(logp, dim=-1, descending=True)
    sorted_p = sorted_logp.exp()
    cum = torch.cumsum(sorted_p, dim=-1)
    out = []
    B = logits.shape[0]
    for b in range(B):
        # K = first position where cumulative prob >= p (always at least 1)
        mask = cum[b] >= p
        k = int(mask.float().argmax().item()) + 1 if mask.any() else logits.shape[-1]
        out.append((sorted_idx[b, :k].to(torch.long).cpu(), sorted_logp[b, :k].to(torch.float16).cpu()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="deepseek-ai/deepseek-coder-6.7b-instruct")
    p.add_argument("--corpus", default="data/mbpp_train_prompts.jsonl")
    p.add_argument("--output", required=True, help="JSONL with text completions")
    p.add_argument("--cache-output", default=None,
                   help="Optional .pt path for top-K/top-P teacher logit cache "
                        "(student-agnostic, reusable across fine-tunes).")
    p.add_argument("--cache-logits-topk", type=int, default=None,
                   help="Per-position fixed top-K cache (e.g., 128). "
                        "Uniform shape, fast to load.")
    p.add_argument("--cache-logits-topp", type=float, default=None,
                   help="Per-position nucleus (top-P) cache (e.g., 0.95). "
                        "Variable K per token, smaller cache for peaky distributions.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-prompt-len", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mask-teacher-tokens", default="",
                   help="Comma-separated TEACHER-vocab token strings to ban during generation "
                        "(via bad_words_ids). Recommended for thinking-mode teachers: "
                        "'<think>,</think>'. Resolved through ctd.util.make_teacher_token_blacklist.")
    p.add_argument("--mask-teacher-token-ids", default="",
                   help="Comma-separated raw teacher token IDs to ban (additive to --mask-teacher-tokens).")
    args = p.parse_args()

    cache_mode = None
    if args.cache_output is not None:
        if (args.cache_logits_topk is None) == (args.cache_logits_topp is None):
            raise SystemExit("--cache-output requires exactly ONE of --cache-logits-topk / --cache-logits-topp")
        cache_mode = "topk" if args.cache_logits_topk is not None else "topp"

    torch.manual_seed(args.seed)
    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[gen] teacher={args.teacher}", flush=True)
    if cache_mode:
        print(f"[gen] LOGIT CACHE -> {args.cache_output} mode={cache_mode} "
              f"({'k='+str(args.cache_logits_topk) if cache_mode=='topk' else 'p='+str(args.cache_logits_topp)})", flush=True)
    tok = AutoTokenizer.from_pretrained(args.teacher)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    print("[gen] loading teacher (bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16, device_map={"": "cuda"})
    model.eval()
    vocab_size = model.config.vocab_size

    blacklist_ids = make_teacher_token_blacklist(tok, args.mask_teacher_tokens, args.mask_teacher_token_ids)
    bad_words = bad_words_ids_for_generate(blacklist_ids)
    if bad_words:
        print(f"[gen] banning {len(blacklist_ids)} teacher token IDs from generation: "
              f"{blacklist_ids[:10]}{'...' if len(blacklist_ids) > 10 else ''}", flush=True)

    records = []
    with open(args.corpus) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"[gen] {len(records)} prompts; bs={args.batch_size} vocab={vocab_size}", flush=True)

    cache_records: list[dict] = []  # only populated if cache_mode

    out_f = open(out_path, "w")
    t0 = time.time(); n_done = 0
    for i in range(0, len(records), args.batch_size):
        batch = records[i:i+args.batch_size]
        prompts = [r["prompt"] for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_prompt_len, add_special_tokens=False).to("cuda")
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            gen_out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                output_scores=cache_mode is not None,
                return_dict_in_generate=cache_mode is not None,
                bad_words_ids=bad_words,
            )
        if cache_mode:
            seqs = gen_out.sequences
            scores = gen_out.scores  # tuple of [B, V] tensors, len = #generated steps
        else:
            seqs = gen_out
            scores = None
        prompt_len_padded = enc["input_ids"].shape[1]
        gen_only = seqs[:, prompt_len_padded:]
        for j, rec in enumerate(batch):
            ids_full = gen_only[j].tolist()
            # find first pad to truncate (left-padded prompts mean rightmost pads here are end-of-sequence)
            try:
                t_eff = next(idx for idx, t in enumerate(ids_full) if t == tok.pad_token_id)
            except StopIteration:
                t_eff = len(ids_full)
            ids = ids_full[:t_eff]
            comp = tok.decode(ids, skip_special_tokens=True)
            out_f.write(json.dumps({
                "prompt": rec["prompt"],
                "teacher_completion": comp,
                "prompt_token_len": int(prompt_lens[j]),
            }) + "\n")

            if cache_mode:
                # scores[t] is [B, V]; extract top-K/top-P for j over t in [0, t_eff)
                if t_eff == 0:
                    cache_records.append({
                        "completion_token_ids": torch.empty(0, dtype=torch.long),
                        "topk_indices": torch.empty(0, 0, dtype=torch.long),
                        "topk_logprobs": torch.empty(0, 0, dtype=torch.float16),
                    })
                    continue
                if cache_mode == "topk":
                    K = args.cache_logits_topk
                    idx_per_step, lp_per_step = [], []
                    for t in range(t_eff):
                        idx_t, lp_t = topk_from_logits(scores[t][j:j+1], K)
                        idx_per_step.append(idx_t[0].cpu())
                        lp_per_step.append(lp_t[0].cpu())
                    cache_records.append({
                        "completion_token_ids": torch.tensor(ids, dtype=torch.long),
                        "topk_indices": torch.stack(idx_per_step, dim=0),     # [T, K]
                        "topk_logprobs": torch.stack(lp_per_step, dim=0),     # [T, K]
                    })
                else:
                    P = args.cache_logits_topp
                    idx_list, lp_list = [], []
                    for t in range(t_eff):
                        out_t = topp_from_logits(scores[t][j:j+1], P)[0]
                        idx_list.append(out_t[0])
                        lp_list.append(out_t[1])
                    cache_records.append({
                        "completion_token_ids": torch.tensor(ids, dtype=torch.long),
                        "topp_indices_per_step": idx_list,    # list[T] of variable-K LongTensors
                        "topp_logprobs_per_step": lp_list,    # list[T] of variable-K Float16
                    })

        n_done += len(batch)
        if (i // args.batch_size) % 5 == 0:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1.0)
            eta = (len(records) - n_done) / max(rate, 0.01)
            print(f"[gen] {n_done}/{len(records)} ({rate:.1f}/s, ETA {eta:.0f}s)", flush=True)
    out_f.close()
    print(f"[gen] DONE -> {out_path} ({time.time()-t0:.0f}s)", flush=True)

    if cache_mode:
        cache_path = Path(args.cache_output); cache_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "teacher": args.teacher,
            "tokenizer": args.teacher,
            "corpus": args.corpus,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "topk": args.cache_logits_topk,
            "topp": args.cache_logits_topp,
            "seed": args.seed,
            "vocab_size": vocab_size,
            "dtype": "float16",
            "n_records": len(cache_records),
        }
        torch.save({"meta": meta, "records": cache_records}, cache_path)
        size_mb = cache_path.stat().st_size / 1024 / 1024
        print(f"[gen] LOGIT CACHE saved -> {cache_path} ({size_mb:.1f} MB, n={len(cache_records)})", flush=True)


if __name__ == "__main__":
    main()
