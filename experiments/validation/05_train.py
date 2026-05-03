"""Step 5 — Train one of the three runs (A/B/C).

Run A: SFT only (no distill, no cache).
Run B: same-vocab distill (cache_B).
Run C: CTD distill (cache_C).

All three share an identical recipe (LoRA r=16, lr=1e-4, 2 epochs,
bs=8 grad-accum 4, bf16). Only `--cache` and `--ctd-weight` differ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from adapters.transformers import CTDTrainerMixin
from ctd.losses import CTDLoss


class CTDTrainer(CTDTrainerMixin, Trainer):
    pass


def make_dataset(corpus_path: str, tokenizer, max_seq_len: int, cache_meta: dict | None):
    """Tokenize corpus → datasets.Dataset with input_ids, labels, ctd_positions.

    ctd_positions are GLOBAL token indices into the cache's flat layout.
    For run A (no cache), we omit ctd_positions entirely.
    """
    texts = []
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])

    records = []
    cache_offset = 0
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)[:max_seq_len]
        if not ids:
            continue
        rec = {"input_ids": ids, "labels": ids}
        if cache_meta is not None:
            # Global positions in the cache for this example.
            rec["ctd_positions"] = list(range(cache_offset, cache_offset + len(ids)))
            cache_offset += len(ids)
        records.append(rec)

    return Dataset.from_list(records)


def collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = []
    labels = []
    ctd_positions = []
    has_ctd = "ctd_positions" in batch[0]
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        if has_ctd:
            ctd_positions.append(b["ctd_positions"] + [0] * pad)
    out = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": (torch.tensor(input_ids, dtype=torch.long) != pad_id).long(),
    }
    if has_ctd:
        out["ctd_positions"] = torch.tensor(ctd_positions, dtype=torch.long)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True, choices=["A", "B", "C"])
    parser.add_argument("--corpus", default="data/corpus_5k.jsonl")
    parser.add_argument("--student", default="deepseek-ai/deepseek-coder-1.3b-instruct")
    parser.add_argument("--cache", default=None,
                        help="Path to teacher cache .pt — required for B and C, omitted for A.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ctd-weight", type=float, default=0.5)
    parser.add_argument("--ctd-weight-warmup-steps", type=int, default=0,
                        help="Linear ramp ctd_weight 0→target over N steps. "
                             "Compensates for LoRA cold-start (lora_B inits to 0).")
    parser.add_argument("--kl-temperature", type=float, default=1.0)
    parser.add_argument("--ctd-kind", default="kl", choices=["kl", "jsd", "uld_sorted_kl"])
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.run_name in ("B", "C") and args.cache is None:
        raise SystemExit(f"Run {args.run_name} requires --cache <path>.")

    output_dir = args.output_dir or f"runs/run_{args.run_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[05-train] Run {args.run_name} → {output_dir}")
    print(f"[05-train] Loading student: {args.student}")
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.student,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    cache = None
    cache_meta = None
    if args.cache:
        print(f"[05-train] Loading cache: {args.cache}")
        cache = torch.load(args.cache, weights_only=False)
        cache_meta = cache["meta"]
        print(f"[05-train] Cache: {cache_meta['n_total_tokens']} tokens, "
              f"top_k={cache_meta['top_k']}")

    print("[05-train] Tokenising corpus...")
    ds = make_dataset(args.corpus, tokenizer, args.max_seq_len, cache_meta)
    print(f"[05-train] Dataset: {len(ds)} examples")

    targs = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_steps=100,
        lr_scheduler_type="cosine",
        seed=args.seed,
        report_to=[],
    )

    trainer_kwargs = dict(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda b: collate(b, tokenizer.pad_token_id),
    )

    if cache is not None:
        ctd_loss = CTDLoss(kind=args.ctd_kind, temperature=args.kl_temperature)
        trainer = CTDTrainer(
            ctd_cache=cache,
            ctd_loss=ctd_loss,
            ctd_weight=args.ctd_weight,
            ctd_weight_warmup_steps=args.ctd_weight_warmup_steps,
            **trainer_kwargs,
        )
    else:
        trainer = Trainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(output_dir)
    print(f"[05-train] Done. LoRA weights → {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
