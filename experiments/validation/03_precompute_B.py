"""Step 3 — build same-vocab teacher cache (Pair B).

Teacher: Qwen2.5-Coder-7B-Instruct (BF16, NF4 if VRAM tight).
Student: Qwen2.5-Coder-0.5B-Instruct.

Same vocab (152064) — no projection needed, but we route through
ctd.precompute to keep the code path identical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ctd.precompute import precompute_aligned_cache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus_5k.jsonl")
    parser.add_argument("--output", default="cache_B/qwen25_coder_7b_top32.pt")
    parser.add_argument("--teacher", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--student-tokenizer", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--quant", choices=["none", "nf4"], default="nf4")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[03-precompute-B] Loading student tokenizer: {args.student_tokenizer}")
    student_tok = AutoTokenizer.from_pretrained(args.student_tokenizer)

    print(f"[03-precompute-B] Loading teacher: {args.teacher} (quant={args.quant})")
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher)

    quant_kwargs = {}
    if args.quant == "nf4":
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        quant_kwargs["torch_dtype"] = torch.bfloat16

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        device_map={"": args.device} if args.quant == "none" else "auto",
        **quant_kwargs,
    )
    teacher.eval()

    print(f"[03-precompute-B] Loading corpus: {args.corpus}")
    corpus_path = Path(args.corpus)
    texts = []
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    print(f"[03-precompute-B] {len(texts)} examples loaded")

    print(f"[03-precompute-B] Precomputing → {out}")
    meta = precompute_aligned_cache(
        teacher_model=teacher,
        teacher_tokenizer=teacher_tok,
        student_tokenizer=student_tok,
        text_corpus=texts,
        output_path=str(out),
        top_k=args.top_k,
        alignment="student_offset",
        suffix_reencode=True,
        projection=None,            # same vocab — no projection needed
        project_at_write_time=False,
        max_seq_len=args.max_seq_len,
        device=args.device,
    )

    print("\n[03-precompute-B] Meta:")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
