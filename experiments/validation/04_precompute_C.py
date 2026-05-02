"""Step 4 — build CTD cross-vocab teacher cache (Pair C).

Teacher: DeepSeek-Coder-V2-Lite-Instruct (vocab 102400, NF4).
Student: Qwen2.5-Coder-0.5B-Instruct (vocab 152064).

Different vocabs → CTD VocabMapper projects teacher's top-K logits
onto student vocab at precompute time. Cache is drop-in compatible
with same-vocab cache built in step 3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ctd.mapper import VocabMapper
from ctd.precompute import precompute_aligned_cache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus_5k.jsonl")
    parser.add_argument("--output", default="cache_C/dscoder_v2_lite_via_ctd_top32.pt")
    parser.add_argument("--teacher", default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    parser.add_argument("--student-tokenizer", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--multi-token", default="distribute",
                        choices=["strict", "distribute", "first_token"])
    parser.add_argument("--alignment", default="student_offset",
                        choices=["byte_anchor", "student_offset"])
    parser.add_argument("--quant", choices=["none", "nf4"], default="nf4")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[04-precompute-C] Loading student tokenizer: {args.student_tokenizer}")
    student_tok = AutoTokenizer.from_pretrained(args.student_tokenizer)

    print(f"[04-precompute-C] Loading teacher: {args.teacher} (quant={args.quant})")
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)

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
        trust_remote_code=True,
        device_map={"": args.device} if args.quant == "none" else "auto",
        **quant_kwargs,
    )
    teacher.eval()

    print(f"[04-precompute-C] Building VocabMapper "
          f"(teacher V={teacher_tok.vocab_size}, student V={student_tok.vocab_size}, "
          f"strategy={args.multi_token})")
    mapper = VocabMapper.from_tokenizers(
        teacher_tokenizer=teacher_tok,
        student_tokenizer=student_tok,
        multi_token=args.multi_token,
    )
    rep = mapper.coverage_report()
    print(rep)
    if rep.coverage < 0.5:
        print(f"[04-precompute-C] WARNING: coverage {rep.coverage:.0%} <50%. "
              "CTD signal will be heavily attenuated. Consider 'distribute' strategy.")

    print(f"[04-precompute-C] Loading corpus: {args.corpus}")
    texts = []
    with open(args.corpus) as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    print(f"[04-precompute-C] {len(texts)} examples")

    print(f"[04-precompute-C] Precomputing CTD cache → {out}")
    meta = precompute_aligned_cache(
        teacher_model=teacher,
        teacher_tokenizer=teacher_tok,
        student_tokenizer=student_tok,
        text_corpus=texts,
        output_path=str(out),
        top_k=args.top_k,
        alignment=args.alignment,
        suffix_reencode=True,
        projection=mapper,
        project_at_write_time=True,    # bake projection into cache
        max_seq_len=args.max_seq_len,
        device=args.device,
    )

    print("\n[04-precompute-C] Meta:")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print(f"\n[04-precompute-C] Suffix re-encode rate: "
          f"{meta['n_suffix_reencode'] / max(meta['n_total_tokens'], 1):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
