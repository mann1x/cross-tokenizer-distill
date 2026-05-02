"""ctd-inspect — diagnostic CLI.

Build a VocabMapper for a given (teacher, student) tokenizer pair,
report coverage stats, and optionally evaluate position-alignment
hit-rate on a sample corpus.

Usage:
    python -m cli.inspect \\
        --teacher-tokenizer Qwen/Qwen3-Coder-Next \\
        --student-tokenizer deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \\
        [--strategies strict,distribute,first_token] \\
        [--sample-corpus path/to/sample.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="CTD diagnostic — coverage report.")
    parser.add_argument("--teacher-tokenizer", required=True, help="HF model id or path.")
    parser.add_argument("--student-tokenizer", required=True, help="HF model id or path.")
    parser.add_argument(
        "--strategies",
        default="distribute",
        help="Comma-separated multi-token strategies to evaluate.",
    )
    parser.add_argument(
        "--sample-corpus",
        default=None,
        help="Optional JSONL of sample texts (each line: {'text': '...'}). "
             "If provided, also reports position-alignment hit-rate.",
    )
    parser.add_argument(
        "--n-corpus-samples",
        type=int,
        default=20,
        help="Max number of texts from --sample-corpus to evaluate.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoTokenizer.",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed.", file=sys.stderr)
        return 1

    print(f"[ctd-inspect] Loading teacher tokenizer: {args.teacher_tokenizer}")
    teacher_tok = AutoTokenizer.from_pretrained(
        args.teacher_tokenizer, trust_remote_code=args.trust_remote_code
    )
    print(f"[ctd-inspect] Loading student tokenizer: {args.student_tokenizer}")
    student_tok = AutoTokenizer.from_pretrained(
        args.student_tokenizer, trust_remote_code=args.trust_remote_code
    )

    from ctd.mapper import VocabMapper

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    reports: dict = {}

    for strategy in strategies:
        print(f"\n=== Building VocabMapper(strategy={strategy}) ===")
        mapper = VocabMapper.from_tokenizers(
            teacher_tokenizer=teacher_tok,
            student_tokenizer=student_tok,
            multi_token=strategy,
            progress=True,
        )
        rep = mapper.coverage_report()
        print(rep)
        reports[strategy] = {
            "single_token_rate": rep.single_token_rate,
            "multi_token_rate": rep.multi_token_rate,
            "dropped_rate": rep.dropped_rate,
            "coverage": rep.coverage,
            "avg_multi_token_fragments": rep.avg_multi_token_fragments,
            "bytewise_roundtrip_ok": rep.bytewise_roundtrip_ok,
            "roundtrip_failures": rep.roundtrip_failures,
        }

    if args.sample_corpus:
        from ctd.alignment import build_alignment

        corpus_path = Path(args.sample_corpus)
        if not corpus_path.exists():
            print(f"WARN: --sample-corpus {corpus_path} not found, skipping alignment eval.")
        else:
            print(f"\n=== Position-alignment hit-rate on {corpus_path} "
                  f"(first {args.n_corpus_samples} samples) ===")
            n_aligned_total = 0
            n_suffix_total = 0
            n_dropped_total = 0
            n_examples = 0

            with open(corpus_path) as f:
                for line in f:
                    if n_examples >= args.n_corpus_samples:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        text = rec.get("text") or rec.get("content") or ""
                    except json.JSONDecodeError:
                        text = line
                    if not text:
                        continue
                    t_ids = teacher_tok.encode(text, add_special_tokens=False)
                    s_ids = student_tok.encode(text, add_special_tokens=False)
                    if not t_ids or not s_ids:
                        continue
                    table = build_alignment(
                        text=text,
                        teacher_token_ids=t_ids,
                        student_token_ids=s_ids,
                        teacher_tokenizer=teacher_tok,
                        student_tokenizer=student_tok,
                        mode="student_offset",
                        suffix_reencode=True,
                    )
                    n_aligned_total += table.n_aligned
                    n_suffix_total += table.n_suffix
                    n_dropped_total += table.n_dropped
                    n_examples += 1

            n_pos = n_aligned_total + n_suffix_total + n_dropped_total
            if n_pos > 0:
                print(f"  examples evaluated:   {n_examples}")
                print(f"  total student positions: {n_pos}")
                print(f"  cleanly aligned:      {100 * n_aligned_total / n_pos:.1f}%  ({n_aligned_total})")
                print(f"  suffix re-encode:     {100 * n_suffix_total / n_pos:.1f}%  ({n_suffix_total})")
                print(f"  dropped (mode=byte):  {100 * n_dropped_total / n_pos:.1f}%  ({n_dropped_total})")

    print("\n=== Recommendation ===")
    distribute = reports.get("distribute", {})
    if distribute:
        cov = distribute["coverage"]
        rt_ok = distribute["bytewise_roundtrip_ok"]
        if not rt_ok:
            print("  NO-GO — byte-roundtrip failed too often. Check tokenizer normalisation.")
            return 1
        if cov < 0.5:
            print(f"  CAUTION — coverage {cov:.0%} <50%. Cross-vocab projection may degrade signal.")
        elif cov < 0.8:
            print(f"  GO with caution — coverage {cov:.0%} (50-80%). Validate on small models first.")
        else:
            print(f"  GO — coverage {cov:.0%} with strategy=distribute. CTD viable for this pair.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
