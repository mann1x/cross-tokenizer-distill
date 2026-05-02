"""ctd-inspect — diagnostic CLI.

Build a VocabMapper for a given (teacher, student) tokenizer pair,
report coverage stats, and recommend GO / NO-GO before committing
to a full precompute run.

Usage:
    ctd-inspect --teacher-tokenizer Qwen/Qwen3-Coder-Next \\
                --student-tokenizer deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \\
                [--strategies strict,distribute,first_token] \\
                [--sample-corpus path/to/sample.jsonl]
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="CTD diagnostic — coverage report.")
    parser.add_argument("--teacher-tokenizer", required=True, help="HF model id or path.")
    parser.add_argument("--student-tokenizer", required=True, help="HF model id or path.")
    parser.add_argument(
        "--strategies",
        default="strict,distribute,first_token",
        help="Comma-separated multi-token strategies to evaluate.",
    )
    parser.add_argument(
        "--sample-corpus",
        default=None,
        help="Optional JSONL of sample texts to also report position-alignment "
             "hit-rate on (each line: {'text': '...'}).",
    )
    args = parser.parse_args()
    print(f"[ctd-inspect] teacher={args.teacher_tokenizer} student={args.student_tokenizer}")
    raise SystemExit("ctd-inspect not yet implemented — depends on ctd.mapper.VocabMapper.")


if __name__ == "__main__":
    raise SystemExit(main())
