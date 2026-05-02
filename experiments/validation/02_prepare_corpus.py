"""Step 2 — prepare the training corpus.

Sample 5K examples from Magicoder-OSS-Instruct-75K. Save as a single
JSONL with one 'text' field per line — concatenated instruction +
response. Same seed for all three runs (A/B/C) so they see identical
data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/corpus_5k.jsonl")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=4000,
                        help="Drop examples longer than this (post-format).")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("[02-prepare] Loading Magicoder-OSS-Instruct-75K...")
    ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    print(f"[02-prepare] Loaded {len(ds)} examples")

    ds = ds.shuffle(seed=args.seed)

    n_kept = 0
    n_dropped_long = 0
    with open(out, "w") as f:
        for ex in ds:
            if n_kept >= args.n_samples:
                break
            problem = ex.get("problem") or ex.get("instruction", "")
            solution = ex.get("solution") or ex.get("response", "")
            text = (
                f"### Problem\n{problem}\n\n"
                f"### Solution\n{solution}\n"
            )
            if len(text) > args.max_chars:
                n_dropped_long += 1
                continue
            f.write(json.dumps({"text": text}) + "\n")
            n_kept += 1

    print(f"[02-prepare] Wrote {n_kept} examples to {out}")
    print(f"[02-prepare] Dropped {n_dropped_long} examples >{args.max_chars} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
