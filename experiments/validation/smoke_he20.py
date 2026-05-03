"""HE-20 smoke control for distill recipe iteration.

20 hand-picked HumanEval problems chosen from the eval3 A/B run:
- 5 problems both A and B pass (success controls — recipe must keep them)
- 2 problems A fails, B passes (signal floor — recipe must keep these wins)
- 6 problems A passes, B fails (regression set — better recipe should recover them)
- 7 problems both fail (stretch set — best recipes might crack one or two)

Usage:
    python smoke_he20.py --adapter runs/run_X --output results/smoke_X.json

Cost ~3 min per run on RTX 6000 Ada vs ~25 min for full HE-164.
"""

from __future__ import annotations
import argparse
import json
import sys
from importlib import import_module
from pathlib import Path
from datasets import load_dataset

# Indices chosen from eval3 results/A.json + results/B.json
SUCCESS_CONTROLS = ["HumanEval/0", "HumanEval/1", "HumanEval/2", "HumanEval/3", "HumanEval/4"]
B_IMPROVES = ["HumanEval/105", "HumanEval/127"]
B_REGRESSIONS = ["HumanEval/62", "HumanEval/65", "HumanEval/67", "HumanEval/73",
                 "HumanEval/120", "HumanEval/152"]
BOTH_FAIL_STRETCH = ["HumanEval/32", "HumanEval/38", "HumanEval/50",
                     "HumanEval/83", "HumanEval/95", "HumanEval/107", "HumanEval/132"]
SMOKE_IDS = SUCCESS_CONTROLS + B_IMPROVES + B_REGRESSIONS + BOTH_FAIL_STRETCH


def main() -> int:
    sys.path.insert(0, str(Path(__file__).parent))
    ev = import_module("06_eval")

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="deepseek-ai/deepseek-coder-1.3b-instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map={"": "cuda"}
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    ds = load_dataset("openai/openai_humaneval", split="test")
    by_id = {p["task_id"]: p for p in ds}

    results = {"adapter": args.adapter, "buckets": {}, "details": []}
    buckets = {
        "success_control": SUCCESS_CONTROLS,
        "b_improves": B_IMPROVES,
        "b_regressions": B_REGRESSIONS,
        "both_fail_stretch": BOTH_FAIL_STRETCH,
    }
    n_pass_total = 0
    for bucket_name, ids in buckets.items():
        n_pass = 0
        for tid in ids:
            prob = by_id[tid]
            gen = ev.generate_one(model, tok, prob["prompt"])
            ok, info = ev.score_humaneval(prob, gen)
            if ok:
                n_pass += 1
                n_pass_total += 1
            results["details"].append({"task_id": tid, "bucket": bucket_name,
                                       "passed": ok, "info": info[:100]})
        results["buckets"][bucket_name] = {"n_total": len(ids), "n_pass": n_pass}
        print(f"  {bucket_name}: {n_pass}/{len(ids)}")
    results["overall"] = {"n_total": len(SMOKE_IDS), "n_pass": n_pass_total,
                          "pass@1": n_pass_total / len(SMOKE_IDS)}
    print(f"  TOTAL: {n_pass_total}/{len(SMOKE_IDS)} = {n_pass_total / len(SMOKE_IDS):.1%}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
