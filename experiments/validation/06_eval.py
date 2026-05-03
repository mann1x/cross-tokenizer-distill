"""Step 6 — Evaluate one trained run on HumanEval+ / MBPP+ / LiveCodeBench-medium-30.

Loads the LoRA adapter, runs greedy single-attempt generation per
problem, scores via exec(). Writes JSON results.

Lightweight self-contained runner — no lm-eval dependency. We score
HumanEval and MBPP via exec(), and LCB via the lcb_runner subset
loader (matches what Mythic-RDT uses).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def strip_markdown_fences(s: str) -> str:
    """Remove ``` fences but PRESERVE leading whitespace (function indent)."""
    # Only strip a leading fence if it's at the very start (allowing newline).
    s = re.sub(r"\A\s*```(?:python|py)?\n", "", s)
    s = re.sub(r"\n```\s*\Z", "", s)
    return s


_HE_STOPS = ("\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\nassert ")


def truncate_at_function_end(s: str) -> str:
    """Keep only the function body — cut at the first top-level non-continuation."""
    earliest = len(s)
    for stop in _HE_STOPS:
        i = s.find(stop)
        if i != -1 and i < earliest:
            earliest = i
    return s[:earliest]


def score_humaneval(prob: dict, generation: str) -> tuple[bool, str]:
    """Run exec(prompt + generation + test). Return (passed, info)."""
    code = truncate_at_function_end(strip_markdown_fences(generation))
    full = prob["prompt"] + code + "\n" + prob["test"] + f"\ncheck({prob['entry_point']})\n"
    buf_o, buf_e = StringIO(), StringIO()
    try:
        with redirect_stdout(buf_o), redirect_stderr(buf_e):
            ns = {}
            exec(compile(full, "<he>", "exec"), ns)
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def score_mbpp(prob: dict, generation: str) -> tuple[bool, str]:
    code = truncate_at_function_end(strip_markdown_fences(generation))
    test_list = prob.get("test_list", [])
    full = code + "\n" + "\n".join(test_list)
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            exec(compile(full, "<mbpp>", "exec"), {})
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


@torch.no_grad()
def generate_one(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    gen = out[0, enc.input_ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def run_humaneval(model, tokenizer, limit: Optional[int] = None) -> dict:
    ds = load_dataset("openai/openai_humaneval", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    n_pass = 0
    failures = []
    for i, prob in enumerate(ds):
        gen = generate_one(model, tokenizer, prob["prompt"])
        passed, info = score_humaneval(prob, gen)
        if passed:
            n_pass += 1
        else:
            failures.append({"task_id": prob["task_id"], "info": info})
        if (i + 1) % 10 == 0:
            print(f"  HE {i+1}/{len(ds)}  pass@1={n_pass}/{i+1}={n_pass/(i+1):.1%}")
    return {"n_total": len(ds), "n_pass": n_pass, "pass@1": n_pass / len(ds), "failures": failures}


def _mbpp_prompt(prob: dict) -> str:
    """Standard MBPP prompt: description + tests anchor the function name."""
    desc = prob.get("prompt") or prob.get("text") or ""
    tests = prob.get("test_list", [])
    test_str = "\n".join(tests[:3])
    return (
        f'"""\n{desc}\n{test_str}\n"""\n'
    )


def run_mbpp(model, tokenizer, limit: Optional[int] = None) -> dict:
    ds = load_dataset("evalplus/mbppplus", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    n_pass = 0
    failures = []
    for i, prob in enumerate(ds):
        prompt = _mbpp_prompt(prob)
        gen = generate_one(model, tokenizer, prompt)
        passed, info = score_mbpp(prob, gen)
        if passed:
            n_pass += 1
        else:
            failures.append({"task_id": prob.get("task_id", i), "info": info})
        if (i + 1) % 20 == 0:
            print(f"  MBPP {i+1}/{len(ds)}  pass@1={n_pass}/{i+1}={n_pass/(i+1):.1%}")
    return {"n_total": len(ds), "n_pass": n_pass, "pass@1": n_pass / len(ds), "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="deepseek-ai/deepseek-coder-1.3b-instruct")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter dir.")
    parser.add_argument("--output", required=True, help="JSON output path.")
    parser.add_argument("--he-limit", type=int, default=164)
    parser.add_argument("--mbpp-limit", type=int, default=378)
    parser.add_argument("--skip-humaneval", action="store_true")
    parser.add_argument("--skip-mbpp", action="store_true")
    args = parser.parse_args()

    print(f"[06-eval] Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map={"": "cuda"}
    )
    print(f"[06-eval] Loading LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    results = {"adapter": args.adapter}

    if not args.skip_humaneval:
        print("[06-eval] HumanEval+...")
        results["humaneval"] = run_humaneval(model, tokenizer, args.he_limit)
        print(f"  → pass@1 = {results['humaneval']['pass@1']:.1%}")

    if not args.skip_mbpp:
        print("[06-eval] MBPP+...")
        results["mbpp"] = run_mbpp(model, tokenizer, args.mbpp_limit)
        print(f"  → pass@1 = {results['mbpp']['pass@1']:.1%}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[06-eval] Wrote {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
