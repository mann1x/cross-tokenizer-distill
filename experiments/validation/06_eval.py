"""Step 6 — Evaluate one trained run on HumanEval+ / MBPP+.

Loads the LoRA adapter, runs greedy single-attempt generation per
problem, scores via exec() with a per-problem timeout, persists every
sample (generation + verdict) to a SQLite cache, and writes a JSON
summary at the end.

Resumability: on restart, all task_ids already in the cache are
skipped. Kill it any time; rerun with the same --cache-db to pick up
where it stopped.

Hang protection: each exec() runs under SIGALRM. Generated code that
loops forever is killed after EXEC_TIMEOUT_S seconds and recorded as
a failure. Without this, a single bad generation freezes the whole
eval (saw HE/30 hang at 100% CPU for 3+ hours).
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import sqlite3
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


EXEC_TIMEOUT_S = 10


class ExecTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise ExecTimeout(f"exec exceeded {EXEC_TIMEOUT_S}s")


def _exec_with_timeout(code: str, label: str) -> tuple[bool, str]:
    """Run code in a fresh namespace under SIGALRM. Returns (passed, info)."""
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(EXEC_TIMEOUT_S)
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            ns: dict = {}
            exec(compile(code, label, "exec"), ns)
        return True, "ok"
    except ExecTimeout as e:
        return False, f"Timeout: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def strip_markdown_fences(s: str) -> str:
    """Remove ``` fences but PRESERVE leading whitespace (function indent)."""
    s = re.sub(r"\A\s*```(?:python|py)?\n", "", s)
    s = re.sub(r"\n```\s*\Z", "", s)
    return s


_HE_STOPS = ("\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\nassert ", "\n```")


def truncate_at_function_end(s: str) -> str:
    """Keep the (possibly leading) def + body — cut at next top-level marker."""
    start = 0
    stripped = s.lstrip("\n ")
    if stripped.startswith(("def ", "class ")):
        leading_offset = len(s) - len(stripped)
        nl = s.find("\n", leading_offset)
        start = nl + 1 if nl != -1 else len(s)

    earliest = len(s)
    for stop in _HE_STOPS:
        i = s.find(stop, start)
        if i != -1 and i < earliest:
            earliest = i
    return s[:earliest]


def score_humaneval(prob: dict, generation: str) -> tuple[bool, str]:
    code = truncate_at_function_end(strip_markdown_fences(generation))
    full = prob["prompt"] + code + "\n" + prob["test"] + f"\ncheck({prob['entry_point']})\n"
    return _exec_with_timeout(full, "<he>")


def score_mbpp(prob: dict, generation: str) -> tuple[bool, str]:
    code = truncate_at_function_end(strip_markdown_fences(generation))
    test_list = prob.get("test_list", [])
    full = code + "\n" + "\n".join(test_list)
    return _exec_with_timeout(full, "<mbpp>")


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


# -------------------- SQLite cache --------------------

def open_cache(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS results ("
        "  task TEXT NOT NULL,"
        "  task_id TEXT NOT NULL,"
        "  generation TEXT,"
        "  passed INTEGER,"
        "  info TEXT,"
        "  PRIMARY KEY (task, task_id)"
        ")"
    )
    # Durable but not paranoid: WAL + NORMAL syncs survive process kill.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.commit()
    return conn


def cache_done_ids(conn: sqlite3.Connection, task: str) -> dict[str, tuple[int, str]]:
    return {
        row[0]: (row[1], row[2])
        for row in conn.execute(
            "SELECT task_id, passed, info FROM results WHERE task = ?", (task,)
        )
    }


def cache_save(
    conn: sqlite3.Connection,
    task: str,
    task_id: str,
    generation: str,
    passed: bool,
    info: str,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO results (task, task_id, generation, passed, info) "
        "VALUES (?, ?, ?, ?, ?)",
        (task, task_id, generation, int(passed), info),
    )
    conn.commit()


# -------------------- Runners --------------------

def run_humaneval(model, tokenizer, conn, limit: Optional[int] = None) -> dict:
    ds = load_dataset("openai/openai_humaneval", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    done = cache_done_ids(conn, "humaneval")
    print(f"  [humaneval] cache hit: {len(done)}/{len(ds)} already done", flush=True)

    n_pass = 0
    n_seen = 0
    for i, prob in enumerate(ds):
        tid = prob["task_id"]
        if tid in done:
            passed, _info = done[tid]
            if passed:
                n_pass += 1
            n_seen += 1
            continue
        t0 = time.monotonic()
        try:
            gen = generate_one(model, tokenizer, prob["prompt"])
            t_gen = time.monotonic() - t0
            t1 = time.monotonic()
            passed, info = score_humaneval(prob, gen)
            t_exec = time.monotonic() - t1
        except Exception as e:
            gen = ""
            t_gen = time.monotonic() - t0
            t_exec = 0.0
            passed, info = False, f"GenError: {type(e).__name__}: {str(e)[:120]}"
            print(f"  HE {i+1}/{len(ds)} GENERATION ERROR on {tid}: {info}", flush=True)
        cache_save(conn, "humaneval", tid, gen, passed, info)
        if passed:
            n_pass += 1
        n_seen += 1
        mark = "PASS" if passed else "FAIL"
        print(f"  HE {i+1}/{len(ds)} {tid:<14} {mark} gen={t_gen:5.1f}s exec={t_exec:5.2f}s "
              f"running={n_pass}/{n_seen}={n_pass/n_seen:.1%} info={info[:60]}", flush=True)

    rows = list(conn.execute(
        "SELECT task_id, passed, info FROM results WHERE task='humaneval'"
    ))
    n_total_rows = len(rows)
    n_pass_total = sum(r[1] for r in rows)
    failures = [{"task_id": r[0], "info": r[2]} for r in rows if not r[1]]
    return {
        "n_total": n_total_rows,
        "n_pass": n_pass_total,
        "pass@1": n_pass_total / max(1, n_total_rows),
        "failures": failures,
    }


def _mbpp_prompt(prob: dict) -> str:
    desc = prob.get("prompt") or prob.get("text") or ""
    tests = prob.get("test_list", [])
    test_str = "\n".join(tests[:3])
    return f'"""\n{desc}\n{test_str}\n"""\n'


def run_mbpp(model, tokenizer, conn, limit: Optional[int] = None) -> dict:
    ds = load_dataset("evalplus/mbppplus", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    done = cache_done_ids(conn, "mbpp")
    print(f"  [mbpp] cache hit: {len(done)}/{len(ds)} already done", flush=True)

    n_pass = 0
    n_seen = 0
    for i, prob in enumerate(ds):
        tid = str(prob.get("task_id", i))
        if tid in done:
            passed, _info = done[tid]
            if passed:
                n_pass += 1
            n_seen += 1
            continue
        prompt = _mbpp_prompt(prob)
        t0 = time.monotonic()
        try:
            gen = generate_one(model, tokenizer, prompt)
            t_gen = time.monotonic() - t0
            t1 = time.monotonic()
            passed, info = score_mbpp(prob, gen)
            t_exec = time.monotonic() - t1
        except Exception as e:
            gen = ""
            t_gen = time.monotonic() - t0
            t_exec = 0.0
            passed, info = False, f"GenError: {type(e).__name__}: {str(e)[:120]}"
            print(f"  MBPP {i+1}/{len(ds)} GENERATION ERROR on {tid}: {info}", flush=True)
        cache_save(conn, "mbpp", tid, gen, passed, info)
        if passed:
            n_pass += 1
        n_seen += 1
        mark = "PASS" if passed else "FAIL"
        print(f"  MBPP {i+1}/{len(ds)} tid={tid:<6} {mark} gen={t_gen:5.1f}s exec={t_exec:5.2f}s "
              f"running={n_pass}/{n_seen}={n_pass/n_seen:.1%} info={info[:60]}", flush=True)

    rows = list(conn.execute(
        "SELECT task_id, passed, info FROM results WHERE task='mbpp'"
    ))
    n_total_rows = len(rows)
    n_pass_total = sum(r[1] for r in rows)
    failures = [{"task_id": r[0], "info": r[2]} for r in rows if not r[1]]
    return {
        "n_total": n_total_rows,
        "n_pass": n_pass_total,
        "pass@1": n_pass_total / max(1, n_total_rows),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="deepseek-ai/deepseek-coder-1.3b-instruct")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter dir.")
    parser.add_argument("--output", required=True, help="JSON summary output path.")
    parser.add_argument(
        "--cache-db",
        default=None,
        help="SQLite cache for resumable per-sample storage. "
             "Default: <output stem>.db next to --output.",
    )
    parser.add_argument("--he-limit", type=int, default=164)
    parser.add_argument("--mbpp-limit", type=int, default=378)
    parser.add_argument("--skip-humaneval", action="store_true")
    parser.add_argument("--skip-mbpp", action="store_true")
    parser.add_argument("--exec-timeout", type=int, default=10,
                        help="Per-problem exec() timeout in seconds.")
    parser.add_argument("--no-load-model", action="store_true",
                        help="Test mode: skip model load (used by self-test).")
    args = parser.parse_args()

    global EXEC_TIMEOUT_S
    EXEC_TIMEOUT_S = args.exec_timeout

    cache_path = args.cache_db or str(Path(args.output).with_suffix(".db"))
    print(f"[06-eval] SQLite cache: {cache_path}")
    conn = open_cache(cache_path)

    if not args.no_load_model:
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
    else:
        model = tokenizer = None

    results = {"adapter": args.adapter, "cache_db": cache_path}

    if not args.skip_humaneval:
        print("[06-eval] HumanEval+...")
        results["humaneval"] = run_humaneval(model, tokenizer, conn, args.he_limit)
        print(f"  → pass@1 = {results['humaneval']['pass@1']:.1%} "
              f"({results['humaneval']['n_pass']}/{results['humaneval']['n_total']})")

    if not args.skip_mbpp:
        print("[06-eval] MBPP+...")
        results["mbpp"] = run_mbpp(model, tokenizer, conn, args.mbpp_limit)
        print(f"  → pass@1 = {results['mbpp']['pass@1']:.1%} "
              f"({results['mbpp']['n_pass']}/{results['mbpp']['n_total']})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[06-eval] Wrote {args.output}")
    conn.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
