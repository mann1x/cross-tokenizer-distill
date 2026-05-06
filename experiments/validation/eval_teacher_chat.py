"""Eval a chat-template teacher (DeepCoder-14B etc) on HE+MBPP.

Same scoring as 06_eval_batched.py, but:
- Loads any HF model (no LoRA), uses chat template (apply_chat_template)
- Strips <think>...</think> blocks from generations before scoring
- Larger max_new_tokens budget for reasoning models
- SQLite cached + resumable
"""
from __future__ import annotations

import argparse, json, re, signal, sqlite3, sys, time
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/cross-tokenizer-distill")
from ctd.util import make_teacher_token_blacklist, bad_words_ids_for_generate

EXEC_TIMEOUT_S = 30
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)\n```", re.DOTALL)


class ExecTimeout(Exception): ...


def _alarm_handler(signum, frame):
    raise ExecTimeout(f"exec exceeded {EXEC_TIMEOUT_S}s")


def _exec_with_timeout(code: str) -> tuple[bool, str]:
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(EXEC_TIMEOUT_S)
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            exec(code, {})
        return True, ""
    except ExecTimeout as e:
        return False, f"Timeout: {e}"
    except AssertionError as e:
        return False, f"AssertionError: {str(e)[:120]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def strip_thinking(s: str) -> str:
    """Remove <think>...</think> blocks (reasoning model artifact)."""
    return THINK_RE.sub("", s).strip()


def extract_code(s: str) -> str:
    """Extract code from fenced markdown or return as-is if no fence."""
    s = strip_thinking(s)
    m = FENCE_RE.search(s)
    if m:
        return m.group(1).strip()
    return s.strip()


def score_humaneval(prob: dict, generation: str) -> tuple[bool, str]:
    code = extract_code(generation)
    # Heuristic: if code starts with `def <entry>`, it's a full function.
    # If it doesn't, treat as a body to append after the prompt.
    entry = prob["entry_point"]
    if re.search(rf"^def\s+{re.escape(entry)}\b", code, re.M):
        full = code + "\n" + prob["test"] + f"\ncheck({entry})"
    else:
        full = prob["prompt"] + code + "\n" + prob["test"] + f"\ncheck({entry})"
    return _exec_with_timeout(full)


def score_mbpp(prob: dict, generation: str) -> tuple[bool, str]:
    code = extract_code(generation)
    tests = "\n".join(prob.get("test_list", []))
    full = code + "\n" + tests
    return _exec_with_timeout(full)


def open_cache(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE IF NOT EXISTS results (
        task TEXT NOT NULL, task_id TEXT NOT NULL,
        generation TEXT, passed INTEGER, info TEXT, ts REAL,
        PRIMARY KEY (task, task_id))""")
    conn.commit()
    return conn


def cache_done_ids(conn, task: str) -> dict[str, tuple[int, str]]:
    return {row[0]: (row[1], row[2] or "") for row in
            conn.execute("SELECT task_id, passed, info FROM results WHERE task = ?", (task,))}


def cache_save(conn, task: str, task_id: str, generation: str, passed: bool, info: str):
    conn.execute("INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?)",
                 (task, str(task_id), generation, int(bool(passed)), info[:500], time.time()))
    conn.commit()


@torch.no_grad()
def generate_batch_chat(model, tokenizer, prompts: list[str], max_new_tokens: int,
                        bad_words: Optional[list[list[int]]] = None) -> list[str]:
    """Apply chat template + generate. No system prompt (per DeepCoder recs)."""
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    inputs = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt",
        padding=True, return_dict=True,
    ).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True, temperature=0.6, top_p=0.95,
        pad_token_id=pad_id,
        bad_words_ids=bad_words,
    )
    prompt_len = inputs["input_ids"].shape[1]
    gen_only = out[:, prompt_len:]
    return [tokenizer.decode(gen_only[i], skip_special_tokens=True) for i in range(len(prompts))]


def he_user_prompt(prob: dict) -> str:
    return (
        "Complete the following Python function. Return only the function "
        "definition (signature + body), no explanations.\n\n```python\n"
        + prob["prompt"] + "\n```"
    )


def mbpp_user_prompt(prob: dict) -> str:
    desc = prob.get("text") or prob.get("prompt") or ""
    tests = prob.get("test_list", [])
    test_lines = "\n".join(tests[:1])
    return (
        f"Write a Python function that solves the following problem:\n\n{desc}\n\n"
        f"It must pass this test:\n```python\n{test_lines}\n```\n\n"
        "Return only the function code, no explanations."
    )


def run_task(name, model, tokenizer, conn, problems, prompt_fn, score_fn,
             batch_size, max_new_tokens, bad_words=None):
    done = cache_done_ids(conn, name)
    print(f"  [{name}] cache hit: {len(done)}/{len(problems)} already done", flush=True)
    pending = [(i, p) for i, p in enumerate(problems) if str(p["task_id"]) not in done]
    n_pass = sum(1 for tid, (passed, _) in done.items() if passed)
    n_seen = len(done)
    pad_side_orig = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(pending), batch_size):
            chunk = pending[start:start+batch_size]
            user_prompts = [prompt_fn(p) for _, p in chunk]
            t0 = time.monotonic()
            try:
                gens = generate_batch_chat(model, tokenizer, user_prompts, max_new_tokens, bad_words=bad_words)
            except Exception as e:
                print(f"  [{name}] batch GenError: {type(e).__name__}: {str(e)[:120]}", flush=True)
                gens = [""] * len(chunk)
            t_gen = time.monotonic() - t0
            for (i, prob), gen in zip(chunk, gens):
                tid = str(prob["task_id"])
                try:
                    passed, info = score_fn(prob, gen)
                except Exception as e:
                    passed, info = False, f"ScoreError: {type(e).__name__}: {str(e)[:120]}"
                cache_save(conn, name, tid, gen, passed, info)
                if passed: n_pass += 1
                n_seen += 1
                mark = "PASS" if passed else "FAIL"
                print(f"  {name.upper()} {n_seen}/{len(problems)} {tid:<14} {mark} "
                      f"gen={t_gen/len(chunk):5.1f}s running={n_pass}/{n_seen}={n_pass/n_seen:.1%} "
                      f"info={info[:60]}", flush=True)
    finally:
        tokenizer.padding_side = pad_side_orig
    rows = list(conn.execute("SELECT passed FROM results WHERE task=?", (name,)))
    return {"n_total": len(rows), "n_pass": sum(r[0] for r in rows),
            "pass@1": sum(r[0] for r in rows) / max(1, len(rows))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id (no LoRA)")
    p.add_argument("--output", required=True)
    p.add_argument("--cache-db", default=None)
    p.add_argument("--he-limit", type=int, default=164)
    p.add_argument("--mbpp-limit", type=int, default=378)
    p.add_argument("--skip-humaneval", action="store_true")
    p.add_argument("--skip-mbpp", action="store_true")
    p.add_argument("--exec-timeout", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--mask-teacher-tokens", default="",
                   help="Comma-separated token strings to ban from generation. "
                        "For non-thinking eval of reasoning teachers, leave empty "
                        "(thinking blocks are stripped post-hoc by THINK_RE). "
                        "Use this to FORCE no-think mode (e.g. '<think>,</think>').")
    p.add_argument("--mask-teacher-token-ids", default="")
    args = p.parse_args()

    global EXEC_TIMEOUT_S
    EXEC_TIMEOUT_S = args.exec_timeout

    cache_path = args.cache_db or str(Path(args.output).with_suffix(".db"))
    print(f"[teacher-eval] cache: {cache_path} | bs={args.batch_size} | mnt={args.max_new_tokens}", flush=True)
    conn = open_cache(cache_path)

    print(f"[teacher-eval] loading {args.model} (bf16)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map={"": "cuda"})
    model.eval()

    blacklist_ids = make_teacher_token_blacklist(tokenizer, args.mask_teacher_tokens, args.mask_teacher_token_ids)
    bad_words = bad_words_ids_for_generate(blacklist_ids)
    if bad_words:
        print(f"[teacher-eval] banning {len(blacklist_ids)} token IDs from generation: "
              f"{blacklist_ids[:10]}{'...' if len(blacklist_ids) > 10 else ''}", flush=True)

    results = {"model": args.model, "cache_db": cache_path, "blacklist_ids": blacklist_ids}

    if not args.skip_humaneval:
        ds = load_dataset("openai/openai_humaneval", split="test")
        if args.he_limit: ds = ds.select(range(min(args.he_limit, len(ds))))
        problems = list(ds)
        print(f"[teacher-eval] HumanEval: {len(problems)}", flush=True)
        results["humaneval"] = run_task("humaneval", model, tokenizer, conn, problems,
                                        he_user_prompt, score_humaneval,
                                        args.batch_size, args.max_new_tokens, bad_words=bad_words)
        print(f"  → pass@1 = {results['humaneval']['pass@1']:.1%}")

    if not args.skip_mbpp:
        ds = load_dataset("google-research-datasets/mbpp", split="test")
        if args.mbpp_limit: ds = ds.select(range(min(args.mbpp_limit, len(ds))))
        problems = list(ds)
        print(f"[teacher-eval] MBPP: {len(problems)}", flush=True)
        results["mbpp"] = run_task("mbpp", model, tokenizer, conn, problems,
                                   mbpp_user_prompt, score_mbpp,
                                   args.batch_size, args.max_new_tokens, bad_words=bad_words)
        print(f"  → pass@1 = {results['mbpp']['pass@1']:.1%}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[teacher-eval] DONE -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
