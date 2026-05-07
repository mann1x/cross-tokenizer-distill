"""Batched eval — drop-in for 06_eval.py with --batch-size N greedy generation.

Same args + same SQLite cache schema as 06_eval.py. Only difference:
generation phase batches uncached prompts (left-pad, single
model.generate call per batch). Sandbox exec still per-task. Should
hit 3-5x wallclock speedup on under-utilized GPUs.
"""
from __future__ import annotations

import argparse, json, re, signal, sqlite3, sys, time
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


EXEC_TIMEOUT_S = 10


class ExecTimeout(Exception): ...


def _alarm_handler(signum, frame):
    raise ExecTimeout(f"exec exceeded {EXEC_TIMEOUT_S}s")


def _exec_with_timeout(code: str, label: str) -> tuple[bool, str]:
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(EXEC_TIMEOUT_S)
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            ns: dict = {}
            exec(code, ns)
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


THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
FENCE_RE_REGEX = re.compile(r"```(?:python)?\s*\n(.*?)\n```", re.DOTALL)

_FENCE_RE_PY = "```python"
_FENCE_RE = "```"

def strip_markdown_fences(s: str) -> str:
    s = s.rstrip()
    if s.startswith(_FENCE_RE_PY): s = s[len(_FENCE_RE_PY):].lstrip("\n")
    elif s.startswith(_FENCE_RE): s = s[len(_FENCE_RE):].lstrip("\n")
    if _FENCE_RE in s: s = s.split(_FENCE_RE)[0]
    return s


def strip_thinking(t):
    return THINK_RE.sub("", t).strip()

def extract_code_chat(t):
    """Extract the LAST python code block from a chat-mode generation.

    Chat models often emit explanatory snippets first and the actual
    answer in the final fence (audit csl-...-bd96). Falling back to
    `findall()[-1]` instead of `search()` recovers those cases.
    Empty fence list → fall back to the whole text (post think-strip).
    """
    t = strip_thinking(t)
    blocks = FENCE_RE_REGEX.findall(t)
    if blocks:
        return blocks[-1].strip()
    return t.strip()

_HE_STOPS = ("\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\nassert ", "\n```")

def truncate_at_function_end(s: str) -> str:
    stripped = s.lstrip()
    if stripped.startswith(("def ", "class ")):
        for stop in _HE_STOPS:
            j = s.find(stop, 1)
            if j != -1:
                s = s[:j]
                break
    else:
        for stop in _HE_STOPS:
            j = s.find(stop)
            if j != -1:
                s = s[:j]
                break
    return s


def score_humaneval(prob, generation, chat_mode=False):
    if chat_mode:
        code = extract_code_chat(generation)
        # Chat path was bypassing truncate (audit csl-...-bd96): trailing
        # prose after the function definition crashes exec(). Apply the same
        # function-end truncation as the raw path, but only when the chat
        # extraction begins with a definition (otherwise truncation would
        # eat the leading prompt-continuation case below).
        if code.lstrip().startswith(("def ", "class ")):
            code = truncate_at_function_end(code)
        entry = prob["entry_point"]
        if re.search(rf"^def\s+{re.escape(entry)}\b", code, re.M):
            full = code + "\n" + prob["test"] + f"\ncheck({entry})"
        else:
            full = prob["prompt"] + code + "\n" + prob["test"] + f"\ncheck({entry})"
    else:
        code = strip_markdown_fences(generation)
        code = truncate_at_function_end(code)
        full = prob["prompt"] + code + "\n" + prob["test"] + f"\ncheck({prob['entry_point']})"
    return _exec_with_timeout(full, prob["task_id"])


def score_mbpp(prob, generation, chat_mode=False):
    if chat_mode:
        code = extract_code_chat(generation)
        # Same fix as score_humaneval (audit csl-...-bd96): truncate trailing
        # prose when the extraction starts with a definition.
        if code.lstrip().startswith(("def ", "class ")):
            code = truncate_at_function_end(code)
    else:
        code = strip_markdown_fences(generation)
        code = truncate_at_function_end(code)
    tests = "\n".join(prob.get("test_list", []))
    full = code + "\n" + tests
    return _exec_with_timeout(full, str(prob.get("task_id", "?")))


def open_cache(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            task TEXT NOT NULL,
            task_id TEXT NOT NULL,
            generation TEXT,
            passed INTEGER,
            info TEXT,
            ts REAL,
            PRIMARY KEY (task, task_id)
        )
    """)
    conn.commit()
    return conn


def cache_done_ids(conn, task: str) -> dict[str, tuple[int, str]]:
    out = {}
    for row in conn.execute("SELECT task_id, passed, info FROM results WHERE task = ?", (task,)):
        out[row[0]] = (row[1], row[2] or "")
    return out


def cache_save(conn, task: str, task_id: str, generation: str, passed: bool, info: str):
    conn.execute(
        "INSERT OR REPLACE INTO results (task, task_id, generation, passed, info, ts) VALUES (?,?,?,?,?,?)",
        (task, str(task_id), generation, int(bool(passed)), info[:500], time.time()),
    )
    conn.commit()


@torch.no_grad()
def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int = 512) -> list[str]:
    """Left-pad prompts, generate greedy in one call, return decoded continuations."""
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False,
                    add_special_tokens=True).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=pad_id,
    )
    prompt_len = enc["input_ids"].shape[1]
    gen_only = out[:, prompt_len:]
    return [tokenizer.decode(gen_only[i], skip_special_tokens=True) for i in range(len(prompts))]


def _mbpp_prompt(prob: dict) -> str:
    desc = prob.get("prompt") or prob.get("text") or ""
    tests = prob.get("test_list", [])
    test_lines = "\n".join(tests[:1]) if tests else ""
    return f'"""\n{desc}\n{test_lines}\n"""\n'


def run_task_batched(name: str, model, tokenizer, conn, problems, prompt_fn, score_fn,
                     batch_size: int, max_new_tokens: int = 512) -> dict:
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
            prompts = [prompt_fn(p) for _, p in chunk]
            t0 = time.monotonic()
            try:
                gens = generate_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
            except Exception as e:
                print(f"  [{name}] batch GenError: {type(e).__name__}: {str(e)[:120]}", flush=True)
                gens = [""] * len(chunk)
            t_gen = time.monotonic() - t0
            for (i, prob), gen in zip(chunk, gens):
                tid = str(prob["task_id"])
                t1 = time.monotonic()
                try:
                    passed, info = score_fn(prob, gen)
                except Exception as e:
                    passed, info = False, f"ScoreError: {type(e).__name__}: {str(e)[:120]}"
                t_exec = time.monotonic() - t1
                cache_save(conn, name, tid, gen, passed, info)
                if passed: n_pass += 1
                n_seen += 1
                mark = "PASS" if passed else "FAIL"
                print(f"  {name.upper()} {n_seen}/{len(problems)} {tid:<14} {mark} "
                      f"gen={t_gen/len(chunk):4.1f}s exec={t_exec:5.2f}s "
                      f"running={n_pass}/{n_seen}={n_pass/n_seen:.1%} info={info[:60]}", flush=True)
    finally:
        tokenizer.padding_side = pad_side_orig

    rows = list(conn.execute("SELECT task_id, passed, info FROM results WHERE task=?", (name,)))
    n_total = len(rows)
    n_pass_total = sum(r[1] for r in rows)
    return {
        "n_total": n_total, "n_pass": n_pass_total,
        "pass@1": n_pass_total / max(1, n_total),
        "failures": [{"task_id": r[0], "info": r[2]} for r in rows if not r[1]],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="deepseek-ai/deepseek-coder-1.3b-instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-db", default=None)
    parser.add_argument("--he-limit", type=int, default=164)
    parser.add_argument("--mbpp-limit", type=int, default=378)
    parser.add_argument("--skip-humaneval", action="store_true")
    parser.add_argument("--skip-mbpp", action="store_true")
    parser.add_argument("--exec-timeout", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--quant", choices=["bf16", "nf4"], default="bf16")
    parser.add_argument("--chat-template", action="store_true",
                        help="Wrap each prompt as a user message via tokenizer.apply_chat_template before generation.")
    args = parser.parse_args()

    global EXEC_TIMEOUT_S
    EXEC_TIMEOUT_S = args.exec_timeout

    cache_path = args.cache_db or str(Path(args.output).with_suffix(".db"))
    print(f"[06-eval-b] SQLite cache: {cache_path} | batch_size={args.batch_size}")
    conn = open_cache(cache_path)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.quant == "nf4":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, quantization_config=bnb, device_map={"": "cuda"}, trust_remote_code=True)
    else:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    if args.adapter:
        print(f"[06-eval-b] Loading LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(base, args.adapter)
    else:
        print("[06-eval-b] BASE-ONLY mode (no adapter)")
        model = base
    model.eval()

    results = {"adapter": args.adapter, "cache_db": cache_path, "batch_size": args.batch_size}

    # If chat template is requested, wrap raw prompts as user messages before tokenization.
    def _maybe_chat(prompt: str) -> str:
        if args.chat_template:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        return prompt

    if not args.skip_humaneval:
        ds = load_dataset("openai/openai_humaneval", split="test")
        if args.he_limit: ds = ds.select(range(min(args.he_limit, len(ds))))
        problems = list(ds)
        print(f"[06-eval-b] HumanEval+: {len(problems)} problems")
        results["humaneval"] = run_task_batched(
            "humaneval", model, tokenizer, conn, problems,
            prompt_fn=lambda p: _maybe_chat(p["prompt"]),
            score_fn=lambda p, g: score_humaneval(p, g, chat_mode=args.chat_template),
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"  → pass@1 = {results['humaneval']['pass@1']:.1%}")

    if not args.skip_mbpp:
        ds = load_dataset("google-research-datasets/mbpp", split="test")
        if args.mbpp_limit: ds = ds.select(range(min(args.mbpp_limit, len(ds))))
        problems = list(ds)
        print(f"[06-eval-b] MBPP: {len(problems)} problems")
        results["mbpp"] = run_task_batched(
            "mbpp", model, tokenizer, conn, problems,
            prompt_fn=lambda p: _maybe_chat(_mbpp_prompt(p)),
            score_fn=lambda p, g: score_mbpp(p, g, chat_mode=args.chat_template),
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"  → pass@1 = {results['mbpp']['pass@1']:.1%}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[06-eval-b] DONE -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
