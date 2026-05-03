# Validation experiment — small-models 3-way A/B/C

See `../../docs/VALIDATION.md` for the full design doc. This README is
the operational quick-reference.

## Run order

```bash
# 0. Setup
conda create -n ctd python=3.11
conda activate ctd
pip install -e ../..
pip install datasets accelerate bitsandbytes

# 1. Diagnostic — confirm vocab pair is workable
bash 01_inspect.sh

# 2. Build same-vocab teacher cache (Qwen2.5-Coder-7B → Qwen2.5-Coder-0.5B)
bash 02_precompute_B.sh

# 3. Build CTD cache (DS-Coder-V2-Lite → Qwen2.5-Coder-0.5B via CTD)
bash 03_precompute_C.sh

# 4-6. Three training runs (sequential on solidPC 3090)
bash 04_train_A.sh   # SFT only
bash 05_train_B.sh   # same-vocab distill
bash 06_train_C.sh   # CTD distill

# 7. Eval all three
bash 07_eval_all.sh

# 8. Decision report
python 08_compare.py
```

## Status

Stub only. Run scripts to be filled in once core CTD modules are
implemented (tasks #75 and #72).

## `06_eval.py` — resumable single-run evaluator

Evaluates one trained adapter on HumanEval+ and MBPP+ via greedy
single-attempt generation, scored with `exec()`. Two operational
properties matter when running this on a flaky pod:

### SQLite cache (resume)

Every per-problem result is committed to a SQLite cache the moment it's
scored — generation text + pass/fail + info string. On restart, all
`task_id`s already in the cache are skipped, so you pick up exactly
where you stopped. Kill the run any time; rerun with the same
`--cache-db` and it continues at the next un-evaluated problem.

```bash
python -u 06_eval.py \
    --adapter runs/run_C \
    --output  results/C.json \
    --cache-db results/C.db \
    --he-limit 164 --mbpp-limit 378 \
    --exec-timeout 30
```

`--cache-db` defaults to `<output stem>.db` next to `--output` if
omitted. The JSON summary is rewritten at the end from the full cache
state (so partial summaries stay consistent if you stop mid-run and
re-summarise later by re-running with `--skip-humaneval --skip-mbpp`).

### Per-problem `exec()` timeout

`score_humaneval` / `score_mbpp` run `exec(prompt + generation +
test)`. A generation containing an infinite loop (`while True`,
unbounded recursion) hangs the worker thread at 100 % CPU forever —
this happened on run_C HE/30 and burned 71 minutes of wall-clock
before being noticed. `--exec-timeout` (default `30` seconds) wraps
each `exec()` under `SIGALRM`; on expiry the problem is recorded as a
`Timeout: …` failure and the loop continues. Mirrors the EvalPlus
default for MBPP+ extended tests.

### Per-problem progress logging

Every problem prints a single line:

```
HE 14/164 HumanEval/13   PASS gen= 17.8s exec= 0.00s running=12/14=85.7% info=ok
```

`gen=` is generation wall-time, `exec=` is the test-execution
wall-time, `running=` is the cumulative pass@1 over evaluated
problems. Run with `python -u` (or `PYTHONUNBUFFERED=1`) so the
log streams in real time under `nohup`/`tee` and remote monitors can
react to milestones immediately.
