# Smoke procedure for SFT-on-teacher experiments

Derived from M21/M22/M23 failures (2026-05-06). The shallow "peek 2
completions" smoke I ran on M22 missed three module-rewriter recipes
that collapsed at full eval (HE 0-32%, eval-distribution mismatch).

## The mistake to avoid

Validating that **teacher outputs match teacher-prompt format** is
*not* the same as validating that **a student trained on those
outputs will produce eval-distribution-correct text on
eval-distribution prompts**. The two questions are genuinely
different and both must be answered before sinking 30+ min of gen
+ train + eval into a recipe.

## Required smoke for any new SFT-on-teacher gen

Run all four checks before launching the train. Each is cheap; the
combined cost is <2 min.

### 1. Full-completion peek (≥5 random samples)

`head -1` reading is too short — pollution often comes after the
first 200 chars (M22's `if __name__ == "__main__":\n    doctest`
killer was at byte 220+).

```bash
python3 -c "
import json, random
random.seed(0)
recs = [json.loads(l) for l in open('data/<teacher>.jsonl')]
for r in random.sample(recs, k=5):
    print('=== task ===')
    print('PROMPT (first 200):', repr(r['prompt'][:200]))
    print('COMPLETION (FULL):', repr(r['teacher_completion']))
    print()
"
```

Look for: trailing `if __name__`, `import doctest`, test functions,
class definitions where you expected a body, `pass`-bodied stubs.

### 2. Distribution-level pollution check

Eyeballing 5 samples doesn't catch a 1-3% pollution rate that still
trains the student to emit pollution 1-3% of the time. Compute these
fractions over the FULL completion set:

```bash
python3 -c "
import json
recs = [json.loads(l) for l in open('data/<teacher>.jsonl')]
n = len(recs)
def frac(pred): return sum(1 for r in recs if pred(r['teacher_completion'])) / n
print(f'n={n}')
print(f'  ends with pass:               {frac(lambda c: c.rstrip().endswith(\"pass\")):.1%}')
print(f'  contains __main__:            {frac(lambda c: \"__main__\" in c):.1%}')
print(f'  contains doctest.testmod:     {frac(lambda c: \"doctest.testmod\" in c):.1%}')
print(f'  starts with from/import:      {frac(lambda c: c.lstrip().startswith((\"from \", \"import \"))):.1%}')
print(f'  starts with class:            {frac(lambda c: c.lstrip().startswith(\"class \")):.1%}')
print(f'  starts with def:              {frac(lambda c: c.lstrip().startswith(\"def \")):.1%}')
print(f'  mean length (chars):          {sum(len(r[\"teacher_completion\"]) for r in recs)/n:.0f}')
"
```

Flag thresholds (any one triggers an abort + corpus reformat):
- `pass`-ending: > 5%  → teacher writing stubs
- `__main__`: > 1%  → test-runner pollution
- `from`/`import` start: > 30% on function-sig prompts → module rewrite
  (acceptable on docstring-only prompts where you EXPECT module shape)
- `class` start: > 5% on function-sig prompts → helper-class generation

### 3. Tiny-train + eval probe (the empirical check)

The cheapest way to validate "student trained on this will succeed"
is to actually train a tiny version and check the student's HE-eval
output shape. Cost: ~3-5 min.

```bash
# Train tiny adapter on first 50 prompts × 1 epoch
python -u 06_train_sft_on_teacher.py --output-dir runs/_smoke/SMOKE_TINY \
    --student deepseek-ai/deepseek-coder-1.3b-instruct \
    --corpus data/<teacher>.jsonl \
    --max-prompt-len 384 --max-total-len 768 \
    --lr 5e-5 --epochs 1 --warmup-steps 1 --lora-rank 8 \
    --batch-size 4 --grad-accum 1 --logging-steps 10

# Eval HE-8 only (fast)
python -u 06_eval_batched.py --adapter runs/_smoke/SMOKE_TINY \
    --output /tmp/_smoke.json --cache-db /tmp/_smoke.db \
    --he-limit 8 --mbpp-limit 0 --exec-timeout 10 \
    --batch-size 8 --max-new-tokens 384

# Inspect what student emits on HE prompts
sqlite3 /tmp/_smoke.db "SELECT task_id, substr(generation, 1, 200) FROM results WHERE task='humaneval' LIMIT 3"
rm -rf runs/_smoke/SMOKE_TINY /tmp/_smoke.json /tmp/_smoke.db
```

What healthy looks like:
- Pass rate ≥30% on HE-8 (random; not a strict bar but a sanity)
- Generations start with code at proper indentation (4-space body)
- No `from typing import...` after a function header
- No `class X:` after a function header

If pass rate is 0 or you see module-level code after function
headers, **abort and reformat the corpus** (e.g., from MBPP-style
docstring → function-signature shape).

### 4. Document the smoke in the run log

Put the four-check output in `logs/onpolicy/smoke_<RUN>.log` so the
post-mortem has data, not just memory. The CTD repo `RESULTS.md` row
should reference the smoke log when reporting collapses.

## Quick reference: what failed without proper smoke

| Run | Smoke I did | Real failure mode |
|---|---|---|
| M21 | None | Qwen-Inst rewrites docstring as module → student emits module after func sig |
| M21b | None | Qwen-base writes `pass` then EOSes at 70 tok → student emits stubs |
| M21c | None | Same as M21b (mnt=768 didn't help, EOS was natural not budgeted) |
| M22 | head-200 peek of 2 samples | trailing `if __name__ + doctest.testmod` polluted student → module-level emission |
| M23 | Full peek of 2 samples | I observed clean module rewrites and approved — but didn't ask "does this match HE eval shape?" → it doesn't |
| M25 | All 4 checks (this doc) | TBD — first run with the procedure |

If M25 succeeds, the procedure earned its keep. If it fails, the
procedure caught it cheap.
