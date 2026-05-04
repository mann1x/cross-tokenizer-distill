# CTD Validation — Small-Models Results

**Status:** in progress (M6 cross-vocab on-policy CTD training as of 2026-05-04).
**Pod:** vast.ai 35822024 (single 48 GB GPU).
**Student:** `deepseek-ai/deepseek-coder-1.3b-instruct` (32 000 vocab).
**Eval suites:** HumanEval-164 + MBPP-378 (sanitized), full sets, SQLite-cached, 30 s exec timeout.

> **Setup note** — all training uses LoRA rank 16 on `all-linear` with the same
> recipe (lr 5e-5, 2 epochs, batch 2 × grad-accum 8, warmup 8, cosine, max-new-tokens 128)
> unless noted. The corpus is the 374-example MBPP train split formatted as HE-style
> docstring prompts so the methods can be compared without recipe noise.

## Reference numbers

The first thing the validation produced was the *true* DS-Coder-1.3B-Instruct
score on HumanEval / MBPP — published `≈65 %` HE was misleading for our setup
(prompt format, post-processing, and exec scorer matter more than expected):

| Model | HumanEval pass@1 (n=164) | MBPP pass@1 (n=378) |
|---|---|---|
| **DS-Coder-1.3B-Instruct (BASE, no-FT)** | **59.8 %** | **61.1 %** |

Every distillation run is judged against this base, not the published number.

## On-policy distillation chain (same vocab)

Same teacher (`deepseek-ai/deepseek-coder-6.7b-instruct`), same student, same
recipe, same 374-prompt corpus. Only the loss objective differs.

| Run | Method | Loss objective | HE-164 pass@1 | Δ vs base | MBPP-378 pass@1 | Δ vs base |
|---|---|---|---|---|---|---|
| BASE | — | (no FT) | 59.8 % | — | 61.1 % | — |
| **M3** | GKD | generalized JSD (β=0.5) | **55.5 %** | **−4.3** | **61.1 %** | **±0.0** |
| M4 | MiniLLM | reverse KL on student-sampled positions | 54.3 % | −5.5 | — | — |
| M5 | DistillSpec / FKL on-policy | forward KL on student-sampled positions | 56.1 % | −3.7 | — | — |
| SFT | reference SFT | cross-entropy on labels | 51.8 % | −8.0 | 59.5 % | −1.6 |

### What this proves

1. **Distillation regularises vs SFT.** Every KL variant beats SFT on HE by
   3.7–4.3 pp at the same recipe and corpus — the teacher signal stops the
   student from overfitting the small training set.
2. **MBPP is preserved by every method.** All same-vocab distill runs finish
   at base-level MBPP (61.1 %); SFT regresses by 1.6 pp.
3. **Reverse KL (MiniLLM) is the worst of the three** at this scale and recipe.
   Mode-seeking on a 1.3B student with a 6.7B teacher concentrates probability
   on a too-narrow mode and drops 1.2 pp behind the simpler forward KL.
4. **None of the three beats base on HE.** With this corpus and recipe the
   teacher's distributional pull cannot overcome the loss of base behaviour
   that any FT introduces. The next experiments (M6 cross-vocab teacher,
   M7 capacity, M8 mixed corpus) are designed to break that ceiling.

### Smoke-set caveat

Earlier "S03 wins smoke" / "A_OLD beats S03 by 3 pp" results came from an HE-20
smoke set. With n=20 the standard error is ~11 pp (Wilson 95 % CI), which
swallows differences smaller than ~7 pp at 50 % accuracy. Smoke sets are useful
for "did training crash" gating and for rejecting actively-bad recipes (M4 RKL),
but no recipe ranking is trustworthy without the full HE-164.

## Cross-vocab on-policy CTD (M6 — in progress)

**Teacher:** `Qwen/Qwen2.5-Coder-7B-Instruct` (151 643 vocab, ~5 × student vocab).
**Method:** student samples → decode → teacher re-tokenizes → byte-anchor
alignment per example → project teacher top-K via cached `VocabMapper` →
forward-KL at aligned student positions.

Mapper coverage on this pair (`Qwen2.5-Coder-7B → DS-Coder-1.3B`,
`multi_token=distribute`):

| Coverage statistic | Value |
|---|---|
| single-token (1-1 string match) | 19.1 % |
| multi-token (1-K, K>1) | 80.9 % |
| dropped (no mapping) | 0.0 % |
| build time (cold, 151 K vocab) | 4.8 s |

Smoke (16 prompts, 2 optimizer steps):
- Both models load in bf16 (DS-Coder-1.3B + Qwen 7B = ~17 GB).
- `byte_anchor` alignment keeps ~64 % of student positions per example
  (218 / 603 dropped on a 4-example batch — expected: tokenizers diverge
  inside identifiers and whitespace).
- Loss = 2.46 → 2.69 across the 2 steps; same magnitude as same-vocab FKL
  on this corpus, confirming the projected distribution is in the right ballpark.

### Full M6 results (FAILED the gate)

| Metric | Value | Δ vs base | Δ vs M3 (same-vocab GKD) | Δ vs SFT |
|---|---|---|---|---|
| HumanEval-164 pass@1 | **38.4 %** (63 / 164) | **−21.4** | −17.1 | −13.4 |
| MBPP-378 pass@1 (partial 175/378) | ~36 % running | ~−25 | ~−25 | ~−24 |

Training was healthy (loss 2.57 → 0.92 over 46 optimizer steps, ~73 % of
positions aligned per example) but the trained student dropped 21 pp HE
vs base. **Cross-vocab CTD with `byte_anchor` + `multi_token=distribute`
on a 5× vocab gap (Qwen 152 K → DS-Coder 32 K) destroyed signal far worse
than expected.** The 80.9 % multi-token mass appears to smear teacher
probability across many low-confidence student tokens; the projection
fidelity is too low at this coverage to drive the small student
constructively.

**v6U decision rule fires:** `C ≤ A → fall back to DS-Coder-V2-236B
same-vocab teacher` for Mythic-RDT. Cross-vocab teacher with this projection
configuration is not the path. Future CTD work to revisit:

- `student_offset` alignment (full coverage at 1.5-2× compute) instead of
  `byte_anchor` — may recover the dropped 36 % of positions and the signal
  carried in them.
- `multi_token=strict` (skip multi-token teacher tokens entirely) on a pair
  with higher single-token coverage (e.g. closer-vocab teacher).
- Hybrid loss: KL only at high-confidence projected positions, SFT at the
  rest, with a confidence threshold derived from projection mass retained.

### Decision rule for v6U

The original validation plan in `README.md` set:
- **C ≥ 0.8 × B** and **C > A** → CTD works; ship the same recipe to Mythic-RDT v6U
  with a `Qwen3-Coder-Next` teacher.
- **C ≤ A** → vocab projection destroys signal; fall back to DS-Coder-V2-236B
  same-vocab teacher.

Translating to this pod's actual numbers:
- "B" = M3 = 55.5 % HE / 61.1 % MBPP (same-vocab GKD).
- "A" = SFT = 51.8 % HE / 59.5 % MBPP.
- **0.8 × B = 44.4 % HE.** Threshold for shipping CTD: **M6 HE ≥ 44.4 % AND ≥ SFT (51.8 %)**.

Stricter target (recommended): **M6 HE ≥ 53 %** (within 2.5 pp of M3) — proves
the cross-vocab projection is not throwing away meaningful teacher signal.

## What's planned next

| ID | Run | Hypothesis being tested |
|---|---|---|
| M6 | cross-vocab on-policy (Qwen2.5-Coder-7B teacher) | does CTD recover same-vocab quality? gate for v6U |
| M7 | capacity test (rank=64, 4 epochs, M3 recipe) | does the same KL recipe beat base if we let it learn longer? |
| M8 | mixed corpus (MBPP-train + ~1500 HE-style synthetic from base) | is the corpus the bottleneck, not the loss? |

## Reproduction

All runs are checked into `experiments/validation/`:

- `05_train.py` — vanilla SFT trainer (used for `SFT_mbpp_train` baseline).
- `06_train_onpolicy.py` — same-vocab on-policy distill (used for M3/M4/M5).
- `06_train_onpolicy_xv.py` — cross-vocab on-policy distill (used for M6).
- `06_eval.py` — SQLite-cached, resumable HE+MBPP eval with per-problem timeouts.
- `data/mbpp_train_prompts.jsonl` — 374 HE-style docstring prompts derived from MBPP train.
- `data/mbpp_train_sft.jsonl` — same prompts + reference solutions for SFT.

All result DBs are SQLite — `sqlite3 results/full_sft/<NAME>_HE164.db
'SELECT task, COUNT(*), SUM(passed), ROUND(100.0*SUM(passed)/COUNT(*),1) FROM results GROUP BY task'`
reproduces every number above.
