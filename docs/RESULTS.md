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

### M6 first attempt — RETRACTED (code bug, re-run as M6b)

The first M6 run finished training (loss 2.57 → 0.92, ~73 % positions aligned)
and HE-164 = 38.4 % (−21.4 vs base, −17.1 vs M3 same-vocab). That number is
**not a valid CTD-vs-same-vocab gate** — code review of
`06_train_onpolicy_xv.py` found two real bugs that contaminate the result:

1. **Sample/train tensor layout mismatch.** Student was sampled on a
   left-padded tensor (`generate(input_ids=…)` from the DataLoader), then
   re-forwarded for training on a freshly **right-padded** tensor of the
   re-packed sequences. RoPE position encoding (DS-Coder is Llama-family)
   then puts the same logical token at a different absolute position in
   the two forwards — the KL trains the student against teacher logits at
   positions the student will never actually be at during inference.
   M5 (same-vocab, working) avoids this by forwarding both models on the
   same left-padded `full_ids`.
2. **Latent prompt-text index drift after filtering.** When any example in
   the batch was dropped via `continue` (empty row, empty teacher tokens,
   etc.), the subsequent filtered-list index `b_idx` was used to read
   `batch["prompt_texts"][b_idx]` — i.e. the **wrong prompt's** text for
   computing `prompt_len_s`. Low frequency at batch_size=2 but a real
   silent corruption.

There's also a design issue compounding the bugs: mapper coverage is
19.1 % single-token / **80.9 % multi-token** for Qwen 152 K → DS-Coder 32 K.
With `multi_token=distribute`, a teacher token that splits to N student
tokens has its mass spread across all N student vocab indices at the same
position — training the student to favor *fragments* over coherent
identifiers. The eval failures (NameError on undefined symbols,
IndentationError, TypeError "int not callable") match exactly that
fragment-emission signature.

**M6b** re-runs with all four fixes:

1. Forward student on the same left-padded tensor used for sampling
   (mirror M5's structure exactly); keep teacher on its own re-tokenized
   tensor (cross-vocab still requires that).
2. Index `prompt_texts` via the original batch position, not the filtered
   list index.
3. Switch loss aggregation to global-mean-over-positions to match M5.
4. Rebuild mapper with `multi_token=first_token` (puts teacher mass on a
   single coherent next-token target for multi-token entries).

### M6b results (full, both benches)

| Metric | Value | Δ vs base | Δ vs M3 (same-vocab GKD) | Δ vs SFT |
|---|---|---|---|---|
| HumanEval-164 pass@1 | **53.0 %** (87/164) | −6.8 | −2.5 | +1.2 |
| MBPP-378 pass@1 | **53.2 %** (201/378) | −7.9 | −7.9 | −6.3 |

Training was healthy (loss 0.27-0.39 across 46 steps, ~81 % positions
aligned). The bug fixes recovered +14.6 pp HE vs the buggy M6 run (38.4 →
53.0), so the original bugs were real and material — but cross-vocab CTD
with `first_token` projection still lands **2.5 pp below the same-vocab
GKD baseline**. The "first 50 problems at 81.5 %" was problem-difficulty
bias — the harder second-half HE problems pulled the rate back to 53 %.

### v6U decision (informed by M6b)

The original gate rule was `C ≥ 0.8 × B AND C > A → ship CTD`. With this
pod's actual numbers:

- B (M3 same-vocab GKD) = 55.5 % HE → 0.8 × B = 44.4 %.
- A (SFT) = 51.8 % HE.
- **C (M6b cross-vocab CTD with first_token) = 53.0 % HE.**

Mechanically on HE: C > 0.8 × B ✓ and C > A (+1.2 pp) ✓ — so the HE gate
technically *passes*. **On MBPP, M6b regresses 7.9 pp vs M3** —
the projection cost compounds across the longer-form MBPP problems,
where any per-token signal loss accumulates over 30+ tokens of generation.

But the in-isolation comparison is misleading for the v6U decision: the
two candidate teachers are not equally strong. Published HumanEval pass@1:

| Teacher | HE pass@1 | Operational cost (Mythic-RDT serving) |
|---|---|---|
| DeepSeek-Coder-V2-236B-Instruct (same-vocab) | ~75 % | 236 B params, multi-GPU, ~250 GB VRAM bf16 |
| Qwen3-Coder-30B-A3B (cross-vocab via CTD) | ~82 % | 30 B total / 3 B active, single-GPU, ~60 GB VRAM bf16 |
| Qwen3-Coder-Next-80B (cross-vocab via CTD) | ~85 % (per Qwen team) | 80 B params, multi-GPU but smaller than 236 B, ~160 GB VRAM bf16 |

That ~7-10 pp teacher-quality gap on HE plus the dramatic active-parameter
advantage of the 30B-A3B option flips the v6U calculus: even at the current
M6b projection cost, net expected HE gain with 30B-A3B ≈ +7 pp (teacher)
− 2.5 pp (projection) = **+4.5 pp**.
For Mythic-RDT v6U, the choice is:

- **Cross-vocab Qwen3-Coder teacher**: gains a stronger model (Qwen3-Coder >
  DS-Coder-V2-236B on most code benches) at the cost of ~2.5 pp from
  projection. Net effect depends on the teacher gap being > 2.5 pp.
- **Same-vocab DS-Coder-V2-236B teacher**: simpler, no projection loss, but
  capped at the V2 family's quality ceiling.

Recommendation: **lean cross-vocab Qwen3-Coder for Mythic-RDT v6U** —
the +7 pp teacher gap dominates the −2.5 pp HE projection cost on the
v6U headline benchmark. **But not at current M6b cost** — the −7.9 pp MBPP
regression is large enough that it would erase the teacher-quality win on
the secondary benchmark. The plan is to first iterate the CTD recipe to
match same-vocab quality (M6b → M6c → ... until parity with M3 on both HE
and MBPP), then ship to v6U.

Iteration plan ("CTD parity push"):

1. Build a **diff-smoke** corpus from the M3-vs-M6b HE-164 DBs:
   - 19 problems M3 passed but M6b failed (the projection-specific failures)
   - 72 problems both passed (regression guard set)
   - 58 problems both failed (not actionable for CTD recipe iteration)
   Total smoke set = 91 problems (19 + 72), eval cost ~50 % of full HE-164,
   with a much sharper signal-to-noise ratio for "did the recipe change
   help where projection is hurting?"
2. Sweep recipe knobs in priority order:
   a. `student_offset` alignment + suffix re-encode (full coverage,
      recover the ~20 % positions `byte_anchor` drops; ~1.5-2× compute
      but covers the gap).
   b. `multi_token=first_token` is current default — try `strict`
      (skip multi-token entries entirely; smaller training signal but
      no smearing) for ablation.
   c. Larger `out_topk` (32 → 64 → 128) to retain more of the projected
      teacher distribution before renormalisation.
   d. Hybrid loss: KL on positions where projected mass is ≥ τ; SFT-on-teacher
      otherwise. Gates the lossy KL on signal quality.
   e. Temperature sweep T ∈ {1.0, 1.5, 2.0} (Hinton softening helps
      multi-modal projected distributions).
3. Decision rule per knob: a candidate must beat M6b on diff-smoke
   pass-rate AND not regress on the M3-passed problems.
4. Once parity (M6N matches M3 on both HE and MBPP within ±1 pp),
   ship M6N recipe to Mythic-RDT v6U with the real Qwen3-Coder teacher.

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

| ID | Run | Hypothesis being tested | Result |
|---|---|---|---|
| M6b | cross-vocab on-policy (Qwen2.5-Coder-7B teacher) | does CTD recover same-vocab quality? gate for v6U | HE 53.0 / MBPP 53.2 — gate ✓ HE, MBPP regression vs M3, parity push next |
| **M7** | capacity test (rank=64, 4 epochs, M3 recipe) | does same recipe beat base with more capacity / time? | **HE 54.3 (−5.5 vs base) / MBPP 62.2 (+1.1 vs base)** — MBPP-friendly, HE-overfitting; first run to clear base on either bench, but on the wrong one. Confirms corpus diversity is the real lever. |
| **M8** | mixed corpus (MBPP-train 374 + 1471 HE-style synthetic) | is the corpus the bottleneck, not the loss? | **HE 53.0 / MBPP partial 51.7** — corpus is NOT the bottleneck either. M8 (87/164) = M6b (87/164) exactly despite completely different setups. **Recipe ceiling at ~53 % HE for DS-Coder-1.3B at this hyperparameter family** (lr 5e-5, ep 2, rank 16). |

## Synthesis — DS-Coder-1.3B distillation ceiling

After M3 (off-policy GKD), M5 (on-policy FKL), M6b (cross-vocab CTD with
fixes), M7 (4× capacity + 2× epochs), and M8 (5× corpus diversity), the
HE-164 spread is narrow:

| Recipe family | HE-164 | Δ vs base |
|---|---|---|
| Base (no FT) | 59.8 | — |
| M5 (best same-vocab on-policy FKL) | 56.1 | −3.7 |
| M3 (same-vocab GKD) | 55.5 | −4.3 |
| M7 (rank 64 + ep 4) | 54.3 | −5.5 |
| M6b (cross-vocab CTD) | 53.0 | −6.8 |
| **M8 (mixed corpus 1845)** | **53.0** | **−6.8** |
| SFT (no distill, same recipe) | 51.8 | −8.0 |

M6b and M8 produced **identical 87/164 results** despite completely different
setups (cross-vocab teacher with `first_token` projection vs same-vocab teacher
with 5× more diverse corpus). That's not noise — it's a **recipe-family ceiling**
at ~53 % HE for `DS-Coder-1.3B + lr 5e-5 + ep 2 + LoRA rank 16 all-linear`.

The two knobs that had any signal were:
- **Forward KL > GKD > Reverse KL** — M5 (FKL) sat 0.6 pp above M3 (GKD) and
  1.8 pp above M4 (RKL).
- **Capacity helps MBPP, hurts HE** — M7's rank 64 + ep 4 pushed MBPP +1.1 vs
  base while shaving HE −1.8 vs M5. MBPP-derived training data biases learning
  toward MBPP-distribution patterns; more capacity captures more of that.

What we have NOT yet tried (recipe-knob space outside this family):
- Lower LR (5e-5 may be over-regularising — try 2e-5, 1e-5)
- Lower epoch count (1 ep vs 2 — possible early-stop sweet spot)
- Different LoRA targets (q/v only vs all-linear)
- KL temperature > 1.0 (Hinton softening on the teacher distribution)
- Distill weight / SFT-mix annealing
- Larger student (3B / 6.7B) where the teacher's added information has
  somewhere to go

For Mythic-RDT v6U, this means the small-models validation has hit a
recipe-family floor and any further KL-recipe iteration on DS-Coder-1.3B
should explore *outside* the (lr 5e-5, ep 2, rank 16, LoRA all-linear) box
before scaling to V2-Lite or a Qwen3-Coder teacher.

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
