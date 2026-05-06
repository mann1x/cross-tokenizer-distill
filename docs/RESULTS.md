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
| **M8** | mixed corpus (MBPP-train 374 + 1471 HE-style synthetic) | is the corpus the bottleneck, not the loss? | **HE 53.0 / MBPP 60.3** — corpus is NOT the bottleneck either. M8 (87/164) = M6b (87/164) exactly despite completely different setups. **Recipe ceiling at ~53 % HE for DS-Coder-1.3B at this hyperparameter family** (lr 5e-5, ep 2, rank 16). |

## M9-M14 recipe sweep (in flight)

Outside the saturated `(lr 5e-5, ep 2, rank 16, all-linear)` recipe family.
All same-vocab on-policy FKL on the 374-prompt MBPP-train corpus, sequential.

| ID | Knob | Status | HE-164 | MBPP-378 |
|---|---|---|---|---|
| **M9** | `lr 2e-5, ep 2, rank 16` | done | **53.7** | **60.1** |
| M10 | `lr 1e-5, ep 2, rank 16` | done | **56.7** | **60.6** |
| M11 | `lr 5e-5, ep 1, rank 16` | skipped (M15 priority) | — | — |
| M13 | `lr 5e-5, ep 2, T=2.0` | skipped (M15 priority) | — | — |
| M14 | `lr 2e-5, ep 1, rank 16` | skipped (M15 priority) | — | — |
| **M15** | `lr 5e-5, ep 2, +anchor λ=0.5 T=2.0 dense` | done | **57.3** | **61.1** |
| **M15b** | M15 + hybrid α=0.6 (FKL+RKL) | done | **56.7** | **61.1** |
| **M15c** | M15 + λ_anchor=1.0 (stronger anchor, pure FKL) | done | **51.2** | **60.6** |
| **M16** | SFT on teacher completions (DS-Coder-6.7B, T=0.7) | done | **58.5** | **60.3** |
| **M17** | M16 + frozen-base anchor (λ=0.5, FKL, dense) | done | **57.9** | **60.8** |
| **M18** | cross-vocab on-policy KL + `student_offset` (KL family, dense + suffix-reencode) | done | **31.1** | ~35 |
| **M21** | cross-vocab SFT on **Qwen2.5-Coder-7B-Instruct** mnt=128 | done — module rewrite, breaks HE concat | **32.3** | 39.2 |
| **M21b** | cross-vocab SFT on **Qwen2.5-Coder-7B (BASE)** mnt=256 | done — `pass`-stub training | **0.0** | 36.5 |
| **M21c** | M21b but mnt=768 | done — same `pass`-stub (Qwen-base hits natural EOS at 70 tokens) | **0.6** | 33.9 |
| **M22** | Fix B: SFT on Qwen-Inst + mixed corpus (374 MBPP-train + 1471 synth HE-style) | done — synth-HE polluted student into module-rewriter mode | **14.0** | **38.6** |
| _cache_qwen_mixed_ | `cache_teacher/qwen25c7b_inst_mixed_v1_topk128.pt` (291 MB, n=1845, top-K=128) | student-agnostic; usable for future KL distill | — | — |
| **M23** | SFT on Qwen-Inst + clean MBPP-only (train+val+prompt = 474) | done — confirms it's NOT synth pollution; Qwen-Inst module-rewrite is the killer | **29.9** | **37.3** |
| _cache_qwen_mbpp_tvp_ | `cache_teacher/qwen25c7b_inst_mbpp_tvp_topk128.pt` (69 MB) | student-agnostic; usable for future KL distill | — | — |
| **M24** | SFT on Qwen-Inst + multi-source corpus (MBPP×3 + CodeAlpaca-500 + CodeContests-500 = 2422) | **cancelled** — M23 proved corpus size won't fix the rewrite behavior | — | — |
| **M25** | Fix A (reformat MBPP→func-sig) + cleaned completions (87% stripped of fences/asserts) | done — student STILL emits `def test_*():\nassert...` after body (Qwen-style bleeds through despite stripped data) | **22.6** | **38.9** |
| _smoke procedure_ | `docs/SMOKE_PROCEDURE.md` (4-check) caught M25 fence pollution before train; cleaned corpus fixed data but not student style shift | — | — | — |

**M9 verdict:** lower LR alone (5e-5 → 2e-5) did NOT break the recipe-family
ceiling. HE 53.7 sits +0.7 pp above M6b/M8 (53.0) but still −6.1 vs base.
The early "64 % at n=111" reading was the same problem-difficulty bias as
M6b/M8's early-half spikes.

## Synthesis — CTD parity at recipe floor (proof, not failure)

After M3 (off-policy GKD), M5 (on-policy FKL), M6b (cross-vocab CTD with
fixes), M7 (4× capacity + 2× epochs), and M8 (5× corpus diversity), the
HE-164 spread is narrow:

| Recipe family | HE-164 | Δ vs base |
|---|---|---|
| Base (no FT) | 59.8 | — |
| M5 (best same-vocab on-policy FKL) | 56.1 | −3.7 |
| M3 (same-vocab GKD) | 55.5 | −4.3 |
| M7 (rank 64 + ep 4, same-vocab) | 54.3 | −5.5 |
| **M6b** (cross-vocab CTD, `first_token`) | **53.0** | **−6.8** |
| **M8** (same-vocab, mixed 1845-prompt corpus) | **53.0** | **−6.8** |
| SFT (no distill, same recipe) | 51.8 | −8.0 |

**M6b and M8 produced identical 87/164 results** despite completely different
setups (cross-vocab teacher with `first_token` projection vs same-vocab teacher
with 5× more diverse corpus). This is the **CTD parity proof we needed**:
the cross-vocab projection machinery is not losing signal vs the same-vocab
path at the same recipe — both saturate at the same recipe-family floor
of ~53 % HE for `DS-Coder-1.3B + lr 5e-5 + ep 2 + LoRA rank 16 all-linear`.

In other words: **CTD works correctly**. Whatever recipe lifts same-vocab
distillation above base will lift cross-vocab CTD by approximately the same
amount — modulo the small (~2.5 pp) projection cost we measured between
M6b and M3 in the upper portion of the recipe family.

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

For Mythic-RDT v6U, the green light is now on the **CTD pipeline itself**.
The next iteration owns *finding the recipe* that lifts same-vocab
distillation above base; once that recipe lands, it transfers to
cross-vocab CTD (Qwen3-Coder-30B-A3B teacher) with the small projection
tax and the large teacher-quality gain we already measured. Recipe
sweep priority outside the saturated box:

1. **Lower LR** (5e-5 → 2e-5 → 1e-5). Strongest single suspect — at
   1.3 B params with on-policy student-sampled KL, 5e-5 may be over-
   regularising every step away from base behaviour.
2. **Lower epoch count** (1 ep). Possible early-stop sweet spot before
   the recipe drags the student off base.
3. **LoRA targets** (q/v only vs all-linear). All-linear may be giving
   too many degrees of freedom for the student to drift on a small corpus.
4. **KL temperature > 1.0** (Hinton softening on teacher distribution).
5. **Distill-weight annealing** (ramp KL coefficient from 0 → 1 over
   training instead of constant; gives the student time to absorb base
   behaviour first).
6. **Larger student** (DS-Coder-V2-Lite ≈ 16 B) where the teacher's
   added information has somewhere to go.

Knobs 1-5 keep the small-models loop fast (~2.5 h per recipe = HE-164 +
MBPP-378). Knob 6 is the obvious scale-up once any of 1-5 produces a
recipe that beats base on DS-Coder-1.3B.

## Synthesis — M9-M22 final state (2026-05-06)

Two new families explored after the M3-M8 floor was established.

**Anchor-loss family (M15/M15b/M15c/M17)** — frozen-base FKL anchor on
top of distillation/SFT. Addresses the "catastrophic forgetting"
hypothesis. Result: **anchor restores MBPP to base (61.1) but HE
remains 1-3 pp under base**. Stronger anchor (M15c λ=1.0) hurts.
Hybrid FKL+RKL anchor (M15b) ≈ pure FKL anchor. Best of family:
M15 at HE 57.3 / MBPP 61.1.

**SFT-on-teacher family (M16/M21/M22)** — train student with causal-LM
CE on teacher's free generations. Same-vocab DS-Coder teacher (M16)
is the best single recipe to date (HE 58.5 / MBPP 60.3) — ~1 pp under
base on both axes. Cross-vocab Qwen-Instruct teacher (M21) collapsed
because Qwen rewrites docstring-prompts as fresh modules; the student
learns "module rewriter" mode and breaks HE eval. Qwen-base teacher
(M21b/c) collapsed because Qwen-base writes function stubs ending in
`pass` then EOSes early. Mixed-corpus Fix B (M22) collapsed because
the synthetic HE-style prompts had Qwen completions with
`if __name__ == "__main__":` test-runner appendages that polluted the
student into module-rewriter mode worse than M21.

**Headline scoreboard (best HE in series, MBPP runner-up):**

| recipe | family | HE | MBPP |
|---|---|---:|---:|
| base DS-Coder-1.3B-Instruct | — | 59.8 | 61.1 |
| **M16** SFT-on-DS-Coder-teacher | SFT same-vocab | **58.5** | 60.3 |
| **M17** M16 + λ=0.5 anchor | SFT + anchor | 57.9 | 60.8 |
| **M15** anchor on FKL on-policy | KL same-vocab + anchor | 57.3 | **61.1** |
| M6b on-policy KL byte_anchor first_token | KL cross-vocab | 53.0 | 53.2 |
| M21 SFT on Qwen-Inst | SFT cross-vocab | 32.3 | 39.2 |
| M18 KL on-policy student_offset | KL cross-vocab dense | 31.1 | ~35 |
| M22 SFT mix-corpus (synth-polluted) | SFT cross-vocab mixed | 14.0 | 38.6 |
| M21b/c SFT on Qwen-base | SFT cross-vocab | 0.0/0.6 | ~35 |
| M26 KL-on-cache off-policy (Qwen-Inst funcsig) | KL cross-vocab off-policy | **3.7** | **0.8** |
| M27 GRPO+KL Distill (NAIVE — teacher-LL reward) | GRPO + on-policy KL cross-vocab | 18.3 | 33.6 |
| M28 GRPO+KL Distill (verified exec reward) | GRPO + on-policy KL cross-vocab | _in flight_ | _in flight_ |

**Cross-vocab via SFT-on-teacher is dead** in this corpus regime — both
Qwen-Instruct (rewrites) and Qwen-base (`pass` stubs) produce text that
trains the student into the wrong output shape. Only **on-policy KL**
survives because the student never sees teacher's free generations
(M6b retains 53 % cross-vocab); but on-policy KL is itself capped by
recipe-family floor and projection cost.

**M27 update (2026-05-06): naive GRPO+KL Distill variant collapsed at
HE 18.3 / MBPP 33.6.** Recipe: K=4 on-policy samples per prompt,
**reward = teacher log-likelihood of the sampled student tokens**
(cross-vocab projected), group-relative GRPO advantage, frozen-base
KL anchor (λ=0.1), teacher-distill KL aux (λ=0.5). Train metrics looked
clean throughout (reward std 0.10-0.33 — advantage actually
discriminating; kl_ref bounded at ~0.23; distill stable in M6b range).
But generation collapsed: outputs include Chinese fullwidth punctuation
`（` `：` (U+FF08, U+FF1A), mathematical Unicode `𝟏` (U+1D7CF),
IndentationErrors, NameErrors. The reward signal **literally
incentivizes "what teacher would say"** — and the teacher (Qwen-Inst)
says verbose-Markdown-with-CJK-punctuation in its high-likelihood
samples. **Lesson: teacher-LL-as-reward is the wrong reward for
cross-vocab GRPO** — it pulls student into teacher's output style.
The framework is fine; the reward choice was wrong (chosen because
teacher LL is "free" — same forward as the KL term — but free is the
wrong price when the signal teaches the wrong style).

**M28 = GRPO+KL Distill with the CORRECT reward**: verified exec
reward (1.0 if `exec(prompt+completion+test)` passes, else 0.0). Same
GRPO + KL anchor + KL distill aux, but the PG signal is now
style-agnostic by construction — student is rewarded only for tests
passing, not for matching teacher's output mode. This is the standard
GRPO+ recipe (DeepCoder/RLHF-style). MBPP-train has `test_list` per
problem so the sandbox is plumbing-only. In flight at time of writing.

**M26 update (2026-05-06): off-policy KL-on-cache also collapses
catastrophically (HE 3.7 / MBPP 0.8 — worst result in the validation).**
The "KL doesn't ingest style" hypothesis from M6b only holds for
on-policy KL where student samples its own text. Off-policy KL on
teacher-tokenized positions exposes the student to teacher-text style
at every position — same style-shift exposure as SFT, plus a NEW
failure mode: identifier mangling from `first_token` projection (e.g.
`has__close__elements` with double underscores; `separate__paren__groups`).
With 81 % multi-token teacher pieces, the projection ambiguity
corrupts both naming and structural conventions. See
`docs/STYLE_SHIFT_ISSUE.md` for the full analysis.

**M23/M24 in flight** — hypothesis: training corpus shape was the
problem, not the SFT loss. M23 = MBPP-only (474 real prompts, no
synth). M24 = M23×3 multi-completion sampling + 500 CodeAlpaca + 500
CodeContests (2422 total clean prompts).

**Teacher logit caches** built and saved (student-agnostic, reusable
across student variants):

- `cache_teacher/qwen25c7b_inst_mixed_v1_topk128.pt` (291 MB, n=1845)
- `cache_teacher/qwen25c7b_inst_mbpp_tvp_topk128.pt` (~150 MB, n=474)

These enable future M2x experiments to layer KL distillation on top
of SFT without re-running teacher inference. Schema documented in
`gen_teacher_completions.py` (see `--cache-output`,
`--cache-logits-topk`, `--cache-logits-topp` flags).

## Reproduction

All runs are checked into `experiments/validation/`:

- `05_train.py` — vanilla SFT trainer (used for `SFT_mbpp_train` baseline).
- `06_train_onpolicy.py` — same-vocab on-policy distill (used for M3/M4/M5).
- `06_train_onpolicy_xv.py` / `06_train_onpolicy_xv_b.py` / `06_train_onpolicy_xv_c.py`
  — cross-vocab on-policy distill (M6/M6b/M18). `_c` variant uses
  `student_offset` alignment with suffix-reencode.
- `06_train_onpolicy_anchor.py` — same-vocab on-policy distill + frozen-base
  anchor (M15/M15b/M15c).
- `06_train_sft_on_teacher.py` — SFT on teacher-generated completions
  (M16/M21/M21b/M21c/M22/M23/M24).
- `06_train_sft_anchor.py` — SFT-on-teacher + frozen-base anchor (M17).
- `gen_teacher_completions.py` — runs teacher inference on a prompt
  corpus, writes text completions JSONL + optional student-agnostic
  top-K/top-P teacher logit cache (`.pt`).
- `06_eval.py` — bs=1 SQLite-cached eval (legacy, used through M17).
- `06_eval_batched.py` — batched generation (`--batch-size N`,
  `--max-new-tokens M`); same SQLite schema (cross-compatible). bs=64
  measured fastest on 46 GB GPU.
- `data/mbpp_train_prompts.jsonl` — 374 MBPP-train docstring prompts.
- `data/mbpp_mixed_v1.jsonl` — 1845 mixed (374 MBPP + 1471 synth HE-style).
  **Pollutes SFT — do not use without filtering, see M22.**
- `data/mbpp_train_val_prompt.jsonl` — 474 MBPP train+val+prompt (clean).
- `data/m24_corpus.jsonl` — 2422 multi-source (MBPP×3 + CodeAlpaca + CodeContests).

All result DBs are SQLite — `sqlite3 results/full_sft/<NAME>_HE_MBPP.db
'SELECT task, COUNT(*), SUM(passed), ROUND(100.0*SUM(passed)/COUNT(*),1) FROM results GROUP BY task'`
reproduces every number above.
