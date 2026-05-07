# CTD v2 — Plan

**Date drafted**: 2026-05-07
**Status**: Awaiting sign-off + 4 prerequisite landings (see § Prerequisites)
**Supersedes**: ad-hoc M1-M35 tracking from v1; the cross-vocab Pareto frontier from v1 (M25 SFT 55.5/39.9, M6b on-policy KL 53.0/33.9 on test split) is the v2 baseline to beat.

## Why v2

Three reasons v1 ran out of headroom:

1. **Teacher was Qwen2.5-Coder-7B (HE 82.9 / MBPP 68.8 BF16 chat).** Student ceiling 53-55 HE leaves ~28 pp headroom unrecovered. With the right teacher signal we should be able to close more of that.
2. **Single student architecture.** v1 only used DS-Coder-1.3B-Instruct (cross-vocab). We never measured what cross-vocab via VocabMapper costs us vs. same-vocab direct KL at the same scale.
3. **Pod GPU is RTX 6000 Ada 48GB**, not 24GB as v1 implicitly assumed. v1 NF4-only choices were forced by 24GB; on 48GB we have room for bigger batches, longer sequences, or a bigger student. Teacher stays NF4 (matches Mythic-RDT operational quant) but with much more breathing room around it.

## What's new in v2

| v1 → v2 | Why |
|---|---|
| Teacher: Qwen2.5-Coder-7B-Instruct → **Qwen2.5-Coder-14B-Instruct** | Measured HE 88.4 / MBPP ≥ 65 (raw, NF4). +5.5 pp HE headroom over QC-7B. Same tokenizer family, mapper cache reusable. |
| Teacher precision: mixed BF16/NF4 → **NF4 chat throughout** | NF4 = BF16 measured equivalent on DS-V2-Lite (HE bit-exact, MBPP ±1 problem). Same conclusion expected on QC-14B. NF4 frees 16GB VRAM (12GB vs 28GB), is ~30% faster forward-pass, and matches Mythic-RDT's operational quant — training and downstream stay congruent. Optional one-shot BF16 A/B if a recipe lands suspiciously close to base. |
| Student: DS-Coder-1.3B-Instruct only → **DS-Coder-1.3B + Qwen2.5-Coder-1.5B-Instruct (dual)** | QC-1.5B shares teacher's vocab → same-vocab direct KL, no VocabMapper. Delta vs DS-Coder-1.3B = quantified cost of cross-vocab. |
| Eval: bs=64 batched, non-deterministic | **Determinism fix in place before Phase 1.** ~8pp run-to-run drift was masking recipe deltas. |
| Dataset split: mixed (some MBPP-full, some test) | **Canonical MBPP test split** for everything. Pre-May-6 results re-evaluated for alignment. |

## Prerequisites blocking v2 launch

| # | Block | Status | Why |
|---|---|---|---|
| 1 | Determinism fix (#98) | smoke queued | A/B variance must be reduced from 8pp → ≤ 2pp before recipe comparisons are valid |
| 2 | Pre-May-6 reeval on test split (#99) | chain queued | Establish v1 baseline numbers on the same dataset as v2 |
| 3 | QC-14B chat-template re-eval (#100) | queued | Confirms the operationally relevant teacher score (chat is what `gen_teacher_completions.py` and on-policy generation use) |
| 4 | Sign-off on this doc | this PR | Architecture-level review before committing ~30 hr of pod compute |

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Teacher: Qwen2.5-Coder-14B-Instruct (NF4)      │
│  HE 88.4 / MBPP ~ 73-75 (chat, expected)         │
│  ~12 GB VRAM, frozen                             │
└────────────────┬────────────────────────────────┘
                 │ logits / completions / chat-mode generation
                 │
        ┌────────┴────────┐
        │                 │
   ┌────▼─────┐      ┌────▼─────┐
   │ Student A│      │ Student B│
   │ DS-Coder │      │ QC-1.5B  │
   │ -1.3B-Inst│      │ -Inst    │
   │ ~3 GB BF16│      │ ~3 GB BF16│
   │ +1 GB LoRA│      │ +1 GB LoRA│
   └─────┬────┘      └────┬─────┘
         │                │
   cross-vocab        same-vocab
   (VocabMapper       (direct KL,
    first_token)       no mapper)
         │                │
         └─────┬──────────┘
               │
        Recipes M36-M44
        evaluated on canonical
        MBPP test split + HE-164
```

VRAM total: ~16-18 GB at bs=2 (12 GB teacher NF4 + 3 GB student BF16 + 1 GB LoRA grads + 2-3 GB activations/cache). Comfortable on 48 GB pod with room for bs=4-8 or rank=32. Also fits the 3090 local 24GB at bs=2 → CTD work can be offloaded to local GPU when pod is busy.

## Phases

### Phase 0 — Setup (~3-4 hours, mostly compute)

1. **VRAM smoke**: 5-step training loop with QC-14B NF4 teacher + DS-Coder-1.3B BF16 student + LoRA grads at bs=2, bs=4, bs=8. Confirm fit + log peak GPU usage on the 48 GB pod (and on the 24 GB 3090 local at bs=2). Repeat with QC-1.5B student.
2. **Teacher logit caches** (NF4 teacher): recompute for `mbpp_train_prompts.jsonl` (374), `mbpp_train_val_prompt.jsonl` (474), `mbpp_funcsig_prompts.jsonl` (474). top-k=64. ~20 min each (NF4 ~30% faster than BF16).
3. **Teacher completions** (chat template, NF4): regenerate for funcsig prompts (M25-style SFT corpus) and `mbpp_train_prompts` (M16-style). ~30 min each.
4. **QC-1.5B base eval** on canonical test split: HE-164 + MBPP-378. Establishes reference for the same-vocab student track.

### Phase 1+2 — Dual-student baseline matrix (~6 hr total, sequential)

The point of v2 is the **A/B comparison between two students at the same recipes with the same teacher artifacts**. Each row of the matrix is one recipe; each column is one student. Same teacher (QC-14B NF4), same corpus, same completions, same hyperparams within a row. Reading down the column = recipe variation within a student. Reading across the row = vocab-path delta on the same recipe.

| Recipe | v1 ancestor | **Student A — DS-Coder-1.3B-Inst** (cross-vocab via VocabMapper first_token) | **Student B — Qwen2.5-Coder-1.5B-Inst** (same-vocab, no mapper) |
|---|---|---|---|
| **R1: On-policy KL** (mbpp_train_prompts, ep=2, bs=2, ga=8, lr=5e-5, rank=16) | M6b | **M36** | **M39** |
| **R2: SFT on funcsig completions** (M25 corpus, clean) | M25 | **M37** | **M41** |
| **R3: SFT on mbpp_train completions** (M16-style) | M16 | **M38** | **M40** |
| **Base reference** | n/a | DS-Coder-1.3B base = HE 56.1 / MBPP 41.0 (test split) | QC-1.5B base = TBD from Phase 0 step 4 |

**The 3 A/B comparisons** that v2 is built to answer:

1. **M36 vs M39** — On-policy KL: how much does the VocabMapper first_token projection cost vs no mapper at all? This is the cleanest measurement of cross-vocab CTD overhead. Both students sample their own text under same teacher KL signal.
2. **M37 vs M41** — SFT on funcsig: same teacher completions used as supervision for both students. Cross-vocab here is just "the student's tokenizer re-tokenizes the teacher's text" — no logit projection. Should be a smaller delta than M36 vs M39.
3. **M38 vs M40** — SFT on mbpp_train: same as #2 but different corpus. Cross-validates the SFT delta across two corpora.

**What the deltas tell us:**
- **All 3 deltas small (< 3 pp)** → VocabMapper first_token is fine; cross-vocab is essentially free; v1's "ceiling at 53-55 HE" was a recipe limitation, not a cross-vocab limitation.
- **M36 vs M39 large (> 5 pp), M37/M38 vs M41/M40 small** → projection ambiguity in on-policy KL is the bottleneck; SFT is the pragmatic path for cross-vocab.
- **All 3 deltas large** → cross-vocab via this mapper costs us substantially across recipes; need a better mapper (or accept the cost).

**Phase 2 follow-up — LoRA capacity test on QC-1.5B**:
- **M41b**: re-run the best Phase 2 recipe (whichever of M39/M40/M41 wins on test split) at **rank=64** instead of rank=16. ~30 min. Tells us whether QC-1.5B's same-vocab numbers are capacity-limited at rank=16. If M41b ≥ M39/M40/M41 winner + 2 pp on either bench → Phase 3 should default to rank=64 for the same-vocab track.

**Phase 1+2 deliverable**: that 3-row, 2-column matrix of HE/MBPP scores on the canonical test split with deterministic eval, plus M41b rank-bump verdict, plus the 3 deltas above and the recommended path for Phase 3.

### Phase 3 — Productive variations (~6-10 hr)

Run on whichever student-track wins Phase 1+2 head-to-head. The decision rule:

- If **same-vocab QC-1.5B substantially outscores** cross-vocab DS-Coder-1.3B at all three recipes → cross-vocab via this VocabMapper is leaving too much on the table. Run Phase 3 on the **same-vocab track** (QC-1.5B student) and accept that v2's Mythic-RDT downstream value depends on a separate "port the recipe to a same-vocab Mythic-RDT student" step.
- If **deltas are small** (cross-vocab competitive) → run Phase 3 on the **cross-vocab track** (DS-Coder-1.3B student) since that maps directly to Mythic-RDT downstream (DS-Coder-V2-Lite-Inst with QC-14B teacher).
- If **deltas are mixed** (e.g., R1 KL fails cross-vocab but R2/R3 SFT works fine) → run Phase 3 on the cross-vocab track using the SFT recipe, drop the on-policy KL recipe.

| ID | Recipe | v1 ancestor | New knob |
|---|---|---|---|
| **M42** | GRPO + verified exec reward, K=8, 4 epochs, KL anchor λ=0.1 | M28 (killed early, K=4 too sparse) | Properly resourced; the only v1 recipe that's principled but never finished |
| **M43** | 2-stage curriculum: M37/M40-equivalent SFT init → M36/M39-equivalent on-policy KL | M31 | Reuse v1's 2-stage finding with new teacher |
| **M44** | Hard-prompt mining: filter teacher corpus to only problems where the student base FAILS. Train on those. | new | Concentrates capacity on actual gaps. Hypothesis: 80% of teacher distribution is wasted on problems the student already gets. |

### Phase 4 — Unified eval + decide (~2-3 hr)

- All recipes evaluated on canonical MBPP test split + HE-164 with deterministic eval
- Tab compare against:
  - DS-Coder-1.3B base (HE 56.1 / MBPP 41.0)
  - QC-1.5B base (TBD from Phase 0 step 4)
  - QC-7B teacher (HE 82.9 / MBPP 68.8 chat)
  - QC-14B teacher (HE 88.4 / MBPP TBD chat)
- Pick the v2 winner for **Mythic-RDT downstream**: the recipe with the best HE+MBPP gain over base, applied to DS-Coder-V2-Lite-Instruct as the next step (CTD-pre-recurrence layer).

## Compute budget

| Phase | Sequential hours |
|---|---|
| 0 — setup | 3-4 |
| 1 — cross-vocab baselines | 3 |
| 2 — same-vocab baselines | 3 |
| 3 — productive variations | 6-10 |
| 4 — unified eval | 2-3 |
| **Total** | **17-23** pod hours |

Sequential because of GPU contention on the 48GB Ada 6000.

## Mythic-RDT follow-on (out of scope here, but motivation)

If v2 CTD lands a recipe that gives DS-Coder-1.3B a meaningful HE+MBPP gain over base (e.g., HE ≥ 65, MBPP ≥ 50), apply that recipe to DS-Coder-V2-Lite-Instruct as a CTD-pre-recurrence step:

- DS-V2-Lite base = HE 75.6 / MBPP 60.6 (measured 2026-05-06 on test split)
- CTD with QC-14B teacher → ?
- Then recurrence-wrap → Mythic-Coder T=8 target ≥ HE 80.6 / MBPP 63.6

The CTD step is upside-only: if it doesn't lift the base, we just skip it. If it does, the recurrence wrap starts from a higher floor.

## Open questions — all closed at plan-sign-off

1. ~~**Hard-prompt mining (M44) selection criterion**~~ → **DECIDED**: filter where student base fails outright (binary `passed=0` from the `06_eval_batched.py` SQLite cache). Per-student corpus (each student trained on its own failures), not union. Concretely: run base eval on full mbpp_train + mbpp_train_val_prompt + funcsig corpora for each student, take the `passed=0` rows as M44 training corpus for that student. The M44 cross-vocab/same-vocab delta is then a "best each student can do" comparison rather than apples-to-apples — that's intentional since M44's point is concentrating capacity on each student's actual gaps.
2. ~~**GRPO+ K=8 vs K=16**~~ → **DECIDED**: defer the K choice to after M42 K=8 lands. If M42 verifies the recipe and lands close to the M36/M39 baseline ceiling, run K=16 as M42b. If M42 K=8 already saturates or fails, skip K=16.
3. ~~**Same-vocab QC-1.5B LoRA capacity**~~ → **DECIDED**: add as a Phase 2 follow-up. After M39/M40/M41 land, identify the best same-vocab recipe and re-run it at **rank=64** as **M41b** (4× capacity vs default rank=16). 14M LoRA params → 56M ≈ 3.7% of QC-1.5B. Tells us whether Phase 2 results are LoRA-capacity-limited.
4. ~~**BF16 vs NF4 student**~~ → **DECIDED**: BF16 throughout v2. Both students are small (DS-Coder-1.3B = 3GB BF16, QC-1.5B = 3GB BF16); NF4 saves ~2GB which doesn't matter on a 48GB pod, and at 1.5B-class scale NF4 quantization noise would muddy recipe-delta interpretation. Cost we want to measure is *recipe* not *quant*. Teacher stays NF4 (different reason: matches Mythic-RDT operational config).

## Decision

**Sign off on**: phases 0-4 as scoped, ~20 hr compute budget, two-student A/B as the v1 → v2 differentiator. **Defer**: open questions 1-4 to in-flight decisions per phase.
