# CTD validation experiment — small-models proof-of-concept

Before applying CTD to a real production training run (Mythic-RDT v6U
with Qwen3-Coder-Next 80B-A3B teacher), we validate the technique on
small models where iteration is fast and cost is near zero.

## Hypothesis under test

**H1**: cross-vocab distillation via byte-aligned projection (CTD)
recovers a meaningful fraction of the quality gain that same-vocab
distillation provides.

**Operational definition**: with all training hyperparameters held
constant, on the same student model, CTD distillation from a
different-vocab teacher produces eval scores within `0.8×` the gap
between no-distill baseline and same-vocab distillation from a
comparable-quality teacher.

## Experimental design

A 3-way comparison on a single small student, varying ONLY the
teacher / distillation method:

| Run | Student | Teacher | Vocab | Method |
|---|---|---|---|---|
| **A** | Qwen2.5-Coder-0.5B | none | n/a | SFT only (no distill) |
| **B** | Qwen2.5-Coder-0.5B | Qwen2.5-Coder-7B | same (152064) | KL distill (standard) |
| **C** | Qwen2.5-Coder-0.5B | DeepSeek-Coder-V2-Lite-Instruct | **different** (102400) | **CTD** |

Why these specific models:

- **Student Qwen2.5-Coder-0.5B**: small, common, well-documented
  baseline; trains in ~1-2h on a 3090 with LoRA.
- **Teacher B Qwen2.5-Coder-7B**: same family, larger, ~88% HE+
  pass@1 — strong same-vocab teacher with comfortable headroom over
  the 0.5B student.
- **Teacher C DeepSeek-Coder-V2-Lite-Instruct**: different vocab
  family entirely, ~81% HE+, comparable-quality teacher to (B). We
  intentionally pick a teacher that's NOT obviously stronger than B
  so quality differences in the student aren't confounded by teacher
  capability.

A teacher pair where C is clearly weaker than B would let us
mistake "CTD works" for "the weak teacher caps quality". Picking
B≈C in raw capability isolates the vocab gap as the only
independent variable.

## Training corpus

5,000 examples sampled from `ise-uiuc/Magicoder-OSS-Instruct-75K`
(Apache 2.0 license, code-instruction pairs, ~600 tokens average
length). Same seed, same shuffle for all three runs.

## Hyperparameters (held constant across A/B/C)

- LoRA rank 16, alpha 32, all linear layers
- LR 1e-4, cosine schedule, 100 warmup steps
- Batch size 8, grad accum 4 (effective 32)
- 2 epochs
- bf16 mixed precision
- Max seq length 1024
- Temperature for distillation: 1.0
- Distill loss weight (B and C): alpha=0.5, hard-label CE weight=0.5

## Eval

Same suite for all three runs:

- **HumanEval+** (164 problems, pass@1)
- **MBPP+** (~378 problems, pass@1)
- **LiveCodeBench-medium-30** (random sample of 30 medium problems
  from after 2024-10-01 to avoid contamination)

All evals via `humaneval_smoke.py` adapted from Mythic-RDT, single
attempt per problem (greedy), batch_size 16, gen_tokens 512,
flash_attention_2.

## Compute budget

| Phase | Hardware | Wall-clock | Cost |
|---|---|---|---|
| Cache precompute (B): Qwen2.5-Coder-7B over 5K samples × 1024 tokens | 1×3090 24GB (NF4) | ~3h | $0 (local) |
| Cache precompute (C): DeepSeek-Coder-V2-Lite over same corpus, projected to Qwen2.5 vocab via CTD | 1×3090 24GB (NF4) | ~4h (incl. suffix re-encode overhead) | $0 (local) |
| Run A (SFT) | 1×3090 | ~1.5h | $0 |
| Run B (same-vocab distill) | 1×3090 | ~2h | $0 |
| Run C (CTD distill) | 1×3090 | ~2h | $0 |
| Eval all 3 (HE+/MBPP+/LCB-30) | 1×3090 | ~3h | $0 |
| **Total wall-clock** | | **~16h** | **$0** |

Runs sequentially on solidPC 3090 over ~2 days, OR on a vast 1×A100
(~$1.5/h × 16h = $24) in one overnight session. solidPC can run
overnight without blocking interactive work.

## Decision rules

After the eval table is in:

```
Δ_B = (B_score - A_score)        # Quality gain from same-vocab distill
Δ_C = (C_score - A_score)        # Quality gain from CTD
ratio = Δ_C / Δ_B                # CTD efficiency
```

| Outcome | ratio | Decision |
|---|---|---|
| **CTD works** | `Δ_C ≥ 0.8 × Δ_B` and `Δ_C > 0` | Apply CTD to Mythic-RDT v6U with Qwen3-Coder-Next teacher |
| **CTD partial** | `0 < Δ_C < 0.8 × Δ_B` | Re-tune projection strategy (try `byte_anchor`, different multi_token policy), re-run C only |
| **CTD broken** | `Δ_C ≤ 0` (no improvement over no-distill) | Fall back to v6T-DS-V2-236B (same-vocab teacher, no CTD) |

Use averaged improvement across HE+/MBPP+/LCB-30, not a single
benchmark.

## Risks & confounders

1. **0.5B student may saturate easily** — if even SFT alone closes
   most of the gap, distillation signal will be small and ratios
   noisy. Mitigation: pick small training corpus (5K), short
   schedule, so distill has measurable headroom to fill.

2. **Vocab pair quality matters** — Qwen ↔ DeepSeek-V2-Lite is
   moderate similarity. For a worst-case stress test we should
   ALSO include a third pair (e.g. Qwen2.5-Coder-0.5B ← Mistral-7B-
   Instruct, vocab=32000 — very different). Defer to phase 2 if
   the primary 3-way is positive.

3. **Eval noise on 30-problem LCB** — SE on n=30 is ~9pp at p=0.3.
   Small differences won't be detectable. Keep HE+/MBPP+ as the
   primary signals (n=164 / n=378 → SE 3-4pp).

4. **Teacher cache projection fidelity** — if VocabMapper coverage
   is < 70%, the cache itself is degraded before training even
   starts. Run `ctd-inspect` before precompute to see coverage.

## Repo location

This experiment lives in `experiments/validation/`:

```
experiments/validation/
├── README.md              # this file in short form
├── 01_inspect.sh          # ctd-inspect on (Qwen, DS) — coverage report
├── 02_precompute_B.sh     # build same-vocab cache (Qwen2.5-7B → Qwen2.5-0.5B)
├── 03_precompute_C.sh     # build CTD cache (DS-V2-Lite → Qwen2.5-0.5B via CTD)
├── 04_train_A.sh          # SFT only baseline
├── 05_train_B.sh          # same-vocab distill
├── 06_train_C.sh          # CTD distill
├── 07_eval_all.sh         # HE+/MBPP+/LCB-30 on all three
├── 08_compare.py          # produce decision table
├── data/                  # Magicoder subset, ~5K examples
└── results/               # JSONs, logs, decision report
```

## Timeline target

- **Day 1**: scaffold + implement core (`mapper`, `alignment`,
  `precompute`). Run `ctd-inspect` on (Qwen2.5, DS-V2-Lite).
- **Day 2**: precompute caches B and C. Run SFT baseline (A).
- **Day 3**: train B and C. Eval all three.
- **Day 4**: decision report. Either: launch v6U-Qwen3 precompute
  on vast pod, or pivot to v6T-DS-V2-236B same-vocab.
