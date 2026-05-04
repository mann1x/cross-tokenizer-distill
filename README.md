# cross-tokenizer-distill (CTD)

A drop-in library for **knowledge distillation across different tokenizer
vocabularies**. Train a student model with logits from a teacher whose
tokenizer doesn't match the student's, without losing alignment between
the two models' token positions.

## Why this exists

Logit-level knowledge distillation requires that teacher and student
share a vocabulary: KL divergence between two distributions is only
meaningful when both distributions live over the same set of token
indices. In practice, the strongest teachers for a given task often
use a different tokenizer than the student you want to train.

Examples we keep hitting in the wild:

| Student | Strongest available teacher | Vocab match? |
|---|---|---|
| DeepSeek-Coder-V2-Lite (102400) | DeepSeek-Coder-V2-236B (102400) | yes |
| DeepSeek-Coder-V2-Lite (102400) | Qwen3-Coder-Next 80B-A3B (151936) | **no** |
| TinyLlama-1.1B (32000) | Qwen2.5-Coder-7B (152064) | **no** |
| Llama-3-8B (128256) | Mistral-Large (32000) | **no** |
| Any small student | Closed-vocab API model (top-K logits exposed) | **no, often** |

When the only "same-vocab" teacher is the student's own larger sibling,
distillation caps the student at the same family's quality. To break
that ceiling you need a cross-vocab teacher — but no clean public tool
exists to handle the alignment correctly.

This library fixes that.

## What's novel

| Component | Public state before this lib |
|---|---|
| Vocab string-mapping (mergekit `tokenizer_source: union`) | exists for **weight merge**, not for distillation |
| Cross-tokenizer KL with naive top-K projection | scattered in research code, no clean lib |
| **Per-position alignment between two tokenizations of the same text** | no clean public solution |
| **Byte-boundary distillation (only at common offsets)** | mentioned in DistillSpec, no library |
| Universal Logit Distillation Loss (ULD, 2024) | paper, no clean impl |
| Sparse mapping cache + HF Trainer integration | doesn't exist as drop-in |
| Diagnostic CLI ("does CTD work for THIS pair before I spend $50?") | doesn't exist |

## Design at a glance

```
text  ──▶ teacher_tokenize ──▶ teacher_input_ids ──▶ teacher.forward ──▶ teacher_logits
                                                                              │
                                                                              ▼
                                                                  ┌──── projection M ────┐
                                                                  │ (student_vocab,      │
                                                                  │  teacher_vocab)      │
                                                                  └──────────────────────┘
                                                                              │
                                                                              ▼
text  ──▶ student_tokenize ──▶ student_input_ids ──┐                  projected_top_K
                                                   │                          │
                                                   ▼                          ▼
                                          alignment_table   ─────▶  distillation_target
                                          (student_pos →             at student positions
                                           teacher_pos)
```

Two pluggable strategies for where to fetch teacher's distribution:

1. `byte_anchor` — only use teacher's logit when teacher and student
   tokenization end on the same byte offset. Skip non-aligned positions.
   Cheap, lossy (~30-50% of positions skipped for typical pairs).

2. `student_offset` (default) — for every student position, materialise
   teacher's distribution at that exact byte offset. For aligned
   positions: use teacher's natural logit. For non-aligned: tokenize
   the suffix string from the previous teacher boundary to the student
   boundary, advance teacher with KV-cache reuse, capture the next-token
   logit. ~1.5-2× the compute of `byte_anchor`, full coverage.

## API sketch (what we're building toward)

```python
from ctd import VocabMapper, precompute_aligned_cache, CTDLoss

# 1. Build the projection matrix once
mapper = VocabMapper.from_tokenizers(
    teacher_tokenizer=qwen_tok,
    student_tokenizer=ds_tok,
    multi_token="distribute",   # or "strict"
)
print(mapper.coverage_report())
# >>> teacher mass with single-token student match: 78.4%
# >>> teacher mass with multi-token mapping:        14.2%
# >>> teacher mass dropped (no mapping):             7.4%

# 2. Precompute aligned cache once (offline, on a beefy pod)
precompute_aligned_cache(
    teacher_model=teacher,
    teacher_tokenizer=qwen_tok,
    student_tokenizer=ds_tok,
    text_corpus=stream,
    output_path="cache.pt",
    top_k=32,
    alignment="student_offset",
    suffix_reencode=True,
    projection=mapper,
)

# 3. Train student with the cache (drop-in for HF Trainer)
loss_fn = CTDLoss(kind="kl", temperature=1.0)
loss = loss_fn(student_logits, cache_top_k_values, cache_top_k_indices)
```

## Repository layout

```
cross-tokenizer-distill/
├── ctd/
│   ├── __init__.py
│   ├── mapper.py          # VocabMapper, sparse remapping matrix
│   ├── strategies.py      # multi-token mapping policies
│   ├── alignment.py       # byte-offset alignment + suffix re-encode
│   ├── projection.py      # project teacher dist → student vocab space
│   ├── losses.py          # KL/JSD/MSE/ULD-style losses on projected dist
│   ├── precompute.py      # standalone teacher-cache builder
│   └── trainer_hooks.py   # generic compute_loss override
├── adapters/
│   ├── transformers.py    # HF Trainer plug-in (drop-in)
│   ├── trl.py             # TRL SFTTrainer plug-in
│   └── lightning.py       # PyTorch Lightning module
├── cli/
│   ├── precompute.py      # ctd-precompute CLI entry
│   ├── inspect.py         # ctd-inspect: coverage report for any (T, S) pair
│   └── benchmark.py       # ctd-benchmark: smoke A/B on toy task
├── experiments/
│   └── validation/        # the small-models proof-of-concept (see docs/VALIDATION.md)
├── tests/
└── docs/
    ├── DESIGN.md          # alignment math, projection semantics, loss derivations
    ├── VALIDATION.md      # small-models experiment plan and results
    └── INTEGRATION.md     # how to wire CTD into your training loop
```

## Validation plan (before any production use)

See `docs/VALIDATION.md` for the full experimental design.

In short — a 3-way A/B/C on small (~0.5-1B) student models across two
tokenizer pairs:

| Run | Student | Teacher | Vocab match | Method |
|---|---|---|---|---|
| A | Qwen2.5-Coder-0.5B | none | n/a | SFT baseline (no distill) |
| B | Qwen2.5-Coder-0.5B | Qwen2.5-Coder-7B | yes | standard KL distill |
| C | Qwen2.5-Coder-0.5B | DeepSeek-Coder-V2-Lite-Instruct | **no** | **CTD** |

Same training corpus (~5K code samples, e.g. Magicoder-OSS-Instruct subset),
same LoRA recipe, same epochs. Eval on HumanEval+ / MBPP+ / LCB-medium-30
on a 3090. ~2h per run on a 3090, ~6h total.

Decision rules:
- **C ≥ 0.8×B and C > A** → CTD works, ship it. Apply to Mythic-RDT v6U with
  Qwen3-Coder-Next teacher.
- **A < C < 0.8×B** → CTD adds signal but loses fidelity. Re-tune (try
  `byte_anchor`, multi-token strategies).
- **C ≤ A** → vocab projection is destroying signal. Fall back to
  same-vocab teacher (DeepSeek-Coder-V2-236B) for Mythic-RDT.

## Status

Pre-alpha. Scaffolded 2026-05-02. First user: Mythic-RDT v6U.

**Validation in progress** — see [`docs/RESULTS.md`](docs/RESULTS.md) for live numbers.
Highlights from the pod 35822024 run (DS-Coder-1.3B-Instruct student,
HumanEval-164 + MBPP-378 full sets):

| Run | Method | HE-164 | Δ vs base |
|---|---|---|---|
| BASE | no-FT | 59.8 % | — |
| M3 (same-vocab GKD, JSD β=0.5) | KL distill | 55.5 % | −4.3 |
| M4 (same-vocab MiniLLM, RKL on-policy) | reverse-KL distill | 54.3 % | −5.5 |
| M5 (same-vocab DistillSpec/FKL on-policy) | forward-KL distill | 56.1 % | −3.7 |
| SFT (mbpp_train, same recipe as M3) | cross-entropy | 51.8 % | −8.0 |
| **M6 (cross-vocab CTD, Qwen2.5-Coder-7B teacher)** | **CTD on-policy FKL** | **38.4 %** | **−21.4 (FAIL)** |

Headline finding from M3/M4/M5: **distillation regularises vs SFT by
3.7–4.3 pp on HE at the same recipe** — the teacher signal stops the small
student from overfitting the narrow training set, but no same-vocab variant
beats the no-FT base on this corpus. M6 (cross-vocab CTD with a 7B teacher),
M7 (capacity test, rank 64 + 4 ep), and M8 (mixed corpus) are in flight to
break the ceiling.

## License

MIT — see [LICENSE](LICENSE).
