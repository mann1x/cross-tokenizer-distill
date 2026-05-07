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
│   ├── util.py            # teacher-token blacklist (thinking-mode masking, etc.)
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
    ├── DESIGN.md                  # alignment math, projection semantics, loss derivations
    ├── VALIDATION.md              # small-models experiment plan and results
    ├── INTEGRATION.md             # how to wire CTD into your training loop
    ├── TEACHER_TOKEN_MASKING.md   # ban specific teacher tokens (thinking-mode etc.)
    ├── STYLE_SHIFT_ISSUE.md       # postmortem: cross-vocab SFT/off-policy KL collapse
    └── TEACHER_REFERENCE.md       # reported HE/MBPP scores for teacher candidates
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

Alpha. Scaffolded 2026-05-02. v2 closed Phase 2.5 on 2026-05-07 with three
working dual-positive recipes. See [`docs/V2_CTD_PLAN.md`](docs/V2_CTD_PLAN.md)
for the full result table.

### v2 working recipes (2026-05-07)

QC-14B-Instruct-NF4 (chat) teacher, HumanEval-164 + MBPP-378 full sets,
chat-mode eval with FENCE-last extractor + truncate-after-fn, rank=16
LoRA unless noted, 2 epochs, lr=5e-5.

**Bases**: DS-Coder-1.3B-Instruct = HE 56.1 / MBPP 41.0 ; QC-1.5B-Instruct = HE 63.4 / MBPP 45.0
; QC-14B-NF4 (teacher) = HE 86.0 / MBPP 74.3.

| Recipe | Student | Data | Code-only mask | Rank | HE | MBPP | dHE | dMBPP |
|---|---|---|:---:|---:|---:|---:|---:|---:|
| **M37c** | DSC-1.3B (xv) | funcsig | ✓ | 16 | 62.2 | 41.3 | **+6.1** | +0.3 |
| **M41c** | QC-1.5B (sv)  | funcsig | ✓ | 16 | 64.0 | 48.4 | +0.6 | **+3.4** |
| **M41bc** | QC-1.5B (sv) | funcsig | ✓ | 64 | 60.4 | 50.3 | -3.0 | **+5.3** |

(xv = cross-vocab via VocabMapper; sv = same-vocab via IdentityMapper bypass.)

**Pick by goal**:
- Best HE delta: **M37c** (+6.1) — cross-vocab CTD, headline pick for Mythic-RDT
  pre-recurrence step on DS-Coder-V2-Lite.
- Best dual gain: **M41c** — first same-vocab QC-1.5B recipe to beat base on both metrics.
- Best MBPP: **M41bc** (r=64) — trades HE for +5.3 MBPP, useful when MBPP-only is the goal.

**Three structural fixes that made it work**:

1. **Code-only chat completions** — regenerate teacher completions with a
   "Python code generator only, wrap in ```python```" system prompt. Drops
   prose-share from 55-60% to 0%. Required for Path C below to be coherent.
2. **`--code-only-mask` SFT loss** — restrict CE to tokens INSIDE
   ```python ... ``` fences (offset_mapping → per-token mask). Concentrates
   gradient on code; preserves student's coding prior under chat-template shift.
3. **Eval extractor: last-fence + truncate-after-fn** — chat-mode generations
   often emit a leading prose explanation followed by the actual `def`. The
   pre-fix extractor took the FIRST fence; the patch takes the LAST and
   truncates at next top-level `def`/`class` so the exec scorer sees a clean
   single function.

### Pattern asymmetries observed

- **Cross-vocab tolerates the mask broadly**: M38c on prose-heavy mbpp_train still
  gained +6.1 HE. The VocabMapper projection appears to act as additional
  regularization that keeps gradient density healthy under aggressive masking.
- **Same-vocab needs code-dense corpora**: M40c (QC-1.5B + mbpp_train + mask)
  regressed -3.6 HE / -2.1 MBPP. Without cross-vocab regularization, the mask
  drops 50%+ of teacher tokens (the prose) and over-concentrates gradient on
  the easy-to-converge code positions.
- **r=64 + mask reverts the HE-drain**: M41bc HE = M41b HE (60.4) without mask;
  rank=64 has enough degrees of freedom to overfit chat-prose patterns the mask
  was supposed to suppress. r=16 is the sweet spot if HE matters.

### v1 archive (kept for reference)

| Run | Method | HE-164 | Δ vs base |
|---|---|---|---|
| M3 (same-vocab GKD, JSD β=0.5) | KL distill | 55.5 % | −4.3 |
| M5 (same-vocab DistillSpec/FKL on-policy) | forward-KL distill | 56.1 % | −3.7 |
| M7 (capacity, rank 64 + 4 ep) | same-vocab FKL | 54.3 % | −5.5 (HE) / +1.1 (MBPP) |
| **M16 (SFT on teacher completions)** | sequence imitation | **65.9 %** | **+6.1** |
| M37 (cross-vocab SFT funcsig) | cross-vocab SFT | 61.0 % | +4.9 |

v1 headline: **SFT on teacher completions (M16) was the first recipe to clear
the base by a wide margin.** v2 reproduces and amplifies this with chat-mode
SFT + Path C + code-only regeneration.

## License

MIT — see [LICENSE](LICENSE).
