# Integration guide — wiring CTD into your training loop

CTD is designed to be a drop-in replacement for same-vocab logit
distillation. If your trainer already accepts a precomputed teacher
cache of shape `[N_tokens, top_K]` for values + indices (the format
established in Mythic-RDT and used by various other recipes), the
only change you need is **how the cache is built**.

## Setup

```bash
# In your project root
git clone https://github.com/<TBD>/cross-tokenizer-distill.git ../cross-tokenizer-distill
pip install -e ../cross-tokenizer-distill
```

Or use as a subdirectory dependency / git submodule.

## The two integration patterns

### Pattern 1 — Cache projected at write time (recommended)

CTD projects teacher's distribution onto student vocab during
precompute. The cache file you ship to your training pod is in
**student vocab indices**. Your trainer doesn't need to know CTD
exists.

```python
# precompute.py (run once, on a beefy pod)
from ctd import VocabMapper, precompute_aligned_cache

mapper = VocabMapper.from_tokenizers(
    teacher_tokenizer=teacher_tok,
    student_tokenizer=student_tok,
)
print(mapper.coverage_report())  # decide go/no-go

precompute_aligned_cache(
    teacher_model=teacher_model,
    teacher_tokenizer=teacher_tok,
    student_tokenizer=student_tok,
    text_corpus=load_corpus(),
    output_path="cache.pt",
    top_k=32,
    alignment="student_offset",
    suffix_reencode=True,
    projection=mapper,
    project_at_write_time=True,   # <-- key flag
)
```

The resulting `cache.pt` has the **same shape** as a same-vocab
top-K cache. Drop it into your existing trainer:

```python
# Your existing training code, unchanged
cache = torch.load("cache.pt")
teacher_top_k_values = cache["values"]    # [N_tokens, top_K], student-vocab logits
teacher_top_k_indices = cache["indices"]  # [N_tokens, top_K], student-vocab IDs
# ... usual KL distillation against student logits at each position
```

### Pattern 2 — Project at training time

If you want flexibility (e.g. swap projection strategies without
re-precompute), keep the cache in teacher vocab and project on the
fly. Higher per-step cost, more flexibility:

```python
# precompute (offline)
precompute_aligned_cache(
    ...,
    project_at_write_time=False,    # store teacher's native top-K
)

# trainer (every step)
from ctd import CTDLoss

loss_fn = CTDLoss(
    mapper=mapper,           # projection applied per-step
    kind="kl",
    temperature=1.0,
)
loss = loss_fn(
    student_logits,           # [B, L, V_student]
    teacher_top_k_values,     # [B, L, top_K], teacher-vocab indices
    teacher_top_k_indices,    # [B, L, top_K]
)
```

## HF Trainer adapter

For HuggingFace `Trainer`, override `compute_loss`:

```python
from ctd.adapters.transformers import CTDTrainerMixin

class MyTrainer(CTDTrainerMixin, Trainer):
    def __init__(self, *args, ctd_cache_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctd_cache = torch.load(ctd_cache_path)
        self.ctd_loss = CTDLoss(...)

    # CTDTrainerMixin overrides compute_loss to add the distill term
```

## Mythic-RDT specific notes

In `Mythic-RDT/src/mythic_rdt/training/trainer.py` the existing teacher
cache loader expects shape `[N_blocks, top_K]` with `(values,
indices)`. CTD's `project_at_write_time=True` produces output in this
exact shape — substitute the cache path and the trainer needs no
code changes.

The training script (`scripts/finetune_phase1.py`) already accepts
`--teacher-logits <path>`. To switch from same-vocab DS-Coder-V2-Lite
teacher to CTD-projected Qwen3-Coder-Next teacher, only the
`--teacher-logits` argument changes.

```bash
# Same-vocab (current, v6Q/v6R)
python scripts/finetune_phase1.py \
    --teacher-logits teacher_cache/dscoder_v2_lite_bf16_top32_seed0.pt \
    ...

# CTD with Qwen3-Coder-Next teacher (v6U candidate)
python scripts/finetune_phase1.py \
    --teacher-logits teacher_cache/qwen3_coder_next_via_ctd_top32_seed0.pt \
    ...
```

## Diagnostic before commit

Always run `ctd-inspect` before launching a precompute:

```bash
ctd-inspect \
    --teacher-tokenizer Qwen/Qwen3-Coder-Next \
    --student-tokenizer deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
    --sample-corpus path/to/sample.jsonl \
    --strategies strict,distribute,first_token
```

Expected output:

```
=== Coverage report (Qwen3-Coder-Next → DS-Coder-V2-Lite) ===

Tokenizer pair: Qwen-tokenizer (V=151936) → DeepseekTokenizerFast (V=102400)
Both byte-level BPE: True
Round-trip byte-exact: True

Strategy: strict
  Single-token map rate: 78.4%
  Average mass per teacher position: 0.74
  Position alignment hit rate (byte_anchor): 62.1%

Strategy: distribute
  Single-token mass:    78.4%
  Multi-token mass:     14.2%  (avg fragments: 2.3)
  Dropped mass:          7.4%
  Average mass retained: 0.93
  Position alignment hit rate (student_offset): 100%
  Suffix re-encode rate (positions needing extra teacher fwd): 38.0%

Recommendation: GO with distribute + student_offset
```

If recommendation is NO-GO (e.g. tokenizers incompatible due to
normalization differences), error out before the user spends $50 on
precompute.
