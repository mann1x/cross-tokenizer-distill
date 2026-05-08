# Plan #115 — Sanity-check M37c recipe on DSC-V2-Lite (Mythic-RDT pre-recurrence)

## Goal

Validate that the M37c gain (+6.1 HE / +0.3 MBPP on DSC-1.3B) **transfers** to DSC-V2-Lite
(16B MoE, 64 experts top-6, MLA + shared experts) before any Mythic-RDT recurrence wrap.

If transfer holds: DSC-V2-Lite + M37c → HE ~81.7 / MBPP ~60.9.
Mythic-Coder T=8 target is HE ≥ 80.6, so M37c alone clears the bar — recurrence becomes
upside, not a requirement.

## Why this matters before recurrence

- DSC-V2-Lite recurrence wrap (Mythic-RDT) is expensive: $35-55 pod time per training run,
  4-5 days of cumulative experimentation per recipe family (v6A→v6R lineage proof).
- If M37c alone clears the target, we have a published artifact (`Mythic-Coder-CTD`) WITHOUT
  needing to debug recurrence — and the recurrence wrap is then a "T=8 bonus" optimization.
- If M37c does NOT transfer (e.g. MLA + shared experts behave differently under code-only
  mask), we know to investigate that BEFORE adding the recurrence complexity layer.

## Recipe (single experiment)

Same hyperparameters as M37c on DSC-1.3B:

| Param | Value | Notes |
|---|---|---|
| student | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | 16B MoE; ~50 GB BF16 |
| teacher | QC-14B-NF4 (cached) | Same as M37c |
| corpus  | `funcsig_prompts_qwen25c14b_codeonly_T07.jsonl` (474 prompts) | Same |
| mask    | `--code-only-mask` | Same |
| LoRA rank | 16 | Same; do NOT increase (M41bc lesson: r=64 reverts the gain) |
| epochs  | 2 | Same |
| lr      | 5e-5 | Same |
| batch_size | 1 (gradient_accum 16) | Reduced from M37c bs=2 ga=8 due to 16B size |
| max-prompt-len | 512 | Same |
| max-total-len | 1408 | Same |

VRAM estimate: 16B MoE BF16 + LoRA r=16 + activations at bs=1/seq=1408 → ~30-35 GB. Fits
the 48 GB Ada 6000 with margin.

## Compute budget

- Training: 474 prompts × 2 epochs × bs=1 ga=16 → ~60 optim steps. At ~5-8 s/step on 16B MoE
  → 5-8 minutes wall-clock.
- Eval HE-164 + MBPP-378 (chat-mode, FENCE-last, truncate-after-fn) → ~25-30 minutes on 16B.
- Total: ~30-40 minutes.

## Decision rule

- **dHE ≥ +3** → commit M37c-on-DSC-V2-Lite as the official Mythic-Coder-CTD pre-recurrence
  step. Tag as `ManniX-ITA/Mythic-Coder-CTD-V2-Lite`. Then proceed to recurrence wrap with
  this as the new starting checkpoint.
- **dHE in [-1, +3]** → marginal transfer. Investigate: try M42-style combined corpus on
  DSC-V2-Lite, then decide.
- **dHE < -1** → transfer fails. Likely cause: MLA / shared experts react differently to
  the loss mask. Rollback decision: skip CTD pre-recurrence, train recurrence directly on
  DSC-V2-Lite-Instruct base.

## Driver template (run AFTER Phase 3 completes)

```bash
#!/bin/bash
set -uo pipefail
cd /workspace/cross-tokenizer-distill/experiments/validation
source /workspace/venv-tf4/bin/activate
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 \
       PYTHONPATH=/workspace/cross-tokenizer-distill

while pgrep -f "06_eval_batched|06_train_sft|06_train_onpolicy" > /dev/null; do sleep 30; done

NAME=M115_DSCV2LITE_M37c
OUT=runs/v2/${NAME}
DB=results/full_sft/${NAME}_HE_MBPP.db
JSON=results/full_sft/${NAME}_HE_MBPP.json

python -u 06_train_sft_on_teacher.py \
    --student deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
    --corpus data/funcsig_prompts_qwen25c14b_codeonly_T07.jsonl \
    --output-dir $OUT \
    --max-prompt-len 512 --max-total-len 1408 \
    --lora-rank 16 --lr 5e-5 --epochs 2 --batch-size 1 --grad-accum 16 \
    --warmup-steps 8 --logging-steps 5 --seed 0 \
    --code-only-mask

ADAPTER=$(ls -d $OUT/epoch-* 2>/dev/null | sort -V | tail -1)
[ -z "$ADAPTER" ] && ADAPTER=$OUT

python -u 06_eval_batched.py \
    --base-model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
    --quant bf16 --chat-template \
    --adapter $ADAPTER --output $JSON --cache-db $DB \
    --he-limit 164 --mbpp-limit 378 --exec-timeout 30 \
    --batch-size 8 --max-new-tokens 1024
```

Note `--batch-size 8` for eval (not 16) — chat-mode generation on 16B is heavier.

## Risk register

1. **trust_remote_code required**: DSC-V2-Lite uses custom modeling. Tokenizer + AutoModel
   load with `trust_remote_code=True`. Already wired in `06_eval_batched.py`; check
   `06_train_sft_on_teacher.py` does the same (it uses `AutoModelForCausalLM.from_pretrained`
   without `trust_remote_code` — patch needed before launching).
2. **MoE expert routing under LoRA**: LoRA on `target_modules="all-linear"` will hit expert
   gates and expert MLPs. This is the canonical config but may produce noisier gradients than
   on dense DSC-1.3B. If train loss plateaus high (>0.5), try `target_modules="q_proj,k_proj,v_proj,o_proj"`
   to skip experts.
3. **Aux load-balance loss**: DSC-V2-Lite has aux loss α₁=0.001 (MoE balance). Don't disable.
4. **Eval batch 16 OOMs**: drop to bs=8 (reflected in template above).
5. **Gradient accumulation at bs=1 ga=16**: same effective batch as M37c bs=2 ga=8. Same
   total tokens per step. Gradient noise should be comparable.

## Out of scope for #115

- LoRA merge to base before recurrence wrap (decide after Mythic-RDT recurrence converges).
- HumanEval+ / MBPP+ extended evals (decide after #115 lands).
- HF Hub upload (only if dHE ≥ +3).


## Reframe (2026-05-08 03:00 UTC) — control experiment for recurrence

CTD is the **enabler** for Mythic-RDT recurrence, not a competing product. The pipeline is:

```
DSC-V2-Lite-Instruct (base, HE 75.6)
        │
        ▼  ← CTD/SFT lift (M37c-style, this task #115)
DSC-V2-Lite + CTD adapter (target HE 78-82)
        │
        ▼  ← Mythic-RDT recurrence wrap (T=2/4/8 sweep, separate task)
Mythic-Coder T=8 (final product, target HE ≥ 80.6)
```

Each layer is additive and independently measurable:
- **Base** = floor we cannot do worse than
- **CTD adapter** (#115) = cheap LoRA SFT lift to harvest the gap to the QC-14B teacher
- **Recurrence** = the research contribution; T=8 lift over T=1 (which equals the CTD checkpoint) is the clean measure of recurrence value

This means the Mythic-Coder paper/release reports:
- CTD lift Δ_CTD = HE(base + M37c) − HE(base)
- Recurrence lift Δ_REC = HE(base + M37c, T=8) − HE(base + M37c, T=1)
- Total = Δ_CTD + Δ_REC

Each is independently attributable. Wrapping recurrence on a stronger starting checkpoint is the right pipeline ordering — never wrap recurrence around a weaker base when a stronger one is one cheap LoRA away.
