#!/bin/bash
# Sweep distill recipes for run_B (same-vocab) — v2 after T² + teacher-T + warmup fixes.
#
# Bugs fixed since v1 sweep:
#   1. losses.py: missing T² rescale on KL/JSD/MSE/ULD — silently shrank
#      distill gradient by 1/T² as T rose. Made T>1 recipes degenerate.
#   2. losses.py: teacher cache was at T=1 but student was softened at T;
#      now both are softened consistently (treats cached top-K log-probs
#      as proxy logits, divides by T, re-log_softmax over the K support).
#   3. adapters/transformers.py + 05_train.py: added --ctd-weight-warmup-steps
#      to ramp distill weight 0→target over N steps. Compensates for LoRA
#      cold-start (lora_B inits to 0 → first ~100 steps the student is
#      base-model-identical and distill gradient is wasted).
#
# v1 reference (still valid since T=1 was unaffected by bug #1/2):
#   A = 11/20  (5 success_control + 6 b_regressions + 0 b_improves + 0 stretch)
#   B (old broken)  = 7/20  (5 + 0 + 2 + 0)
#   B02_w0.3_T1.0   = 11/20 (4 + 1 + 3 + 3)  ← best v1 recipe
#
# Target: ≥12/20 with success_control = 5/5 AND ≥3 b_regressions recovered.

set -uo pipefail
cd /workspace/cross-tokenizer-distill/experiments/validation
source /workspace/venv-tf4/bin/activate
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/cross-tokenizer-distill

CACHE_B=cache_B/qwen25_coder_7b_top32.pt
SWEEP_DIR=runs/sweep_v2
SMOKE_DIR=results/smoke_v2
mkdir -p $SWEEP_DIR $SMOKE_DIR

# Each recipe: NAME CTD_WEIGHT KL_TEMP KIND EPOCHS LR WARMUP_STEPS
RECIPES=(
    "R01_w0.3_T1.0_warm0     0.3 1.0 kl  2 1e-4 0"
    "R02_w0.3_T1.0_warm100   0.3 1.0 kl  2 1e-4 100"
    "R03_w0.3_T2.0_warm100   0.3 2.0 kl  2 1e-4 100"
    "R04_w0.5_T2.0_warm100   0.5 2.0 kl  2 1e-4 100"
    "R05_w0.5_T2.0_warm100j  0.5 2.0 jsd 2 1e-4 100"
    "R06_w0.3_T1.0_warm100_ep4 0.3 1.0 kl 4 1e-4 100"
)

LOG=results/sweep_v2_summary.tsv
echo -e "name\tctd_w\tkl_T\tkind\tepochs\tlr\twarmup\tsuccess\tb_imp\tb_reg\tstretch\tTOTAL\toverall_pct" > $LOG

for entry in "${RECIPES[@]}"; do
    read NAME W T KIND EP LR WS <<<"$entry"
    OUT_DIR=$SWEEP_DIR/$NAME
    SMOKE_OUT=$SMOKE_DIR/${NAME}.json
    if [ -f "$SMOKE_OUT" ]; then
        echo "[sweep-v2] SKIP $NAME (smoke exists)"
        continue
    fi
    echo "=== $NAME (w=$W T=$T kind=$KIND ep=$EP lr=$LR warm=$WS) ==="
    python -u 05_train.py --run-name B --output-dir $OUT_DIR \
        --cache $CACHE_B \
        --ctd-weight $W --kl-temperature $T --ctd-kind $KIND \
        --epochs $EP --lr $LR \
        --ctd-weight-warmup-steps $WS \
        2>&1 | tail -3
    python -u smoke_he20.py --adapter $OUT_DIR --output $SMOKE_OUT 2>&1 | tail -7
    python3 -c "
import json
r = json.load(open('$SMOKE_OUT'))
b = r['buckets']
o = r['overall']
print(f\"$NAME\t$W\t$T\t$KIND\t$EP\t$LR\t$WS\t{b['success_control']['n_pass']}\t{b['b_improves']['n_pass']}\t{b['b_regressions']['n_pass']}\t{b['both_fail_stretch']['n_pass']}\t{o['n_pass']}\t{o['pass@1']:.1%}\"" >> $LOG
done

echo ""
echo "=== SWEEP-V2 DONE ==="
cat $LOG
