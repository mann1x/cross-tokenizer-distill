#!/bin/bash
# Sweep distill recipes for run_B (same-vocab) until one beats run_A on HE-20 smoke.
#
# For each recipe: re-train (~14 min) → HE-20 smoke (~3 min) → log.
# Reference baselines (HE-20 control):
#   A = 11/20 (5 success + 6 b_regressions + 0 b_improves + 0 stretch)
#   B = 7/20  (5 success + 0 b_regressions + 2 b_improves + 0 stretch)
# Target: ≥12/20 with success_control = 5/5 AND b_regressions recovered.

set -uo pipefail
cd /workspace/cross-tokenizer-distill/experiments/validation
source /workspace/venv-tf4/bin/activate
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/cross-tokenizer-distill

CACHE_B=cache_B/qwen25_coder_7b_top32.pt
SWEEP_DIR=runs/sweep
SMOKE_DIR=results/smoke
mkdir -p $SWEEP_DIR $SMOKE_DIR

# Each recipe: NAME CTD_WEIGHT KL_TEMP KIND EPOCHS LR
RECIPES=(
    "B01_w0.1_T1.0  0.1 1.0 kl 2 1e-4"
    "B02_w0.3_T1.0  0.3 1.0 kl 2 1e-4"
    "B03_w0.5_T2.0  0.5 2.0 kl 2 1e-4"
    "B04_w0.5_T4.0  0.5 4.0 kl 2 1e-4"
    "B05_w1.0_T2.0  1.0 2.0 kl 2 1e-4"
    "B06_w0.3_T2.0_ep4  0.3 2.0 kl 4 1e-4"
    "B07_w0.5_T1.0_lr5e5  0.5 1.0 kl 2 5e-5"
    "B08_jsd_w0.5_T1.0  0.5 1.0 jsd 2 1e-4"
)

LOG=results/sweep_summary.tsv
echo -e "name\tctd_w\tkl_T\tkind\tepochs\tlr\tsuccess\tb_imp\tb_reg\tstretch\tTOTAL\toverall_pct" > $LOG

for entry in "${RECIPES[@]}"; do
    read NAME W T KIND EP LR <<<"$entry"
    OUT_DIR=$SWEEP_DIR/$NAME
    SMOKE_OUT=$SMOKE_DIR/${NAME}.json
    if [ -f "$SMOKE_OUT" ]; then
        echo "[sweep] SKIP $NAME (smoke exists)"
        continue
    fi
    echo "=== $NAME (w=$W T=$T kind=$KIND ep=$EP lr=$LR) ==="
    python 05_train.py --run-name B --output-dir $OUT_DIR \
        --cache $CACHE_B \
        --ctd-weight $W --kl-temperature $T --ctd-kind $KIND \
        --epochs $EP --lr $LR \
        2>&1 | tail -3
    python smoke_he20.py --adapter $OUT_DIR --output $SMOKE_OUT 2>&1 | tail -7
    python3 -c "
import json
r = json.load(open('$SMOKE_OUT'))
b = r['buckets']
o = r['overall']
print(f\"$NAME\t$W\t$T\t$KIND\t$EP\t$LR\t{b['success_control']['n_pass']}\t{b['b_improves']['n_pass']}\t{b['b_regressions']['n_pass']}\t{b['both_fail_stretch']['n_pass']}\t{o['n_pass']}\t{o['pass@1']:.1%}\"" >> $LOG
done

echo ""
echo "=== SWEEP DONE ==="
column -t -s $'\t' $LOG
