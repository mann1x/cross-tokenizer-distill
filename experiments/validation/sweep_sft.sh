#!/bin/bash
# Sweep SFT recipes — find a recipe where SFT-only beats DS-Coder-1.3B base.
#
# Hypothesis: lr=1e-4 / cosine / warmup=100 / ep=2 with the small 5K corpus
# is too aggressive — peaks LR for too long, never anneals enough, causes
# mild catastrophic forgetting on Python code. Try gentler LRs, fewer epochs,
# different LoRA ranks, no-cosine schedule.
#
# Target: BASE smoke score (computed first by base_baseline.sh).
#         Winner = highest TOTAL with success_control >= 4 and TOTAL > BASE.
set -uo pipefail
cd /workspace/cross-tokenizer-distill/experiments/validation
source /workspace/venv-tf4/bin/activate
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/cross-tokenizer-distill

SWEEP_DIR=runs/sweep_sft
SMOKE_DIR=results/smoke_sft
mkdir -p $SWEEP_DIR $SMOKE_DIR

# Each: NAME LR EPOCHS LORA_R WARMUP SCHED
RECIPES=(
    "S01_lr5e5_ep2_r16        5e-5 2 16  30 cosine"
    "S02_lr2e5_ep2_r16        2e-5 2 16  30 cosine"
    "S03_lr5e5_ep1_r16        5e-5 1 16  15 cosine"
    "S04_lr1e4_ep2_r32        1e-4 2 32  30 cosine"
    "S05_lr5e5_ep2_r64        5e-5 2 64  30 cosine"
    "S06_lr1e4_ep2_r8         1e-4 2 8   30 cosine"
    "S07_lr5e5_ep2_r16_const  5e-5 2 16  30 constant_with_warmup"
)

LOG=results/sweep_sft_summary.tsv
echo -e "name\tlr\tep\trank\tws\tsched\tsuccess\tb_imp\tb_reg\tstretch\tTOTAL\toverall_pct" > $LOG

for entry in "${RECIPES[@]}"; do
    read NAME LR EP RANK WS SCHED <<<"$entry"
    OUT_DIR=$SWEEP_DIR/$NAME
    SMOKE_OUT=$SMOKE_DIR/${NAME}.json
    if [ -f "$SMOKE_OUT" ]; then
        echo "[sft-sweep] SKIP $NAME (smoke exists)"
        continue
    fi
    echo ""
    echo "=== $NAME (lr=$LR ep=$EP rank=$RANK ws=$WS sched=$SCHED) ==="
    python -u 05_train.py --run-name A --output-dir $OUT_DIR \
        --lora-rank $RANK \
        --lr $LR --epochs $EP \
        --warmup-steps $WS --lr-scheduler $SCHED \
        2>&1 | tail -3
    python -u smoke_he20.py --adapter $OUT_DIR --output $SMOKE_OUT 2>&1 | tail -7
    SMOKE_PATH=$SMOKE_OUT NAME=$NAME LR=$LR EP=$EP RANK=$RANK WS=$WS SCHED=$SCHED \
    python3 << "PYEOF" >> $LOG
import json, os
r = json.load(open(os.environ["SMOKE_PATH"]))
b = r["buckets"]; o = r["overall"]
pct = "{:.1%}".format(o["pass@1"])
fields = [os.environ["NAME"], os.environ["LR"], os.environ["EP"], os.environ["RANK"],
          os.environ["WS"], os.environ["SCHED"],
          str(b["success_control"]["n_pass"]), str(b["b_improves"]["n_pass"]),
          str(b["b_regressions"]["n_pass"]), str(b["both_fail_stretch"]["n_pass"]),
          str(o["n_pass"]), pct]
print("\t".join(fields))
PYEOF
done

echo ""
echo "=== SFT-SWEEP DONE ==="
cat $LOG
