#!/bin/bash
# Phase 9 (post-Phase 8): retrain SFT winner (S03) + R04 distill correctly.
# Phase 4 in the orchestrator failed because the broken TSV writer left the
# winner config vars empty. Fixed here by hardcoding the S03 config:
#   lr=5e-5, ep=1, rank=16, ws=15, sched=cosine
# Then runs HE-20 smoke + full HE-164 vs base.
set -uo pipefail
cd /workspace/cross-tokenizer-distill/experiments/validation
source /workspace/venv-tf4/bin/activate
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/cross-tokenizer-distill

LOGDIR=logs/overnight
mkdir -p $LOGDIR results/full_sft results/smoke_sft

# Wait for Phase 8 done (= the last expected file appears or phase8 process exits)
echo "[p9] waiting for Phase 8 done marker..."
until grep -q "\\[p8\\] DONE" logs/overnight/phase8.log 2>/dev/null; do sleep 60; done
echo "[p9] Phase 8 done at $(date -u +%FT%TZ)"

WINNER=S03_lr5e5_ep1_r16
RUN_DIR=runs/B_${WINNER}_R04_v2

if [ ! -d "$RUN_DIR" ]; then
    echo "[p9] Train ${WINNER} + R04 distill (lr=5e-5 ep=1 rank=16 ws=15 cosine, ctd_w=0.5 T=2.0 warmup=100)"
    python -u 05_train.py --run-name B --output-dir $RUN_DIR \
        --cache cache_B/qwen25_coder_7b_top32.pt \
        --lora-rank 16 \
        --lr 5e-5 --epochs 1 \
        --warmup-steps 15 --lr-scheduler cosine \
        --ctd-weight 0.5 --kl-temperature 2.0 --ctd-kind kl \
        --ctd-weight-warmup-steps 100 \
        2>&1 | tee $LOGDIR/p9_train.log | tail -3
else
    echo "[p9] SKIP train (run dir exists)"
fi

SMOKE_OUT=results/smoke_sft/B_${WINNER}_R04_v2.json
if [ ! -f "$SMOKE_OUT" ]; then
    echo "[p9] Smoke HE-20"
    python -u smoke_he20.py --adapter $RUN_DIR --output $SMOKE_OUT 2>&1 | tee $LOGDIR/p9_smoke.log | tail -7
fi

FULL_OUT=results/full_sft/B_${WINNER}_R04_v2_HE164.json
if [ ! -f "$FULL_OUT" ]; then
    echo "[p9] Full HE-164"
    python -u 06_eval.py \
        --adapter $RUN_DIR \
        --output  $FULL_OUT \
        --cache-db results/full_sft/B_${WINNER}_R04_v2_HE164.db \
        --he-limit 164 --skip-mbpp --exec-timeout 30 \
        2>&1 | tee $LOGDIR/p9_full_he.log | tail -3
fi

# Append to scoreboard
python3 << "PYEOF" >> results/decision_overnight.md
import json
print()
print("## Phase 9: SFT winner + R04 distill (Phase 4 retry)")
b = json.load(open("results/full_sft/BASE_HE164.json"))["humaneval"]
sft_winner = json.load(open("results/full_sft/S03_lr5e5_ep1_r16_HE164.json"))["humaneval"]
r04 = json.load(open("results/full_sft/B_S03_lr5e5_ep1_r16_R04_v2_HE164.json"))["humaneval"]
print(f"- BASE                    : HE {b[\"n_pass\"]}/{b[\"n_total\"]} = {b[\"pass@1\"]:.1%}")
print(f"- S03 (SFT only, winner)  : HE {sft_winner[\"n_pass\"]}/{sft_winner[\"n_total\"]} = {sft_winner[\"pass@1\"]:.1%} (delta_vs_base = {sft_winner[\"n_pass\"]-b[\"n_pass\"]:+d})")
print(f"- S03 + R04 distill (B)   : HE {r04[\"n_pass\"]}/{r04[\"n_total\"]} = {r04[\"pass@1\"]:.1%} (delta_vs_base = {r04[\"n_pass\"]-b[\"n_pass\"]:+d}, delta_vs_S03 = {r04[\"n_pass\"]-sft_winner[\"n_pass\"]:+d})")
PYEOF

echo "[p9] DONE $(date -u +%FT%TZ)"
