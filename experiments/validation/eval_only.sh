#!/bin/bash
set -euo pipefail
cd /workspace/cross-tokenizer-distill/experiments/validation
source /workspace/venv-tf4/bin/activate
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/cross-tokenizer-distill
echo "[eval-only] Step 6 — eval all three (with truncation fix)"
for R in A B C; do
    python 06_eval.py         --adapter runs/run_${R}         --output results/${R}.json         --he-limit 164 --mbpp-limit 378
done
echo "[eval-only] Step 7 — comparison + decision"
python 07_compare.py     --A results/A.json --B results/B.json --C results/C.json     --output results/decision.md
echo "[eval-only] DONE."
cat results/decision.md
