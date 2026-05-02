#!/bin/bash
# Wrapper — run the entire A/B/C validation end-to-end.
#
# Expects:
#   - solidPC 3090 idle (will block ~16h)
#   - conda env 'ctd' active (or pip-installed package)
#   - HF datasets accessible (Magicoder corpus, HumanEval, MBPP)
#
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p results data cache_B cache_C runs

echo "[run_all] Step 1 — inspect both vocab pairs"
bash 01_inspect.sh

echo "[run_all] Step 2 — prepare 5K corpus"
python 02_prepare_corpus.py --n-samples 5000 --output data/corpus_5k.jsonl

echo "[run_all] Step 3 — precompute SAME-VOCAB cache (Pair B)"
python 03_precompute_B.py \
    --corpus data/corpus_5k.jsonl \
    --output cache_B/qwen25_coder_7b_top32.pt \
    --top-k 32 --max-seq-len 1024 --quant nf4

echo "[run_all] Step 4 — precompute CTD cache (Pair C)"
python 04_precompute_C.py \
    --corpus data/corpus_5k.jsonl \
    --output cache_C/dscoder_v2_lite_via_ctd_top32.pt \
    --top-k 32 --max-seq-len 1024 --quant nf4 \
    --multi-token distribute --alignment student_offset

echo "[run_all] Step 5A — train SFT-only baseline (run A)"
python 05_train.py --run-name A --output-dir runs/run_A

echo "[run_all] Step 5B — train same-vocab distill (run B)"
python 05_train.py --run-name B --output-dir runs/run_B \
    --cache cache_B/qwen25_coder_7b_top32.pt --ctd-weight 0.5

echo "[run_all] Step 5C — train CTD distill (run C)"
python 05_train.py --run-name C --output-dir runs/run_C \
    --cache cache_C/dscoder_v2_lite_via_ctd_top32.pt --ctd-weight 0.5

echo "[run_all] Step 6 — eval all three"
for R in A B C; do
    python 06_eval.py \
        --adapter runs/run_${R} \
        --output results/${R}.json \
        --he-limit 164 --mbpp-limit 378
done

echo "[run_all] Step 7 — comparison + decision"
python 07_compare.py \
    --A results/A.json --B results/B.json --C results/C.json \
    --output results/decision.md

echo "[run_all] DONE. See results/decision.md."
cat results/decision.md
