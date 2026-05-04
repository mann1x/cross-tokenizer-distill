#!/bin/bash
# Phase 8 (post-Phase 7): full HE-164 on top-2 SFT recipes by smoke total
# (in case the smoke set is biased; gives us multiple data points vs base).
# Skips the recipe Phase 5 already evaluated (Phase 5 picks rank-1 by smoke).
set -uo pipefail
cd /workspace/cross-tokenizer-distill/experiments/validation
source /workspace/venv-tf4/bin/activate
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/cross-tokenizer-distill

LOGDIR=logs/overnight
mkdir -p $LOGDIR results/full_sft

echo "[p8] waiting for BASE_HE164.json (Phase 7 done)..."
until [ -f results/full_sft/BASE_HE164.json ]; do sleep 60; done
echo "[p8] Phase 7 done at $(date -u +%FT%TZ)"

# Pick top-3 SFT recipes by smoke total, skip those already eval-ed (Phase 5)
TOP3=$(python3 << "PYEOF"
import json
from pathlib import Path
ranked = []
for f in Path("results/smoke_sft").glob("S*.json"):
    r = json.load(open(f))
    if r["buckets"]["success_control"]["n_pass"] < 3: continue
    ranked.append((r["overall"]["n_pass"], f.stem))
ranked.sort(reverse=True)
print(" ".join(name for _,name in ranked[:3]))
PYEOF
)
echo "[p8] top-3 recipes: $TOP3"

for NAME in $TOP3; do
    OUT=results/full_sft/${NAME}_HE164.json
    if [ -f "$OUT" ]; then
        echo "[p8] SKIP $NAME (full HE exists)"
        continue
    fi
    if [ ! -d "runs/sweep_sft/$NAME" ]; then
        echo "[p8] SKIP $NAME (no adapter dir)"
        continue
    fi
    echo "[p8] full HE-164 on $NAME"
    python -u 06_eval.py \
        --adapter runs/sweep_sft/$NAME \
        --output  $OUT \
        --cache-db results/full_sft/${NAME}_HE164.db \
        --he-limit 164 --skip-mbpp --exec-timeout 30 \
        2>&1 | tee $LOGDIR/p8_${NAME}.log | tail -3
done

# Append final scoreboard
python3 << "PYEOF" >> results/decision_overnight.md
import json
from pathlib import Path
print()
print("## Phase 8: top-3 SFT recipes — full HE-164 vs BASE")
b = json.load(open("results/full_sft/BASE_HE164.json"))["humaneval"]
base_pass = b["n_pass"]
print(f"- BASE = {base_pass}/{b[\"n_total\"]} = {b[\"pass@1\"]:.1%}")
rows = []
for f in sorted(Path("results/full_sft").glob("*_HE164.json")):
    if f.stem == "BASE_HE164": continue
    r = json.load(open(f))["humaneval"]
    delta = r["n_pass"] - base_pass
    rows.append((delta, f.stem, r["n_pass"], r["n_total"], r["pass@1"]))
rows.sort(reverse=True)
for delta, name, n_pass, n_total, pct in rows:
    flag = "WIN" if delta > 0 else ("TIE" if delta == 0 else "LOSE")
    print(f"- **{name}** ({flag}): {n_pass}/{n_total} = {pct:.1%} (delta_vs_base = {delta:+d})")
PYEOF

echo "[p8] DONE $(date -u +%FT%TZ)"
