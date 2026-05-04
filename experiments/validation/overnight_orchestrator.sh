#!/bin/bash
# Overnight orchestrator (Auto-mode launched 2026-05-04 ~01:30 CEST):
# 1. Wait for C_R04 decision_R04.md (full eval still in flight on launch)
# 2. Compute BASE smoke (no LoRA) — gives the ceiling SFT must beat
# 3. Derive A_old smoke from existing results/A.json (the failing SFT recipe)
# 4. Run sweep_sft.sh (7 SFT recipes)
# 5. Pick SFT winner: highest TOTAL with success_control >= 4 AND TOTAL > BASE
# 6. If a winner exists: train winner + R04 distill (w=0.5 T=2.0 warm=100), smoke it
# 7. Run full HE-164 on the SFT winner (no MBPP — time budget) for confirmation
# 8. Write results/decision_overnight.md with full scoreboard
set -uo pipefail
cd /workspace/cross-tokenizer-distill/experiments/validation
source /workspace/venv-tf4/bin/activate
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/cross-tokenizer-distill

LOGDIR=logs/overnight
mkdir -p $LOGDIR results/smoke_sft results/full_sft

echo "=== OVERNIGHT START $(date -u +%FT%TZ) ==="

# ---- Phase 0: wait for C_R04 to finish ----
echo "[ovn] Phase 0: waiting for C_R04 full eval..."
until [ -f results/decision_R04.md ]; do
    sleep 60
done
echo "[ovn] C_R04 done at $(date -u +%FT%TZ)"

# ---- Phase 1: BASE smoke ----
if [ ! -f results/smoke_sft/BASE.json ]; then
    echo "[ovn] Phase 1: BASE smoke (no LoRA)"
    python -u smoke_he20_base.py --output results/smoke_sft/BASE.json 2>&1 | tee $LOGDIR/base_smoke.log | tail -7
else
    echo "[ovn] Phase 1: SKIP BASE smoke (exists)"
fi
BASE_TOTAL=$(python3 -c "import json; print(json.load(open(\"results/smoke_sft/BASE.json\"))[\"overall\"][\"n_pass\"])")
echo "[ovn] BASE smoke TOTAL = $BASE_TOTAL/20"

# ---- Phase 1b: A_OLD smoke derived from results/A.json (existing full HE-164) ----
python3 << "PYEOF" > results/smoke_sft/A_OLD_derived.json
import json
A = json.load(open("results/A.json"))
SUCC = ["HumanEval/0","HumanEval/1","HumanEval/2","HumanEval/3","HumanEval/4"]
BIMP = ["HumanEval/105","HumanEval/127"]
BREG = ["HumanEval/62","HumanEval/65","HumanEval/67","HumanEval/73","HumanEval/120","HumanEval/152"]
STR  = ["HumanEval/32","HumanEval/38","HumanEval/50","HumanEval/83","HumanEval/95","HumanEval/107","HumanEval/132"]
buckets = {"success_control":SUCC,"b_improves":BIMP,"b_regressions":BREG,"both_fail_stretch":STR}
he = {d["task_id"]: d.get("passed", d.get("ok", False)) for d in A.get("humaneval",{}).get("details",[])}
out = {"adapter":"A_OLD_DERIVED","buckets":{},"overall":{}}
total = 0
for name, ids in buckets.items():
    n = sum(1 for tid in ids if he.get(tid, False))
    out["buckets"][name] = {"n_total": len(ids), "n_pass": n}
    total += n
out["overall"] = {"n_total": 20, "n_pass": total, "pass@1": total/20.0}
print(json.dumps(out, indent=2))
PYEOF
A_OLD_TOTAL=$(python3 -c "import json; print(json.load(open(\"results/smoke_sft/A_OLD_derived.json\"))[\"overall\"][\"n_pass\"])")
echo "[ovn] A_OLD (current SFT) derived smoke TOTAL = $A_OLD_TOTAL/20"

# ---- Phase 2: SFT sweep ----
echo "[ovn] Phase 2: SFT sweep (7 recipes, ~140 min)"
bash sweep_sft.sh 2>&1 | tee $LOGDIR/sweep_sft.log

# ---- Phase 3: pick winner ----
WINNER=$(python3 << "PYEOF"
import json, os
from pathlib import Path
base = json.load(open("results/smoke_sft/BASE.json"))["overall"]["n_pass"]
best = None; best_score = -1
for f in sorted(Path("results/smoke_sft").glob("S*.json")):
    r = json.load(open(f))
    succ = r["buckets"]["success_control"]["n_pass"]
    total = r["overall"]["n_pass"]
    if succ < 4: continue
    if total <= base: continue
    if total > best_score:
        best_score = total
        best = f.stem
print(best or "")
PYEOF
)
echo "[ovn] SFT winner: ${WINNER:-NONE}"

if [ -z "$WINNER" ]; then
    echo "[ovn] NO SFT recipe beat base — writing partial scoreboard and stopping early."
fi

# ---- Phase 4: train winner+R04 distill (only if winner exists) ----
if [ -n "$WINNER" ]; then
    echo "[ovn] Phase 4: train ${WINNER}+R04 distill"
    # Parse winner config from sweep TSV
    WINNER_ROW=$(grep "^${WINNER}\b" results/sweep_sft_summary.tsv)
    WINNER_LR=$(echo "$WINNER_ROW" | awk "{print \$2}")
    WINNER_EP=$(echo "$WINNER_ROW" | awk "{print \$3}")
    WINNER_RANK=$(echo "$WINNER_ROW" | awk "{print \$4}")
    WINNER_WS=$(echo "$WINNER_ROW" | awk "{print \$5}")
    WINNER_SCHED=$(echo "$WINNER_ROW" | awk "{print \$6}")
    echo "[ovn] Winner config: lr=$WINNER_LR ep=$WINNER_EP rank=$WINNER_RANK ws=$WINNER_WS sched=$WINNER_SCHED"

    # Run B (same-vocab distill) with winner SFT recipe + R04 distill knobs
    BR04_DIR=runs/B_${WINNER}_R04
    BR04_SMOKE=results/smoke_sft/B_${WINNER}_R04.json
    if [ ! -f "$BR04_SMOKE" ]; then
        python -u 05_train.py --run-name B --output-dir $BR04_DIR \
            --cache cache_B/qwen25_coder_7b_top32.pt \
            --lora-rank $WINNER_RANK \
            --lr $WINNER_LR --epochs $WINNER_EP \
            --warmup-steps $WINNER_WS --lr-scheduler $WINNER_SCHED \
            --ctd-weight 0.5 --kl-temperature 2.0 --ctd-kind kl \
            --ctd-weight-warmup-steps 100 \
            2>&1 | tee $LOGDIR/B_winner_R04_train.log | tail -3
        python -u smoke_he20.py --adapter $BR04_DIR --output $BR04_SMOKE 2>&1 | tee $LOGDIR/B_winner_R04_smoke.log | tail -7
    fi

    # Phase 5: full HE-164 on winner SFT (~30 min)
    WINNER_FULL=results/full_sft/${WINNER}_HE164.json
    if [ ! -f "$WINNER_FULL" ]; then
        echo "[ovn] Phase 5: full HE-164 on SFT winner $WINNER"
        python -u 06_eval.py \
            --adapter runs/sweep_sft/$WINNER \
            --output  $WINNER_FULL \
            --cache-db results/full_sft/${WINNER}_HE164.db \
            --he-limit 164 --mbpp-limit 0 \
            --exec-timeout 30 \
            2>&1 | tee $LOGDIR/winner_full_he.log | tail -3
    fi
fi

# ---- Phase 6: scoreboard ----
echo "[ovn] Phase 6: write decision_overnight.md"
python3 << "PYEOF" > results/decision_overnight.md
import json, os
from pathlib import Path

def load(p):
    try: return json.load(open(p))
    except Exception: return None

print("# Overnight SFT-recovery experiment\n")
print(f"_Run finished {os.popen(\"date -u +%FT%TZ\").read().strip()}_\n")

print("## HE-20 smoke scoreboard\n")
print("| Recipe | Total/20 | success | b_imp | b_reg | stretch | vs BASE | vs A_OLD |")
print("|---|---|---|---|---|---|---|---|")
base = load("results/smoke_sft/BASE.json")
a_old = load("results/smoke_sft/A_OLD_derived.json")
def row(name, r, base_t, a_t):
    if r is None: return f"| {name} | ? | ? | ? | ? | ? | ? | ? |"
    b = r["buckets"]; o = r["overall"]
    delta_b = o["n_pass"] - base_t if base_t is not None else "?"
    delta_a = o["n_pass"] - a_t if a_t is not None else "?"
    return (f"| {name} | {o[\"n_pass\"]} | {b[\"success_control\"][\"n_pass\"]}/5 | "
            f"{b[\"b_improves\"][\"n_pass\"]}/2 | {b[\"b_regressions\"][\"n_pass\"]}/6 | "
            f"{b[\"both_fail_stretch\"][\"n_pass\"]}/7 | {delta_b:+d} | {delta_a:+d} |")
base_t = base["overall"]["n_pass"] if base else None
a_t = a_old["overall"]["n_pass"] if a_old else None
print(row("BASE (no LoRA)", base, base_t, a_t))
print(row("A_OLD (current SFT)", a_old, base_t, a_t))
for f in sorted(Path("results/smoke_sft").glob("S*.json")):
    print(row(f.stem, load(f), base_t, a_t))

# Winner+R04 row
for f in sorted(Path("results/smoke_sft").glob("B_*_R04.json")):
    print(row(f.stem, load(f), base_t, a_t))

print("\n## Full HE-164 (SFT winner only)\n")
for f in sorted(Path("results/full_sft").glob("*_HE164.json")):
    r = load(f)
    if r is None: continue
    he = r.get("humaneval",{})
    n_pass = sum(1 for d in he.get("details",[]) if d.get("passed", d.get("ok", False)))
    n_total = len(he.get("details",[]))
    print(f"- **{f.stem}**: HE pass@1 = {n_pass}/{n_total} = {n_pass/n_total:.1%}" if n_total else f"- {f.stem}: empty")

print("\n## Reference numbers (full HE-164 from previous runs)\n")
for label, path in [("A (old SFT)","results/A.json"),("B_R04","results/B_R04.json"),("C_R04","results/C_R04.json")]:
    r = load(path)
    if r is None: print(f"- {label}: missing"); continue
    he = r.get("humaneval",{})
    mb = r.get("mbpp",{})
    he_n = sum(1 for d in he.get("details",[]) if d.get("passed", d.get("ok", False)))
    he_t = len(he.get("details",[]))
    mb_n = sum(1 for d in mb.get("details",[]) if d.get("passed", d.get("ok", False)))
    mb_t = len(mb.get("details",[]))
    he_pct = f"{he_n/he_t:.1%}" if he_t else "?"
    mb_pct = f"{mb_n/mb_t:.1%}" if mb_t else "?"
    print(f"- **{label}**: HE {he_n}/{he_t} ({he_pct}), MBPP {mb_n}/{mb_t} ({mb_pct})")
PYEOF

echo ""
echo "=== OVERNIGHT DONE $(date -u +%FT%TZ) ==="
cat results/decision_overnight.md
