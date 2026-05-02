#!/bin/bash
# Step 1 — diagnostic. Confirm both vocab pairs are workable BEFORE
# spending compute on precomputes.
#
# Pair B (same-vocab): Qwen2.5-Coder-7B → Qwen2.5-Coder-0.5B-Instruct
#   Trivial 100% coverage expected (identical tokenizer).
#
# Pair C (CTD): DeepSeek-Coder-V2-Lite-Instruct → Qwen2.5-Coder-0.5B-Instruct
#   The real test. We expect ~80-100% coverage with multi-token=distribute.
#
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p results

STUDENT="Qwen/Qwen2.5-Coder-0.5B-Instruct"
TEACHER_B="Qwen/Qwen2.5-Coder-7B-Instruct"
TEACHER_C="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"

echo "================================================================"
echo "[01-inspect] Pair B (same-vocab) coverage report"
echo "================================================================"
python -m cli.inspect \
    --teacher-tokenizer "$TEACHER_B" \
    --student-tokenizer "$STUDENT" \
    --strategies distribute \
    2>&1 | tee results/01_inspect_B.log

echo
echo "================================================================"
echo "[01-inspect] Pair C (CTD) coverage report"
echo "================================================================"
python -m cli.inspect \
    --teacher-tokenizer "$TEACHER_C" \
    --student-tokenizer "$STUDENT" \
    --strategies strict,distribute,first_token \
    --trust-remote-code \
    2>&1 | tee results/01_inspect_C.log

echo
echo "[01-inspect] Done. Review results/01_inspect_{B,C}.log."
echo "  Pair B should show ~100% coverage (same tokenizer)."
echo "  Pair C — if coverage > 80%, GO for precompute. < 50%, stop here."
