#!/bin/bash
# M118: M37c recipe + --chat-template (Option B step 1).
#
# Hypothesis: M115's -4.3 HE regression on DSC-V2-Lite-Instruct is from
# training on raw prompts while eval applies tokenizer.apply_chat_template.
# M118 trains with the SAME chat wrap so train and eval distributions match.
# If HE recovers to >= base (75.6) → format mismatch was the cause.
# If HE still drops → step 2 (lower LR / smaller rank / corpus rebalance).
#
# Reuses the M37c recipe verbatim except for two flags:
#   --chat-template
#   --system-prompt "You are a Python coding assistant. Reply with code only."
#
# WANDB_API_KEY must be inlined by the caller (sed -i 's|__WANDB_KEY__|<key>|').
# Idle-barrier blocks until any 06_eval_batched / 06_train_sft is done.

set -uo pipefail
cd /workspace/cross-tokenizer-distill/experiments/validation
source /workspace/venv-tf4/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/workspace/cross-tokenizer-distill

while pgrep -f "06_eval_batched|06_train_sft|06_train_onpolicy" > /dev/null; do sleep 30; done
echo "[M118] idle barrier passed $(date -u +%FT%TZ)"

NAME=M118_DSCV2LITE_M37c_CHAT
OUT=runs/v2/${NAME}
JSON=results/full_sft/${NAME}_HE_MBPP.json
DB=results/full_sft/${NAME}_HE_MBPP.db
LOG=/tmp/118_train.out

export WANDB_API_KEY=__WANDB_KEY__
export WANDB_PROJECT=cross-tokenizer-distill

echo "=== M118 TRAIN (M37c + --chat-template + system code-only) $(date -u +%FT%TZ) ===" | tee $LOG
python -u 06_train_sft_on_teacher.py \
  --student deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
  --corpus data/funcsig_prompts_qwen25c14b_codeonly_T07.jsonl \
  --output-dir $OUT \
  --max-prompt-len 512 --max-total-len 1408 \
  --lora-rank 16 --lr 5e-5 --epochs 2 --batch-size 1 --grad-accum 16 \
  --warmup-steps 8 --logging-steps 5 --seed 0 \
  --code-only-mask \
  --chat-template \
  --system-prompt "You are a Python coding assistant. Reply with code only." \
  --wandb --wandb-run-name $NAME \
  2>&1 | tee -a $LOG

ADAPTER=$(ls -d $OUT/checkpoint-* 2>/dev/null | sort -V | tail -1)
[ -z "$ADAPTER" ] && ADAPTER=$(ls -d $OUT/epoch-* 2>/dev/null | sort -V | tail -1)
[ -z "$ADAPTER" ] && ADAPTER=$OUT
echo "=== M118 EVAL (FA2 + merge_and_unload + chat-template, bs=8) $(date -u +%FT%TZ) ===" | tee -a $LOG

python -u 06_eval_batched.py \
  --base-model /workspace/mythic-rdt/base/DeepSeek-Coder-V2-Lite-Instruct \
  --quant bf16 --chat-template \
  --adapter $ADAPTER --output $JSON --cache-db $DB \
  --he-limit 164 --mbpp-limit 378 --exec-timeout 30 \
  --batch-size 8 --max-new-tokens 1024 \
  2>&1 | tee -a $LOG

echo "=== M118 DONE $(date -u +%FT%TZ) ===" | tee -a $LOG
sqlite3 $DB "SELECT task,COUNT(*),SUM(passed),ROUND(100.0*SUM(passed)/COUNT(*),1) FROM results GROUP BY task" | tee -a $LOG
