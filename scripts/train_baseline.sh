#!/bin/bash
# Baselines: InfoNCE / XLCO (cross_infonce) / LaKDA from raw pretrained models.
# No teacher, no Stage 1 — identical data, batch size, and schedule as MIMO.
#
# Usage:
#   bash scripts/train_baseline.sh <student_model> <pooling> <loss_type> [output_dir]
#   bash scripts/train_baseline.sh FacebookAI/xlm-roberta-large mean infonce
#   bash scripts/train_baseline.sh jhu-clsp/mmBERT-base cls lakda
#
# Environment overrides:
#   DATA_BASE (default: data/stage2)  NPROC

set -euo pipefail

STUDENT_MODEL="${1:?Usage: train_baseline.sh <student_model> <pooling> <loss_type> [output_dir]}"
POOLING="${2:?Usage: train_baseline.sh <student_model> <pooling> <loss_type> [output_dir]}"
LOSS_TYPE="${3:?loss_type must be one of: infonce, cross_infonce, lakda}"

DATA_BASE="${DATA_BASE:-data/stage2}"
NPROC="${NPROC:-2}"

MODEL_SHORT=$(basename "$STUDENT_MODEL")
RUN_NAME="baseline-${LOSS_TYPE}-${MODEL_SHORT}-${POOLING}"
OUTPUT_DIR="${4:-outputs/baselines/${RUN_NAME}}"
TRAIN_DATA="${DATA_BASE}/${LOSS_TYPE}"

LOSS_ARGS=(--loss_type "$LOSS_TYPE" --infonce_scale 20.0)
if [ "$LOSS_TYPE" = "lakda" ]; then
    LOSS_ARGS+=(--lakda_alpha 0.5)
fi

uv run torchrun --nproc_per_node="$NPROC" train.py \
    --model_name_or_path "$STUDENT_MODEL" \
    --train_data "$TRAIN_DATA" \
    --stage 2 \
    --pooling_mode "$POOLING" \
    "${LOSS_ARGS[@]}" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 2048 \
    --mini_batch_size 64 \
    --max_seq_length 512 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --bf16 \
    --logging_steps 10 \
    --save_strategy no \
    --eval_strategy steps \
    --eval_steps 0.05 \
    --eval_on_start true \
    --dataloader_num_workers 4

echo "Done: ${RUN_NAME} -> ${OUTPUT_DIR}"
