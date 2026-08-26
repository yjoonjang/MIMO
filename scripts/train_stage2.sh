#!/bin/bash
# Stage 2: Joint optimization (MIMO) from a Stage 1 checkpoint.
#
# Usage:
#   bash scripts/train_stage2.sh <stage1_dir> [lambda] [output_dir]
#   bash scripts/train_stage2.sh outputs/stage1/stage1-qwen3-xlm-roberta-large-mean 0.2
#
# Environment overrides:
#   TRAIN_DATA (default: data/stage2/mimo)  TEACHER_MODEL  NPROC

set -euo pipefail

STAGE1_DIR="${1:?Usage: train_stage2.sh <stage1_dir> [lambda] [output_dir]}"
LAMBDA="${2:-0.2}"

TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-Embedding-8B}"
TRAIN_DATA="${TRAIN_DATA:-data/stage2/mimo}"
NPROC="${NPROC:-2}"

RUN_NAME="stage2-$(basename "$STAGE1_DIR" | sed 's/^stage1-//')"
OUTPUT_DIR="${3:-outputs/stage2/${RUN_NAME}}"
PROJECTION_PATH="${STAGE1_DIR}/projection.pt"

uv run torchrun --nproc_per_node="$NPROC" train.py \
    --model_name_or_path "$STAGE1_DIR" \
    --teacher_model_name_or_path "$TEACHER_MODEL" \
    --train_data "$TRAIN_DATA" \
    --stage 2 \
    --projection_path "$PROJECTION_PATH" \
    --lambda_weight "$LAMBDA" \
    --infonce_scale 20.0 \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 2048 \
    --mini_batch_size 32 \
    --max_seq_length 512 \
    --teacher_max_seq_length 512 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --bf16 \
    --logging_steps 10 \
    --save_strategy no \
    --eval_strategy steps \
    --eval_steps 0.2 \
    --eval_on_start true \
    --dataloader_num_workers 4

echo "Done: ${RUN_NAME} -> ${OUTPUT_DIR}"
