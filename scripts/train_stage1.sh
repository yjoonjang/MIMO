#!/bin/bash
# Stage 1: Cross-lingual distillation warmup.
#
# Usage:
#   bash scripts/train_stage1.sh <student_model> <pooling> [output_dir]
#   bash scripts/train_stage1.sh FacebookAI/xlm-roberta-large mean
#
# Environment overrides:
#   TRAIN_DATA (default: data/stage1_parallel)  TEACHER_MODEL  NPROC

set -euo pipefail

STUDENT_MODEL="${1:?Usage: train_stage1.sh <student_model> <pooling> [output_dir]}"
POOLING="${2:?Usage: train_stage1.sh <student_model> <pooling> [output_dir]}"

TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-Embedding-8B}"
TRAIN_DATA="${TRAIN_DATA:-data/stage1_parallel}"
NPROC="${NPROC:-2}"

MODEL_SHORT=$(basename "$STUDENT_MODEL")
RUN_NAME="stage1-qwen3-${MODEL_SHORT}-${POOLING}"
OUTPUT_DIR="${3:-outputs/stage1/${RUN_NAME}}"

uv run torchrun --nproc_per_node="$NPROC" train.py \
    --model_name_or_path "$STUDENT_MODEL" \
    --teacher_model_name_or_path "$TEACHER_MODEL" \
    --train_data "$TRAIN_DATA" \
    --stage 1 \
    --pooling_mode "$POOLING" \
    --distance_metric cosine \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 128 \
    --max_seq_length 256 \
    --teacher_max_seq_length 256 \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
    --bf16 \
    --logging_steps 50 \
    --save_strategy no \
    --eval_strategy steps \
    --eval_steps 0.05 \
    --eval_on_start true \
    --dataloader_num_workers 4

echo "Done: ${RUN_NAME} -> ${OUTPUT_DIR}"
