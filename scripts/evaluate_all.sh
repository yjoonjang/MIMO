#!/bin/bash
# Evaluate one model on all benchmarks reported in the paper:
#   MLIR (Belebele, MLQA, XQuAD, MultiEuP-v2), NeuCLIR'22/'23, and MIRACL.
#
# Usage:
#   bash scripts/evaluate_all.sh <model_path> [output_dir]

set -euo pipefail

MODEL="${1:?Usage: evaluate_all.sh <model_path> [output_dir]}"
OUTPUT_DIR="${2:-results}"

uv run python evaluate.py \
    --model_name_or_path "$MODEL" \
    --output_dir "$OUTPUT_DIR"

uv run python evaluate_neuclir.py \
    --model_name_or_path "$MODEL" \
    --output_dir "$OUTPUT_DIR"

uv run python evaluate_miracl.py \
    --model_name_or_path "$MODEL" \
    --output_dir "$OUTPUT_DIR"

echo "All evaluations complete. Results in ${OUTPUT_DIR}/"
