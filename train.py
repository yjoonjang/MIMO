"""MIMO training entry point.

Usage:
    # Stage 1: Cross-lingual distillation warmup
    torchrun --nproc_per_node=2 train.py \
        --stage 1 \
        --model_name_or_path FacebookAI/xlm-roberta-large \
        --teacher_model_name_or_path Qwen/Qwen3-Embedding-8B \
        --train_data /path/to/stage1_parallel \
        --output_dir outputs/stage1

    # Stage 2: Joint optimization (MIMO)
    torchrun --nproc_per_node=2 train.py \
        --stage 2 \
        --model_name_or_path outputs/stage1 \
        --teacher_model_name_or_path Qwen/Qwen3-Embedding-8B \
        --projection_path outputs/stage1/projection.pt \
        --train_data /path/to/stage2/mimo \
        --output_dir outputs/stage2

    # Baselines (no teacher): --loss_type infonce | cross_infonce | lakda
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field

import torch
from sentence_transformers import SentenceTransformer
from transformers import HfArgumentParser

from mimo.data import MIMODataCollator, load_train_dataset
from mimo.losses import (
    CachedDistillInfoNCELoss,
    CachedInfoNCELoss,
    CachedLaKDALoss,
    EmbedDistillLoss,
)
from mimo.trainer import MIMOTrainer, MIMOTrainingArguments

logger = logging.getLogger(__name__)

QWEN3_QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
)


@dataclass
class ModelArguments:
    """Arguments for model and data paths."""

    model_name_or_path: str = field(metadata={"help": "Path to student model or HuggingFace model ID"})
    train_data: str = field(metadata={"help": "Path to training data (save_to_disk dir or HF dataset id)"})
    teacher_model_name_or_path: str = field(
        default="Qwen/Qwen3-Embedding-8B",
        metadata={"help": "Path to teacher model or HuggingFace model ID"},
    )


def main():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = HfArgumentParser((ModelArguments, MIMOTrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    # Resolve the teacher query prompt (only relevant for the MIMO loss)
    if training_args.loss_type == "mimo":
        if training_args.teacher_query_prompt == "auto" or (
            training_args.stage == 2 and training_args.teacher_query_prompt is None
        ):
            training_args.teacher_query_prompt = QWEN3_QUERY_PROMPT

    logger.info("=== MIMO Training ===")
    logger.info("Stage: %d", training_args.stage)
    logger.info("Loss type: %s", training_args.loss_type)
    logger.info("Student model: %s", model_args.model_name_or_path)
    if training_args.loss_type == "mimo":
        logger.info("Teacher model: %s", model_args.teacher_model_name_or_path)
    logger.info("Output dir: %s", training_args.output_dir)

    # --- Load student model ---
    student_model = SentenceTransformer(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    student_model.max_seq_length = training_args.max_seq_length

    # Override the pooling mode when starting from a raw HF model (Stage 1,
    # baseline losses, or when no SentenceTransformer config exists).
    needs_pooling_override = (
        training_args.stage == 1
        or training_args.loss_type in ("infonce", "cross_infonce", "lakda")
        or not os.path.exists(os.path.join(model_args.model_name_or_path, "modules.json"))
    )
    if needs_pooling_override:
        from sentence_transformers.models import Pooling as STPooling

        for i, module in enumerate(student_model):
            if isinstance(module, STPooling):
                student_model[i] = STPooling(
                    student_model[0].get_word_embedding_dimension(),
                    pooling_mode=training_args.pooling_mode,
                )
                logger.info("Student pooling mode: %s", training_args.pooling_mode)
                break

    student_dim = student_model.get_sentence_embedding_dimension()
    logger.info("Student dim: %d", student_dim)

    # --- Load teacher model (MIMO loss only) ---
    if training_args.loss_type == "mimo":
        attn_implementation = training_args.teacher_attn_implementation
        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                logger.warning("flash-attn not installed; falling back to sdpa for the teacher")
                attn_implementation = "sdpa"

        teacher_model = SentenceTransformer(
            model_args.teacher_model_name_or_path,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": attn_implementation},
        )
        teacher_model.max_seq_length = training_args.teacher_max_seq_length
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_dim = teacher_model.get_sentence_embedding_dimension()
        logger.info("Teacher dim: %d", teacher_dim)

        # Qwen3 requires left padding
        teacher_model.tokenizer.padding_side = "left"

        # --- Projection layer ---
        projection = torch.nn.Linear(student_dim, teacher_dim)
        if training_args.projection_path and os.path.exists(training_args.projection_path):
            projection.load_state_dict(torch.load(training_args.projection_path, weights_only=True))
            logger.info("Loaded projection from %s", training_args.projection_path)
        else:
            logger.info("Initialized new projection layer: %d -> %d", student_dim, teacher_dim)

        projection = projection.to(device=student_model.device, dtype=torch.float32)
    else:
        teacher_model = None
        projection = None

    # --- Data collator ---
    collator = MIMODataCollator(
        student_tokenizer=student_model.tokenizer,
        teacher_tokenizer=teacher_model.tokenizer if teacher_model is not None else None,
        max_seq_length=training_args.max_seq_length,
        teacher_max_seq_length=training_args.teacher_max_seq_length,
        stage=training_args.stage,
        teacher_query_prompt=training_args.teacher_query_prompt if training_args.stage == 2 else None,
        student_query_prompt=training_args.student_query_prompt if training_args.stage == 2 else None,
        student_doc_prompt=training_args.student_doc_prompt if training_args.stage == 2 else None,
    )

    # --- Dataset ---
    train_dataset = load_train_dataset(model_args.train_data)
    logger.info(
        "Stage %d dataset: %d samples (loss_type=%s)",
        training_args.stage, len(train_dataset), training_args.loss_type,
    )
    if training_args.stage == 2 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
        sample0 = train_dataset[0]
        logger.info("Sample[0] keys=%s", list(sample0.keys()))

    # --- Loss function ---
    if training_args.stage == 1:
        loss_fn = EmbedDistillLoss(
            model=student_model,
            teacher_model=teacher_model,
            projection=projection,
            distance_metric=training_args.distance_metric,
        )
    elif training_args.stage == 2:
        if training_args.loss_type == "mimo":
            loss_fn = CachedDistillInfoNCELoss(
                model=student_model,
                teacher_model=teacher_model,
                projection=projection,
                scale=training_args.infonce_scale,
                infonce_weight=training_args.lambda_weight,
                distill_weight=1.0 - training_args.lambda_weight,
                mini_batch_size=training_args.mini_batch_size,
            )
            logger.info(
                "MIMO loss weights: lambda=%.2f (XLCO=%.2f, Distill=%.2f)",
                training_args.lambda_weight, training_args.lambda_weight, 1.0 - training_args.lambda_weight,
            )
        elif training_args.loss_type in ("infonce", "cross_infonce"):
            loss_fn = CachedInfoNCELoss(
                model=student_model,
                scale=training_args.infonce_scale,
                mini_batch_size=training_args.mini_batch_size,
            )
        elif training_args.loss_type == "lakda":
            loss_fn = CachedLaKDALoss(
                model=student_model,
                scale=training_args.infonce_scale,
                alpha=training_args.lakda_alpha,
                mini_batch_size=training_args.mini_batch_size,
            )
        else:
            raise ValueError(f"Unknown loss_type: {training_args.loss_type}")
    else:
        raise ValueError(f"Invalid stage: {training_args.stage}")

    # --- Evaluator (NanoMIRACL mid-training validation) ---
    evaluator = None
    if training_args.eval_strategy != "no":
        from mimo.evaluation import NanoMIRACLEvaluator

        evaluator = NanoMIRACLEvaluator(batch_size=256)
        logger.info(
            "NanoMIRACL evaluator enabled (eval every %s)",
            f"{training_args.eval_steps} steps" if training_args.eval_strategy == "steps" else "epoch",
        )

    # --- Trainer ---
    trainer = MIMOTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss_fn,
        data_collator=collator,
        projection=projection,
        teacher_model=teacher_model,
        evaluator=evaluator,
    )

    logger.info("Starting Stage %d training...", training_args.stage)
    trainer.train()

    trainer.save_model(training_args.output_dir)
    logger.info("Training complete! Model saved to %s", training_args.output_dir)


if __name__ == "__main__":
    main()
