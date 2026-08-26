"""MIMO training arguments extending SentenceTransformerTrainingArguments."""

from __future__ import annotations

from dataclasses import dataclass, field

from sentence_transformers.training_args import SentenceTransformerTrainingArguments


@dataclass
class MIMOTrainingArguments(SentenceTransformerTrainingArguments):
    """Training arguments for the MIMO framework.

    Args:
        stage: Training stage (1 = cross-lingual distillation warmup,
            2 = joint optimization / baseline training).
        loss_type: "mimo", "infonce", "cross_infonce", or "lakda".
        distance_metric: Distance metric for distillation ("cosine", "mse", "l2").
        lambda_weight: Stage 2 loss weight: L = lambda * L_XLCO + (1 - lambda) * L_Distill.
        infonce_scale: Scale factor for InfoNCE similarity (1/temperature).
        mini_batch_size: Mini-batch size for gradient caching (Stage 2).
        teacher_query_prompt: Query prompt for the teacher model (Stage 2).
            "auto" resolves to the Qwen3-Embedding query prompt.
        projection_path: Path to pre-trained projection weights (Stage 2).
        pooling_mode: Pooling mode for the student model.
    """

    max_seq_length: int = field(default=256, metadata={"help": "Max sequence length for student tokenizer"})
    stage: int = field(default=1, metadata={"help": "Training stage: 1 or 2"})
    loss_type: str = field(default="mimo", metadata={"help": "Loss type: mimo, infonce, cross_infonce, lakda"})
    distance_metric: str = field(default="cosine", metadata={"help": "Distance metric for distillation"})
    lambda_weight: float = field(default=0.2, metadata={"help": "Lambda: L = lambda*L_XLCO + (1-lambda)*L_Distill (Stage 2)"})
    infonce_scale: float = field(default=20.0, metadata={"help": "Scale for InfoNCE similarity"})
    mini_batch_size: int = field(default=32, metadata={"help": "Mini-batch size for gradient caching"})
    teacher_query_prompt: str | None = field(
        default=None,
        metadata={"help": "Query prompt for teacher (Stage 2). 'auto' uses the Qwen3 default."},
    )
    teacher_max_seq_length: int = field(default=512, metadata={"help": "Max seq length for teacher tokenizer"})
    teacher_attn_implementation: str = field(
        default="flash_attention_2",
        metadata={"help": "Attention implementation for the teacher (flash_attention_2 or sdpa)"},
    )
    projection_path: str | None = field(default=None, metadata={"help": "Path to pre-trained projection weights"})
    pooling_mode: str = field(default="mean", metadata={"help": "Pooling mode for the student: mean, cls, max, lasttoken"})
    student_query_prompt: str | None = field(default=None, metadata={"help": "Prompt prefix for student query encoding (Stage 2)"})
    student_doc_prompt: str | None = field(default=None, metadata={"help": "Prompt prefix for student document encoding (Stage 2)"})
    lakda_alpha: float = field(default=0.5, metadata={"help": "LaKDA alpha: weight for KL vs InfoNCE"})
