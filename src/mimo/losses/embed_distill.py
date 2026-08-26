"""Stage 1 loss: cross-lingual embedding distillation.

    Student(text_XX) -> student_dim -> Projection -> teacher_dim
                                            | cosine distance
    Teacher(text_en) -> teacher_dim  (frozen, torch.no_grad)

The collator provides:
    sentence_features[0] = student-tokenized text (any language)
    sentence_features[1] = teacher-tokenized English text
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from sentence_transformers import SentenceTransformer


class EmbedDistillLoss(nn.Module):
    """Embedding distillation loss for Stage 1 cross-lingual alignment.

    Args:
        model: Student SentenceTransformer.
        teacher_model: Teacher SentenceTransformer (frozen).
        projection: Linear projection from student_dim to teacher_dim.
        distance_metric: Distance metric ("cosine", "mse", "l2").
    """

    def __init__(
        self,
        model: SentenceTransformer,
        teacher_model: SentenceTransformer,
        projection: nn.Linear,
        distance_metric: str = "cosine",
    ) -> None:
        super().__init__()
        self.model = model
        self.teacher_model = teacher_model
        self.projection = projection
        self.distance_metric = distance_metric

        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor | None = None,
    ) -> Tensor:
        sentence_features = list(sentence_features)
        student_sf, teacher_sf = sentence_features[0], sentence_features[1]

        student_emb = self.model(student_sf)["sentence_embedding"]

        self.projection = self.projection.to(device=student_emb.device, dtype=student_emb.dtype)
        projected = self.projection(student_emb)

        with torch.no_grad():
            teacher_emb = self.teacher_model(teacher_sf)["sentence_embedding"]
        teacher_emb = teacher_emb.to(device=projected.device, dtype=projected.dtype)

        if self.distance_metric == "cosine":
            loss = (1 - nn.functional.cosine_similarity(projected, teacher_emb, dim=-1)).mean()
        elif self.distance_metric == "mse":
            loss = nn.functional.mse_loss(projected, teacher_emb)
        elif self.distance_metric == "l2":
            loss = torch.norm(projected - teacher_emb, dim=-1).mean()
        else:
            raise ValueError(f"Unknown distance_metric: {self.distance_metric}")

        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "distance_metric": self.distance_metric,
            "projection_in": self.projection.in_features,
            "projection_out": self.projection.out_features,
        }
