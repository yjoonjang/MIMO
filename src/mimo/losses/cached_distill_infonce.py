"""Stage 2 MIMO loss: cross-lingual InfoNCE + teacher distillation, gradient cached.

    L = lambda * L_XLCO + (1 - lambda) * L_Distill

L_XLCO:    CrossEntropy(cos_sim(student(q_XX), student(p_YY)) * scale) with
           in-batch negatives, computed in the student embedding space.
L_Distill: cosine distance between projected student embeddings and the frozen
           teacher's English embeddings, computed in the teacher space.

Data flow:
    sentence_features[0] = anchor   (query_XX,   student tokenizer)
    sentence_features[1] = positive (positive_YY, student tokenizer)
    sentence_features[2] = anchor_en   (query_en,    teacher tokenizer)
    sentence_features[3] = positive_en (positive_en, teacher tokenizer)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import tqdm
from torch import Tensor, nn

from sentence_transformers import SentenceTransformer, util

from mimo.losses.grad_cache import CachedLossBase


class CachedDistillInfoNCELoss(CachedLossBase):
    """Combined InfoNCE + distillation loss with gradient caching.

    Args:
        model: Student SentenceTransformer.
        teacher_model: Teacher SentenceTransformer (frozen).
        projection: Linear projection from student_dim to teacher_dim.
        scale: Scaling factor for InfoNCE similarity (1/temperature).
        infonce_weight: Weight for the InfoNCE loss (lambda in the paper).
        distill_weight: Weight for the distillation loss (1 - lambda).
        mini_batch_size: Mini-batch size for gradient caching.
        show_progress_bar: Show progress during mini-batch processing.
    """

    num_grad_inputs = 2  # anchor and positive; teacher inputs get no gradients

    def __init__(
        self,
        model: SentenceTransformer,
        teacher_model: SentenceTransformer,
        projection: nn.Linear,
        scale: float = 20.0,
        infonce_weight: float = 0.5,
        distill_weight: float = 0.5,
        mini_batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> None:
        super().__init__(model=model, mini_batch_size=mini_batch_size, show_progress_bar=show_progress_bar)
        self.teacher_model = teacher_model
        self.projection = projection
        self.scale = scale
        self.infonce_weight = infonce_weight
        self.distill_weight = distill_weight

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # For logging individual loss components
        self._last_infonce_loss: float = 0.0
        self._last_distill_loss: float = 0.0

    def _embed_teacher_minibatch(self, sentence_feature: dict[str, Tensor]) -> Tensor:
        """Embed with the teacher model in mini-batches to bound VRAM usage."""
        input_ids: Tensor = sentence_feature["input_ids"]
        batch_size = input_ids.shape[0]
        embeddings = []
        for begin in range(0, batch_size, self.mini_batch_size):
            end = begin + self.mini_batch_size
            mb = {
                key: value[begin:end] if isinstance(value, Tensor) else value
                for key, value in sentence_feature.items()
            }
            embeddings.append(self.teacher_model(mb)["sentence_embedding"])
        return torch.cat(embeddings)

    def _compute_loss(
        self,
        anchors: Tensor,
        positives: Tensor,
        teacher_query_emb: Tensor,
        teacher_pos_emb: Tensor,
        minibatched_infonce: bool,
    ) -> Tensor:
        batch_size = anchors.shape[0]

        teacher_query_emb = teacher_query_emb.to(dtype=anchors.dtype, device=anchors.device)
        teacher_pos_emb = teacher_pos_emb.to(dtype=anchors.dtype, device=anchors.device)
        self.projection = self.projection.to(device=anchors.device, dtype=anchors.dtype)

        # --- InfoNCE loss (student embedding space) ---
        if minibatched_infonce:
            infonce_losses = []
            for begin in tqdm.trange(
                0, batch_size, self.mini_batch_size,
                desc="Calculating loss",
                disable=not self.show_progress_bar,
            ):
                end = min(begin + self.mini_batch_size, batch_size)
                local_anchors = anchors[begin:end]
                scores = util.cos_sim(local_anchors, positives) * self.scale  # (mbs, B)
                labels = torch.arange(begin, end, device=scores.device)
                loss_mb = self.cross_entropy_loss(scores, labels) * len(local_anchors) / batch_size
                infonce_losses.append(loss_mb)
            infonce_loss = sum(infonce_losses)
        else:
            scores = util.cos_sim(anchors, positives) * self.scale
            labels = torch.arange(batch_size, device=scores.device)
            infonce_loss = self.cross_entropy_loss(scores, labels)

        # --- Distillation loss (projected teacher space) ---
        proj_anchors = self.projection(anchors)
        proj_positives = self.projection(positives)
        distill_anchor = (1 - nn.functional.cosine_similarity(proj_anchors, teacher_query_emb, dim=-1)).mean()
        distill_positive = (1 - nn.functional.cosine_similarity(proj_positives, teacher_pos_emb, dim=-1)).mean()
        distill_loss = (distill_anchor + distill_positive) / 2

        self._last_infonce_loss = float(infonce_loss.detach())
        self._last_distill_loss = float(distill_loss.detach())

        return self.infonce_weight * infonce_loss + self.distill_weight * distill_loss

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor | None = None,
    ) -> Tensor:
        sentence_features = list(sentence_features)
        if len(sentence_features) < 4:
            raise ValueError(
                f"Expected 4 sentence features (anchor, positive, anchor_en, positive_en), "
                f"got {len(sentence_features)}"
            )

        # Step 1: student embeddings without grad (gradient caching)
        student_reps = self.embed_inputs_without_grad(sentence_features)

        # Step 2: teacher embeddings in mini-batches (no grad needed)
        with torch.no_grad():
            teacher_query_emb = self._embed_teacher_minibatch(sentence_features[2])
            teacher_pos_emb = self._embed_teacher_minibatch(sentence_features[3])

        anchors = torch.cat(student_reps[0])
        positives = torch.cat(student_reps[1])

        if torch.is_grad_enabled():
            # Steps 3-4: full-batch loss, cache gradients, register re-forward hook
            loss = self._compute_loss(
                anchors, positives, teacher_query_emb, teacher_pos_emb, minibatched_infonce=True
            )
            loss = self.cache_gradients_and_register_hook(loss, student_reps, sentence_features)
        else:
            loss = self._compute_loss(
                anchors, positives, teacher_query_emb, teacher_pos_emb, minibatched_infonce=False
            )

        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "infonce_weight": self.infonce_weight,
            "distill_weight": self.distill_weight,
            "mini_batch_size": self.mini_batch_size,
        }
