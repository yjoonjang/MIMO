"""Baseline loss: InfoNCE with gradient caching (no teacher, no projection).

    L = CrossEntropy(cos_sim(anchor, positive) * scale)

Used for both the InfoNCE baseline (monolingual pairs) and the XLCO baseline
(cross-lingual pairs); the pair construction is decided by the dataset, not
the loss.

Data flow:
    sentence_features[0] = anchor   (query, student tokenizer)
    sentence_features[1] = positive (document, student tokenizer)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import tqdm
from torch import Tensor, nn

from sentence_transformers import SentenceTransformer, util

from mimo.losses.grad_cache import CachedLossBase


class CachedInfoNCELoss(CachedLossBase):
    """Pure InfoNCE loss with gradient caching.

    Args:
        model: Student SentenceTransformer.
        scale: Scaling factor for cosine similarity (1/temperature).
        mini_batch_size: Mini-batch size for gradient caching.
        show_progress_bar: Show progress during mini-batch processing.
    """

    num_grad_inputs = 2

    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        mini_batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> None:
        super().__init__(model=model, mini_batch_size=mini_batch_size, show_progress_bar=show_progress_bar)
        self.scale = scale
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor | None = None,
    ) -> Tensor:
        sentence_features = list(sentence_features)
        if len(sentence_features) < 2:
            raise ValueError(
                f"Expected 2 sentence features (anchor, positive), got {len(sentence_features)}"
            )

        student_reps = self.embed_inputs_without_grad(sentence_features)
        anchors = torch.cat(student_reps[0])
        positives = torch.cat(student_reps[1])
        batch_size = anchors.shape[0]

        if torch.is_grad_enabled():
            loss = torch.tensor(0.0, device=anchors.device, requires_grad=True)
            for begin in tqdm.trange(
                0, batch_size, self.mini_batch_size,
                desc="Calculating loss",
                disable=not self.show_progress_bar,
            ):
                end = min(begin + self.mini_batch_size, batch_size)
                local_anchors = anchors[begin:end]
                scores = util.cos_sim(local_anchors, positives) * self.scale
                labels_mb = torch.arange(begin, end, device=scores.device)
                loss_mb = self.cross_entropy_loss(scores, labels_mb) * len(local_anchors) / batch_size
                loss = loss + loss_mb
            loss = self.cache_gradients_and_register_hook(loss, student_reps, sentence_features)
        else:
            scores = util.cos_sim(anchors, positives) * self.scale
            info_labels = torch.arange(batch_size, device=scores.device)
            loss = self.cross_entropy_loss(scores, info_labels)

        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "mini_batch_size": self.mini_batch_size,
        }
