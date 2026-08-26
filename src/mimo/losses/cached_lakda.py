"""Baseline loss: LaKDA with gradient caching (no teacher, no projection).

    L = (1 - alpha) * L_DPR(q_a, p) + alpha * D_KL(p_b || p_a)

where p_a = softmax(cos_sim(q_a, D)) and p_b = softmax(cos_sim(q_b, D)) are the
similarity distributions of two language variants of the same query over the
in-batch documents. All three inputs receive gradients (no gradient stopping
on q_b), and the KL term uses unscaled cosine similarities, following the
original formulation.

Reference: Yang et al., "Language Bias in Multilingual Information Retrieval:
The Nature of the Beast and Mitigation Methods", MRL 2024.

Data flow:
    sentence_features[0] = anchor   (query in language A, student tokenizer)
    sentence_features[1] = positive (document in a random language, student tokenizer)
    sentence_features[2] = anchor_b (query in language B, student tokenizer)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import tqdm
from torch import Tensor, nn

from sentence_transformers import SentenceTransformer, util

from mimo.losses.grad_cache import CachedLossBase


class CachedLaKDALoss(CachedLossBase):
    """LaKDA loss with gradient caching.

    Args:
        model: Student SentenceTransformer.
        scale: Scaling factor for the InfoNCE cosine similarity (1/temperature).
        alpha: Weight for the KL term ((1 - alpha) for InfoNCE).
        mini_batch_size: Mini-batch size for gradient caching.
        show_progress_bar: Show progress during mini-batch processing.
    """

    num_grad_inputs = 3  # anchor_a, positive, anchor_b

    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        alpha: float = 0.5,
        mini_batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> None:
        super().__init__(model=model, mini_batch_size=mini_batch_size, show_progress_bar=show_progress_bar)
        self.scale = scale
        self.alpha = alpha

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        self._last_infonce_loss: float = 0.0
        self._last_kl_loss: float = 0.0

    def _compute_loss(
        self, anchors_a: Tensor, positives: Tensor, anchors_b: Tensor, minibatched_infonce: bool
    ) -> Tensor:
        batch_size = anchors_a.shape[0]

        # --- InfoNCE (L_DPR): cos_sim(q_a, p) * scale ---
        if minibatched_infonce:
            infonce_loss = torch.tensor(0.0, device=anchors_a.device, requires_grad=True)
            for begin in tqdm.trange(
                0, batch_size, self.mini_batch_size,
                desc="Calculating loss",
                disable=not self.show_progress_bar,
            ):
                end = min(begin + self.mini_batch_size, batch_size)
                local_anchors = anchors_a[begin:end]
                scores = util.cos_sim(local_anchors, positives) * self.scale
                labels = torch.arange(begin, end, device=scores.device)
                loss_mb = self.cross_entropy_loss(scores, labels) * len(local_anchors) / batch_size
                infonce_loss = infonce_loss + loss_mb
        else:
            scores = util.cos_sim(anchors_a, positives) * self.scale
            labels = torch.arange(batch_size, device=scores.device)
            infonce_loss = self.cross_entropy_loss(scores, labels)

        # --- KL divergence: D_KL(p_b || p_a), unscaled cosine similarities ---
        sim_a = util.cos_sim(anchors_a, positives)  # (B, B)
        sim_b = util.cos_sim(anchors_b, positives)  # (B, B)
        log_p_a = torch.log_softmax(sim_a, dim=-1)
        p_b = torch.softmax(sim_b, dim=-1)
        kl_loss = self.kl_loss(log_p_a, p_b)

        self._last_infonce_loss = float(infonce_loss.detach())
        self._last_kl_loss = float(kl_loss.detach())

        return (1 - self.alpha) * infonce_loss + self.alpha * kl_loss

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor | None = None,
    ) -> Tensor:
        sentence_features = list(sentence_features)
        if len(sentence_features) < 3:
            raise ValueError(
                f"Expected 3 sentence features (anchor_a, positive, anchor_b), "
                f"got {len(sentence_features)}"
            )

        student_reps = self.embed_inputs_without_grad(sentence_features)
        anchors_a = torch.cat(student_reps[0])
        positives = torch.cat(student_reps[1])
        anchors_b = torch.cat(student_reps[2])

        if torch.is_grad_enabled():
            loss = self._compute_loss(anchors_a, positives, anchors_b, minibatched_infonce=True)
            loss = self.cache_gradients_and_register_hook(loss, student_reps, sentence_features)
        else:
            loss = self._compute_loss(anchors_a, positives, anchors_b, minibatched_infonce=False)

        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "alpha": self.alpha,
            "mini_batch_size": self.mini_batch_size,
        }
