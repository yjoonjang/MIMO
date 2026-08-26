"""Loss sanity tests: gradient-cached losses must match their direct formulas.

Uses a tiny linear stand-in for SentenceTransformer so no model downloads are
required. "input_ids" carries float features; the fake model embeds them with
a single Linear layer.
"""

import pytest
import torch
from torch import nn

from mimo.losses import CachedDistillInfoNCELoss, CachedInfoNCELoss, CachedLaKDALoss
from sentence_transformers import util

BATCH, DIM, TEACHER_DIM = 8, 16, 24
MINI_BATCH = 4


class FakeEncoder(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, features):
        return {"sentence_embedding": self.linear(features["input_ids"])}


def features(seed):
    g = torch.Generator().manual_seed(seed)
    return {"input_ids": torch.randn(BATCH, DIM, generator=g)}


def test_cached_infonce_matches_direct():
    torch.manual_seed(0)
    model = FakeEncoder()
    loss_fn = CachedInfoNCELoss(model=model, scale=20.0, mini_batch_size=MINI_BATCH)

    sf = [features(1), features(2)]
    loss = loss_fn(sf, None)

    with torch.no_grad():
        anchors = model(sf[0])["sentence_embedding"]
        positives = model(sf[1])["sentence_embedding"]
        scores = util.cos_sim(anchors, positives) * 20.0
        expected = nn.CrossEntropyLoss()(scores, torch.arange(BATCH))

    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)

    # Backward through the gradient cache must reach the model parameters
    loss.backward()
    assert model.linear.weight.grad is not None
    assert model.linear.weight.grad.abs().sum() > 0


def test_cached_lakda_matches_direct():
    torch.manual_seed(0)
    model = FakeEncoder()
    alpha = 0.5
    loss_fn = CachedLaKDALoss(model=model, scale=20.0, alpha=alpha, mini_batch_size=MINI_BATCH)

    sf = [features(1), features(2), features(3)]
    loss = loss_fn(sf, None)

    with torch.no_grad():
        q_a = model(sf[0])["sentence_embedding"]
        p = model(sf[1])["sentence_embedding"]
        q_b = model(sf[2])["sentence_embedding"]
        infonce = nn.CrossEntropyLoss()(util.cos_sim(q_a, p) * 20.0, torch.arange(BATCH))
        log_p_a = torch.log_softmax(util.cos_sim(q_a, p), dim=-1)
        p_b = torch.softmax(util.cos_sim(q_b, p), dim=-1)
        kl = nn.KLDivLoss(reduction="batchmean")(log_p_a, p_b)
        expected = (1 - alpha) * infonce + alpha * kl

    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)

    loss.backward()
    assert model.linear.weight.grad is not None


def test_cached_distill_infonce_matches_direct():
    torch.manual_seed(0)
    student = FakeEncoder()
    teacher = FakeEncoder()
    projection = nn.Linear(DIM, DIM)
    lam = 0.2
    loss_fn = CachedDistillInfoNCELoss(
        model=student,
        teacher_model=teacher,
        projection=projection,
        scale=20.0,
        infonce_weight=lam,
        distill_weight=1.0 - lam,
        mini_batch_size=MINI_BATCH,
    )

    sf = [features(1), features(2), features(3), features(4)]
    loss = loss_fn(sf, None)

    with torch.no_grad():
        anchors = student(sf[0])["sentence_embedding"]
        positives = student(sf[1])["sentence_embedding"]
        t_query = teacher(sf[2])["sentence_embedding"]
        t_pos = teacher(sf[3])["sentence_embedding"]
        infonce = nn.CrossEntropyLoss()(util.cos_sim(anchors, positives) * 20.0, torch.arange(BATCH))
        proj_a, proj_p = projection(anchors), projection(positives)
        distill = (
            (1 - nn.functional.cosine_similarity(proj_a, t_query, dim=-1)).mean()
            + (1 - nn.functional.cosine_similarity(proj_p, t_pos, dim=-1)).mean()
        ) / 2
        expected = lam * infonce + (1 - lam) * distill

    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)

    loss.backward()
    assert student.linear.weight.grad is not None
    # Teacher stays frozen
    assert all(p.grad is None for p in teacher.parameters())
