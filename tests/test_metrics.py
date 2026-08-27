"""MRC metric tests against hand-computable cases (Eq. 1-2 of the paper)."""

import numpy as np
import pytest

from mimo.evaluation.metrics import _parse_doc_lang, _parse_query_hash, _parse_query_lang, compute_mrc


def test_query_id_parsing():
    # Belebele MMLIR format: {lang}_q_{hash}
    assert _parse_query_lang("ar_q_005a") == "ar"
    assert _parse_query_hash("ar_q_005a") == "005a"
    # XQuAD/MLQA MMLIR format: {lang}_{original_id}
    assert _parse_query_lang("de_56beb4343aeaaa14008c925b") == "de"
    assert _parse_query_hash("de_56beb4343aeaaa14008c925b") == "56beb4343aeaaa14008c925b"
    # Original ids containing underscores stay intact
    assert _parse_query_hash("zh_q_ab_cd") == "ab_cd"
    # Doc ids: doc_{lang}_{n}
    assert _parse_doc_lang("doc_en_1466") == "en"


def test_mrc_perfectly_consistent_rankings():
    """Parallel queries with identical score rows must give MRC@k = 100."""
    corpus_ids = [f"d{i}" for i in range(6)]
    query_ids = ["en_q_h1", "de_q_h1", "fr_q_h1"]
    row = np.array([0.9, 0.5, 0.8, 0.1, 0.3, 0.7])
    sim = np.stack([row, row, row])

    result = compute_mrc(sim, query_ids, corpus_ids, k=3)
    for lang in ("en", "de", "fr"):
        assert result[f"MRC@3_{lang}"] == pytest.approx(100.0)
    assert result["MRC@3_avg"] == pytest.approx(100.0)


def test_mrc_reversed_rankings():
    """Two parallel queries with exactly opposite rankings give MRC@k = -100."""
    corpus_ids = [f"d{i}" for i in range(5)]
    query_ids = ["en_q_h1", "de_q_h1"]
    row = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    sim = np.stack([row, row[::-1]])

    # Union of top-k sets covers the whole corpus; Spearman over opposite
    # score orders is exactly -1.
    result = compute_mrc(sim, query_ids, corpus_ids, k=3)
    assert result["MRC@3_avg"] == pytest.approx(-100.0)


def test_mrc_averages_pairwise_correlations():
    """RC for a language is the mean correlation against the other languages.

    en and de rank identically (corr +1); fr ranks exactly opposite
    (corr -1 against both). So RC(en) = RC(de) = (1 - 1) / 2 = 0 and
    RC(fr) = (-1 - 1) / 2 = -1.
    """
    corpus_ids = [f"d{i}" for i in range(5)]
    query_ids = ["en_q_h1", "de_q_h1", "fr_q_h1"]
    row = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    sim = np.stack([row, row, row[::-1]])

    result = compute_mrc(sim, query_ids, corpus_ids, k=3)
    assert result["MRC@3_en"] == pytest.approx(0.0)
    assert result["MRC@3_de"] == pytest.approx(0.0)
    assert result["MRC@3_fr"] == pytest.approx(-100.0)


def test_mrc_ignores_singleton_groups():
    """Queries without a parallel counterpart contribute nothing."""
    corpus_ids = [f"d{i}" for i in range(4)]
    query_ids = ["en_q_h1", "de_q_h2"]  # different hashes -> no pairs
    sim = np.random.RandomState(0).rand(2, 4)

    result = compute_mrc(sim, query_ids, corpus_ids, k=2)
    assert result == {}
