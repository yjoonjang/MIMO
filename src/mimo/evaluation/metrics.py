"""MLIR language fairness metrics: MRC and PEER.

MRC (Mean Rank Correlation): Measures ranking consistency across parallel queries
in different languages. From Yang et al., EMNLP MRL 2024.

PEER (Probability of Equal Expected Rank): Measures whether relevant documents
are ranked fairly regardless of their language. Adapted from Yang et al., SIGIR 2024.
"""

from __future__ import annotations

import logging
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr, kruskal

logger = logging.getLogger(__name__)


def _parse_query_lang(qid: str) -> str:
    """Extract language from query ID (e.g., 'ar_q_005a' -> 'ar')."""
    return qid.split("_")[0]


def _parse_query_hash(qid: str) -> str:
    """Extract semantic hash from query ID.

    Handles both '{lang}_q_{hash}' (Belebele) and '{lang}_{hash}' (XQuAD) formats.
    Parallel queries share the same hash.
    """
    parts = qid.split("_")
    if len(parts) > 2 and parts[1] == "q":
        return "_".join(parts[2:])
    return "_".join(parts[1:])


def _parse_doc_lang(doc_id: str) -> str:
    """Extract language from doc ID (e.g., 'doc_en_1466' -> 'en')."""
    parts = doc_id.split("_")
    return parts[1] if len(parts) >= 3 else "unknown"


def compute_mrc(
    sim_matrix: np.ndarray,
    query_ids: list[str],
    corpus_ids: list[str],
    k: int = 10,
) -> dict[str, float]:
    """Compute Mean Rank Correlation (MRC@k).

    For each group of parallel queries (same meaning, different languages),
    computes pairwise Spearman rank correlation on the union of top-k documents.

    Args:
        sim_matrix: Similarity matrix (num_queries, num_docs), pre-normalized.
        query_ids: Query IDs aligned with sim_matrix rows.
        corpus_ids: Doc IDs aligned with sim_matrix columns.
        k: Top-k cutoff for ranking comparison.

    Returns:
        Dict with MRC@k per language and average. Scale: -100 to 100 (higher = better).
    """
    # Group parallel queries by hash
    query_groups = defaultdict(dict)  # hash -> {lang: row_idx}
    for idx, qid in enumerate(query_ids):
        lang = _parse_query_lang(qid)
        hash_part = _parse_query_hash(qid)
        query_groups[hash_part][lang] = idx

    per_lang_scores = defaultdict(list)

    for hash_id, lang_to_idx in query_groups.items():
        if len(lang_to_idx) < 2:
            continue

        langs = sorted(lang_to_idx.keys())

        # Top-k indices for each language
        topk = {}
        for lang in langs:
            topk[lang] = set(np.argsort(-sim_matrix[lang_to_idx[lang]])[:k].tolist())

        # Pairwise Spearman on union of top-k
        for i, lang_a in enumerate(langs):
            pair_corrs = []
            for j, lang_b in enumerate(langs):
                if i == j:
                    continue

                union_idx = sorted(topk[lang_a] | topk[lang_b])
                if len(union_idx) < 2:
                    continue

                s_a = sim_matrix[lang_to_idx[lang_a], union_idx]
                s_b = sim_matrix[lang_to_idx[lang_b], union_idx]

                corr, _ = spearmanr(s_a, s_b)
                if not np.isnan(corr):
                    pair_corrs.append(corr)

            if pair_corrs:
                # RC(a)^i = mean over all other languages
                per_lang_scores[lang_a].append(np.mean(pair_corrs))

    # MRC@k(a) = mean over all queries for language a
    results = {}
    all_scores = []
    for lang in sorted(per_lang_scores.keys()):
        score = float(np.mean(per_lang_scores[lang]) * 100)
        results[f"MRC@{k}_{lang}"] = score
        all_scores.append(score)

    if all_scores:
        results[f"MRC@{k}_avg"] = float(np.mean(all_scores))

    return results


def compute_peer(
    sim_matrix: np.ndarray,
    query_ids: list[str],
    corpus_ids: list[str],
    relevant_docs: dict[str, set[str]],
    k: int = 100,
) -> dict[str, float]:
    """Compute PEER (Probability of Equal Expected Rank) for language fairness.

    Following Yang et al., SIGIR 2024: for each query, ranks documents and applies
    Kruskal-Wallis test on relevant document ranks grouped by language.

    Key implementation details per the original paper:
    - Per-query KW test with p-values averaged across queries
    - Rank cutoff: docs outside top-k get assigned rank k+1 (tied rank)
    - If no relevant docs in top-k: p-value = 1.0 (no discrimination detected)
    - Binary relevance (single level, w=1.0)

    The cutoff is crucial: without it, per-query KW with 1 doc/language/query is
    degenerate (H always equals N-1). With cutoff, ties between retrieved vs
    non-retrieved docs create meaningful group differences.

    Args:
        sim_matrix: Similarity matrix (num_queries, num_docs), pre-normalized.
        query_ids: Query IDs aligned with sim_matrix rows.
        corpus_ids: Doc IDs aligned with sim_matrix columns.
        relevant_docs: Mapping query_id -> set of relevant doc_ids.
        k: Rank cutoff. Docs outside top-k get rank k+1.

    Returns:
        Dict with:
        - PEER@{k}: overall PEER score (mean of per-query p-values, higher = fairer)
        - PEER@{k}_{lang}: per query-language PEER score
        - mean_rank_{lang}: mean rank of relevant docs per doc language (with cutoff)
    """
    did_to_idx = {did: i for i, did in enumerate(corpus_ids)}

    per_qlang_pvalues = defaultdict(list)  # query_lang -> [p-values]
    all_pvalues = []
    pooled_doc_ranks = defaultdict(list)  # doc_lang -> [ranks with cutoff]

    for q_idx, qid in enumerate(query_ids):
        if qid not in relevant_docs:
            continue

        q_lang = _parse_query_lang(qid)

        # Full ranking (1-indexed, lower = better)
        ranking = np.argsort(np.argsort(-sim_matrix[q_idx])) + 1

        # Collect relevant doc ranks with cutoff
        doc_lang_ranks = defaultdict(list)  # doc_lang -> [rank]
        for rel_did in relevant_docs[qid]:
            if rel_did not in did_to_idx:
                continue
            d_idx = did_to_idx[rel_did]
            d_lang = _parse_doc_lang(rel_did)
            raw_rank = int(ranking[d_idx])
            rank = min(raw_rank, k + 1)  # Apply cutoff
            doc_lang_ranks[d_lang].append(rank)
            pooled_doc_ranks[d_lang].append(rank)

        # Per-query KW test
        groups = [ranks for ranks in doc_lang_ranks.values() if len(ranks) > 0]

        if len(groups) < 2:
            continue

        all_ranks = [r for g in groups for r in g]

        if len(set(all_ranks)) <= 1:
            # All ranks identical (e.g., all k+1 or all same rank) → no discrimination
            p_value = 1.0
        else:
            try:
                _, p_value = kruskal(*groups)
            except ValueError:
                p_value = 1.0

        all_pvalues.append(p_value)
        per_qlang_pvalues[q_lang].append(p_value)

    results = {}

    # Overall PEER = mean of per-query p-values
    if all_pvalues:
        results[f"PEER@{k}"] = float(np.mean(all_pvalues))

    # Per query-language PEER
    for q_lang in sorted(per_qlang_pvalues.keys()):
        results[f"PEER@{k}_{q_lang}"] = float(np.mean(per_qlang_pvalues[q_lang]))

    # Mean rank per doc language (with cutoff applied)
    for dl in sorted(pooled_doc_ranks.keys()):
        results[f"mean_rank@{k}_{dl}"] = float(np.mean(pooled_doc_ranks[dl]))

    return results
