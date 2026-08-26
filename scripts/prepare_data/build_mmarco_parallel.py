"""Build the mMARCO parallel table with ID-based alignment across 14 languages.

Downloads ``unicamp-dl/mmarco`` (Google Translate version) from the
HuggingFace Hub and builds a parallel Arrow dataset where each row contains
aligned (query, positive) pairs across ALL 14 languages.

Uses ``triples.train.ids.small.tsv`` for unique (qid, pid_pos) pairs, then
looks up the query and passage text in each language's TSV files.

Output columns:
    query_ar, ..., query_zh, positive_ar, ..., positive_zh (14 languages)

Note: "dt" is the mMARCO code for Dutch.

Usage:
    python scripts/prepare_data/build_mmarco_parallel.py --output_dir data/mmarco_parallel
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import OrderedDict

from datasets import Dataset
from huggingface_hub import hf_hub_download

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

REPO_ID = "unicamp-dl/mmarco"

# Language name in the mmarco HF repo -> short code
LANG_MAP = OrderedDict([
    ("arabic", "ar"),
    ("chinese", "zh"),
    ("dutch", "dt"),
    ("english", "en"),
    ("french", "fr"),
    ("german", "de"),
    ("hindi", "hi"),
    ("indonesian", "id"),
    ("italian", "it"),
    ("japanese", "ja"),
    ("portuguese", "pt"),
    ("russian", "ru"),
    ("spanish", "es"),
    ("vietnamese", "vi"),
])


def load_tsv_dict(path: str, needed_ids: set[str] | None = None) -> dict[str, str]:
    """Load a TSV file (id\\ttext) into a dict, optionally filtering by needed IDs."""
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            idx, text = parts
            if needed_ids is None or idx in needed_ids:
                d[idx] = text
    return d


def main():
    parser = argparse.ArgumentParser(description="Build the aligned mMARCO parallel dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the Arrow dataset")
    args = parser.parse_args()

    # Step 1: unique (qid, pid_pos) pairs from the training triples
    logger.info("Downloading triples.train.ids.small.tsv...")
    triples_path = hf_hub_download(REPO_ID, "data/triples.train.ids.small.tsv", repo_type="dataset")

    logger.info("Extracting unique (qid, pid_pos) pairs...")
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    needed_qids: set[str] = set()
    needed_pids: set[str] = set()

    with open(triples_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            qid, pid_pos, _ = line.strip().split("\t")
            pair = (qid, pid_pos)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
                needed_qids.add(qid)
                needed_pids.add(pid_pos)
            if (i + 1) % 10_000_000 == 0:
                logger.info("  Processed %dM triples...", (i + 1) // 1_000_000)

    logger.info("Unique pairs: %d (from %d qids, %d pids)", len(pairs), len(needed_qids), len(needed_pids))

    # Step 2: per-language queries and collections
    data: dict[str, list[str]] = {}
    for short_code in LANG_MAP.values():
        data[f"query_{short_code}"] = []
        data[f"positive_{short_code}"] = []

    for lang_full, lang_short in LANG_MAP.items():
        logger.info("Processing language: %s (%s)", lang_full, lang_short)

        query_file = f"data/google/queries/train/{lang_full}_queries.train.tsv"
        query_path = hf_hub_download(REPO_ID, query_file, repo_type="dataset")
        queries = load_tsv_dict(query_path, needed_qids)
        logger.info("  Loaded %d queries (needed %d)", len(queries), len(needed_qids))

        collection_file = f"data/google/collections/{lang_full}_collection.tsv"
        logger.info("  Downloading %s collection (this may take a while)...", lang_full)
        collection_path = hf_hub_download(REPO_ID, collection_file, repo_type="dataset")
        collection = load_tsv_dict(collection_path, needed_pids)
        logger.info("  Loaded %d passages (needed %d)", len(collection), len(needed_pids))

        q_col = [queries.get(qid, "") for qid, _ in pairs]
        p_col = [collection.get(pid, "") for _, pid in pairs]

        missing_q = sum(1 for t in q_col if t == "")
        missing_p = sum(1 for t in p_col if t == "")
        if missing_q:
            logger.warning("  %s: %d missing queries!", lang_short, missing_q)
        if missing_p:
            logger.warning("  %s: %d missing passages!", lang_short, missing_p)

        data[f"query_{lang_short}"] = q_col
        data[f"positive_{lang_short}"] = p_col

        del queries, collection
        logger.info("  Done with %s", lang_full)

    # Step 3: build and save
    logger.info("Building Arrow dataset with %d rows...", len(pairs))
    dataset = Dataset.from_dict(data)

    logger.info("=== Alignment verification (first 3 rows) ===")
    sample_langs = list(LANG_MAP.values())[:3]
    for i in range(min(3, len(dataset))):
        qid, pid = pairs[i]
        logger.info("Row %d: qid=%s, pid=%s", i, qid, pid)
        for lang in sample_langs:
            logger.info("  %s query:    %s", lang.upper(), dataset[i][f"query_{lang}"][:80])
        logger.info("  %s positive: %s", sample_langs[0].upper(), dataset[i][f"positive_{sample_langs[0]}"][:80])

    dataset.save_to_disk(args.output_dir)
    logger.info("Saved %d rows to %s", len(dataset), args.output_dir)


if __name__ == "__main__":
    main()
