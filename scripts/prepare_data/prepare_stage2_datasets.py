"""Pre-sample the Stage 2 training datasets with uniform language distribution.

Reads the aligned mMARCO parallel table (see build_mmarco_parallel.py) and
materializes one dataset per loss type, so that all training runs see the
exact same deterministic language assignments:

  - cross-lingual (mimo, cross_infonce): 182 ordered pairs (14x13) via
    round-robin cycling, each appearing floor(N/182) or ceil(N/182) times
  - mono-lingual (infonce): 14 languages, uniform round-robin
  - lakda: (lang_a, lang_b) uniform over 182 ordered pairs; lang_p uniform
    over 14 languages

Output datasets (under --output_dir):
    mimo/          — anchor, positive (cross-lang), anchor_en, positive_en
    infonce/       — anchor, positive (same language)
    cross_infonce/ — anchor, positive (different languages)
    lakda/         — anchor, positive, anchor_b (two different query languages)

Usage:
    python scripts/prepare_data/prepare_stage2_datasets.py \
        --source data/mmarco_parallel --output_dir data/stage2 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import Counter

from datasets import Dataset, load_from_disk

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

LANG_CODES = [
    "ar", "de", "dt", "en", "es", "fr", "hi",
    "id", "it", "ja", "pt", "ru", "vi", "zh",
]


def make_uniform_cross_langs(n: int, rng: random.Random) -> list[tuple[str, str]]:
    """n cross-lingual (lang_q, lang_p) pairs, uniform over all 182 ordered pairs."""
    all_pairs = [(a, b) for a in LANG_CODES for b in LANG_CODES if a != b]
    result = []
    while len(result) < n:
        cycle = list(all_pairs)
        rng.shuffle(cycle)
        result.extend(cycle)
    return result[:n]


def make_uniform_mono_langs(n: int, rng: random.Random) -> list[str]:
    """n language selections, uniform over the 14 languages."""
    result = []
    while len(result) < n:
        cycle = list(LANG_CODES)
        rng.shuffle(cycle)
        result.extend(cycle)
    return result[:n]


def main():
    parser = argparse.ArgumentParser(description="Pre-sample Stage 2 training datasets")
    parser.add_argument("--source", type=str, required=True,
                        help="Aligned mMARCO parallel dataset (build_mmarco_parallel.py output)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("Loading source dataset from %s", args.source)
    src = load_from_disk(args.source)
    n = len(src)
    logger.info("Source: %d rows", n)

    # Separate RNGs for each distribution to ensure independence
    rng_cross = random.Random(args.seed)
    rng_mono = random.Random(args.seed + 1)
    rng_lakda_q = random.Random(args.seed + 2)
    rng_lakda_p = random.Random(args.seed + 3)

    logger.info("Pre-sampling language assignments (seed=%d, uniform round-robin)...", args.seed)
    cross_langs = make_uniform_cross_langs(n, rng_cross)
    mono_langs = make_uniform_mono_langs(n, rng_mono)
    lakda_q_langs = make_uniform_cross_langs(n, rng_lakda_q)
    lakda_p_langs = make_uniform_mono_langs(n, rng_lakda_p)

    outputs: dict[str, Dataset] = {}

    # 1. MIMO: cross-lingual student pairs + English teacher pairs
    logger.info("Building MIMO dataset...")
    mimo_data = {"anchor": [], "positive": [], "anchor_en": [], "positive_en": [], "lang_q": [], "lang_p": []}
    for i in range(n):
        row = src[i]
        lang_q, lang_p = cross_langs[i]
        mimo_data["anchor"].append(row[f"query_{lang_q}"])
        mimo_data["positive"].append(row[f"positive_{lang_p}"])
        mimo_data["anchor_en"].append(row["query_en"])
        mimo_data["positive_en"].append(row["positive_en"])
        mimo_data["lang_q"].append(lang_q)
        mimo_data["lang_p"].append(lang_p)
    outputs["mimo"] = Dataset.from_dict(mimo_data)

    # 2. InfoNCE: same language for query and positive
    logger.info("Building InfoNCE dataset...")
    infonce_data = {"anchor": [], "positive": [], "lang": []}
    for i in range(n):
        row = src[i]
        lang = mono_langs[i]
        infonce_data["anchor"].append(row[f"query_{lang}"])
        infonce_data["positive"].append(row[f"positive_{lang}"])
        infonce_data["lang"].append(lang)
    outputs["infonce"] = Dataset.from_dict(infonce_data)

    # 3. Cross-InfoNCE (XLCO): same cross-lingual assignments as MIMO
    logger.info("Building Cross-InfoNCE dataset...")
    cross_data = {"anchor": [], "positive": [], "lang_q": [], "lang_p": []}
    for i in range(n):
        row = src[i]
        lang_q, lang_p = cross_langs[i]
        cross_data["anchor"].append(row[f"query_{lang_q}"])
        cross_data["positive"].append(row[f"positive_{lang_p}"])
        cross_data["lang_q"].append(lang_q)
        cross_data["lang_p"].append(lang_p)
    outputs["cross_infonce"] = Dataset.from_dict(cross_data)

    # 4. LaKDA: two different query languages + independent positive language
    logger.info("Building LaKDA dataset...")
    lakda_data = {"anchor": [], "positive": [], "anchor_b": [], "lang": [], "lang_b": [], "lang_p": []}
    for i in range(n):
        row = src[i]
        lang_a, lang_b = lakda_q_langs[i]
        lang_p = lakda_p_langs[i]
        lakda_data["anchor"].append(row[f"query_{lang_a}"])
        lakda_data["positive"].append(row[f"positive_{lang_p}"])
        lakda_data["anchor_b"].append(row[f"query_{lang_b}"])
        lakda_data["lang"].append(lang_a)
        lakda_data["lang_b"].append(lang_b)
        lakda_data["lang_p"].append(lang_p)
    outputs["lakda"] = Dataset.from_dict(lakda_data)

    for name, ds in outputs.items():
        path = f"{args.output_dir}/{name}"
        ds.save_to_disk(path)
        logger.info("Saved %s: %d rows -> %s", name, len(ds), path)

    # Uniformity verification
    logger.info("=== Uniformity Verification ===")
    pair_counts = Counter(cross_langs)
    logger.info("Cross-lingual pairs: %d unique, min=%d, max=%d",
                len(pair_counts), min(pair_counts.values()), max(pair_counts.values()))
    mono_counts = Counter(mono_langs)
    logger.info("Mono-lingual langs: %d unique, min=%d, max=%d",
                len(mono_counts), min(mono_counts.values()), max(mono_counts.values()))


if __name__ == "__main__":
    main()
