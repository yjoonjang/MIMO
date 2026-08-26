"""Stage 1 data preparation: download and filter parallel sentence pairs.

Source: HuggingFace ``sentence-transformers/parallel-sentences`` (8 OPUS-based
corpora: Europarl, GlobalVoices, JW300, News-Commentary, OpenSubtitles, Talks,
Tatoeba, WikiMatrix), TSV.gz files formatted as ``english\tnon_english``.

Filters: minimum 10 chars, maximum 512 chars, length ratio < 3:1,
English-side deduplication, and a per-language cap of 500k pairs.
English-English identity pairs are added from the collected English texts.

Output: Arrow dataset with columns {anchor: text_XX, positive: text_en, lang}.

Usage:
    python scripts/prepare_data/prepare_stage1_parallel.py --output_dir data/stage1_parallel
"""

from __future__ import annotations

import argparse
import gzip
import logging
import sys
from pathlib import Path

from datasets import Dataset, concatenate_datasets
from huggingface_hub import hf_hub_download

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

REPO_ID = "sentence-transformers/parallel-sentences"

DATASET_NAMES = [
    "Europarl",
    "GlobalVoices",
    "JW300",
    "News-Commentary",
    "OpenSubtitles",
    "Talks",
    "Tatoeba",
    "WikiMatrix",
]

# Non-English target languages (English identity pairs are added separately).
# Note: parallel-sentences uses "nl" for Dutch; mMARCO (Stage 2) uses "dt".
TARGET_LANGUAGES = [
    "ar", "de", "es", "fr", "hi", "id", "it", "ja", "nl", "pt", "ru", "vi", "zh",
]

MIN_LENGTH = 10
MAX_LENGTH = 512
MAX_LENGTH_RATIO = 3.0
MAX_PAIRS_PER_LANG = 500_000

# Note: the repo also carries "zh_cn" / "zh-cn" variant files for Chinese;
# the paper's dataset (Table 5: zh = 35,407 pairs) uses only the "zh" files.


def _passes_quality_filter(en_text: str, other_text: str) -> bool:
    if not en_text or not other_text:
        return False
    len_en, len_other = len(en_text), len(other_text)
    if len_en < MIN_LENGTH or len_other < MIN_LENGTH:
        return False
    if len_en > MAX_LENGTH or len_other > MAX_LENGTH:
        return False
    ratio = max(len_en, len_other) / max(min(len_en, len_other), 1)
    return ratio <= MAX_LENGTH_RATIO


def _download_and_parse(dataset_name: str, lang: str) -> list[dict[str, str]]:
    """Download one TSV.gz file and parse it into filtered (en, non_en) pairs."""
    filename = f"{dataset_name}/{dataset_name}-en-{lang}-train.tsv.gz"
    try:
        local_path = hf_hub_download(REPO_ID, filename, repo_type="dataset")
    except Exception:
        logger.debug("File not found: %s", filename)
        return []

    all_pairs = []
    with gzip.open(local_path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            en_text, other_text = parts[0], parts[1]
            if _passes_quality_filter(en_text, other_text):
                all_pairs.append({
                    "anchor": other_text,
                    "positive": en_text,
                    "lang": lang,
                })
    return all_pairs


def main():
    parser = argparse.ArgumentParser(description="Prepare Stage 1 parallel sentence data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for the Arrow dataset")
    parser.add_argument("--max_pairs_per_lang", type=int, default=MAX_PAIRS_PER_LANG)
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_pairs: dict[str, list[dict]] = {lang: [] for lang in TARGET_LANGUAGES}
    seen_en_texts: set[str] = set()
    en_texts_in_order: list[str] = []  # deterministic (first-seen) order

    for ds_name in DATASET_NAMES:
        for lang in TARGET_LANGUAGES:
            pairs = _download_and_parse(ds_name, lang)
            if not pairs:
                continue

            count_added = 0
            for pair in pairs:
                en_text = pair["positive"]
                if en_text in seen_en_texts:
                    continue
                seen_en_texts.add(en_text)
                en_texts_in_order.append(en_text)
                all_pairs[lang].append(pair)
                count_added += 1

            logger.info("Loaded %d pairs from %s for lang=%s (%d after dedup)",
                        len(pairs), ds_name, lang, count_added)

    # English-English identity pairs from the collected English texts
    en_pairs = [
        {"anchor": en_text, "positive": en_text, "lang": "en"}
        for en_text in en_texts_in_order[: args.max_pairs_per_lang]
    ]
    all_pairs["en"] = en_pairs
    logger.info("Added %d English-English identity pairs", len(en_pairs))

    datasets_per_lang = []
    for lang in TARGET_LANGUAGES + ["en"]:
        pairs = all_pairs.get(lang, [])[: args.max_pairs_per_lang]
        if not pairs:
            logger.warning("No pairs found for language: %s", lang)
            continue
        datasets_per_lang.append(Dataset.from_list(pairs))
        logger.info("Language %s: %d pairs (capped at %d)", lang, len(pairs), args.max_pairs_per_lang)

    if not datasets_per_lang:
        raise ValueError("No data found for any target language!")

    combined = concatenate_datasets(datasets_per_lang).shuffle(seed=42)
    combined.save_to_disk(str(output_path))
    logger.info("Stage 1 data saved to %s: %d total pairs", output_path, len(combined))


if __name__ == "__main__":
    main()
