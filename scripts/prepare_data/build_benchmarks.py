"""Construct the MLIR evaluation benchmarks from public datasets.

Adapts four multilingual QA/retrieval datasets to the MLIR setting: all
language versions of the context passages are pooled into one mixed corpus,
and for each query the gold context in every language is a positive.

Datasets:
    belebele  — facebook/belebele (14 languages, fully parallel)
    xquad     — XQuAD-R from LAReQA (8 languages, fully parallel)
    mlqa      — facebook/mlqa (7 languages, partially parallel)
    multieup  — unimelb-nlp/MultiEup-v2 (parliamentary speeches, EU languages)

Variants:
    --query_lang en            queries in one source language (default: en);
                               saved under {output_dir}/MLIR-{lang}/
    --multilingual_queries     queries in ALL languages, ids prefixed with the
                               language ("{lang}_{qid}"); saved under
                               {output_dir}/MMLIR/  (used by the MIMO paper and
                               the M2MIR experiments)

Output format per benchmark directory:
    queries.jsonl: {"_id": ..., "text": ...[, "lang": ...]}
    corpus.jsonl:  {doc_id: {"text": ..., "lang": ...}}
    qrels.jsonl:   {query_id: {doc_id: 1}}

Usage:
    python scripts/prepare_data/build_benchmarks.py --output_dir data/benchmarks
    python scripts/prepare_data/build_benchmarks.py --output_dir data/benchmarks --query_lang zh
    python scripts/prepare_data/build_benchmarks.py --output_dir data/benchmarks --multilingual_queries
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import urllib.request
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BELEBELE_CONFIGS = {
    "ar": "acm_Arab", "de": "deu_Latn", "dt": "nld_Latn", "en": "eng_Latn",
    "es": "spa_Latn", "fr": "fra_Latn", "hi": "hin_Deva", "id": "ind_Latn",
    "it": "ita_Latn", "ja": "jpn_Jpan", "pt": "por_Latn", "ru": "rus_Cyrl",
    "vi": "vie_Latn", "zh": "zho_Hans",
}

XQUAD_LANGS = ["ar", "de", "en", "es", "hi", "ru", "vi", "zh"]
XQUAD_R_URL = "https://raw.githubusercontent.com/google-research-datasets/lareqa/master/xquad-r/{lang}.json"

MLQA_LANGS = ["ar", "de", "en", "es", "hi", "vi", "zh"]

MULTIEUP_LANGS = ["ar", "de", "dt", "en", "es", "fr", "hi", "id", "it", "ja", "pt", "ru", "vi", "zh"]


def save_benchmark(output_dir: str, queries: dict, corpus: dict, qrels: dict) -> None:
    """queries: {qid: text} or {qid: {"text", "lang"}}; corpus: {doc_id: {"text", "lang"}}."""
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/queries.jsonl", "w", encoding="utf-8") as f:
        for qid in sorted(queries.keys()):
            q = queries[qid]
            record = {"_id": qid, "text": q["text"], "lang": q["lang"]} if isinstance(q, dict) else {"_id": qid, "text": q}
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    with open(f"{output_dir}/corpus.jsonl", "w", encoding="utf-8") as f:
        for doc_id in sorted(corpus.keys()):
            json.dump({doc_id: corpus[doc_id]}, f, ensure_ascii=False)
            f.write("\n")

    with open(f"{output_dir}/qrels.jsonl", "w", encoding="utf-8") as f:
        for qid in sorted(qrels.keys()):
            json.dump({qid: dict(qrels[qid])}, f, ensure_ascii=False)
            f.write("\n")

    n_qrels = sum(len(d) for d in qrels.values())
    logger.info("Saved %s: %d queries, %d docs, %d qrels", output_dir, len(queries), len(corpus), n_qrels)


# ---------------------------------------------------------------------------
# Belebele
# ---------------------------------------------------------------------------
def build_belebele(query_langs: list[str], multilingual: bool):
    datasets = {}
    for lang, config in BELEBELE_CONFIGS.items():
        logger.info("Loading belebele %s...", config)
        datasets[lang] = load_dataset("facebook/belebele", config, split="test")

    queries: dict = {}
    corpus: dict = {}
    qrels: dict = defaultdict(dict)
    context_hash_to_docid: dict[str, str] = {}
    key_to_base_qid: dict[tuple, str] = {}

    # Anchor pass (English) defines the query id space
    for item in datasets["en"]:
        unique_key = (item["link"], item["question_number"])
        q_hash_str = f"{item['link']}_{item['question_number']}"
        key_to_base_qid[unique_key] = hashlib.md5(q_hash_str.encode()).hexdigest()[:8]

    # Corpus + qrels over all languages
    for lang in BELEBELE_CONFIGS:
        for item in tqdm(datasets[lang], desc=f"Belebele corpus {lang}"):
            unique_key = (item["link"], item["question_number"])
            if unique_key not in key_to_base_qid:
                continue
            base_qid = key_to_base_qid[unique_key]

            passage = item["flores_passage"]
            pass_hash = hashlib.md5(passage.encode()).hexdigest()
            if pass_hash not in context_hash_to_docid:
                doc_id = f"doc_{lang}_{len(corpus)}"
                context_hash_to_docid[pass_hash] = doc_id
                corpus[doc_id] = {"text": passage, "lang": lang}
            else:
                doc_id = context_hash_to_docid[pass_hash]

            for q_lang in query_langs:
                qid = f"{q_lang}_q_{base_qid}" if multilingual else f"q_{base_qid}"
                qrels[qid][doc_id] = 1

    # Queries
    for q_lang in query_langs:
        for item in datasets[q_lang]:
            unique_key = (item["link"], item["question_number"])
            if unique_key not in key_to_base_qid:
                continue
            base_qid = key_to_base_qid[unique_key]
            if multilingual:
                queries[f"{q_lang}_q_{base_qid}"] = {"text": item["question"], "lang": q_lang}
            else:
                queries[f"q_{base_qid}"] = item["question"]

    qrels = {qid: docs for qid, docs in qrels.items() if qid in queries}
    return queries, corpus, qrels


# ---------------------------------------------------------------------------
# XQuAD (XQuAD-R from LAReQA)
# ---------------------------------------------------------------------------
def _download_xquad_r(cache_dir: str) -> dict[str, dict]:
    os.makedirs(cache_dir, exist_ok=True)
    data = {}
    for lang in XQUAD_LANGS:
        path = os.path.join(cache_dir, f"{lang}.json")
        if not os.path.exists(path):
            url = XQUAD_R_URL.format(lang=lang)
            logger.info("Downloading %s...", url)
            urllib.request.urlretrieve(url, path)
        with open(path, encoding="utf-8") as f:
            data[lang] = json.load(f)
    return data


def build_xquad(query_langs: list[str], multilingual: bool, cache_dir: str):
    lang_data = _download_xquad_r(cache_dir)

    queries: dict = {}
    corpus: dict = {}
    qrels: dict = defaultdict(dict)
    qid_to_docs: dict[str, set] = defaultdict(set)

    # Contexts + original-qid -> doc mapping
    for lang, data in lang_data.items():
        for data_idx, item in enumerate(data["data"]):
            for para_idx, paragraph in enumerate(item["paragraphs"]):
                doc_id = f"{lang}_{data_idx}_{para_idx}"
                corpus[doc_id] = {"text": paragraph["context"], "lang": lang}
                for qa in paragraph["qas"]:
                    qid_to_docs[qa["id"]].add(doc_id)

    # Queries + qrels
    for q_lang in query_langs:
        for item in lang_data[q_lang]["data"]:
            for paragraph in item["paragraphs"]:
                for qa in paragraph["qas"]:
                    qid = f"{q_lang}_{qa['id']}" if multilingual else qa["id"]
                    if multilingual:
                        queries[qid] = {"text": qa["question"], "lang": q_lang}
                    else:
                        queries[qid] = qa["question"]
                    for doc_id in qid_to_docs[qa["id"]]:
                        qrels[qid][doc_id] = 1

    return queries, corpus, dict(qrels)


# ---------------------------------------------------------------------------
# MLQA
# ---------------------------------------------------------------------------
def build_mlqa(query_langs: list[str], multilingual: bool):
    queries: dict = {}
    corpus: dict = {}
    qrels: dict = defaultdict(dict)
    context_hash_map: dict[str, str] = {}
    original_qid_to_docs: dict[str, set] = defaultdict(set)

    def load_config(ctx_lang, q_lang):
        try:
            return load_dataset("facebook/mlqa", f"mlqa.{ctx_lang}.{q_lang}", split="test")
        except Exception:
            return None

    # Pass 1: collect contexts and original-qid -> doc mapping over all pairs
    for ctx_lang in MLQA_LANGS:
        for q_lang in MLQA_LANGS if multilingual else query_langs:
            ds = load_config(ctx_lang, q_lang)
            if ds is None:
                continue
            for item in tqdm(ds, desc=f"MLQA contexts {ctx_lang}.{q_lang}"):
                ctx_hash = hashlib.md5(item["context"].encode("utf-8")).hexdigest()
                if ctx_hash not in context_hash_map:
                    doc_id = f"doc_{ctx_lang}_{len(corpus)}"
                    context_hash_map[ctx_hash] = doc_id
                    corpus[doc_id] = {"text": item["context"], "lang": ctx_lang}
                else:
                    doc_id = context_hash_map[ctx_hash]
                original_qid_to_docs[item["id"]].add(doc_id)

    # Pass 2: queries + qrels
    if multilingual:
        query_hash_map: dict[tuple, str] = {}
        for ctx_lang in MLQA_LANGS:
            for q_lang in MLQA_LANGS:
                ds = load_config(ctx_lang, q_lang)
                if ds is None:
                    continue
                for item in tqdm(ds, desc=f"MLQA queries {ctx_lang}.{q_lang}"):
                    query_key = (item["id"], q_lang)
                    if query_key in query_hash_map:
                        continue
                    qid = f"{q_lang}_{item['id']}"
                    query_hash_map[query_key] = qid
                    queries[qid] = {"text": item["question"], "lang": q_lang}
                    for doc_id in original_qid_to_docs[item["id"]]:
                        qrels[qid][doc_id] = 1
    else:
        q_lang = query_langs[0]
        query_hash_to_qid: dict[str, str] = {}
        for ctx_lang in MLQA_LANGS:
            ds = load_config(ctx_lang, q_lang)
            if ds is None:
                continue
            for item in tqdm(ds, desc=f"MLQA queries {ctx_lang}.{q_lang}"):
                q_hash = hashlib.md5(item["question"].strip().encode("utf-8")).hexdigest()
                if q_hash not in query_hash_to_qid:
                    qid = f"q_{len(queries)}"
                    query_hash_to_qid[q_hash] = qid
                    queries[qid] = item["question"]
                else:
                    qid = query_hash_to_qid[q_hash]
                ctx_hash = hashlib.md5(item["context"].encode("utf-8")).hexdigest()
                qrels[qid][context_hash_map[ctx_hash]] = 1

    return queries, corpus, dict(qrels)


# ---------------------------------------------------------------------------
# MultiEuP-v2
# ---------------------------------------------------------------------------
def build_multieup(query_langs: list[str], multilingual: bool):
    logger.info("Loading unimelb-nlp/MultiEup-v2 (streaming)...")
    # Non-streaming load fails on this dataset's mixed-type columns
    ds = load_dataset("unimelb-nlp/MultiEup-v2", split="train", streaming=True)

    entries = [e for e in tqdm(ds, desc="MultiEup stream") if e["LANGUAGE"].lower() in MULTIEUP_LANGS]
    logger.info("Filtered to %d entries in mMARCO languages", len(entries))

    queries: dict = {}
    corpus: dict = {}
    qrels: dict = defaultdict(dict)
    original_qid_to_docs: dict[str, set] = defaultdict(set)

    # Corpus + original-qid -> doc mapping (anchored on the English query id)
    for entry in entries:
        doc_id, doc_text = entry["did"], entry["TEXT"]
        qid_en = entry.get("qid_EN")
        if doc_id is None or doc_text is None or qid_en is None:
            continue
        corpus[doc_id] = {"text": doc_text, "lang": entry["LANGUAGE"].lower()}
        original_qid_to_docs[qid_en.split("#")[0]].add(doc_id)

    # Queries + qrels
    for entry in entries:
        for q_lang in query_langs:
            title = entry.get(f"title_{q_lang.upper()}")
            qid_raw = entry.get(f"qid_{q_lang.upper()}")
            if title is None or qid_raw is None:
                continue
            original_qid = qid_raw.split("#")[0]
            qid = f"{q_lang}_{original_qid}" if multilingual else qid_raw
            if multilingual:
                queries[qid] = {"text": title, "lang": q_lang}
            else:
                queries[qid] = title
            for doc_id in original_qid_to_docs[original_qid]:
                qrels[qid][doc_id] = 1

    qrels = {qid: docs for qid, docs in qrels.items() if docs}
    queries = {qid: q for qid, q in queries.items() if qid in qrels}
    return queries, corpus, qrels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
BUILDERS = {
    "belebele": ("Belebele_test", build_belebele, list(BELEBELE_CONFIGS.keys())),
    "xquad": ("XQuAD_test", build_xquad, XQUAD_LANGS),
    "mlqa": ("MLQA_test", build_mlqa, MLQA_LANGS),
    "multieup": ("MultiEup_test", build_multieup, MULTIEUP_LANGS),
}


def main():
    parser = argparse.ArgumentParser(description="Build the MLIR evaluation benchmarks")
    parser.add_argument("--datasets", type=str, nargs="+", default=list(BUILDERS.keys()),
                        choices=list(BUILDERS.keys()))
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--query_lang", type=str, default="en",
                        help="Source query language (single-language variant)")
    parser.add_argument("--multilingual_queries", action="store_true",
                        help="Build the MMLIR variant with queries in all languages")
    parser.add_argument("--cache_dir", type=str, default="data/raw",
                        help="Cache directory for raw downloads (XQuAD-R)")
    args = parser.parse_args()

    variant_dir = "MMLIR" if args.multilingual_queries else f"MLIR-{args.query_lang}"

    for key in args.datasets:
        name, builder, available_langs = BUILDERS[key]
        if args.multilingual_queries:
            query_langs = available_langs
        else:
            if args.query_lang not in available_langs:
                logger.warning("%s does not cover query_lang=%s, skipping", name, args.query_lang)
                continue
            query_langs = [args.query_lang]

        logger.info("=== Building %s (%s) ===", name, variant_dir)
        if key == "xquad":
            queries, corpus, qrels = builder(query_langs, args.multilingual_queries, args.cache_dir)
        else:
            queries, corpus, qrels = builder(query_langs, args.multilingual_queries)

        out_dir = os.path.join(args.output_dir, variant_dir, name)
        save_benchmark(out_dir, queries, corpus, qrels)


if __name__ == "__main__":
    main()
