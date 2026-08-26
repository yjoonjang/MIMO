"""MLIR benchmark resolution and loading.

Benchmarks are directories of BEIR-style JSONL files:
    queries.jsonl: {"_id": ..., "text": ..., "lang": ...}
    corpus.jsonl:  {doc_id: {"text": ..., "lang": ...}}
    qrels.jsonl:   {query_id: {doc_id: relevance}}

Benchmarks are read from a local directory when present, and otherwise
downloaded from the HuggingFace Hub (the exact files used in the paper).
They can also be rebuilt from the original sources with:
    python scripts/prepare_data/build_benchmarks.py \
        --output_dir data/benchmarks --multilingual_queries

MIMO evaluates the multilingual-query (MMLIR) variant: queries in all
languages, positives are all language versions of the gold context.
"""

from __future__ import annotations

import json
import os

from huggingface_hub import snapshot_download

DEFAULT_DATA_DIR = "data/benchmarks"
DEFAULT_BENCHMARK_REPO = "yjoonjang/mlir-benchmarks"

ALL_BENCHMARKS = ["Belebele_test", "XQuAD_test", "MLQA_test", "MultiEup_test"]


def resolve_benchmark_dir(
    benchmark_name: str,
    data_dir: str = DEFAULT_DATA_DIR,
    repo_id: str = DEFAULT_BENCHMARK_REPO,
) -> str:
    """Return a local directory containing the benchmark's JSONL files.

    Resolution order: a local root produced by build_benchmarks.py (with an
    ``MMLIR/`` variant subdirectory), a directory containing the benchmark
    folders directly, then the HuggingFace Hub.
    """
    candidates = [
        os.path.join(data_dir, "MMLIR", benchmark_name),
        os.path.join(data_dir, benchmark_name),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    root = snapshot_download(
        repo_id, repo_type="dataset", allow_patterns=[f"MMLIR/{benchmark_name}/*"]
    )
    hub_dir = os.path.join(root, "MMLIR", benchmark_name)
    if os.path.isdir(hub_dir):
        return hub_dir
    raise FileNotFoundError(
        f"Benchmark '{benchmark_name}' not found under {data_dir} "
        f"or in https://huggingface.co/datasets/{repo_id}"
    )


def load_eval_benchmark(benchmark_dir: str) -> tuple[dict, dict, dict]:
    """Load (queries, corpus, relevant_docs) from a benchmark directory."""
    queries, corpus, relevant_docs = {}, {}, {}

    with open(os.path.join(benchmark_dir, "queries.jsonl")) as f:
        for line in f:
            item = json.loads(line)
            queries[item["_id"]] = item["text"]

    with open(os.path.join(benchmark_dir, "corpus.jsonl")) as f:
        for line in f:
            item = json.loads(line)
            for doc_id, doc_data in item.items():
                corpus[doc_id] = doc_data["text"]

    with open(os.path.join(benchmark_dir, "qrels.jsonl")) as f:
        for line in f:
            item = json.loads(line)
            for query_id, doc_scores in item.items():
                relevant_docs[query_id] = set(doc_scores.keys())

    return queries, corpus, relevant_docs
