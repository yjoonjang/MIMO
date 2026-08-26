"""Evaluate models on the MLIR benchmarks (Belebele, MLQA, XQuAD, MultiEuP-v2).

Each benchmark mixes all language versions of the context passages into a
single corpus; queries appear in every language, and all language versions of
the gold context are positives.

Saves per-benchmark JSON: {output_dir}/{model_name}/{benchmark}.json
Metrics: nDCG@{10,20,100}, Recall@{10,20,100}, MRR@{10,20,100}, plus
language-fairness metrics (MRC, PEER).

Usage:
    python evaluate.py --model_name_or_path /path/to/model
    python evaluate.py --model_name_or_path /path/to/model --benchmarks Belebele_test,XQuAD_test
    python evaluate.py --model_name_or_path /path/to/model --data_dir /path/to/benchmarks
"""

import json
import logging
import os
from collections import defaultdict

import fire
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from mimo.evaluation import load_eval_benchmark, resolve_benchmark_dir
from mimo.evaluation.benchmarks import ALL_BENCHMARKS, DEFAULT_DATA_DIR
from mimo.evaluation.metrics import _parse_query_hash, _parse_query_lang, compute_mrc, compute_peer

logger = logging.getLogger(__name__)


def _has_parallel_queries(queries: dict[str, str]) -> bool:
    hashes = defaultdict(set)
    for qid in queries:
        hashes[_parse_query_hash(qid)].add(_parse_query_lang(qid))
    multi_lang = sum(1 for v in hashes.values() if len(v) >= 2)
    return multi_lang > len(hashes) * 0.5


def evaluate(
    model_name_or_path: str,
    benchmarks: list[str] | str | None = None,
    batch_size: int = 512,
    output_dir: str = "results",
    data_dir: str = DEFAULT_DATA_DIR,
    compute_fairness: bool = True,
    mrc_ks: list[int] | None = None,
    peer_ks: list[int] | None = None,
    trust_remote_code: bool = False,
    query_prompt: str | None = None,
    doc_prompt: str | None = None,
):
    logging.basicConfig(level=logging.INFO)

    if benchmarks is None:
        benchmarks = ALL_BENCHMARKS
    elif isinstance(benchmarks, str):
        benchmarks = [b.strip() for b in benchmarks.split(",")]

    mrc_ks = mrc_ks or [5, 10, 20]
    peer_ks = peer_ks or [20, 100]

    model_name = os.path.basename(model_name_or_path.rstrip("/"))

    logger.info("Loading model: %s", model_name_or_path)
    model = SentenceTransformer(model_name_or_path, trust_remote_code=trust_remote_code)
    model.max_seq_length = 512

    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    for benchmark_name in benchmarks:
        out_path = os.path.join(model_output_dir, f"{benchmark_name}.json")
        if os.path.exists(out_path):
            logger.info("Already evaluated %s, skipping (delete %s to re-run)", benchmark_name, out_path)
            continue

        benchmark_dir = resolve_benchmark_dir(benchmark_name, data_dir=data_dir)

        logger.info("Evaluating: %s on %s", model_name, benchmark_name)

        queries, corpus, relevant_docs = load_eval_benchmark(benchmark_dir)
        logger.info("  Queries: %d, Corpus: %d", len(queries), len(corpus))

        # === Standard IR metrics ===
        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=benchmark_name,
            show_progress_bar=True,
            batch_size=batch_size,
            mrr_at_k=[10, 20, 100],
            ndcg_at_k=[10, 20, 100],
            precision_recall_at_k=[10, 20, 100],
            write_csv=False,
            query_prompt=query_prompt,
            corpus_prompt=doc_prompt,
        )
        ir_results = evaluator(model, output_path="")

        prefix = f"{benchmark_name}_cosine"
        result = {
            "model": model_name,
            "benchmark": benchmark_name,
            "ir": {
                f"{metric}@{k}": ir_results.get(f"{prefix}_{metric}@{k}")
                for metric in ("ndcg", "recall", "mrr")
                for k in (10, 20, 100)
            },
        }
        result["ir"]["map@100"] = ir_results.get(f"{prefix}_map@100")

        logger.info("  nDCG@20:   %.4f", result["ir"]["ndcg@20"])
        logger.info("  Recall@20: %.4f", result["ir"]["recall@20"])

        # === Language-fairness metrics (MRC + PEER) ===
        if compute_fairness:
            query_ids = list(queries.keys())
            query_texts = [queries[qid] for qid in query_ids]
            corpus_ids = list(corpus.keys())
            corpus_texts = [corpus[did] for did in corpus_ids]

            logger.info("  Encoding for fairness metrics...")
            q_embs = model.encode(
                query_texts, batch_size=batch_size, normalize_embeddings=True,
                show_progress_bar=True, prompt=query_prompt,
            )
            c_embs = model.encode(
                corpus_texts, batch_size=batch_size, normalize_embeddings=True,
                show_progress_bar=True, prompt=doc_prompt,
            )
            q_embs = np.asarray(q_embs, dtype=np.float32)
            c_embs = np.asarray(c_embs, dtype=np.float32)
            sim_matrix = q_embs @ c_embs.T

            fairness = {}
            if _has_parallel_queries(queries):
                for k in mrc_ks:
                    fairness.update(compute_mrc(sim_matrix, query_ids, corpus_ids, k=k))
            else:
                logger.info("  Skipping MRC (no parallel queries in %s)", benchmark_name)
            for k in peer_ks:
                fairness.update(compute_peer(sim_matrix, query_ids, corpus_ids, relevant_docs, k=k))

            result["fairness"] = fairness
            for key in sorted(k for k in fairness if k.startswith("MRC@") and k.endswith("_avg")):
                logger.info("  %s: %.2f", key, fairness[key])

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        logger.info("  Saved: %s", out_path)

    logger.info("Done. Results in %s/", model_output_dir)


if __name__ == "__main__":
    fire.Fire(evaluate)
