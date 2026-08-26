"""NeuCLIR 2022/2023 MLIR evaluation — true "English query × mixed-corpus" setting.

Per the official NeuCLIRBench spec (https://huggingface.co/datasets/neuclir/bench),
the multilingual retrieval task uses English queries and a multilingual mixed
corpus. We follow that recipe with two adaptations:

  1. fas is excluded from the corpus (not in our 14-lang training set, matching
     the paper's NeuCLIR setup).
  2. NeuCLIRBench's `qrels.mlir` only covers the 2023 query IDs (qid≥200);
     2022 has no `mlir` qrels. To keep both tracks under the same evaluation
     setting, we use the *union of per-language CLIR qrels*
     (`qrels.rus ∪ qrels.zho`) for both tracks. With binary relevance, this is
     equivalent to NeuCLIRBench's mlir qrels for the 2023 track and gives a
     directly comparable mixed-corpus evaluation for 2022.

Setup:
  Documents : pooled mteb HardNegatives corpora (rus + zho)
  Queries   : English (`eng` split of neuclir/bench:queries), looked up by qid
  Qrels     : (qrels.rus ∪ qrels.zho) of neuclir/bench, filtered to the track's
              query IDs and the pooled corpus
  Metric    : nDCG@20 (paper main metric)

Note: mteb's per-language NeuCLIR subsets translate the queries into each
corpus language and evaluate each language separately — that is monolingual IR
macro-averaged across languages, not MLIR. This script instead swaps in the
official English queries from neuclir/bench, pools the corpora, and uses
cross-lingual qrels — the paper's true MLIR setting.

Track separation: NeuCLIR 2022 and 2023 query IDs are disjoint, so we run two
independent MLIR retrievals (one per track) and write one JSON per track,
mirroring the paper's two columns (NeuCLIR'22 / NeuCLIR'23).

Output:
    {output_dir}/{model_name}/{NeuCLIR2022,NeuCLIR2023}RetrievalHardNegatives.json

Usage:
    python evaluate_neuclir.py --model_name_or_path /path/to/model
    python evaluate_neuclir.py --model_name_or_path /path/to/model --batch_size 64
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict

import fire
import mteb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

logger = logging.getLogger(__name__)

NEUCLIR_LANGS = ["ru", "zh"]
LANG_TO_ISO = {"ru": "rus", "zh": "zho"}

TASK_NAMES = [
    "NeuCLIR2022RetrievalHardNegatives",
    "NeuCLIR2023RetrievalHardNegatives",
]

# Cache the neuclir/bench loads (eng queries + per-lang qrels) once per process.
_BENCH_ENG_QUERIES: dict[str, str] | None = None
_BENCH_QRELS_ROWS: list | None = None  # union of qrels.rus + qrels.zho


def _load_bench_once():
    global _BENCH_ENG_QUERIES, _BENCH_QRELS_ROWS
    if _BENCH_ENG_QUERIES is None:
        logger.info("Loading neuclir/bench: queries (eng) + qrels (rus ∪ zho) ...")
        eng = load_dataset("neuclir/bench", "queries")["eng"]
        _BENCH_ENG_QUERIES = {str(r["id"]): r["query"] for r in eng}
        qr = load_dataset("neuclir/bench", "qrels")
        # Cross-lingual qrels for both ru and zh corpora. Doc_ids are unique
        # across languages so a flat list works directly.
        _BENCH_QRELS_ROWS = list(qr["rus"]) + list(qr["zho"])
        logger.info(
            "Bench loaded: %d English queries, %d cross-lingual qrel rows (rus+zho)",
            len(_BENCH_ENG_QUERIES), len(_BENCH_QRELS_ROWS),
        )
    return _BENCH_ENG_QUERIES, _BENCH_QRELS_ROWS


def _build_track_data(task_name: str, iso_subsets: list[str]):
    """For one NeuCLIR track, build (eng_queries, mixed_corpus, mlir_qrels).

    Doc IDs are prefixed with their subset code ("rus::", "zho::") to avoid
    any collision in the pooled corpus, and qrels are matched using the
    `ignore` field of each qrel row (which holds the doc's language).
    """
    tasks = mteb.get_tasks(tasks=[task_name], languages=iso_subsets)
    t = tasks[0]
    t.load_data()

    # Pool the per-language hard-negatives corpora into one mixed corpus
    # with subset-prefixed doc IDs.
    mixed_corpus: dict[str, str] = {}
    track_query_ids: set[str] = set()
    per_subset_sizes: dict[str, dict[str, int]] = {}

    for sub in iso_subsets:
        if sub not in t.dataset:
            logger.warning("Subset %s missing from %s", sub, task_name)
            continue
        d = t.dataset[sub]["test"]
        n_docs_before = len(mixed_corpus)
        for row in d["corpus"]:
            doc_id = row["id"]
            title = (row.get("title") or "").strip()
            text = (row.get("text") or "").strip()
            content = (title + "\n" + text).strip() if title else text
            mixed_corpus[f"{sub}::{doc_id}"] = content
        n_q = 0
        for row in d["queries"]:
            track_query_ids.add(str(row["id"]))
            n_q += 1
        per_subset_sizes[sub] = {
            "corpus_added": len(mixed_corpus) - n_docs_before,
            "queries": n_q,
        }

    # English queries from neuclir/bench, filtered to this track's IDs.
    eng_all, qrel_rows = _load_bench_once()
    queries: dict[str, str] = {qid: eng_all[qid] for qid in track_query_ids if qid in eng_all}
    missing_eng = sorted(track_query_ids - set(queries.keys()))
    if missing_eng:
        logger.warning(
            "%d query IDs in %s have no English text in neuclir/bench (will be dropped): %s",
            len(missing_eng), task_name, missing_eng,
        )

    # MLIR qrels = qrels.rus ∪ qrels.zho, filtered by track query IDs and pooled
    # corpus. Each qrel row's `ignore` field is the doc's language ("rus"/"zho"),
    # used to look up the prefixed doc id.
    relevant: dict[str, set[str]] = defaultdict(set)
    for row in qrel_rows:
        qid = str(row["id"])
        if qid not in queries:
            continue
        lang = row.get("ignore")  # "rus" or "zho"
        if lang not in iso_subsets:
            continue
        prefixed_doc_id = f"{lang}::{row['docid']}"
        if prefixed_doc_id not in mixed_corpus:
            continue
        if int(row["relevance"]) > 0:
            relevant[qid].add(prefixed_doc_id)

    # Drop queries without any positives in the pooled corpus.
    # (InformationRetrievalEvaluator can deflate nDCG to 0 for such queries on
    # some sentence-transformers versions; explicit drop is safer.)
    queries = {q: t for q, t in queries.items() if q in relevant}

    stats = {
        "per_subset_sizes": per_subset_sizes,
        "mixed_corpus_docs": len(mixed_corpus),
        "track_query_count_in_mteb": len(track_query_ids),
        "english_queries_matched_with_text": len(queries) + len(missing_eng),
        "queries_evaluated": len(queries),  # after dropping no-positive queries
    }
    return queries, mixed_corpus, dict(relevant), stats


def evaluate_neuclir_mlir(
    model_name_or_path: str,
    languages: list[str] | str | None = None,
    batch_size: int = 32,
    output_dir: str = "results_neuclir",
    trust_remote_code: bool = False,
    overwrite: bool = False,
    query_prompt: str | None = None,
    doc_prompt: str | None = None,
):
    logging.basicConfig(level=logging.INFO)

    if languages is None:
        languages = NEUCLIR_LANGS
    elif isinstance(languages, str):
        languages = [s.strip() for s in languages.split(",")]
    iso_subsets = [LANG_TO_ISO[l] for l in languages if l in LANG_TO_ISO]

    model_name = os.path.basename(model_name_or_path.rstrip("/")) \
                 or model_name_or_path.rstrip("/").split("/")[-1]
    model_output_dir = os.path.join(output_dir, model_name)

    tasks_to_run = []
    for tn in TASK_NAMES:
        out = os.path.join(model_output_dir, f"{tn}.json")
        if os.path.exists(out) and not overwrite:
            logger.info("Already evaluated %s → skip", out)
        else:
            tasks_to_run.append(tn)

    if not tasks_to_run:
        logger.info("All MLIR-mixed-corpus NeuCLIR done for %s", model_name)
        return

    logger.info("Loading model: %s", model_name_or_path)
    model = SentenceTransformer(
        model_name_or_path,
        model_kwargs={"attn_implementation": "sdpa"},
        trust_remote_code=trust_remote_code,
    )
    model.max_seq_length = 512

    if query_prompt:
        model.prompts = {"query": query_prompt}
    if doc_prompt:
        model.prompts = {**getattr(model, "prompts", {}), "passage": doc_prompt}

    for tn in tasks_to_run:
        logger.info("=" * 70)
        logger.info("  %s — true MLIR (English query × mixed ru∪zh corpus)", tn)
        logger.info("=" * 70)

        queries, corpus, relevant, stats = _build_track_data(tn, iso_subsets)
        logger.info("Stats: %s", stats)

        if not queries or not relevant:
            logger.warning("No queries or qrels for %s — skipping", tn)
            continue

        eval_name = f"{tn}_mlir"
        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant,
            name=eval_name,
            show_progress_bar=True,
            batch_size=batch_size,
            mrr_at_k=[10, 20, 100],
            ndcg_at_k=[10, 20, 100],
            precision_recall_at_k=[10, 20, 100],
            write_csv=False,
            query_prompt=query_prompt,
            corpus_prompt=doc_prompt,
        )
        ir = evaluator(model, output_path=None)
        prefix = f"{eval_name}_cosine"

        # Output schema mirrors evaluate_neuclir.py (`average` / `per_language`)
        # so downstream tooling reads
        # average.ndcg_at_20 uniformly. Single mixed-corpus pass → per_language
        # is intentionally empty.
        result = {
            "model": model_name,
            "benchmark": tn,
            "setting": "mixed_corpus_MLIR_NeuCLIRBench",
            "languages": languages,
            "n_corpus_docs": len(corpus),
            "n_queries": len(queries),
            "n_queries_with_qrels": len(relevant),
            "stats": stats,
            "average": {
                "ndcg_at_10": ir.get(f"{prefix}_ndcg@10"),
                "ndcg_at_20": ir.get(f"{prefix}_ndcg@20"),
                "ndcg_at_100": ir.get(f"{prefix}_ndcg@100"),
                "recall_at_10": ir.get(f"{prefix}_recall@10"),
                "recall_at_20": ir.get(f"{prefix}_recall@20"),
                "recall_at_100": ir.get(f"{prefix}_recall@100"),
                "mrr_at_10": ir.get(f"{prefix}_mrr@10"),
                "mrr_at_20": ir.get(f"{prefix}_mrr@20"),
                "mrr_at_100": ir.get(f"{prefix}_mrr@100"),
                "map_at_100": ir.get(f"{prefix}_map@100"),
            },
            "per_language": {},
        }

        def _safe(x):
            return x if isinstance(x, (int, float)) else float("nan")

        n20 = _safe(result["average"]["ndcg_at_20"])
        r20 = _safe(result["average"]["recall_at_20"])
        m20 = _safe(result["average"]["mrr_at_20"])
        logger.info("[%s MLIR] nDCG@20=%.4f Recall@20=%.4f MRR@20=%.4f", tn, n20, r20, m20)

        os.makedirs(model_output_dir, exist_ok=True)
        out_path = os.path.join(model_output_dir, f"{tn}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    fire.Fire(evaluate_neuclir_mlir)
