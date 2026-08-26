"""NanoMIRACL Evaluator for mid-training multilingual IR validation.

Uses hotchpotch/NanoMIRACL dataset (BEIR format). Each language split has
~50 queries and ~10,000 corpus documents.

Only evaluates on languages supported by MIMO that are also in NanoMIRACL:
ar, de, en, es, fr, hi, id, ja, ru, zh (10 languages out of 18).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

import numpy as np
from torch import Tensor
from tqdm import tqdm

from sentence_transformers.evaluation.InformationRetrievalEvaluator import InformationRetrievalEvaluator
from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction

logger = logging.getLogger(__name__)

DATASET_ID = "hotchpotch/NanoMIRACL"

# Intersection of MIMO target languages and NanoMIRACL available languages
SUPPORTED_LANGUAGES = ["ar", "de", "en", "es", "fr", "hi", "id", "ja", "ru", "zh"]


class NanoMIRACLEvaluator(SentenceEvaluator):
    """Evaluator for multilingual IR using NanoMIRACL during training.

    Creates one InformationRetrievalEvaluator per target language, then
    aggregates metrics (mean across languages) for the primary score.

    Args:
        languages: Language codes to evaluate. Defaults to MIMO-supported languages.
        dataset_id: HuggingFace dataset ID. Defaults to "hotchpotch/NanoMIRACL".
        ndcg_at_k: k values for nDCG. Defaults to [10].
        mrr_at_k: k values for MRR. Defaults to [10].
        accuracy_at_k: k values for Accuracy. Defaults to [1, 3, 5, 10].
        precision_recall_at_k: k values for Precision/Recall. Defaults to [1, 10].
        map_at_k: k values for MAP. Defaults to [100].
        batch_size: Batch size for encoding. Defaults to 256.
        show_progress_bar: Whether to show progress bar during encoding.
        write_csv: Whether to write CSV results.
        score_functions: Score functions. Defaults to model's similarity function.
        main_score_function: Main score function for primary metric selection.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        dataset_id: str = DATASET_ID,
        ndcg_at_k: list[int] | None = None,
        mrr_at_k: list[int] | None = None,
        accuracy_at_k: list[int] | None = None,
        precision_recall_at_k: list[int] | None = None,
        map_at_k: list[int] | None = None,
        batch_size: int = 256,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        score_functions: dict[str, Callable[[Tensor, Tensor], Tensor]] | None = None,
        main_score_function: str | SimilarityFunction | None = None,
        query_prompt: str | None = None,
        corpus_prompt: str | None = None,
    ):
        super().__init__()
        self.languages = languages or SUPPORTED_LANGUAGES
        self.dataset_id = dataset_id
        self.ndcg_at_k = ndcg_at_k or [10]
        self.mrr_at_k = mrr_at_k or [10]
        self.accuracy_at_k = accuracy_at_k or [1, 3, 5, 10]
        self.precision_recall_at_k = precision_recall_at_k or [1, 10]
        self.map_at_k = map_at_k or [100]
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys())) if score_functions else []
        self.main_score_function = main_score_function
        self.name = "NanoMIRACL_mean"

        ir_evaluator_kwargs = {
            "ndcg_at_k": self.ndcg_at_k,
            "mrr_at_k": self.mrr_at_k,
            "accuracy_at_k": self.accuracy_at_k,
            "precision_recall_at_k": self.precision_recall_at_k,
            "map_at_k": self.map_at_k,
            "batch_size": batch_size,
            "show_progress_bar": show_progress_bar,
            "write_csv": write_csv,
            "score_functions": score_functions,
            "main_score_function": main_score_function,
            "query_prompt": query_prompt,
            "corpus_prompt": corpus_prompt,
        }

        self.evaluators: list[tuple[str, InformationRetrievalEvaluator]] = []
        for lang in tqdm(self.languages, desc="Loading NanoMIRACL datasets", leave=False):
            evaluator = self._load_language(lang, **ir_evaluator_kwargs)
            self.evaluators.append((lang, evaluator))

        logger.info(
            "NanoMIRACL evaluator ready: %d languages (%s)",
            len(self.evaluators),
            ", ".join(self.languages),
        )

        self.csv_file = "NanoMIRACL_evaluation_results.csv"
        self.csv_headers = ["epoch", "steps"]

    def _load_language(
        self, lang: str, **ir_evaluator_kwargs
    ) -> InformationRetrievalEvaluator:
        """Load NanoMIRACL data for a single language and create IR evaluator."""
        from datasets import load_dataset

        corpus_ds = load_dataset(self.dataset_id, "corpus", split=lang)
        queries_ds = load_dataset(self.dataset_id, "queries", split=lang)
        qrels_ds = load_dataset(self.dataset_id, "qrels", split=lang)

        corpus_dict = {row["_id"]: row["text"] for row in corpus_ds if row["text"]}
        queries_dict = {row["_id"]: row["text"] for row in queries_ds if row["text"]}

        # Binary relevance: each (query-id, corpus-id) pair is relevant
        relevant_docs: dict[str, set[str]] = {}
        for row in qrels_ds:
            qid = row["query-id"]
            cid = row["corpus-id"]
            if qid not in relevant_docs:
                relevant_docs[qid] = set()
            if isinstance(cid, list):
                relevant_docs[qid].update(cid)
            else:
                relevant_docs[qid].add(cid)

        logger.debug(
            "NanoMIRACL-%s: %d queries, %d corpus, %d qrels",
            lang, len(queries_dict), len(corpus_dict), len(relevant_docs),
        )

        # Use hyphen in name to avoid underscore conflicts in metric key parsing
        return InformationRetrievalEvaluator(
            queries=queries_dict,
            corpus=corpus_dict,
            relevant_docs=relevant_docs,
            name=f"NanoMIRACL-{lang}",
            **ir_evaluator_kwargs,
        )

    def __call__(
        self,
        model,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
        *args,
        **kwargs,
    ) -> dict[str, float]:
        per_metric_results: dict[str, list[float]] = {}
        per_dataset_results: dict[str, float] = {}

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        logger.info("NanoMIRACL Evaluation on %d languages%s:", len(self.evaluators), out_txt)

        # Initialize score functions from model if not set
        if self.score_functions is None:
            self.score_functions = {model.similarity_fn_name: model.similarity}
            self.score_function_names = [model.similarity_fn_name]

        num_underscores_in_name = self.name.count("_")

        for lang, evaluator in self.evaluators:
            logger.info("Evaluating NanoMIRACL-%s", lang)
            evaluation = evaluator(model, output_path, epoch, steps)
            for full_key, metric_value in evaluation.items():
                splits = full_key.split("_", maxsplit=num_underscores_in_name)
                metric = splits[-1]
                if metric not in per_metric_results:
                    per_metric_results[metric] = []
                per_dataset_results[full_key] = metric_value
                per_metric_results[metric].append(metric_value)

        # Aggregate across languages
        agg_results: dict[str, float] = {
            metric: float(np.mean(values))
            for metric, values in per_metric_results.items()
        }

        # Set primary metric on first call
        if not self.primary_metric:
            if self.main_score_function is None:
                score_function = max(
                    [
                        (name, agg_results.get(f"{name}_ndcg@{max(self.ndcg_at_k)}", 0))
                        for name in self.score_function_names
                    ],
                    key=lambda x: x[1],
                )[0]
                self.primary_metric = f"{score_function}_ndcg@{max(self.ndcg_at_k)}"
            else:
                self.primary_metric = (
                    f"{self.main_score_function}_ndcg@{max(self.ndcg_at_k)}"
                )

        # Log per-language nDCG@10
        logger.info("Per-language nDCG@10:")
        for lang, evaluator in self.evaluators:
            for name in self.score_function_names:
                key = f"NanoMIRACL-{lang}_{name}_ndcg@{max(self.ndcg_at_k)}"
                val = per_dataset_results.get(key, 0)
                logger.info("  %s: %.4f", lang, val)

        # Log aggregated results
        for name in self.score_function_names:
            logger.info("Aggregated (%d languages) for %s:", len(self.evaluators), name)
            for k in self.ndcg_at_k:
                logger.info("  NDCG@%d: %.4f", k, agg_results.get(f"{name}_ndcg@{k}", 0))
            for k in self.mrr_at_k:
                logger.info("  MRR@%d: %.4f", k, agg_results.get(f"{name}_mrr@{k}", 0))

        # Prefix with evaluator name and store in model card
        agg_results = self.prefix_name_to_metrics(agg_results, self.name)
        self.store_metrics_in_model_card_data(model, agg_results, epoch, steps)

        per_dataset_results.update(agg_results)
        return per_dataset_results
