"""Evaluate models on MIRACL retrieval benchmark via MTEB.

Evaluates on the subset of MIRACL languages that overlap with our 14 training languages:
ar, de, en, es, fr, hi, id, ja, ru, zh (10 out of 14; dt, it, pt, vi not in MIRACL).

Usage:
	python evaluate_miracl.py --model_name_or_path /path/to/model
	python evaluate_miracl.py --model_name_or_path /path/to/model --languages ar,de,zh
"""

import json
import logging
import os

import fire
import mteb
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Our training languages that exist in MIRACL
MIRACL_LANGS = ["ar", "de", "en", "es", "fr", "hi", "id", "ja", "ru", "zh"]

LANG_TO_ISO = {
	"ar": "ara", "de": "deu", "en": "eng", "es": "spa", "fr": "fra",
	"hi": "hin", "id": "ind", "ja": "jpn", "ru": "rus", "zh": "zho",
}


def evaluate_miracl(
	model_name_or_path: str,
	languages: list[str] | str | None = None,
	batch_size: int = 512,
	output_dir: str = "results",
	trust_remote_code: bool = False,
	overwrite: bool = False,
	query_prompt: str | None = None,
	doc_prompt: str | None = None,
):
	logging.basicConfig(level=logging.INFO)

	if languages is None:
		languages = MIRACL_LANGS
	elif isinstance(languages, str):
		languages = [l.strip() for l in languages.split(",")]

	model_name = os.path.basename(model_name_or_path.rstrip("/"))
	if not model_name:
		model_name = model_name_or_path.rstrip("/").split("/")[-1]

	model_output_dir = os.path.join(output_dir, model_name)
	out_path = os.path.join(model_output_dir, "MIRACLRetrievalHardNegatives.v2.json")

	if os.path.exists(out_path) and not overwrite:
		logger.info("Already evaluated MIRACL, skipping (delete %s or use --overwrite to re-run)", out_path)
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

	# Get MIRACL task filtered to our languages
	iso_langs = [LANG_TO_ISO[l] for l in languages if l in LANG_TO_ISO]
	tasks = mteb.get_tasks(tasks=["MIRACLRetrievalHardNegatives.v2"], languages=iso_langs)
	task = tasks[0]

	logger.info("Evaluating MIRACL on languages: %s (subsets: %s)", languages, task.hf_subsets)

	encode_kwargs = {"batch_size": batch_size}

	model_result = mteb.evaluate(
		model=model,
		tasks=task,
		encode_kwargs=encode_kwargs,
		cache=None,
		overwrite_strategy="always",
	)

	# Parse results from TaskResult.scores
	task_result = model_result.task_results[0]

	per_lang = {}
	for split, subset_scores_list in task_result.scores.items():
		for subset_scores in subset_scores_list:
			lang = subset_scores.get("hf_subset", "unknown")
			per_lang[lang] = {
				k: subset_scores.get(k)
				for k in [
					"main_score", "ndcg_at_1", "ndcg_at_3", "ndcg_at_5",
					"ndcg_at_10", "ndcg_at_20", "ndcg_at_100", "ndcg_at_1000",
					"recall_at_1", "recall_at_3", "recall_at_5",
					"recall_at_10", "recall_at_20", "recall_at_100", "recall_at_1000",
					"map_at_1", "map_at_3", "map_at_5",
					"map_at_10", "map_at_20", "map_at_100", "map_at_1000",
					"mrr_at_1", "mrr_at_3", "mrr_at_5",
					"mrr_at_10", "mrr_at_20", "mrr_at_100", "mrr_at_1000",
				]
				if subset_scores.get(k) is not None
			}

	# Compute averages
	avg = {}
	if per_lang:
		all_keys = set()
		for v in per_lang.values():
			all_keys.update(v.keys())
		for mk in all_keys:
			vals = [v[mk] for v in per_lang.values() if mk in v]
			avg[mk] = float(np.mean(vals)) if vals else None

	result = {
		"model": model_name,
		"benchmark": "MIRACLRetrievalHardNegatives.v2",
		"languages": languages,
		"average": avg,
		"per_language": per_lang,
	}

	# Print summary
	logger.info("")
	logger.info("=" * 70)
	logger.info("  MIRACL Results: %s", model_name)
	logger.info("=" * 70)
	logger.info("  %-5s  %10s  %10s  %10s  %10s", "Lang", "nDCG@10", "nDCG@20", "Recall@100", "MRR@10")
	logger.info("  " + "-" * 50)
	for lang in sorted(per_lang.keys()):
		m = per_lang[lang]
		logger.info("  %-5s  %10.4f  %10.4f  %10.4f  %10.4f",
					 lang,
					 m.get("ndcg_at_10", 0),
					 m.get("ndcg_at_20", 0),
					 m.get("recall_at_100", 0),
					 m.get("mrr_at_10", 0))
	logger.info("  " + "-" * 50)
	logger.info("  %-5s  %10.4f  %10.4f  %10.4f  %10.4f",
				"AVG",
				avg.get("ndcg_at_10", 0),
				avg.get("ndcg_at_20", 0),
				avg.get("recall_at_100", 0),
				avg.get("mrr_at_10", 0))
	logger.info("=" * 70)

	# Save
	os.makedirs(model_output_dir, exist_ok=True)
	with open(out_path, "w") as f:
		json.dump(result, f, indent=2, ensure_ascii=False, default=str)
	logger.info("Saved: %s", out_path)


if __name__ == "__main__":
	fire.Fire(evaluate_miracl)
