import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


BENCHMARK = "MIRACLRetrievalHardNegatives.v2"


@pytest.fixture
def evaluator(monkeypatch, tmp_path):
    for name in ("fire", "mteb", "numpy", "sentence_transformers"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))

    model = sys.modules["sentence_transformers"].SentenceTransformer = Mock()
    task = SimpleNamespace(hf_subsets=["zh"])
    get_tasks = sys.modules["mteb"].get_tasks = Mock(return_value=[task])
    scores = {"test": [{"hf_subset": "zh", "ndcg_at_10": 0.75}]}
    evaluate = sys.modules["mteb"].evaluate = Mock(
        return_value=SimpleNamespace(task_results=[SimpleNamespace(scores=scores)])
    )
    sys.modules["numpy"].mean = lambda values: sum(values) / len(values)

    source = Path(__file__).resolve().parents[1] / "evaluate_miracl.py"
    spec = importlib.util.spec_from_file_location("miracl_evaluator_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, model=model, get_tasks=get_tasks, evaluate=evaluate,
                           task=task, output_dir=tmp_path, path=tmp_path / "model" / f"{BENCHMARK}.json")


def write_cache(path, **metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": "model", "benchmark": BENCHMARK,
                                "average": {}, "per_language": {}, **metadata}), encoding="utf-8")


def run(evaluator, languages=None, overwrite=False):
    return evaluator.module.evaluate_miracl("org/model", languages=languages,
                                            output_dir=str(evaluator.output_dir), overwrite=overwrite)


def assert_skipped(evaluator):
    evaluator.model.assert_not_called()
    evaluator.get_tasks.assert_not_called()
    evaluator.evaluate.assert_not_called()


@pytest.mark.parametrize(("saved", "requested"), [
    (["ar"], "ar"),
    (["ar", "zh", "ar"], " zh, ar, zh "),
    (["ar", "zh"], ["zh", "ar"]),
])
def test_matching_cache_skips(evaluator, saved, requested):
    write_cache(evaluator.path, languages=saved)
    run(evaluator, requested)
    assert_skipped(evaluator)


def test_default_selection_matches_cache(evaluator):
    write_cache(evaluator.path, languages=list(reversed(evaluator.module.MIRACL_LANGS)))
    run(evaluator)
    assert_skipped(evaluator)


@pytest.mark.parametrize("requested", ["zh", None])
def test_mismatch_reports_languages_and_preserves_bytes(evaluator, requested):
    write_cache(evaluator.path, languages=["ar"])
    original = evaluator.path.read_bytes()
    with pytest.raises(ValueError, match=r"cached languages.*requested languages.*--overwrite") as error:
        run(evaluator, requested)
    assert "['ar']" in str(error.value)
    assert "'zh'" in str(error.value)
    assert evaluator.path.read_bytes() == original
    assert_skipped(evaluator)


@pytest.mark.parametrize("contents", [b"{", b"{}", b"[]", b'{"languages": "ar"}',
                                       b'{"languages": []}', b'{"languages": [null]}'])
def test_unusable_cache_requires_overwrite(evaluator, contents):
    evaluator.path.parent.mkdir()
    evaluator.path.write_bytes(contents)
    with pytest.raises(ValueError, match=r"MIRACL cache.*--overwrite"):
        run(evaluator, "ar")
    assert evaluator.path.read_bytes() == contents
    assert_skipped(evaluator)


@pytest.mark.parametrize("overwrite", [False, True])
def test_evaluation_writes_existing_schema(evaluator, overwrite):
    if overwrite:
        write_cache(evaluator.path, languages=["ar"])
    run(evaluator, "zh", overwrite=overwrite)
    evaluator.model.assert_called_once_with("org/model", model_kwargs={"attn_implementation": "sdpa"},
                                           trust_remote_code=False)
    evaluator.get_tasks.assert_called_once_with(tasks=[BENCHMARK], languages=["zho"])
    evaluator.evaluate.assert_called_once_with(model=evaluator.model.return_value, tasks=evaluator.task,
                                               encode_kwargs={"batch_size": 512}, cache=None,
                                               overwrite_strategy="always")
    assert json.loads(evaluator.path.read_text(encoding="utf-8")) == {
        "model": "model", "benchmark": BENCHMARK, "languages": ["zh"],
        "average": {"ndcg_at_10": 0.75}, "per_language": {"zh": {"ndcg_at_10": 0.75}},
    }
