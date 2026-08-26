"""Collator dispatch tests using a stub tokenizer (no downloads)."""

import torch

from mimo.data import MIMODataCollator


class StubTokenizer:
    def __call__(self, texts, **kwargs):
        max_len = max(len(t) for t in texts)
        ids = torch.zeros(len(texts), max_len, dtype=torch.long)
        for i, t in enumerate(texts):
            ids[i, : len(t)] = torch.tensor([ord(c) % 100 for c in t])
        return {"input_ids": ids, "attention_mask": (ids != 0).long()}


def test_stage1_collate():
    collator = MIMODataCollator(student_tokenizer=StubTokenizer(), teacher_tokenizer=StubTokenizer(), stage=1)
    batch = collator([{"anchor": "hallo welt", "positive": "hello world", "lang": "de"}])
    assert len(batch) == 2


def test_stage2_mimo_collate():
    collator = MIMODataCollator(
        student_tokenizer=StubTokenizer(),
        teacher_tokenizer=StubTokenizer(),
        stage=2,
        teacher_query_prompt="Query: ",
    )
    batch = collator([
        {"anchor": "q_de", "positive": "p_fr", "anchor_en": "q_en", "positive_en": "p_en"}
    ])
    assert len(batch) == 4


def test_stage2_infonce_collate():
    collator = MIMODataCollator(student_tokenizer=StubTokenizer(), stage=2)
    batch = collator([{"anchor": "q", "positive": "p", "lang": "de"}])
    assert len(batch) == 2


def test_stage2_lakda_collate():
    collator = MIMODataCollator(student_tokenizer=StubTokenizer(), stage=2)
    batch = collator([{"anchor": "q_a", "positive": "p", "anchor_b": "q_b"}])
    assert len(batch) == 3
