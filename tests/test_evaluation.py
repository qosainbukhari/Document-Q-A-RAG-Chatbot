import json

import pytest

from evaluation.judges import LLMJudge, is_true, parse_json
from evaluation.metrics import mean, recall_at_k, reciprocal_rank

from tests.conftest import RecordingLLM


def test_recall_at_k():
    assert recall_at_k([5, 3, 9], {3}, 1) == 0.0
    assert recall_at_k([5, 3, 9], {3}, 2) == 1.0
    assert recall_at_k([], {3}, 4) == 0.0


def test_reciprocal_rank_uses_first_relevant_hit():
    assert reciprocal_rank([3, 5], {3}) == 1.0
    assert reciprocal_rank([5, 9, 3, 3], {3}) == pytest.approx(1 / 3)
    assert reciprocal_rank([5, 9], {3}) == 0.0


def test_mean_ignores_missing_values():
    assert mean([1.0, None, 0.0]) == 0.5
    assert mean([None]) is None


def test_parse_json_inside_prose_and_fences():
    text = 'Sure!\n```json\n{"relevant": [true, false]}\n```'

    assert parse_json(text) == {"relevant": [True, False]}


def test_is_true_does_not_treat_the_string_false_as_true():
    assert is_true(True) and is_true("true") and is_true("Yes")
    assert not is_true(False) and not is_true("false") and not is_true(None)


def judge_with(reply):
    return LLMJudge(RecordingLLM(replies=[json.dumps(reply)]), sleep_seconds=0)


def test_faithfulness_is_supported_claims_over_all_claims():
    judge = judge_with({"claims": [
        {"claim": "a", "supported": True},
        {"claim": "b", "supported": "false"},
        {"claim": "c", "supported": True},
        {"claim": "d", "supported": True},
    ]})

    assert judge.faithfulness("some answer", ["ctx"]) == 0.75


def test_faithfulness_skips_abstentions():
    judge = judge_with({})

    assert judge.faithfulness("I don't know based on the document.", ["ctx"]) is None


def test_context_precision_ignores_extra_verdicts():
    judge = judge_with({"relevant": [True, False, True, True, True]})

    # only 4 contexts -> the 5th verdict is dropped
    assert judge.context_precision("q", ["a", "b", "c", "d"]) == 0.75


def test_context_recall():
    judge = judge_with({"statements": [
        {"statement": "x", "in_context": True},
        {"statement": "y", "in_context": False},
    ]})

    assert judge.context_recall("reference", ["ctx"]) == 0.5


def test_dataset_is_well_formed():
    with open("evaluation/eval_dataset.json") as file:
        dataset = json.load(file)

    questions = dataset["questions"]

    assert len(questions) >= 25
    for item in questions:
        assert item["question"] and item["reference_answer"]
        assert all(isinstance(page, int) and page >= 1 for page in item["relevant_pages"])
