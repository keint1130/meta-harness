from pathlib import Path

import pytest

from inner_loop import (
    iter_batches,
    load_candidate,
    load_partial,
    parse_judge_response,
    partial_path,
    save_partial,
)


def test_no_memory_candidate_contract():
    path = Path(__file__).parents[1] / "agents" / "no_memory.py"
    candidate = load_candidate(path)(test_problems=[])
    prompt = candidate.build_prompt("Find x.")
    assert "Find x." in prompt
    assert "\\boxed{}" in prompt


def test_candidate_requires_candidate_class(tmp_path):
    path = tmp_path / "invalid.py"
    path.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(TypeError, match="Candidate"):
        load_candidate(path)


def test_iter_batches_reports_bounded_work_units():
    assert list(iter_batches(["a", "b", "c", "d", "e"], 2)) == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
    ]


def test_iter_batches_rejects_invalid_size():
    with pytest.raises(ValueError, match="at least 1"):
        list(iter_batches(["a"], 0))


def test_partial_result_round_trip(tmp_path):
    output = tmp_path / "result.json"
    identity = {"candidate": "x", "problem_ids": ["p1", "p2"]}
    initial = load_partial(output, identity)
    assert initial["responses"] == []
    initial["responses"] = [["answer"]]
    initial["input_tokens"] = 10
    save_partial(output, initial)
    assert partial_path(output).exists()
    assert load_partial(output, identity)["responses"] == [["answer"]]


def test_partial_result_rejects_different_run(tmp_path):
    output = tmp_path / "result.json"
    save_partial(output, {"candidate": "x", "responses": []})
    with pytest.raises(ValueError, match="mismatch"):
        load_partial(output, {"candidate": "y"})


def test_parse_judge_response_is_fail_closed():
    assert parse_judge_response(r"\boxed{correct}") == 1.0
    assert parse_judge_response(r"\boxed{incorrect}") == 0.0
    assert parse_judge_response("probably correct") == 0.0

