import json
from pathlib import Path

import pytest

from inner_loop import (
    iter_batches,
    judge_prompt,
    load_candidate,
    load_partial,
    load_reused_responses,
    parse_judge_response,
    partial_responses_by_id,
    partial_path,
    save_partial,
)
from math_harness import SolvedMathMemory, candidate_observations, instantiate_candidate


def test_no_memory_candidate_contract():
    path = Path(__file__).parents[1] / "agents" / "no_memory.py"
    candidate = instantiate_candidate(load_candidate(path), [], domain=None, memory=[])
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
    assert initial["responses_by_problem_id"] == {}
    initial["responses_by_problem_id"] = {"p1": ["answer"]}
    initial["input_tokens"] = 10
    save_partial(output, initial)
    assert partial_path(output).exists()
    assert load_partial(output, identity)["responses_by_problem_id"] == {
        "p1": ["answer"]
    }


def test_partial_result_rejects_different_run(tmp_path):
    output = tmp_path / "result.json"
    save_partial(output, {"candidate": "x", "responses": []})
    with pytest.raises(ValueError, match="mismatch"):
        load_partial(output, {"candidate": "y"})


def test_old_prefix_partial_is_migrated_to_problem_ids():
    partial = {"responses": [["a"], ["b"]]}
    assert partial_responses_by_id(partial, ["p1", "p2", "p3"]) == {
        "p1": ["a"],
        "p2": ["b"],
    }


def test_reuse_result_excludes_regenerated_problem(tmp_path):
    source = tmp_path / "result.json"
    source.write_text(
        json.dumps(
            {
                "solver_input_tokens": 10,
                "solver_output_tokens": 20,
                "details": [
                    {"problem_id": "p1", "response": "old one"},
                    {"problem_id": "p2", "response": "old two"},
                ],
            }
        ),
        encoding="utf-8",
    )
    reused, input_tokens, output_tokens = load_reused_responses(
        source, ["p1", "p2"], {"p2"}
    )
    assert reused == {"p1": "old one"}
    assert (input_tokens, output_tokens) == (10, 20)


def test_parse_judge_response_is_fail_closed():
    assert parse_judge_response(r"\boxed{correct}") == 1.0
    assert parse_judge_response(r"\boxed{incorrect}") == 0.0
    assert parse_judge_response("probably correct") == 0.0


def test_judge_prompt_rejects_incomplete_answer_forms():
    prompt = judge_prompt("Find the value.", "42", "Use this recurrence.")
    assert "recurrence" in prompt
    assert "explicit value" in prompt
    assert "infinite family" in prompt


def test_candidate_observations_remove_all_evaluator_metadata():
    rows = [
        {
            "problem_id": "secret-id",
            "domain": "geometry",
            "subcategory": "circles",
            "groundtruth": "42",
            "problem": "Find x.",
        }
    ]
    assert candidate_observations(rows) == [{"problem": "Find x."}]


def test_global_and_task_specific_candidate_contexts_are_distinct():
    class NewCandidate:
        def __init__(self, memory, task_context=None):
            self.memory = memory
            self.task_context = task_context

    memory = []
    global_candidate = instantiate_candidate(NewCandidate, [], domain=None, memory=memory)
    specific_candidate = instantiate_candidate(
        NewCandidate, [], domain="algebra", memory=memory
    )
    assert global_candidate.task_context == {}
    assert specific_candidate.task_context == {"domain": "algebra"}


def test_memory_rows_are_allowlisted():
    row = {
        "problem": "p",
        "solution": "s",
        "answer": "a",
        "source": "external",
        "domain": "hidden",
        "subcategory": "hidden",
        "groundtruth": "hidden",
    }
    assert SolvedMathMemory._sanitize(row) == {
        "problem": "p",
        "solution": "s",
        "answer": "a",
        "source": "external",
    }


def test_memory_revision_is_recorded_without_loading():
    memory = SolvedMathMemory("example/repository", revision="fixed-commit")
    assert memory.revision == "fixed-commit"

