import json
from pathlib import Path

import pytest

import optimize


def baseline(domain="algebra"):
    return {
        "candidate": "no_memory",
        "split": "search",
        "domain": domain,
        "accuracy": 0.5,
        "macro_accuracy": 0.5,
        "model": "openai/gpt-6-luna",
        "details": [{"problem_id": "p1"}],
    }


def baseline_config():
    return {
        "experiment": {"model": "openai/gpt-6-luna"},
        "judge": {},
        "memory": {},
        "splits": {"search": 1},
        "domains": ["algebra", "geometry"],
        "baseline": {"allow_legacy_without_evaluation_config": True},
    }


def test_text_classification_directory_does_not_pollute_import_path():
    assert str(optimize.TEXT_EXAMPLE) not in optimize.sys.path


def test_seed_baseline_copies_and_validates(tmp_path):
    source = tmp_path / "source.json"
    destination = tmp_path / "run" / "result.json"
    source.write_text(json.dumps(baseline()), encoding="utf-8")

    sources = [{"domain": "algebra", "path": str(source), "sha256": "hash"}]
    result = optimize.seed_baseline(sources, destination, "algebra", baseline_config())

    assert result["accuracy"] == 0.5
    assert json.loads(destination.read_text(encoding="utf-8")) == result
    assert result["baseline_sources"][0]["compatibility"] == "legacy_partial"


def test_seed_baseline_rejects_wrong_domain(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(baseline("geometry")), encoding="utf-8")

    with pytest.raises(ValueError, match="domain mismatch"):
        optimize.seed_baseline(
            [{"domain": "algebra", "path": str(source), "sha256": "hash"}],
            tmp_path / "result.json",
            "algebra",
            baseline_config(),
        )


def test_candidate_path_is_confined_to_run_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(optimize, "ROOT", tmp_path)
    candidate_dir = tmp_path / "logs" / "run" / "agents" / "iter1"
    candidate_dir.mkdir(parents=True)
    candidate = candidate_dir / "candidate.py"
    candidate.write_text("class Candidate: pass\n", encoding="utf-8")

    relative = candidate.relative_to(tmp_path).as_posix()
    assert optimize.candidate_path({"file": relative}, candidate_dir) == candidate

    outside = tmp_path / "agents" / "shared.py"
    outside.parent.mkdir()
    outside.write_text("class Candidate: pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be inside"):
        optimize.candidate_path(
            {"file": outside.relative_to(tmp_path).as_posix()}, candidate_dir
        )


def test_candidate_validation_does_not_import_math_runtime(tmp_path):
    candidate = tmp_path / "retrieval.py"
    candidate.write_text(
        "from math_retriever import MathBM25\n\nclass Candidate:\n    pass\n",
        encoding="utf-8",
    )
    optimize.validate_candidate_source(candidate)


def test_candidate_validation_requires_candidate_class(tmp_path):
    candidate = tmp_path / "invalid.py"
    candidate.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(TypeError, match="Candidate"):
        optimize.validate_candidate_source(candidate)


def test_existing_run_requires_resume(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "evolution_summary.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--resume"):
        optimize.ensure_new_or_resumable(run_dir, resume=False)
    optimize.ensure_new_or_resumable(run_dir, resume=True)


def test_run_manifest_records_config_code_and_invocations(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = {
        "experiment": {"model": "openai/gpt-6-luna"},
        "paths": {"math_eval_repo": "math-eval"},
        "memory": {"revision": "fixed-revision"},
    }
    monkeypatch.setenv("MOHARNESS_PROPOSER_AUTH_METHOD", "chatgpt")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-recorded")
    monkeypatch.setattr(optimize, "math_repo_path", lambda unused: tmp_path / "math")
    monkeypatch.setattr(
        optimize,
        "git_state",
        lambda path: {"commit": "abc123", "dirty": False},
    )

    optimize.initialize_run_manifest(
        run_dir,
        config,
        run_name="main-run",
        mode="task-specific",
        domain="combinatorics",
        candidates_per_iteration=1,
        requested_iterations=2,
        baseline_sources=[{"domain": "combinatorics", "sha256": "base"}],
    )

    manifest_text = (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["config"] == config
    assert manifest["authentication"]["proposer"] == "chatgpt"
    assert manifest["authentication"]["solver_and_judge"] == "OPENAI_API_KEY"
    assert manifest["code"]["math_frontier_eval"]["commit"] == "abc123"
    assert "must-not-be-recorded" not in manifest_text
    invocation = json.loads((run_dir / "run_invocations.jsonl").read_text())
    assert invocation["requested_iterations"] == 2


def test_run_manifest_rejects_changed_invariant_config(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(optimize, "math_repo_path", lambda unused: tmp_path / "math")
    monkeypatch.setattr(optimize, "git_state", lambda path: {})
    arguments = {
        "run_name": "run",
        "mode": "global",
        "domain": None,
        "candidates_per_iteration": 1,
        "requested_iterations": 1,
        "baseline_sources": [],
    }
    optimize.initialize_run_manifest(run_dir, {"version": 1}, **arguments)
    with pytest.raises(ValueError, match="manifest mismatch"):
        optimize.initialize_run_manifest(run_dir, {"version": 2}, **arguments)


def test_search_can_use_dev_but_cannot_continue_after_test(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "frontier_val.json").write_text("{}\n", encoding="utf-8")
    optimize.ensure_search_mutable(run_dir)
    (run_dir / "finalized.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen"):
        optimize.ensure_search_mutable(run_dir)


def test_completed_names_is_iteration_scoped():
    records = [
        {"iteration": 1, "candidate": "a"},
        {"iteration": 2, "candidate": "b"},
    ]
    assert optimize.completed_names(records, 1) == {"a"}


def test_result_summary_preserves_cost_and_empty_metrics():
    result = {
        **baseline(),
        "solver_input_tokens": 10,
        "solver_output_tokens": 20,
        "judge_input_tokens": 3,
        "judge_output_tokens": 4,
        "empty_responses": 1,
    }
    summary = optimize.result_summary(result, 0)
    assert summary["solver_output_tokens"] == 20
    assert summary["judge_output_tokens"] == 4
    assert summary["empty_responses"] == 1
    assert summary["search_accuracy"] == 0.5


def test_global_proposer_view_removes_domain_labels_and_ids(tmp_path, monkeypatch):
    root = tmp_path / "math_reasoning"
    (root / "agents").mkdir(parents=True)
    (root / "agents" / "no_memory.py").write_text(
        "class Candidate: pass\n", encoding="utf-8"
    )
    (root / "math_harness.py").write_text("# contract\n", encoding="utf-8")
    run_dir = root / "logs" / "run"
    results = run_dir / "results"
    results.mkdir(parents=True)
    (results / "iter0_search.json").write_text(
        json.dumps(
            {
                "candidate": "no_memory",
                "details": [
                    {
                        "problem_id": "secret-id",
                        "domain": "geometry",
                        "subcategory": "circles",
                        "problem": "Find x.",
                        "prompt": "Find x.",
                        "response": "42",
                        "groundtruth": "42",
                        "score": 1.0,
                        "deterministic_score": 1.0,
                        "judge_score": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(optimize, "ROOT", root)
    view, _ = optimize.prepare_proposer_view(
        run_dir,
        run_dir / "agents" / "iter1",
        [{"candidate": "no_memory", "dev_accuracy": 0.75}],
        mode="global",
        domain=None,
    )
    trace = json.loads((view / "anonymous_traces.jsonl").read_text())
    assert trace["problem"] == "Find x."
    assert "problem_id" not in trace
    assert "domain" not in trace
    assert "subcategory" not in trace
    state = json.loads((view / "optimizer_state.json").read_text())
    assert state["visible_task_context"] == {}
    assert state["history"][0]["dev_accuracy"] == 0.75


def test_dev_feedback_updates_summary_without_problem_traces(tmp_path, monkeypatch):
    root = tmp_path / "math_reasoning"
    run = root / "logs" / "run"
    (root / "agents").mkdir(parents=True)
    (root / "agents" / "no_memory.py").write_text(
        "class Candidate: pass\n", encoding="utf-8"
    )
    run.mkdir(parents=True)
    summary = run / "evolution_summary.jsonl"
    summary.write_text(
        json.dumps({"iteration": 0, "candidate": "no_memory"}) + "\n",
        encoding="utf-8",
    )

    def fake_evaluate(candidate, split, domain, output):
        result = {
            "candidate": "no_memory",
            "split": "dev",
            "domain": "algebra",
            "accuracy": 0.75,
            "macro_accuracy": 0.75,
            "n_problems": 20,
            "solver_input_tokens": 100,
            "solver_output_tokens": 900,
            "empty_responses": 1,
            "details": [{"problem": "must stay private", "groundtruth": "secret"}],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr(optimize, "ROOT", root)
    monkeypatch.setattr(optimize, "evaluate", fake_evaluate)
    records = optimize.update_dev_frontier(run, summary, "task-specific", "algebra")
    assert records[0]["dev_accuracy"] == 0.75
    assert "details" not in records[0]
    frontier = json.loads((run / "frontier_val.json").read_text())
    assert frontier["feedback_policy"] == "aggregate_only"
