import json

import pytest

import select_frontier


def point(name, accuracy, cost):
    return {
        "system": name,
        "accuracy": accuracy,
        "solver_tokens_per_problem": cost,
    }


def test_pareto_frontier_removes_dominated_candidates():
    points = [
        point("cheap", 0.7, 100),
        point("dominated", 0.65, 120),
        point("accurate", 0.8, 200),
        point("wasteful", 0.8, 250),
    ]
    assert [row["system"] for row in select_frontier.pareto_frontier(points)] == [
        "accurate",
        "cheap",
    ]


def test_solver_cost_excludes_judge_tokens():
    result = {
        "n_problems": 2,
        "solver_input_tokens": 20,
        "solver_output_tokens": 80,
        "judge_input_tokens": 1000,
        "judge_output_tokens": 1000,
    }
    assert select_frontier.solver_tokens_per_problem(result) == 50


def test_inventory_keeps_baseline_and_run_local_candidates(tmp_path, monkeypatch):
    root = tmp_path / "math_reasoning"
    run = root / "logs" / "run"
    (root / "agents").mkdir(parents=True)
    (root / "agents" / "no_memory.py").write_text("class Candidate: pass\n")
    candidate = run / "agents" / "iter1" / "retrieval.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("class Candidate: pass\n")
    (run / "evolution_summary.jsonl").write_text(
        json.dumps({"iteration": 0, "candidate": "no_memory"})
        + "\n"
        + json.dumps(
            {
                "iteration": 1,
                "candidate": "retrieval",
                "candidate_file": candidate.relative_to(root).as_posix(),
            }
        )
        + "\n"
    )
    monkeypatch.setattr(select_frontier, "ROOT", root)
    inventory = select_frontier.candidate_inventory(run)
    assert [row["name"] for row in inventory] == ["no_memory", "retrieval"]


def test_inventory_rejects_candidate_outside_run(tmp_path, monkeypatch):
    root = tmp_path / "math_reasoning"
    run = root / "logs" / "run"
    outside = root / "agents" / "outside.py"
    outside.parent.mkdir(parents=True)
    outside.write_text("class Candidate: pass\n")
    run.mkdir(parents=True)
    (run / "evolution_summary.jsonl").write_text(
        json.dumps(
            {
                "iteration": 1,
                "candidate": "outside",
                "candidate_file": outside.relative_to(root).as_posix(),
            }
        )
        + "\n"
    )
    monkeypatch.setattr(select_frontier, "ROOT", root)
    with pytest.raises(ValueError, match="outside its run"):
        select_frontier.candidate_inventory(run)


def test_smoke_uses_search_and_does_not_finalize(tmp_path, monkeypatch):
    run = tmp_path / "run"
    candidate = run / "candidate.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("class Candidate: pass\n", encoding="utf-8")
    point_data = {
        "system": "candidate",
        "candidate_path": str(candidate),
        "accuracy": 0.8,
        "solver_tokens_per_problem": 100,
        "solver_input_tokens": 10,
        "solver_output_tokens": 90,
        "empty_responses": 0,
        "n_problems": 20,
    }
    (run / "frontier_val.json").write_text(
        json.dumps(
            {
                "mode": "task-specific",
                "domain": "algebra",
                "best_accuracy": point_data,
                "_pareto": [point_data],
                "systems": [point_data],
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_evaluate(path, split, domain, output, problem_limit=None):
        calls.append((split, domain, problem_limit))
        result = {
            "candidate": "candidate",
            "split": split,
            "domain": domain,
            "accuracy": 1.0,
            "macro_accuracy": 1.0,
            "n_problems": problem_limit,
            "solver_input_tokens": 10,
            "solver_output_tokens": 20,
            "empty_responses": 0,
            "details": [{"problem_id": "exposed-search-row"}],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr(select_frontier, "evaluate", fake_evaluate)
    result = select_frontier.run_smoke(
        run, "task-specific", "algebra", problem_limit=2
    )
    assert calls == [("search", "algebra", 2)]
    assert result["final_test_opened"] is False
    assert not (run / "finalized.json").exists()
