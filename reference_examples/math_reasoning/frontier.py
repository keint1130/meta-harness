"""Shared accuracy/cost frontier helpers."""


def objective_accuracy(result: dict, mode: str) -> float:
    return float(result["macro_accuracy"] if mode == "global" else result["accuracy"])


def solver_tokens_per_problem(result: dict) -> float:
    n = int(result.get("n_problems", 0))
    if n <= 0:
        raise ValueError("Result has no evaluated problems")
    return (
        int(result.get("solver_input_tokens", 0))
        + int(result.get("solver_output_tokens", 0))
    ) / n


def result_point(result: dict, mode: str, candidate_path) -> dict:
    return {
        "system": result["candidate"],
        "candidate_path": str(candidate_path.resolve()),
        "accuracy": objective_accuracy(result, mode),
        "solver_tokens_per_problem": solver_tokens_per_problem(result),
        "solver_input_tokens": int(result.get("solver_input_tokens", 0)),
        "solver_output_tokens": int(result.get("solver_output_tokens", 0)),
        "empty_responses": int(result.get("empty_responses", 0)),
        "n_problems": int(result["n_problems"]),
    }


def pareto_frontier(points: list[dict]) -> list[dict]:
    """Keep points not dominated by higher accuracy and lower solver usage."""
    frontier = []
    for point in points:
        dominated = any(
            other["accuracy"] >= point["accuracy"]
            and other["solver_tokens_per_problem"] <= point["solver_tokens_per_problem"]
            and (
                other["accuracy"] > point["accuracy"]
                or other["solver_tokens_per_problem"] < point["solver_tokens_per_problem"]
            )
            for other in points
            if other is not point
        )
        if not dominated:
            frontier.append(point)
    return sorted(
        frontier,
        key=lambda row: (-row["accuracy"], row["solver_tokens_per_problem"], row["system"]),
    )
