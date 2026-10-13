"""Deterministic, leakage-aware splits for IMO-AnswerBench."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
PILOT_PER_DOMAIN = 3


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def normalize_domain(category: str) -> str:
    value = category.strip().lower().replace(" ", "_")
    return "algebra" if value == "functional_equation" else value


def load_problem_rows(config: dict | None = None) -> list[dict]:
    config = config or load_config()
    math_repo = (ROOT / config["paths"]["math_eval_repo"]).resolve()
    csv_path = math_repo / "data" / "imobench" / "answerbench_v2.csv"
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "problem_id": row["Problem ID"],
            "domain": normalize_domain(row["Category"]),
            "problem": row["Problem"].strip(),
            "groundtruth": row["Short Answer"].strip(),
            "subcategory": row.get("Subcategory", ""),
        }
        for row in rows
    ]


def _rank(problem_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{problem_id}".encode()).hexdigest()


def build_splits(config: dict | None = None) -> dict[str, dict[str, list[str]]]:
    """Return split -> domain -> IDs, forcing the prior pilot into search."""
    config = config or load_config()
    domains = config["domains"]
    sizes = config["splits"]
    seed = int(config["experiment"]["seed"])
    rows = load_problem_rows(config)
    by_domain = {domain: [] for domain in domains}
    for row in rows:
        if row["domain"] in by_domain:
            by_domain[row["domain"]].append(row["problem_id"])

    result = {name: {domain: [] for domain in domains} for name in sizes}
    for domain in domains:
        ids = by_domain[domain]
        prefix = f"imo-bench-{domain}-"
        pilot = [f"{prefix}{number:03d}" for number in range(1, PILOT_PER_DOMAIN + 1)]
        if not set(pilot).issubset(ids):
            raise ValueError(f"Pilot IDs missing for {domain}")
        remaining = sorted((item for item in ids if item not in pilot), key=lambda x: _rank(x, seed))
        search_extra = sizes["search"] - len(pilot)
        result["search"][domain] = pilot + remaining[:search_extra]
        cursor = search_extra
        result["dev"][domain] = remaining[cursor : cursor + sizes["dev"]]
        cursor += sizes["dev"]
        result["test"][domain] = remaining[cursor : cursor + sizes["test"]]

    validate_splits(result, config)
    return result


def validate_splits(splits: dict, config: dict | None = None) -> None:
    config = config or load_config()
    all_seen: set[str] = set()
    for split_name, expected_size in config["splits"].items():
        for domain in config["domains"]:
            ids = splits[split_name][domain]
            if len(ids) != expected_size:
                raise ValueError(f"{split_name}/{domain}: expected {expected_size}, got {len(ids)}")
            overlap = all_seen.intersection(ids)
            if overlap:
                raise ValueError(f"Split leakage detected: {sorted(overlap)}")
            all_seen.update(ids)
    expected_total = sum(config["splits"].values()) * len(config["domains"])
    if len(all_seen) != expected_total:
        raise ValueError(f"Expected {expected_total} unique IDs, got {len(all_seen)}")


def select_rows(
    split_name: str,
    domain: str | None = None,
    config: dict | None = None,
) -> list[dict]:
    config = config or load_config()
    splits = build_splits(config)
    if split_name not in splits:
        raise ValueError(f"Unknown split: {split_name}")
    domains = [domain] if domain else config["domains"]
    wanted = {
        problem_id
        for selected_domain in domains
        for problem_id in splits[split_name][selected_domain]
    }
    rows_by_id = {row["problem_id"]: row for row in load_problem_rows(config)}
    return [rows_by_id[problem_id] for problem_id in sorted(wanted)]

