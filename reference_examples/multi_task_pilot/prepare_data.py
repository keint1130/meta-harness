from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset
from huggingface_hub import HfApi

from multi_task_pilot.splits import stable_rank


ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.yaml"
MANIFEST_DIR = ROOT / "manifests"
PREPARED_DIR = ROOT / "data" / "prepared"


DATASET_SPECS = {
    "livecodebench": {"id_field": "question_id"},
    "popqa": {"id_field": "question_id"},
    "math500": {"id_field": "unique_id"},
}


def _smallest_rows(stream, *, id_field: str, seed: int, count: int):
    """Keep only the deterministically lowest-ranked rows in memory."""
    heap: list[tuple[int, str, dict[str, Any]]] = []
    seen: set[str] = set()
    for row in stream:
        identifier = str(row[id_field])
        if identifier in seen:
            raise ValueError(f"Duplicate {id_field}: {identifier}")
        seen.add(identifier)
        rank = int(stable_rank(identifier, seed), 16)
        entry = (-rank, identifier, dict(row))
        if len(heap) < count:
            heapq.heappush(heap, entry)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, entry)
    selected = [entry[2] for entry in heap]
    selected.sort(key=lambda row: stable_rank(str(row[id_field]), seed))
    return selected, len(seen)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def prepare_one(name: str, cfg: dict[str, Any]) -> Path:
    dataset_cfg = cfg["datasets"][name]
    source = dataset_cfg["source"]
    requested_revision = dataset_cfg.get("revision")
    resolved_revision = HfApi().dataset_info(source, revision=requested_revision).sha
    dataset_config = dataset_cfg.get("config")

    if name == "livecodebench":
        # The pinned release still contains a legacy loader script, which
        # datasets 5.x intentionally refuses to execute. Read JSONL directly
        # from the same immutable commit. `v6` means only the latest release
        # window; `release_v6` means all six cumulative windows.
        if dataset_config == "v6":
            filenames = ["test6.jsonl"]
        elif dataset_config == "release_v6":
            filenames = ["test.jsonl", *[f"test{i}.jsonl" for i in range(2, 7)]]
        else:
            raise ValueError(f"Unsupported LiveCodeBench config: {dataset_config}")
        urls = [
            f"https://huggingface.co/datasets/{source}/resolve/"
            f"{resolved_revision}/{filename}"
            for filename in filenames
        ]
        stream = load_dataset(
            "json", data_files={"test": urls}, split="test", streaming=True
        )
    else:
        kwargs = {
            "path": source,
            "split": dataset_cfg.get("split", "test"),
            "revision": resolved_revision,
            "streaming": True,
        }
        if dataset_config:
            kwargs["name"] = dataset_config
        stream = load_dataset(**kwargs)

    source_filter = dataset_cfg.get("source_filter", {})
    if source_filter:
        stream = (
            row
            for row in stream
            if all(row.get(key) == value for key, value in source_filter.items())
        )

    dev_size = int(cfg["splits"]["dev"])
    test_size = int(cfg["splits"]["test"])
    id_field = DATASET_SPECS[name]["id_field"]
    rows, source_count = _smallest_rows(
        stream,
        id_field=id_field,
        seed=int(cfg["seed"]),
        count=dev_size + test_size,
    )
    dev = rows[:dev_size]
    test = rows[dev_size:]

    _write_jsonl(PREPARED_DIR / name / "dev.jsonl", dev)
    _write_jsonl(PREPARED_DIR / name / "test.jsonl", test)

    manifest = {
        "dataset": name,
        "source": source,
        "config": dataset_config,
        "base_source": dataset_cfg.get("base_source"),
        "source_filter": source_filter,
        "revision": resolved_revision,
        "seed": int(cfg["seed"]),
        "source_count": source_count,
        "id_field": id_field,
        "splits": {
            "dev": [str(row[id_field]) for row in dev],
            "test": [str(row[id_field]) for row in test],
        },
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"{name}.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=sorted(DATASET_SPECS),
        default=sorted(DATASET_SPECS),
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    for name in args.datasets:
        path = prepare_one(name, cfg)
        print(path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
