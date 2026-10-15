"""Candidate contract and leakage-safe math memory access."""

from __future__ import annotations

import inspect
from collections.abc import Iterator, Sequence
from typing import Any


class SolvedMathMemory(Sequence):
    """Lazy, read-only view of an external solved-mathematics corpus.

    Evaluation labels and benchmark metadata are never included.  Rows expose
    only material that a retrieval system could use at deployment time:
    problem, solution, answer, and source.
    """

    SAFE_FIELDS = ("problem", "solution", "answer", "source")

    def __init__(
        self,
        repository: str,
        split: str = "train",
        revision: str | None = None,
    ):
        self.repository = repository
        self.split = split
        self.revision = revision
        self._dataset = None

    def _load(self):
        if self._dataset is None:
            from datasets import load_dataset

            self._dataset = load_dataset(
                self.repository,
                split=self.split,
                revision=self.revision,
            )
        return self._dataset

    @classmethod
    def _sanitize(cls, row: dict[str, Any]) -> dict[str, Any]:
        return {key: row.get(key) for key in cls.SAFE_FIELDS if key in row}

    def __len__(self) -> int:
        return len(self._load())

    def __getitem__(self, index):
        dataset = self._load()
        if isinstance(index, slice):
            return [self._sanitize(dataset[i]) for i in range(*index.indices(len(dataset)))]
        return self._sanitize(dataset[index])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for row in self._load():
            yield self._sanitize(row)


def candidate_observations(rows: list[dict]) -> list[dict[str, str]]:
    """Legacy constructor input with all labels and IDs removed."""
    return [{"problem": row["problem"]} for row in rows]


def instantiate_candidate(
    candidate_class,
    rows: list[dict],
    *,
    domain: str | None,
    memory: Sequence,
):
    """Construct new and legacy candidates without revealing evaluator state.

    New candidates may accept ``memory`` and ``task_context``.  Global runs get
    an empty context; task-specific runs get only the run-level domain.  The
    legacy ``test_problems`` argument contains problem text only.
    """
    parameters = inspect.signature(candidate_class).parameters
    kwargs = {}
    if "memory" in parameters:
        kwargs["memory"] = memory
    if "task_context" in parameters:
        kwargs["task_context"] = {"domain": domain} if domain else {}
    if "test_problems" in parameters:
        kwargs["test_problems"] = candidate_observations(rows)
    return candidate_class(**kwargs)
