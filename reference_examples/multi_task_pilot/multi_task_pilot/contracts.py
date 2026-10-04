from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class TaskFamily(StrEnum):
    CODING = "coding"
    RETRIEVAL = "retrieval"
    MATH = "math"


@dataclass(frozen=True)
class Task:
    id: str
    family: TaskFamily
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalCase:
    """Evaluator-owned data. Only ``task`` is passed to a candidate harness."""

    task: Task
    reference: Any
    resources: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    output: str
    duration_seconds: float


@dataclass
class HarnessResult:
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolSet(Protocol):
    def python_exec(self, source: str, stdin: str = "") -> str: ...

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]: ...


class Harness(Protocol):
    def solve(self, task: Task, tools: ToolSet) -> HarnessResult: ...
