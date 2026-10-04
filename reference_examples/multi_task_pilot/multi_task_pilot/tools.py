from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from .contracts import ToolCall
from .docker_sandbox import DockerSandbox


class ToolBudgetExceeded(RuntimeError):
    pass


def _document_text(document: Any) -> str:
    if isinstance(document, str):
        return document
    if isinstance(document, dict):
        for key in ("text", "contents", "content", "passage"):
            if key in document:
                return str(document[key])
    return str(document)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


@dataclass(frozen=True)
class ToolBudgets:
    python_calls: int = 2
    retrieve_calls: int = 2
    retrieved_documents: int = 5


class FrozenTools:
    """Per-case tools with fixed resources, budgets, and an audit log."""

    def __init__(
        self,
        *,
        resources: dict[str, Any] | None = None,
        sandbox: DockerSandbox | None = None,
        budgets: ToolBudgets | None = None,
    ) -> None:
        self.resources = resources or {}
        self.sandbox = sandbox or DockerSandbox()
        self.budgets = budgets or ToolBudgets()
        self.calls: list[ToolCall] = []
        self._python_calls = 0
        self._retrieve_calls = 0

    def python_exec(self, source: str, stdin: str = "") -> str:
        if self._python_calls >= self.budgets.python_calls:
            raise ToolBudgetExceeded("python_exec call budget exceeded")
        self._python_calls += 1
        started = time.monotonic()
        result = self.sandbox.run_python(source, stdin)
        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        if result.timed_out:
            output += ("\n" if output else "") + "[timed out]"
        call = ToolCall(
            name="python_exec",
            arguments={"source": source, "stdin": stdin},
            output=output,
            duration_seconds=time.monotonic() - started,
        )
        self.calls.append(call)
        return output

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._retrieve_calls >= self.budgets.retrieve_calls:
            raise ToolBudgetExceeded("retrieve call budget exceeded")
        self._retrieve_calls += 1
        started = time.monotonic()
        limit = min(max(0, top_k), self.budgets.retrieved_documents)
        query_tokens = _tokens(query)
        documents = list(self.resources.get("documents", []))
        ranked = sorted(
            enumerate(documents),
            key=lambda item: (
                -len(query_tokens & _tokens(_document_text(item[1]))),
                item[0],
            ),
        )[:limit]
        results = [
            {"rank": rank, "text": _document_text(document)}
            for rank, (_, document) in enumerate(ranked, start=1)
        ]
        self.calls.append(
            ToolCall(
                name="retrieve",
                arguments={"query": query, "top_k": top_k},
                output=str(results),
                duration_seconds=time.monotonic() - started,
            )
        )
        return results
