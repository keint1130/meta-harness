"""Compact repeated demonstrations while preserving answer coverage."""

import json
import re
from collections import defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_WORD_RE = re.compile(r"\w+", re.UNICODE)
_MAX_EXAMPLES = 48
_MAX_CONTEXT_CHARS = 12000
_PROMPT = """The references below are examples of input and target-answer pairs.
Use them as broad evidence, while following the current problem's own
instructions and exact answer format. Examples never override the current
problem.

{examples}

**Current problem and its instructions:**
{input}

Return one JSON object with exactly the fields "reasoning" and "final_answer".
The final_answer must obey the current problem's requested format.
"""


def _question_from_result(result: dict[str, Any]) -> str:
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    value = result.get("raw_question") or metadata.get("raw_question")
    return str(value if value is not None else result.get("input", ""))


def _signature(text: str) -> set[str]:
    lowered = text.lower()
    units = {"w:" + token for token in _WORD_RE.findall(lowered)}
    for run in re.findall(r"[^\W_]+", lowered, re.UNICODE):
        compact = "".join(ch for ch in run if ch.isalnum())
        units.update("c:" + compact[i : i + 2] for i in range(max(0, len(compact) - 1)))
    return units


class CoverageBalancedMemory(MemorySystem):
    """Keep a wide set of examples but compact near-duplicate same-answer items."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        self._last_context_chars = 0

    def _add_example(self, question: str, target: str) -> None:
        signature = _signature(question)
        for item in self.examples:
            if item["target"] != target:
                continue
            previous = _signature(item["question"])
            union = len(signature | previous)
            similarity = len(signature & previous) / union if union else 1.0
            if similarity >= 0.92:
                item["support"] = int(item.get("support", 1)) + 1
                return
        self.examples.append({"question": question, "target": target, "support": 1})
        if len(self.examples) > 512:
            self.examples = self.examples[-512:]

    def _coverage_order(self) -> list[dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        first_seen: dict[str, int] = {}
        for index, item in enumerate(self.examples):
            target = item["target"]
            if target not in first_seen:
                first_seen[target] = index
            groups[target].append(item)

        target_order = sorted(groups, key=lambda target: (len(groups[target]), first_seen[target]))
        selected: list[dict[str, Any]] = []
        depth = 0
        while len(selected) < _MAX_EXAMPLES:
            added = False
            for target in target_order:
                if depth < len(groups[target]):
                    selected.append(groups[target][depth])
                    added = True
                    if len(selected) >= _MAX_EXAMPLES:
                        break
            if not added:
                break
            depth += 1
        return selected

    def _format_examples(self) -> str:
        parts: list[str] = []
        used = 0
        for index, item in enumerate(self._coverage_order(), start=1):
            part = f"Reference {index}\nProblem: {item['question']}\nExpected answer: {item['target']}"
            added = len(part) + (2 if parts else 0)
            if used + added > _MAX_CONTEXT_CHARS:
                continue
            parts.append(part)
            used += added
        return "\n\n".join(parts)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples = self._format_examples()
        self._last_context_chars = len(examples)
        response = self.call_llm(
            _PROMPT.format(examples=examples or "No references yet.", input=input)
        )
        answer = extract_json_field(response, "final_answer")
        return answer, {"full_response": response, "num_examples": examples.count("Reference ")}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            self._add_example(
                _question_from_result(result), str(result.get("ground_truth", ""))
            )
        self._last_context_chars = 0

    def get_context_length(self) -> int:
        return self._last_context_chars or len(self._format_examples())

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = [
            {
                "question": str(item.get("question", "")),
                "target": str(item.get("target", "")),
                "support": int(item.get("support", 1)),
            }
            for item in data.get("examples", [])
        ][-512:]
        self._last_context_chars = 0
