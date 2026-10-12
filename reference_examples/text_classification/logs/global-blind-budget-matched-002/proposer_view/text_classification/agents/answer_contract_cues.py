"""Repeat concise output-contract cues from the live query after its references."""

import json
import re
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_MAX_CONTEXT_CHARS = 30000
_MAX_STORED_EXAMPLES = 512
_DIRECTIVE_RE = re.compile(
    r"\b(format|output|answer|respond|return|exact|json|serialize|delimiter|"
    r"separated|must|should|between)\b",
    re.IGNORECASE,
)
_MARKUP_RE = re.compile(r"(?<!\w)(?:\[[^\]\r\n]{1,48}\]|<[^>\r\n]{1,48}>)")

_PROMPT = """The references are labeled examples. Use them to reason about the current
problem, while following the current problem's own requirements.

{examples}

**Current problem:**
{input}

**Output-contract cues quoted from the current problem:**
{contract}

Use these cues to preserve any requested wrapper, separator, serialization, or
answer format. They summarize the current problem and do not change its meaning.
Return one JSON object with "reasoning" and "final_answer" fields; put the
requested answer in final_answer exactly as the current problem requires.
"""


def _question_from_result(result: dict[str, Any]) -> str:
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    value = result.get("raw_question") or metadata.get("raw_question")
    return str(value if value is not None else result.get("input", ""))


def _output_cues(text: str) -> str:
    cues: list[str] = []
    seen: set[str] = set()
    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) <= 700 and _DIRECTIVE_RE.search(stripped):
            cue = stripped[:600]
            if cue not in seen:
                cues.append(cue)
                seen.add(cue)
    for line in lines:
        if len(line) > 1600:
            continue
        markers = list(_MARKUP_RE.finditer(line))
        if len(markers) < 2:
            continue
        cue = line.strip()[:600]
        if cue and cue not in seen:
            cues.append(cue)
            seen.add(cue)
    return "\n".join(cues[:8]) or "Follow the output format stated in the current problem."


class AnswerContractCues(MemorySystem):
    """Retain broad examples and restate current-query formatting cues at the end."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._last_context_chars = 0

    def _format_examples(self) -> str:
        parts: list[str] = []
        used = 0
        for index, item in enumerate(self.examples, start=1):
            part = (
                f"Reference {index}\nProblem: {item['question']}\n"
                f"Expected answer: {item['target']}"
            )
            added = len(part) + (2 if parts else 0)
            if used + added > _MAX_CONTEXT_CHARS:
                break
            parts.append(part)
            used += added
        return "\n\n".join(parts)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples = self._format_examples() or "No references yet."
        contract = _output_cues(input)
        self._last_context_chars = len(examples) + len(contract)
        response = self.call_llm(
            _PROMPT.format(examples=examples, input=input, contract=contract)
        )
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "num_examples": examples.count("Reference "),
            "num_contract_cues": len(contract.splitlines()),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        seen = {(item["question"], item["target"]) for item in self.examples}
        for result in batch_results:
            item = {
                "question": _question_from_result(result),
                "target": str(result.get("ground_truth", "")),
            }
            key = (item["question"], item["target"])
            if key not in seen:
                self.examples.append(item)
                seen.add(key)
        self.examples = self.examples[-_MAX_STORED_EXAMPLES:]
        self._last_context_chars = 0

    def get_context_length(self) -> int:
        return self._last_context_chars or len(self._format_examples())

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = [
            {"question": str(item.get("question", "")), "target": str(item.get("target", ""))}
            for item in data.get("examples", [])
        ][-_MAX_STORED_EXAMPLES:]
        self._last_context_chars = 0
