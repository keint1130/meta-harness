"""Present demonstrations with an explicit priority rule for the live query."""

import json
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_MAX_CONTEXT_CHARS = 30000
_PROMPT = """The examples are references for recognizing useful patterns. They do not
override the instructions or answer format in the current problem. Treat each
example's expected answer as the target value; follow any serialization or
wrapper required by the current problem when writing your own final_answer.

{examples}

**Current problem and its instructions:**
{input}

Return one JSON object with exactly the fields \"reasoning\" and \"final_answer\".
The final_answer must satisfy the current problem's requested answer format.
"""


def _question_from_result(result: dict[str, Any]) -> str:
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    value = result.get("raw_question") or metadata.get("raw_question")
    return str(value if value is not None else result.get("input", ""))


class InstructionAnchoredMemory(MemorySystem):
    """Keep a broad reference set while making current-query constraints clear."""

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
        examples = self._format_examples()
        self._last_context_chars = len(examples)
        response = self.call_llm(
            _PROMPT.format(examples=examples or "No references yet.", input=input)
        )
        answer = extract_json_field(response, "final_answer")
        return answer, {"full_response": response, "num_examples": examples.count("Reference ")}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            self.examples.append(
                {
                    "question": _question_from_result(result),
                    "target": str(result.get("ground_truth", "")),
                }
            )

    def get_context_length(self) -> int:
        return self._last_context_chars or len(self._format_examples())

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = [
            {"question": str(item.get("question", "")), "target": str(item.get("target", ""))}
            for item in data.get("examples", [])
        ]
        self._last_context_chars = 0
