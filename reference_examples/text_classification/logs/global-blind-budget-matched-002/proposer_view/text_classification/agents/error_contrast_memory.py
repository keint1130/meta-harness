"""Store corrected mistakes as contrastive references for future predictions."""

import json
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_MAX_CONTEXT_CHARS = 14000
_PROMPT = """The references below include past attempts that were marked incorrect and
their corrected answers. Use them as warnings about plausible mistakes, and learn
from the correction rather than repeating the prior answer. If there are only
ordinary references, use them as examples. Always follow the current problem's
instructions.

{references}

**Current problem:**
{input}

Return a JSON object with \"reasoning\" and \"final_answer\" fields. Keep the
final_answer concise and obey the answer format requested in the current problem.
"""


def _question_from_result(result: dict[str, Any]) -> str:
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    value = result.get("raw_question") or metadata.get("raw_question")
    return str(value if value is not None else result.get("input", ""))


class ErrorContrastMemory(MemorySystem):
    """Prioritize input, failed answer, and correction triples from learning."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.errors: list[dict[str, str]] = []
        self.successes: list[dict[str, str]] = []
        self._last_context_chars = 0

    def _format_references(self) -> str:
        is_error = bool(self.errors)
        items = self.errors if is_error else self.successes[-5:]
        parts: list[str] = []
        used = 0
        for index, item in enumerate(reversed(items), start=1):
            if is_error:
                part = (
                    f"Corrected mistake {index}\nProblem: {item['question']}\n"
                    f"Prior incorrect answer: {item['prediction']}\n"
                    f"Correct answer: {item['target']}"
                )
            else:
                part = (
                    f"Reference {index}\nProblem: {item['question']}\n"
                    f"Expected answer: {item['target']}"
                )
            added = len(part) + (2 if parts else 0)
            if used + added > _MAX_CONTEXT_CHARS:
                continue
            parts.append(part)
            used += added
        return "\n\n".join(parts)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        references = self._format_references()
        self._last_context_chars = len(references)
        response = self.call_llm(
            _PROMPT.format(references=references or "No references yet.", input=input)
        )
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_error_examples": references.count("Corrected mistake "),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            target = str(result.get("ground_truth", ""))
            prediction = str(result.get("prediction", ""))
            is_correct = result.get("was_correct")
            if is_correct is None:
                is_correct = prediction == target
            item = {
                "question": _question_from_result(result),
                "prediction": prediction,
                "target": target,
            }
            destination = self.successes if bool(is_correct) else self.errors
            destination.append(item)
        # Bound checkpoint growth while retaining recent corrections.
        self.errors = self.errors[-256:]
        self.successes = self.successes[-64:]

    def get_context_length(self) -> int:
        return self._last_context_chars or len(self._format_references())

    def get_state(self) -> str:
        return json.dumps(
            {"errors": self.errors, "successes": self.successes}, ensure_ascii=False
        )

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.errors = [
            {
                "question": str(item.get("question", "")),
                "prediction": str(item.get("prediction", "")),
                "target": str(item.get("target", "")),
            }
            for item in data.get("errors", [])
        ]
        self.successes = [
            {
                "question": str(item.get("question", "")),
                "prediction": str(item.get("prediction", "")),
                "target": str(item.get("target", "")),
            }
            for item in data.get("successes", [])
        ]
        self._last_context_chars = 0
