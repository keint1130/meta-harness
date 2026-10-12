"""Show corrections only when their problem is a close match to the query."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_PROMPT = """Some references are ordinary labeled examples. A corrected mistake
records a prior answer that was judged wrong for a similar problem. Use a
correction only when its details match; otherwise set it aside. The complete
current problem and its requested answer format govern.

{references}

**Current problem and instructions:**
{input}

Return one JSON object with "reasoning" and "final_answer" fields. Follow the
current problem's requested format exactly.
"""


def _question(result: dict[str, Any]) -> str:
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return str(result.get("raw_question") or metadata.get("raw_question") or result.get("input", ""))


def _terms(text: str) -> Counter[str]:
    return Counter(token.lower() for token in _TOKEN_RE.findall(text))


def _cosine(left: Counter[str], right: Counter[str], df: Counter[str], n: int) -> float:
    def vector(terms: Counter[str]) -> dict[str, float]:
        vals = {key: (1 + math.log(count)) * (1 + math.log((n + 1) / (df[key] + 1))) for key, count in terms.items()}
        norm = math.sqrt(sum(value * value for value in vals.values())) or 1.0
        return {key: value / norm for key, value in vals.items()}
    a, b = vector(left), vector(right)
    return sum(value * b.get(key, 0.0) for key, value in a.items())


class SimilarityGatedCorrectionMemory(MemorySystem):
    """Combine query-near successes with corrections above a relevance gate."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.errors: list[dict[str, str]] = []
        self.successes: list[dict[str, str]] = []
        self._last_context_chars = 0

    def _references(self, query: str) -> tuple[str, int, int]:
        combined = [("success", item) for item in self.successes] + [("error", item) for item in self.errors]
        if not combined:
            return "No learned examples yet.", 0, 0
        docs = [_terms(item[1]["question"]) for item in combined]
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(doc.keys())
        ranked = sorted(
            ((_cosine(_terms(query), doc, df, len(docs)), kind, item) for (kind, item), doc in zip(combined, docs)),
            key=lambda row: -row[0],
        )
        best = ranked[0][0]
        # Weak matches are omitted: they can distract without offering a useful correction.
        selected = [row for row in ranked if row[0] >= max(0.10, best * 0.62)][:5]
        if not selected:
            return "No sufficiently similar learned examples.", 0, 0
        parts = []
        error_count = success_count = 0
        for rank, (score, kind, item) in enumerate(selected, 1):
            if kind == "error":
                error_count += 1
                part = (
                    f"Relevant corrected mistake {rank} (similarity {score:.3f})\n"
                    f"Problem: {item['question']}\nPrior incorrect answer: {item['prediction']}\n"
                    f"Correct answer: {item['target']}"
                )
            else:
                success_count += 1
                part = f"Relevant reference {rank} (similarity {score:.3f})\nProblem: {item['question']}\nExpected answer: {item['target']}"
            parts.append(part)
        return "\n\n".join(parts), error_count, success_count

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        refs, errors, successes = self._references(input)
        self._last_context_chars = len(refs)
        response = self.call_llm(_PROMPT.format(references=refs, input=input))
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "num_relevant_corrections": errors,
            "num_relevant_successes": successes,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            target = str(result.get("ground_truth", ""))
            prediction = str(result.get("prediction", ""))
            correct = result.get("was_correct")
            if correct is None:
                correct = prediction == target
            item = {"question": _question(result), "prediction": prediction, "target": target}
            (self.successes if bool(correct) else self.errors).append(item)
        self.errors = self.errors[-256:]
        self.successes = self.successes[-256:]
        self._last_context_chars = 0

    def get_context_length(self) -> int:
        return self._last_context_chars or len(self._references("")[0])

    def get_state(self) -> str:
        return json.dumps({"errors": self.errors, "successes": self.successes}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        def clean(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
            return [{"question": str(x.get("question", "")), "prediction": str(x.get("prediction", "")), "target": str(x.get("target", ""))} for x in rows][-256:]
        self.errors = clean(data.get("errors", []))
        self.successes = clean(data.get("successes", []))
        self._last_context_chars = 0
