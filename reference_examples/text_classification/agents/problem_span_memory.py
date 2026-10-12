"""Represent examples by their task-specific problem spans for retrieval."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_HEADING_RE = re.compile(
    r"^(?:#{1,6}\s*)?(?:current\s+problem(?:\s+and\s+its\s+instructions)?|"
    r"problem|patient\s+symptoms|case\s+facts|facts|input|query)\s*:?\s*$",
    re.IGNORECASE,
)
_PROMPT = """The references show task-specific problem details and their expected
answers. Use them when the details match; the complete current problem and its
instructions remain authoritative.

{examples}

**Current problem and instructions:**
{input}

Return a JSON object with "reasoning" and "final_answer" fields. Follow the
requested answer format exactly.
"""


def _question(result: dict[str, Any]) -> str:
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return str(result.get("raw_question") or metadata.get("raw_question") or result.get("input", ""))


def _core(text: str) -> str:
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if _HEADING_RE.match(line.strip())]
    if starts:
        # Use the last recognizable problem heading so wrappers and generic instructions
        # before the actual case do not dominate similarity.
        start = starts[-1] + 1
        body = "\n".join(lines[start:]).strip()
        if len(body) >= 24:
            return body
    # Inline field labels are useful for compact structured inputs.
    match = re.search(r"(?im)^\s*(?:patient symptoms|case facts|facts|input)\s*:\s*", text)
    if match and len(text[match.end():].strip()) >= 24:
        return text[match.end():].strip()
    return text


def _vec(features: Counter[str], df: Counter[str], n: int) -> dict[str, float]:
    values = {k: (1 + math.log(c)) * (1 + math.log((n + 1) / (df[k] + 1))) for k, c in features.items()}
    norm = math.sqrt(sum(v * v for v in values.values())) or 1.0
    return {k: v / norm for k, v in values.items()}


class ProblemSpanMemory(MemorySystem):
    """Index the variable body of a problem instead of repeated prompt boilerplate."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._last_context_chars = 0

    def _ranked(self, query: str) -> list[tuple[float, dict[str, str], str]]:
        if not self.examples:
            return []
        cores = [_core(item["question"]) for item in self.examples]
        docs = [Counter("t:" + token.lower() for token in _WORD_RE.findall(core)) for core in cores]
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(doc.keys())
        q = _vec(Counter("t:" + token.lower() for token in _WORD_RE.findall(_core(query))), df, len(docs))
        ranked = []
        for item, core, doc in zip(self.examples, cores, docs):
            d = _vec(doc, df, len(docs))
            ranked.append((sum(v * d.get(k, 0.0) for k, v in q.items()), item, core))
        return sorted(ranked, key=lambda x: -x[0])[:6]

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        parts = [f"Reference {i} (problem-span similarity {score:.3f})\nProblem details: {core}\nExpected answer: {item['target']}" for i, (score, item, core) in enumerate(self._ranked(input), 1)]
        examples = "\n\n".join(parts) or "No references yet."
        self._last_context_chars = len(examples)
        response = self.call_llm(_PROMPT.format(examples=examples, input=input))
        return extract_json_field(response, "final_answer"), {"full_response": response, "num_examples": len(parts)}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        seen = {(x["question"], x["target"]) for x in self.examples}
        for result in batch_results:
            item = {"question": _question(result), "target": str(result.get("ground_truth", ""))}
            key = (item["question"], item["target"])
            if key not in seen:
                self.examples.append(item)
                seen.add(key)
        self.examples = self.examples[-512:]
        self._last_context_chars = 0

    def get_context_length(self) -> int:
        return self._last_context_chars or sum(len(_core(x["question"])) + len(x["target"]) + 36 for x in self.examples[-6:])

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = [{"question": str(x.get("question", "")), "target": str(x.get("target", ""))} for x in data.get("examples", [])][-512:]
        self._last_context_chars = 0
