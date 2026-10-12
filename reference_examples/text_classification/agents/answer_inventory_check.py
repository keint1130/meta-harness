"""Use retrieved answer components as a coverage checklist before responding."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_RUNS = re.compile(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+")
_CJK = re.compile(r"^[\u3400-\u9fff]+$")
_PROMPT = """The examples are labeled cases. Their verified answer components form a checklist of possible outputs, not a list to copy. For the current problem, decide which components are supported by its own facts, check whether another distinct component is warranted, and omit unsupported components. Preserve the exact answer form requested by the problem.

Examples:
{examples}

Current problem:
{query}

Return JSON with final_answer.
{{"reasoning":"check support for each output component and completeness","final_answer":"answer"}}"""


def _source(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return str(row.get("raw_question") or metadata.get("raw_question") or metadata.get("question") or row.get("input", ""))


def _features(text: str) -> Counter[str]:
    out: Counter[str] = Counter()
    for run in _RUNS.findall(text.lower()):
        if _CJK.match(run):
            out.update(run[i:i + 2] for i in range(len(run) - 1))
            out.update(run[i:i + 3] for i in range(len(run) - 2))
        else:
            out[run] += 1
    return out


def _components(target: str) -> list[str]:
    return sorted({part.strip() for part in re.split(r"[;；、]", target) if part.strip()})


class Candidate(MemorySystem):
    """Retrieve examples to build a frequency-aware answer inventory."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        self.budget = 8000
        self._last = ""

    def _retrieve(self, query: str) -> list[tuple[float, int]]:
        if not self.examples:
            return []
        docs = [_features(x["input"]) for x in self.examples]
        df: Counter[str] = Counter()
        for d in docs:
            df.update(d.keys())
        n = len(docs)
        idf = {t: math.log(1 + (n + 1) / (f + 1)) for t, f in df.items()}
        q = _features(query)

        def vec(c: Counter[str]) -> dict[str, float]:
            raw = {t: (1 + math.log(v)) * idf.get(t, 0) for t, v in c.items() if v}
            norm = math.sqrt(sum(v * v for v in raw.values())) or 1.0
            return {t: v / norm for t, v in raw.items()}

        qv = vec(q)
        ranked = []
        for i, d in enumerate(docs):
            dv = vec(d)
            score = sum(v * dv.get(t, 0) for t, v in qv.items())
            ranked.append((score, i))
        return sorted((x for x in ranked if x[0] > 0), reverse=True)

    def _format(self, query: str) -> str:
        ranked = self._retrieve(query)
        # Aggregate only a short list of labels from close cases, with case-level provenance.
        component_support: Counter[str] = Counter()
        selected: list[int] = []
        for _, i in ranked[:12]:
            selected.append(i)
            component_support.update(self.examples[i]["components"])
        inventory = sorted(component_support, key=lambda label: (-component_support[label], label))[:18]
        parts = ["Nearby verified answer components (frequency reflects nearby cases only): " + " ; ".join(inventory)] if inventory else []
        used = len(parts[0]) if parts else 0
        for i in selected:
            ex = self.examples[i]
            problem = ex["input"]
            if len(problem) > 850:
                problem = problem[:600] + " … " + problem[-247:]
            block = f"Case facts: {problem}\nVerified answer: {ex['target']}"
            if used + len(block) > self.budget:
                continue
            parts.append(block)
            used += len(block) + 2
        self._last = "\n\n".join(parts)
        return self._last or "No verified answer inventory is available yet."

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        examples = self._format(input)
        response = self.call_llm(_PROMPT.format(examples=examples, query=input))
        return extract_json_field(response, "final_answer"), {"full_response": response, "retrieved_examples": len(self._retrieve(input))}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        seen = {(x["input"], x["target"]) for x in self.examples}
        for row in batch_results:
            text, target = _source(row), str(row["ground_truth"])
            if (text, target) not in seen:
                self.examples.append({"input": text, "target": target, "components": _components(target)})
                seen.add((text, target))

    def get_context_length(self) -> int:
        return len(self._last)

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])
        self._last = ""
