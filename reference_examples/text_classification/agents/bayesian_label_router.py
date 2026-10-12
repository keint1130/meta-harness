"""Use a lexical Bayes router to focus the final answer prompt."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT_TEMPLATE = """Solve the problem using the ranked label candidates as a guide.

{router_section}

**Problem:**
{input}

The ranking is a word-frequency estimate learned from labeled examples. Check
it against the full meaning of the problem and the task instructions. It is
guidance, not a constraint. Respond in JSON:
{{"reasoning": "[brief explanation]", "final_answer": "[your answer]"}}"""

SMOOTHING = 0.1
MAX_CANDIDATES = 5
MAX_QUERY_CUES = 3


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]{2,}", text.lower())


class BayesianLabelRouter(MemorySystem):
    """Rank learned labels with a smoothed multinomial Bayes evidence model."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.class_token_counts: dict[str, Counter[str]] = {}
        self.class_docs: Counter[str] = Counter()
        self.global_token_counts: Counter[str] = Counter()

    def _rank(self, query: str) -> list[tuple[str, float, list[str]]]:
        labels = sorted(self.class_token_counts, key=str.casefold)
        if not labels:
            return []

        vocabulary_size = max(1, len(self.global_token_counts))
        query_counts = Counter(_tokens(query))
        scores: dict[str, float] = {}
        token_logs: dict[str, dict[str, float]] = {}
        total_docs = max(1, sum(self.class_docs.values()))

        for label in labels:
            counts = self.class_token_counts[label]
            token_total = sum(counts.values())
            denominator = token_total + SMOOTHING * vocabulary_size
            score = math.log((self.class_docs[label] + 1) / (total_docs + len(labels)))
            token_logs[label] = {}
            for token, query_count in query_counts.items():
                if token not in self.global_token_counts:
                    continue
                log_probability = math.log(
                    (counts[token] + SMOOTHING) / denominator
                )
                score += query_count * log_probability
                token_logs[label][token] = log_probability
            scores[label] = score

        ranked_labels = sorted(labels, key=lambda label: scores[label], reverse=True)
        ranked = []
        for label in ranked_labels:
            rivals = [other for other in labels if other != label]
            cues = []
            if rivals:
                differences = []
                for token, value in token_logs[label].items():
                    rival_value = max(
                        (token_logs[other].get(token, 0.0) for other in rivals),
                        default=0.0,
                    )
                    difference = value - rival_value
                    if difference > 0:
                        differences.append((difference, token))
                cues = [token for _, token in sorted(differences, reverse=True)[:MAX_QUERY_CUES]]
            ranked.append((label, scores[label], cues))
        return ranked

    def _format_router(self, query: str) -> str:
        ranked = self._rank(query)
        if not ranked:
            return "No labeled category evidence has been learned yet."

        lines = ["Provisional label ranking (strongest first):"]
        for index, (label, _, cues) in enumerate(ranked[:MAX_CANDIDATES], start=1):
            cue_text = ", ".join(cues) if cues else "no distinct matching word"
            lines.append(
                f"{index}. {label} | training cases: {self.class_docs[label]} | "
                f"matching terms: {cue_text}"
            )
        return "\n".join(lines)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        router_section = self._format_router(input)
        response = self.call_llm(PROMPT_TEMPLATE.format(router_section=router_section, input=input))
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "num_labels": len(self.class_token_counts),
            "num_candidates": min(len(self.class_token_counts), MAX_CANDIDATES),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            label = str(result["ground_truth"])
            text = str(result.get("raw_question") or result["input"])
            counts = Counter(_tokens(text))
            self.class_token_counts.setdefault(label, Counter()).update(counts)
            self.class_docs[label] += 1
            self.global_token_counts.update(counts)

    def get_context_length(self) -> int:
        return len(self._format_router(""))

    def get_state(self) -> str:
        return json.dumps(
            {
                "class_token_counts": {
                    label: dict(counts)
                    for label, counts in self.class_token_counts.items()
                },
                "class_docs": dict(self.class_docs),
                "global_token_counts": dict(self.global_token_counts),
            },
            ensure_ascii=False,
            indent=2,
        )

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.class_token_counts = {
            label: Counter(counts)
            for label, counts in data.get("class_token_counts", {}).items()
        }
        self.class_docs = Counter(data.get("class_docs", {}))
        self.global_token_counts = Counter(data.get("global_token_counts", {}))
