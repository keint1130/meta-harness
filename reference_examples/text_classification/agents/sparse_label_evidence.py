"""Learn sparse, contrastive feature evidence for each verified output label."""

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


_RUNS = re.compile(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+")
_CJK = re.compile(r"^[\u3400-\u9fff]+$")
_PROMPT = """The evidence cards summarize which text features have been associated with verified outputs in prior examples. They are candidate guidance, not substitutes for reading the current problem. Compare the current facts and follow its instructions.

{cards}

Current problem:
{input}

Return JSON with a concise final_answer.
{{"reasoning": "[brief reasoning]", "final_answer": "[answer]}}"""


def _features(text: str) -> set[str]:
    result: set[str] = set()
    for run in _RUNS.findall(text.lower()):
        if _CJK.match(run):
            result.update(run[i : i + 2] for i in range(len(run) - 1))
            result.update(run[i : i + 3] for i in range(len(run) - 2))
        else:
            result.add(run)
    return result


def _source_text(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return str(
        row.get("raw_question")
        or metadata.get("raw_question")
        or metadata.get("question")
        or row.get("input", "")
    )


class Candidate(MemorySystem):
    """Rank labels with smoothed feature likelihood ratios, not case similarity."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self.context_budget = 5000
        self.max_labels = 8
        self._last_context = ""

    def _evidence_cards(self, query: str) -> str:
        if not self.examples:
            self._last_context = ""
            return "No learned label evidence yet."

        document_features = [_features(example["input"]) for example in self.examples]
        labels: dict[str, list[int]] = defaultdict(list)
        global_df: Counter[str] = Counter()
        label_df: dict[str, Counter[str]] = defaultdict(Counter)
        for index, (example, features) in enumerate(zip(self.examples, document_features)):
            label = example["target"]
            labels[label].append(index)
            global_df.update(features)
            label_df[label].update(features)

        n_docs = len(self.examples)
        query_features = _features(query)
        label_scores: list[tuple[float, str, list[tuple[float, str]]]] = []
        for label, indexes in labels.items():
            class_size = len(indexes)
            contributions: list[tuple[float, str]] = []
            for token in query_features:
                df_all = global_df[token]
                # Features repeated in nearly every example carry little class evidence.
                if df_all == 0 or df_all / max(1, n_docs) >= 0.92:
                    continue
                within = label_df[label][token]
                p_label = (within + 0.5) / (class_size + 1.0)
                p_other = (df_all - within + 0.5) / (n_docs - class_size + 1.0)
                contribution = math.log(p_label / p_other)
                if abs(contribution) >= 0.05:
                    contributions.append((contribution, token))
            # Normalize for query length so long inputs do not dominate by volume.
            score = sum(value for value, _ in contributions) / math.sqrt(max(1, len(query_features)))
            supporting = sorted(
                ((value, token) for value, token in contributions if value > 0),
                reverse=True,
            )[:8]
            label_scores.append((score, label, supporting))

        label_scores.sort(key=lambda row: (row[0], len(labels[row[1]]), row[1]), reverse=True)
        cards: list[str] = []
        used = 0
        for score, label, supporting in label_scores[: self.max_labels]:
            matched = ", ".join(token for _, token in supporting) or "no strong matching features"
            card = f"Candidate verified output: {label}\nMatching evidence features: {matched}\nRelative feature score: {score:.3f}"
            if used + len(card) + 2 > self.context_budget:
                continue
            cards.append(card)
            used += len(card) + 2

        self._last_context = "\n\n".join(cards)
        return self._last_context or "No matching label evidence yet."

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        cards = self._evidence_cards(input)
        response = self.call_llm(_PROMPT.format(cards=cards, input=input))
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "candidate_labels": len(self._last_context.split("\n\n")) if self._last_context else 0,
            "memory_examples": len(self.examples),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        seen = {(example["input"], example["target"]) for example in self.examples}
        for row in batch_results:
            example = {"input": _source_text(row), "target": str(row["ground_truth"])}
            key = (example["input"], example["target"])
            if key not in seen:
                self.examples.append(example)
                seen.add(key)

    def get_context_length(self) -> int:
        return len(self._last_context)

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])
        self._last_context = ""
