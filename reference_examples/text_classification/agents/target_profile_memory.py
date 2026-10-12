"""Learn compact answer-associated term profiles from labeled examples."""

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_PROMPT = """The memory includes compact patterns linking recurring problem terms
to expected answers, followed by nearby labeled examples. Treat the patterns as
evidence rather than rules; compare the complete current problem and obey its
instructions.

{profiles}

{examples}

**Current problem and instructions:**
{input}

Return one JSON object with "reasoning" and "final_answer" fields. Follow the
current problem's requested answer format exactly.
"""


def _question(result: dict[str, Any]) -> str:
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return str(result.get("raw_question") or metadata.get("raw_question") or result.get("input", ""))


def _tokens(text: str) -> Counter[str]:
    return Counter(token.lower() for token in _TOKEN_RE.findall(text) if len(token) > 2)


class TargetProfileMemory(MemorySystem):
    """Represent repeated targets by their distinctive vocabulary, then retrieve cases."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, str]] = []
        self._last_context_chars = 0

    def _profiles(self, query: str) -> str:
        if not self.examples:
            return "No learned answer profiles yet."
        label_counts: Counter[str] = Counter()
        label_terms: dict[str, Counter[str]] = defaultdict(Counter)
        global_df: Counter[str] = Counter()
        docs: list[Counter[str]] = []
        for item in self.examples:
            terms = _tokens(item["question"])
            docs.append(terms)
            global_df.update(terms.keys())
            label_counts[item["target"]] += 1
            label_terms[item["target"]].update(terms)
        qterms = _tokens(query)
        ranked = []
        n = len(self.examples)
        for target, counts in label_terms.items():
            distinctive = []
            for term, qcount in qterms.items():
                if counts[term] == 0:
                    continue
                within = (counts[term] + 0.5) / (sum(counts.values()) + 0.5 * max(1, len(counts)))
                overall = (global_df[term] + 1) / (sum(global_df.values()) + max(1, len(global_df)))
                lift = math.log(within / overall)
                if lift > 0:
                    distinctive.append((lift * min(qcount, 2), term))
            score = sum(value for value, _ in distinctive)
            ranked.append((score, target, sorted(distinctive, reverse=True)[:4]))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        lines = []
        for score, target, terms in ranked[:5]:
            if not terms:
                continue
            evidence = ", ".join(f"{term} ({value:.1f})" for value, term in terms)
            lines.append(f"Answer pattern: {target} | support {label_counts[target]} | matching distinctive terms: {evidence}")
        return "\n".join(lines) or "No recurring answer profile matched the current problem."

    def _examples(self, query: str) -> str:
        if not self.examples:
            return "No labeled examples yet."
        q = _tokens(query)
        docs = [_tokens(item["question"]) for item in self.examples]
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(doc.keys())
        def score(doc: Counter[str]) -> float:
            dot = sum(q[t] * doc[t] * (1 + math.log((len(docs) + 1) / (df[t] + 1))) for t in q.keys() & doc.keys())
            return dot / math.sqrt((sum(q.values()) + 1) * (sum(doc.values()) + 1))
        chosen = sorted(zip(self.examples, docs), key=lambda pair: -score(pair[1]))[:3]
        return "\n\n".join(f"Reference {i}\nProblem: {item['question']}\nExpected answer: {item['target']}" for i, (item, _) in enumerate(chosen, 1)) or "No labeled examples yet."

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        profiles = self._profiles(input)
        examples = self._examples(input)
        self._last_context_chars = len(profiles) + len(examples)
        response = self.call_llm(_PROMPT.format(profiles=profiles, examples=examples, input=input))
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "num_examples": examples.count("Reference "),
            "num_profiles": profiles.count("Answer pattern:"),
        }

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
        return self._last_context_chars or sum(len(x["target"]) + 24 for x in self.examples[-3:])

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = [{"question": str(x.get("question", "")), "target": str(x.get("target", ""))} for x in data.get("examples", [])][-512:]
        self._last_context_chars = 0
