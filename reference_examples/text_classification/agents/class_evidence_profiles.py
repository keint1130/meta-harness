"""Compressed class-wise lexical evidence learned from labeled examples."""

import json
import math
import re
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

PROMPT_TEMPLATE = """Answer the problem using the learned class evidence profiles.

{profiles_section}

**Problem:**
{input}

The listed words are recurring clues from labeled examples, not rules. Compare
the complete meaning of the problem with every plausible profile, including
similar alternatives. Answer in JSON:
{{"reasoning": "[brief comparison]", "final_answer": "[your answer]"}}"""

PROFILE_CUES = 5


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]{2,}", text.lower())


class ClassEvidenceProfiles(MemorySystem):
    """Keep sufficient per-label token counts instead of storing raw cases."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.class_counts: dict[str, Counter[str]] = {}
        self.class_docs: Counter[str] = Counter()
        self.global_counts: Counter[str] = Counter()

    def _profile_rows(self) -> list[tuple[str, int, list[str]]]:
        vocabulary_size = max(1, len(self.global_counts))
        global_total = sum(self.global_counts.values())
        rows = []
        for label in sorted(self.class_counts):
            counts = self.class_counts[label]
            class_total = sum(counts.values())
            scored = []
            for token, count in counts.items():
                class_rate = (count + 0.5) / (class_total + 0.5 * vocabulary_size)
                global_rate = (self.global_counts[token] + 0.5) / (
                    global_total + 0.5 * vocabulary_size
                )
                score = math.log(class_rate / global_rate)
                scored.append((score, count, token))
            cues = [
                token
                for score, _, token in sorted(scored, reverse=True)
                if score > 0
            ][:PROFILE_CUES]
            if not cues:
                cues = [token for token, _ in counts.most_common(PROFILE_CUES)]
            rows.append((label, self.class_docs[label], cues))
        return rows

    def _format_profiles_section(self) -> str:
        rows = self._profile_rows()
        if not rows:
            return "No labeled evidence profiles have been learned yet."
        lines = ["Learned label profiles (label, number of cases, recurring cues):"]
        for label, support, cues in rows:
            cue_text = ", ".join(cues) if cues else "no recurring cues yet"
            lines.append(f"- {label} | cases: {support} | cues: {cue_text}")
        return "\n".join(lines)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        profiles_section = self._format_profiles_section()
        prompt = PROMPT_TEMPLATE.format(profiles_section=profiles_section, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_profiles": len(self.class_counts),
            "profile_chars": len(profiles_section),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            label = str(result["ground_truth"])
            text = result.get("raw_question") or result["input"]
            tokens = Counter(_tokens(text))
            if not tokens:
                continue
            self.class_counts.setdefault(label, Counter()).update(tokens)
            self.class_docs[label] += 1
            self.global_counts.update(tokens)

    def get_context_length(self) -> int:
        return len(self._format_profiles_section())

    def get_state(self) -> str:
        return json.dumps(
            {
                "class_counts": {
                    label: dict(counts) for label, counts in self.class_counts.items()
                },
                "class_docs": dict(self.class_docs),
                "global_counts": dict(self.global_counts),
            },
            indent=2,
        )

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.class_counts = {
            label: Counter(counts)
            for label, counts in data.get("class_counts", {}).items()
        }
        self.class_docs = Counter(data.get("class_docs", {}))
        self.global_counts = Counter(data.get("global_counts", {}))
