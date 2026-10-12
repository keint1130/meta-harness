"""Maintain a compact, incrementally updated profile for each output label."""

import json
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


LEARNING_PROMPT = """Update the compact profile for one correct output using a new labeled example.

Preserve recurring evidence and constraints already supported by the previous
profile. Add only information from the new example that could help distinguish
this output in a future problem. Do not infer unsupported rules or copy
incidental identifiers. Keep the profile to at most three short sentences.
Return JSON: {{"profile": "..."}}

Correct output: {label}
Previous profile: {previous}
New input: {input}
"""

PREDICT_PROMPT = """Answer the problem by comparing it with the learned output profiles.

{profiles_section}

Profiles summarize examples associated with each output. They are evidence,
not rules. Compare the complete input with plausible alternatives, and use the
required answer format from the problem. Return JSON:
{{"reasoning": "[brief explanation]", "final_answer": "[your answer]"}}

Problem:
{input}
"""


class LabelPrototypeMemory(MemorySystem):
    """Compress examples into a semantic prototype maintained per output."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.profiles: dict[str, dict[str, Any]] = {}

    def _update_profile(self, label: str, text: str) -> None:
        current = self.profiles.get(label, {"profile": "", "support": 0})
        prompt = LEARNING_PROMPT.format(
            label=label,
            previous=current["profile"] or "None yet",
            input=text,
        )
        response = self.call_llm(prompt)
        profile = extract_json_field(response, "profile").strip()
        if not profile:
            profile = text[:600]
        self.profiles[label] = {
            "profile": profile[:1200],
            "support": int(current["support"]) + 1,
        }

    def _format_profiles(self) -> str:
        if not self.profiles:
            return "No labeled output profiles have been learned yet."
        rows = ["Learned output profiles:"]
        for label in sorted(self.profiles, key=str.casefold):
            record = self.profiles[label]
            rows.append(
                f"- Output: {label} (examples: {record['support']})\n"
                f"  Profile: {record['profile']}"
            )
        return "\n".join(rows)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        profiles_section = self._format_profiles()
        response = self.call_llm(
            PREDICT_PROMPT.format(profiles_section=profiles_section, input=input)
        )
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "num_profiles": len(self.profiles),
            "profile_chars": len(profiles_section),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            label = str(result["ground_truth"])
            text = str(result.get("raw_question") or result["input"])
            self._update_profile(label, text)

    def get_context_length(self) -> int:
        return len(self._format_profiles())

    def get_state(self) -> str:
        return json.dumps({"profiles": self.profiles}, ensure_ascii=False, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.profiles = data.get("profiles", {})
