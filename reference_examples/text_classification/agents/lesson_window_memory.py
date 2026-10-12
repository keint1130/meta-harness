"""Consolidate labeled feedback into periodic LLM-authored lessons."""

import json
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


LESSON_WINDOW = 8
RECENT_CASES = 4
MAX_LESSONS = 16

LEARNING_PROMPT = """Review these problem, attempt, and correct-output records.

Find one concise lesson that could transfer to future problems. Give priority
to distinctions shown by incorrect attempts. Avoid memorizing individual
answers or asserting a rule that the records do not support. Return JSON:
{{"lesson": "[one concise transferable lesson]"}}

Records:
{records}
"""

PREDICT_PROMPT = """Solve the problem using relevant learned lessons and recent examples.

{memory_section}

**Problem:**
{input}

Lessons summarize patterns from earlier feedback. Check that each applies to
this problem before using it. Treat individual examples as evidence, not rules.
Respond in JSON:
{{"reasoning": "[brief explanation]", "final_answer": "[your answer]"}}"""


class LessonWindowMemory(MemorySystem):
    """Replace completed feedback windows with short learned lessons."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.lessons: list[str] = []
        self.pending: list[dict[str, Any]] = []
        self.recent: list[dict[str, str]] = []

    def _distill(self, records: list[dict[str, Any]]) -> str:
        prompt = LEARNING_PROMPT.format(
            records=json.dumps(records, ensure_ascii=False, indent=2)
        )
        response = self.call_llm(prompt)
        lesson = extract_json_field(response, "lesson").strip()
        if not lesson:
            lesson = response.strip()
        return lesson[:1000]

    def _format_memory(self) -> str:
        sections = []
        if self.lessons:
            sections.append(
                "Lessons learned from earlier feedback:\n"
                + "\n".join(f"- {lesson}" for lesson in self.lessons)
            )

        unfinished = self.pending[:-RECENT_CASES] if len(self.pending) > RECENT_CASES else []
        cases = [
            {"input": item["input"], "target": item["ground_truth"]}
            for item in unfinished
        ]
        cases.extend(self.recent[-RECENT_CASES:])
        if cases:
            lines = ["Recent or not-yet-consolidated examples:"]
            for case in cases:
                lines.append(f"- Input: {case['input']}\n  Correct output: {case['target']}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections) if sections else "No feedback has been learned yet."

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        memory_section = self._format_memory()
        prompt = PREDICT_PROMPT.format(memory_section=memory_section, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_lessons": len(self.lessons),
            "num_pending": len(self.pending),
            "num_recent_cases": min(len(self.recent), RECENT_CASES),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            item = {
                "input": str(result.get("raw_question") or result["input"]),
                "prediction": str(result["prediction"]),
                "ground_truth": str(result["ground_truth"]),
                "was_correct": bool(result["was_correct"]),
            }
            self.pending.append(item)
            self.recent.append({"input": item["input"], "target": item["ground_truth"]})
            self.recent = self.recent[-RECENT_CASES:]

            while len(self.pending) >= LESSON_WINDOW:
                window = self.pending[:LESSON_WINDOW]
                lesson = self._distill(window)
                if lesson:
                    self.lessons.append(lesson)
                    self.lessons = self.lessons[-MAX_LESSONS:]
                self.pending = self.pending[LESSON_WINDOW:]

    def get_context_length(self) -> int:
        return len(self._format_memory())

    def get_state(self) -> str:
        return json.dumps(
            {"lessons": self.lessons, "pending": self.pending, "recent": self.recent},
            ensure_ascii=False,
            indent=2,
        )

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.lessons = data.get("lessons", [])
        self.pending = data.get("pending", [])
        self.recent = data.get("recent", [])

