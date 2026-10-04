from __future__ import annotations

import re
import string
from collections.abc import Iterable
from dataclasses import dataclass

from .contracts import TaskFamily


@dataclass(frozen=True)
class Score:
    correct: bool
    value: float
    detail: str = ""


def normalize_qa_answer(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def score_popqa(prediction: str, aliases: Iterable[str]) -> Score:
    normalized_prediction = normalize_qa_answer(prediction)
    normalized_aliases = {
        normalize_qa_answer(alias) for alias in aliases if str(alias).strip()
    }
    matched = any(
        alias == normalized_prediction
        or re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized_prediction)
        for alias in normalized_aliases
    )
    return Score(matched, float(matched), "alias_match" if matched else "no_alias_match")


def extract_boxed_answer(text: str) -> str:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return text.strip()
    depth = 1
    chars: list[str] = []
    for char in text[start + len(marker) :]:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
        chars.append(char)
    return "".join(chars).strip()


def normalize_math_answer(text: str) -> str:
    text = extract_boxed_answer(text)
    text = text.replace("$", "").replace("\\,", "").replace("\\!", "")
    text = re.sub(r"\s+", "", text)
    # MATH-500 sometimes formats large integers as ``11,\! 111``. Remove
    # thousands separators only when the entire answer is numeric, so commas
    # in tuples, sets, and expressions retain their meaning.
    if re.fullmatch(r"[+-]?[\d,]+(?:\.\d+)?", text):
        text = text.replace(",", "")
    return text


def score_math(prediction: str, reference: str) -> Score:
    """Grade MATH answers using Math-Verify, with exact match as a fallback."""
    pred = normalize_math_answer(prediction)
    gold = normalize_math_answer(reference)
    if pred == gold:
        return Score(True, 1.0, "normalized_exact")
    try:
        from math_verify import parse, verify

        # Math-Verify's internal timeout uses multiprocessing on Windows and is
        # unreliable under captured subprocesses. The benchmark worker supplies
        # the outer wall-time limit instead.
        # Raw MATH-500 references such as ``2\sqrt{113}`` can be misread as
        # just their first/last number. Give gold and prediction the same
        # explicit final-answer anchor before symbolic parsing.
        gold_parsed = parse(r"\boxed{" + gold + "}", parsing_timeout=None)
        prediction_parsed = parse(r"\boxed{" + pred + "}", parsing_timeout=None)
        correct = bool(verify(gold_parsed, prediction_parsed, timeout_seconds=None))
        return Score(correct, float(correct), "math_verify_boxed")
    except Exception as exc:
        correct = pred == gold
        return Score(
            correct,
            float(correct),
            f"normalized_fallback={pred!r}; error={type(exc).__name__}",
        )


def macro_average(scores: dict[TaskFamily, list[Score]]) -> float:
    family_means = []
    for family in TaskFamily:
        values = scores.get(family, [])
        if not values:
            raise ValueError(f"Missing scores for task family {family.value}")
        family_means.append(sum(score.value for score in values) / len(values))
    return sum(family_means) / len(family_means)
