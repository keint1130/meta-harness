"""Distill feature-conditioned corrections from mistakes and retrieve them by cue overlap."""

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT = """Use a correction only when its listed cues actually appear in the current problem.
The verified label is authoritative for that earlier case; the old model output records what
was corrected. Also compare the nearby verified examples. Do not transfer a correction when its
trigger details are absent. Follow the requested answer format and return one exact label.

Matched correction rules:
{rules}

Nearby verified examples:
{examples}

Problem:
{input}

Return JSON only: {{\"reasoning\": \"brief rationale\", \"final_answer\": \"exact label\"}}"""

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "i", "in", "into", "is", "it", "its",
    "me", "my", "of", "on", "or", "our", "she", "that", "the", "their", "them", "there",
    "they", "this", "to", "up", "was", "we", "were", "when", "which", "with", "you", "your",
    "please", "provide", "answer", "problem", "question", "label", "format", "final", "based",
    "following", "example", "examples",
}


def _edges(texts: list[str]) -> tuple[str, str]:
    if not texts:
        return "", ""
    prefix = suffix = texts[0]
    for text in texts[1:]:
        n = min(len(prefix), len(text))
        i = 0
        while i < n and prefix[i] == text[i]:
            i += 1
        prefix = prefix[:i]
        n = min(len(suffix), len(text))
        i = 0
        while i < n and suffix[-i - 1] == text[-i - 1]:
            i += 1
        suffix = suffix[-i:] if i else ""
    return (prefix if len(prefix) >= 32 else "", suffix if len(suffix) >= 32 else "")


def _body(text: str, edges: tuple[str, str]) -> str:
    prefix, suffix = edges
    if prefix and text.startswith(prefix):
        text = text[len(prefix):]
    if suffix and text.endswith(suffix):
        text = text[:-len(suffix)]
    return text.strip()


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in _STOP]


def _canonical_label(answer: str, labels: set[str]) -> str:
    cleaned = re.sub(r"\[[^\]]+\]", " ", str(answer)).lower()
    cleaned = " ".join(re.findall(r"[a-z0-9]+", cleaned))
    matches = [label for label in labels if re.search(r"\b" + re.escape(label.lower()) + r"\b", cleaned)]
    return max(matches, key=len) if len(matches) == 1 else ""


def _cosine(a: Counter[str], b: Counter[str], idf: dict[str, float]) -> float:
    dot = sum(a[t] * b.get(t, 0) * idf.get(t, 0.0) ** 2 for t in a)
    na = math.sqrt(sum(v * v * idf.get(t, 0.0) ** 2 for t, v in a.items()))
    nb = math.sqrt(sum(v * v * idf.get(t, 0.0) ** 2 for t, v in b.items()))
    return dot / (na * nb) if na and nb else 0.0


class CueConditionedCorrection(MemorySystem):
    """Learn mistake-specific discriminative cues alongside verified case memory."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples: list[dict[str, Any]] = []
        self.rules: list[dict[str, Any]] = []
        self.max_rules = 200
        self.max_examples = 5
        self._last_context_chars = 0

    def _rebuild_rules(self) -> None:
        if not self.examples:
            self.rules = []
            return
        raw = [str(ex.get("raw_question", ex["input"])) for ex in self.examples]
        edges = _edges(raw)
        docs = [_tokens(_body(text, edges)) for text in raw]
        labels = {str(ex["target"]) for ex in self.examples}
        label_docs: dict[str, list[int]] = defaultdict(list)
        for i, ex in enumerate(self.examples):
            label_docs[str(ex["target"])].append(i)
        doc_freq = {label: Counter(t for i in indexes for t in set(docs[i])) for label, indexes in label_docs.items()}

        built = []
        for i, ex in enumerate(self.examples):
            if ex.get("was_correct", False):
                continue
            target = str(ex["target"])
            source_label = _canonical_label(str(ex.get("prediction", "")), labels)
            query_terms = set(docs[i])
            contrasts = []
            for term in query_terms:
                target_n = len(label_docs[target])
                target_yes = doc_freq[target][term]
                p_target = (target_yes + 0.5) / (target_n + 1.0)
                if source_label and source_label != target and source_label in label_docs:
                    source_n = len(label_docs[source_label])
                    source_yes = doc_freq[source_label][term]
                else:
                    other_indexes = [j for j, item in enumerate(self.examples) if str(item["target"]) != target]
                    source_n = len(other_indexes)
                    source_yes = sum(term in set(docs[j]) for j in other_indexes)
                p_source = (source_yes + 0.5) / (source_n + 1.0)
                contrast = math.log(p_target / (1.0 - p_target)) - math.log(p_source / (1.0 - p_source))
                contrasts.append((contrast, term))
            contrasts.sort(reverse=True)
            cues = [term for score, term in contrasts if score > 0.02][:4]
            if not cues:
                cues = [term for _, term in sorted(contrasts, key=lambda row: (-row[0], row[1]))[:2]]
            if not cues:
                continue
            built.append({
                "target": target,
                "source": str(ex.get("prediction", ""))[:120],
                "source_label": source_label,
                "cues": cues,
                "strength": round(sum(score for score, term in contrasts if term in cues), 3),
                "raw_question": raw[i],
            })
        # Keep the most cue-specific rules while preserving deterministic order on ties.
        built.sort(key=lambda rule: (rule["strength"], len(rule["cues"])), reverse=True)
        self.rules = built[: self.max_rules]

    def _retrieve(self, input: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not self.examples:
            return [], []
        raw = [str(ex.get("raw_question", ex["input"])) for ex in self.examples]
        edges = _edges(raw + [input])
        query = set(_tokens(_body(input, edges)))
        matches = []
        for index, rule in enumerate(self.rules):
            overlap = query & set(rule.get("cues", []))
            if overlap:
                matches.append((len(overlap), float(rule.get("strength", 0)), index, rule))
        matches.sort(reverse=True, key=lambda row: (row[0], row[1], row[2]))

        docs = [_tokens(_body(text, edges)) for text in raw]
        df = Counter(t for doc in docs for t in set(doc))
        idf = {t: max(0.15, math.log((len(docs) + 1) / (freq + 1))) for t, freq in df.items()}
        qvec = Counter(_tokens(_body(input, edges)))
        ranked = sorted(
            ((_cosine(qvec, Counter(doc), idf), i) for i, doc in enumerate(docs)),
            reverse=True,
        )
        selected = [self.examples[i] for score, i in ranked if score > 0][: self.max_examples]
        return [row[3] for row in matches[:5]], selected

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        matched, selected = self._retrieve(input)
        rule_text = "\n".join(
            f"Cues present: {', '.join(rule['cues'])}; earlier output: {rule['source']}; verified label: {rule['target']}"
            for rule in matched
        ) or "No learned correction cue matches this problem."
        example_text = "\n\n".join(
            f"Problem details: {_body(str(ex.get('raw_question', ex['input'])), _edges([str(item.get('raw_question', item['input'])) for item in self.examples] + [input])).replace(chr(10), ' ')[:220]}\n"
            f"Earlier output: {ex.get('prediction', '(none)')}\nVerified label: {ex['target']}"
            for ex in selected
        ) or "No nearby verified examples have been stored yet."
        self._last_context_chars = len(rule_text) + len(example_text)
        response = self.call_llm(PROMPT.format(rules=rule_text, examples=example_text, input=input))
        return extract_json_field(response, "final_answer"), {
            "num_examples": len(self.examples),
            "num_matched_rules": len(matched),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            metadata = result.get("metadata") or {}
            raw = result.get("raw_question") or metadata.get("raw_question") or result["input"]
            self.examples.append({
                "input": str(result["input"]),
                "raw_question": str(raw),
                "target": str(result["ground_truth"]),
                "prediction": str(result.get("prediction", "")),
                "was_correct": bool(result.get("was_correct", False)),
            })
        self.examples = self.examples[-500:]
        self._rebuild_rules()

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])[-500:]
        self._rebuild_rules()

    def get_context_length(self) -> int:
        return self._last_context_chars
