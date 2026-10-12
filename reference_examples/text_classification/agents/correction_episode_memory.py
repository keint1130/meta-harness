"""Retrieve compact error-and-correction episodes with nearby verified anchors."""
import json
import math
import re
from collections import Counter

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT = """Use verified examples as calibration. In a correction episode, the recorded
answer was wrong and the verified label is the correction; treat it as a contrast, not a
universal rule. Check whether its details actually match the current problem. Follow the
requested format and return one exact label.

Relevant correction episodes:
{episodes}

Nearby verified examples:
{anchors}

Problem:
{problem}

Return JSON only: {{\"reasoning\":\"brief rationale\",\"final_answer\":\"exact label\"}}"""
STOP = set("a an and are as at be been but by for from had has have he her his i in into is it its me my of on or our she that the their them there they this to up was we were when which with you your please provide diagnosis symptoms patient problem based following answer final".split())


def body(text):
    m = re.search(r"## Patient Symptoms\s*(.*?)(?:\n\nPlease provide|$)", str(text), re.I | re.S)
    return m.group(1).strip() if m else str(text)


def terms(text):
    return set(w for w in re.findall(r"[a-z0-9]+", body(text).lower()) if len(w) > 2 and w not in STOP)


def short(text, n):
    return " ".join(str(text).split())[:n]


def answer_label(prediction, labels):
    text = " ".join(re.findall(r"[a-z0-9]+", str(prediction).lower()))
    hits = [label for label in labels if re.search(r"\b" + re.escape(" ".join(re.findall(r"[a-z0-9]+", label.lower()))) + r"\b", text)]
    return max(hits, key=len) if hits else str(prediction)


class Candidate(MemorySystem):
    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples = []
        self._last = 0

    def _evidence(self, query):
        q = terms(query)
        raw = [terms(x.get("raw_question", x["input"])) for x in self.examples]
        df = Counter(t for d in raw for t in d)
        n = max(1, len(raw))
        idf = {t: math.log((n + 1) / (c + 1)) + .2 for t, c in df.items()}

        def sim(doc):
            common = q & doc
            numerator = sum(idf.get(t, .2) ** 2 for t in common)
            denom = math.sqrt(sum(idf.get(t, .2) ** 2 for t in q) * sum(idf.get(t, .2) ** 2 for t in doc)) or 1
            return numerator / denom

        mistakes = [(sim(doc), i, x) for i, (doc, x) in enumerate(zip(raw, self.examples)) if not x["was_correct"] and x.get("prediction")]
        mistakes.sort(reverse=True, key=lambda z: (z[0], z[1]))
        selected, pairs = [], set()
        labels = sorted({x["target"] for x in self.examples}, key=len, reverse=True)
        for score, _, x in mistakes:
            pair = (answer_label(x["prediction"], labels), x["target"])
            if score <= 0 or pair in pairs:
                continue
            overlap = sorted(q & terms(x.get("raw_question", x["input"])), key=lambda t: (-idf.get(t, .2), t))[:4]
            selected.append((x, overlap))
            pairs.add(pair)
            if len(selected) == 3:
                break
        episode_text = "\n\n".join(
            f"Similar problem: {short(body(x.get('raw_question', x['input'])), 240)}\n"
            f"Wrong recorded answer: {x['prediction']}\nVerified correction: {x['target']}\n"
            f"Details shared with current problem: {', '.join(overlap) or 'few'}"
            for x, overlap in selected
        ) or "No similar correction episodes."

        anchors = []
        counts = Counter()
        verified = [(sim(doc), x) for doc, x in zip(raw, self.examples) if x["was_correct"]]
        verified.sort(reverse=True, key=lambda z: z[0])
        for score, x in verified:
            if score <= 0 or counts[x["target"]] >= 2:
                continue
            anchors.append(f"Problem: {short(body(x.get('raw_question', x['input'])), 210)}\nVerified label: {x['target']}")
            counts[x["target"]] += 1
            if len(anchors) == 4:
                break
        anchor_text = "\n\n".join(anchors) or "No nearby verified cases."
        return episode_text, anchor_text, len(selected), len(anchors)

    def predict(self, input: str):
        episodes, anchors, ne, na = self._evidence(input)
        self._last = len(episodes) + len(anchors)
        out = self.call_llm(PROMPT.format(episodes=episodes, anchors=anchors, problem=input))
        return extract_json_field(out, "final_answer"), {"num_episodes": ne, "num_anchors": na}

    def learn_from_batch(self, batch_results):
        for r in batch_results:
            md = r.get("metadata") or {}
            self.examples.append({"input": str(r["input"]), "raw_question": str(r.get("raw_question") or md.get("raw_question") or r["input"]), "prediction": str(r.get("prediction", "")), "target": str(r["ground_truth"]), "was_correct": bool(r.get("was_correct", False))})
        self.examples = self.examples[-500:]

    def get_state(self):
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state):
        self.examples = json.loads(state).get("examples", [])[-500:]

    def get_context_length(self):
        return self._last
