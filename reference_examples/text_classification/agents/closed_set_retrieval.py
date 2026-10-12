"""Retrieve examples, then normalize the response against the problem's label set."""
import json
import math
import re
from collections import Counter

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT = """Use nearby verified examples as evidence, but decide from the current problem.
Return exactly one label from the allowed labels when that list is present. Follow any
requested output format only inside the JSON field.

Nearby examples:
{cases}

Problem:
{problem}

Return JSON only: {{\"reasoning\":\"brief rationale\",\"final_answer\":\"one exact label\"}}"""
STOP = set("a an and are as at be been but by for from had has have he her his i in into is it its me my of on or our she that the their them there they this to up was we were when which with you your please provide diagnosis symptoms patient problem based following answer final feeling really also".split())


def tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if len(w) > 2 and w not in STOP]


def labels_in(text):
    match = re.search(r"(?:possible\s+(?:diagnoses|labels|categories|classes|answers)|allowed\s+labels)\s+include\s*:\s*(.*?)(?:\n\n|\nPlease|$)", str(text), re.I | re.S)
    if not match:
        return []
    return [x.strip(" .\t\n") for x in match.group(1).split(",") if x.strip(" .\t\n")]


def canonical(answer, labels):
    norm = " ".join(re.findall(r"[a-z0-9]+", str(answer).lower()))
    hits = [label for label in labels if re.search(r"\b" + re.escape(" ".join(re.findall(r"[a-z0-9]+", label.lower()))) + r"\b", norm)]
    return max(hits, key=len) if hits else answer


class Candidate(MemorySystem):
    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples = []
        self._last = 0

    def predict(self, input: str):
        q = Counter(tokens(input))
        docs = [Counter(tokens(x.get("raw_question", x["input"]))) for x in self.examples]
        df = Counter(t for d in docs for t in d)
        n = len(docs)
        idf = {t: math.log((n + 1) / (c + 1)) + .2 for t, c in df.items()}
        scored = []
        for i, (ex, doc) in enumerate(zip(self.examples, docs)):
            overlap = sum(min(q[t], doc[t]) * idf.get(t, .2) for t in q.keys() & doc.keys())
            denom = math.sqrt(sum(v * v for v in q.values()) * sum(v * v for v in doc.values())) or 1
            scored.append((overlap / denom, i, ex))
        scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
        selected, counts = [], Counter()
        for score, _, ex in scored:
            if score <= 0 or counts[ex["target"]] >= 2:
                continue
            selected.append(ex)
            counts[ex["target"]] += 1
            if len(selected) == 6:
                break
        cases = "\n\n".join(
            f"Problem: {re.sub(r'\s+', ' ', x.get('raw_question', x['input']))[-500:]}\nVerified label: {x['target']}"
            for x in selected
        ) or "No prior verified examples."
        self._last = len(cases)
        raw = extract_json_field(self.call_llm(PROMPT.format(cases=cases, problem=input)), "final_answer")
        allowed = labels_in(input)
        return canonical(raw, allowed), {"num_retrieved": len(selected), "label_set_size": len(allowed)}

    def learn_from_batch(self, batch_results):
        for r in batch_results:
            md = r.get("metadata") or {}
            self.examples.append({"input": str(r["input"]), "raw_question": str(r.get("raw_question") or md.get("raw_question") or r["input"]), "target": str(r["ground_truth"])})
        self.examples = self.examples[-500:]

    def get_state(self):
        return json.dumps({"examples": self.examples}, ensure_ascii=False)

    def set_state(self, state):
        self.examples = json.loads(state).get("examples", [])[-500:]

    def get_context_length(self):
        return self._last
