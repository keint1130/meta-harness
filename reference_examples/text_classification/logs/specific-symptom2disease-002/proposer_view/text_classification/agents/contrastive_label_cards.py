"""Represent each label by cues that distinguish it from the other labels."""
import json
import math
import re
from collections import Counter, defaultdict

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field


PROMPT = """Each label card summarizes evidence that distinguishes that label from others.
Treat individual cue matches as weak evidence; prefer several matching cues that fit together.
Use the example only as context and return one exact label.

Label cards:
{cards}

Problem:
{problem}

Return JSON only: {{\"reasoning\":\"brief rationale\",\"final_answer\":\"exact label\"}}"""
STOP = set("a an and are as at be been but by for from had has have he her his i in into is it its me my of on or our she that the their them there they this to up was we were when which with you your please provide diagnosis symptoms patient problem following answer final".split())


def body(text):
    m = re.search(r"## Patient Symptoms\s*(.*?)(?:\n\nPlease provide|$)", str(text), re.I | re.S)
    return m.group(1).strip() if m else str(text)


def terms(text):
    return [w for w in re.findall(r"[a-z0-9]+", body(text).lower()) if len(w) > 2 and w not in STOP]


class Candidate(MemorySystem):
    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.examples = []
        self._last = 0

    def _cards(self, query):
        groups = defaultdict(list)
        for ex in self.examples:
            groups[ex["target"]].append(ex)
        if not groups:
            return "No label cards have been learned."
        docs = [set(terms(x.get("raw_question", x["input"]))) for x in self.examples]
        df = Counter(t for d in docs for t in d)
        total = len(docs)
        q = set(terms(query))
        global_rate = {t: (c + .5) / (total + 1) for t, c in df.items()}
        cards = []
        for label, examples in groups.items():
            own = [set(terms(x.get("raw_question", x["input"]))) for x in examples]
            own_df = Counter(t for d in own for t in d)
            scored = []
            for cue in q & own_df.keys():
                rate = (own_df[cue] + .5) / (len(own) + 1)
                # Smoothed enrichment rewards cues common within a label and rare overall.
                weight = math.log(rate / global_rate.get(cue, .5 / (total + 1)))
                if weight > 0:
                    scored.append((weight, cue, own_df[cue]))
            scored.sort(reverse=True)
            selected = scored[:5]
            if not selected:
                continue
            exemplar = max(examples, key=lambda x: len(q & set(terms(x.get("raw_question", x["input"])))))
            cues = ", ".join(f"{t} ({c}/{len(examples)})" for _, t, c in selected)
            snippet = " ".join(body(exemplar.get("raw_question", exemplar["input"])).split())[:190]
            cards.append((sum(x[0] for x in selected), f"Label: {label}\nQuery-matched cues: {cues}\nVerified example: {snippet}"))
        cards.sort(reverse=True)
        return "\n\n".join(x[1] for x in cards[:8]) or "No learned cue matches this problem. Compare labels from the problem directly."

    def predict(self, input: str):
        cards = self._cards(input)
        self._last = len(cards)
        out = self.call_llm(PROMPT.format(cards=cards, problem=input))
        return extract_json_field(out, "final_answer"), {"num_labels": len({x['target'] for x in self.examples})}

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
