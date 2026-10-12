"""Retrieve examples using salient symptom phrases and query-term coverage."""
import json, math, re
from collections import Counter
from typing import Any
from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

PROMPT = """Use the retrieved cases as evidence. Match the specific details in the problem, especially distinctive phrases, and compare examples with different labels. These are examples, not an answer key. Follow the requested format and give one exact label.

Retrieved cases:
{cases}

Problem:
{problem}

Return JSON only: {{"reasoning":"brief rationale","final_answer":"exact label"}}"""
STOP=set("a an and are as at be been but by for from had has have he her his i in into is it its me my of on or our she that the their them there they this to up was we were when which with you your please provide diagnosis symptoms patient problem based following answer final feeling feel really also been having have".split())
def _body(s):
    m=re.search(r"## Patient Symptoms\s*(.*?)(?:\n\nPlease provide|$)",s,re.I|re.S)
    return m.group(1).strip() if m else s

def _tokens(s): return re.findall(r"[a-z0-9]+",_body(s).lower())
def _features(s):
    t=[x for x in _tokens(s) if len(x)>1 and x not in STOP]
    # Unigrams preserve broad matches; bigrams retain short symptom phrases.
    return Counter(t + [a+"_"+b for a,b in zip(t,t[1:])])
def _cos(a,b):
    d=sum(v*b.get(k,0) for k,v in a.items()); na=math.sqrt(sum(v*v for v in a.values())); nb=math.sqrt(sum(v*v for v in b.values()))
    return d/(na*nb) if na and nb else 0.0
class PhraseCoverageMemory(MemorySystem):
    def __init__(self,llm:LLMCallable):
        super().__init__(llm); self.examples=[]; self.limit=500; self._last=0
    def _rank(self,query):
        if not self.examples: return []
        docs=[_features(x.get("raw_question",x["input"])) for x in self.examples]
        df=Counter(k for d in docs for k in d)
        n=len(docs); idf={k:max(.15,math.log((n+1)/(v+1))) for k,v in df.items()}
        q=_features(query)
        # Apply corpus rarity to both query and document feature weights.
        q={k:v*idf.get(k,.15) for k,v in q.items()}
        vecs=[{k:v*idf.get(k,.15) for k,v in d.items()} for d in docs]
        scores=[_cos(q,v) for v in vecs]
        ranked=sorted(range(n),key=lambda i:(scores[i],i),reverse=True)
        # Greedily add a case that covers query phrases not already represented.
        chosen=[]; covered=set(); labels=Counter()
        for i in ranked:
            if scores[i]<=0: continue
            label=self.examples[i]["target"]
            if labels[label]>=2: continue
            terms=set(q)&set(vecs[i]); gain=sum(q[t] for t in terms-covered)
            utility=.65*scores[i]+.35*(gain/(sum(q.values()) or 1))
            if not chosen or utility>0:
                chosen.append(i); covered.update(terms); labels[label]+=1
            if len(chosen)>=7: break
        return [self.examples[i] for i in chosen]
    def predict(self,input:str):
        selected=self._rank(input)
        cases="\n\n".join(f"Problem details: {_body(x.get('raw_question',x['input']))[:280]}\nVerified label: {x['target']}" for x in selected) or "No prior cases."
        self._last=len(cases)
        out=self.call_llm(PROMPT.format(cases=cases,problem=input))
        return extract_json_field(out,"final_answer"),{"num_examples":len(self.examples),"num_retrieved":len(selected)}
    def learn_from_batch(self,batch_results):
        for r in batch_results:
            md=r.get("metadata") or {}; self.examples.append({"input":str(r["input"]),"raw_question":str(r.get("raw_question") or md.get("raw_question") or r["input"]),"target":str(r["ground_truth"])})
        self.examples=self.examples[-self.limit:]
    def get_state(self): return json.dumps({"examples":self.examples},ensure_ascii=False)
    def set_state(self,state): self.examples=json.loads(state).get("examples",[])[-self.limit:]
    def get_context_length(self): return self._last
