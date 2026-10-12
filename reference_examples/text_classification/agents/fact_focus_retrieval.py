"""Retrieve demonstrations after removing repeated instruction boilerplate."""
import json, math, re
from collections import Counter
from typing import Any
from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field
RUNS=re.compile(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+")
CJK=re.compile(r"^[\u3400-\u9fff]+$")
PROMPT="""Use the verified examples as analogies, paying attention to the operative facts and requested answer format. Similar wording alone does not establish the same answer.
{examples}
Current problem:
{query}
Return JSON with a concise final_answer. {{"reasoning":"brief","final_answer":"answer"}}"""
def focus(text:str)->str:
    for marker in ("## 案件事实", "案件事实", "事实:"):
        pos=text.find(marker)
        if pos>=0:
            text=text[pos+len(marker):]
            break
    text=re.split(r"\*\*Instructions:\*\*|\nInstructions:",text,maxsplit=1)[0]
    return " ".join(text.split())
def feats(text:str)->Counter:
    out=Counter()
    for run in RUNS.findall(text.lower()):
        if CJK.match(run):
            out.update(run[i:i+2] for i in range(len(run)-1)); out.update(run[i:i+3] for i in range(len(run)-2))
        else: out[run]+=1
    return out
def source(row:dict[str,Any])->str:
    m=row.get("metadata") if isinstance(row.get("metadata"),dict) else {}
    return str(row.get("raw_question") or m.get("raw_question") or m.get("question") or row.get("input", ""))
def clip(s:str,n:int=1050)->str:
    return s if len(s)<=n else s[:740]+" … "+s[-(n-743):]
class Candidate(MemorySystem):
    """Focus retrieval on case facts and greedily diversify query coverage."""
    def __init__(self,llm:LLMCallable):
        super().__init__(llm); self.examples=[]; self.budget=7600; self._last=""
    def _rank(self,q):
        docs=[feats(x['case']) for x in self.examples]; df=Counter()
        for d in docs: df.update(d.keys())
        n=max(1,len(docs)); idf={t:math.log(1+(n+1)/(v+1)) for t,v in df.items()}
        def vec(c):
            z={t:(1+math.log(v))*idf.get(t,0) for t,v in c.items() if v}; norm=math.sqrt(sum(x*x for x in z.values())) or 1
            return {t:x/norm for t,x in z.items()}
        qv=vec(feats(focus(q))); dv=[vec(d) for d in docs]
        scores=[sum(v*x.get(t,0) for t,v in qv.items()) for x in dv]
        overlaps=[set(qv)&set(x) for x in dv]
        return scores,overlaps,set(qv)
    def _render(self,q):
        if not self.examples: self._last=""; return "No learned demonstrations."
        scores,overlaps,terms=self._rank(q); chosen=[]; labels=set(); coverage=set(); parts=[]; used=0
        while len(chosen)<14:
            options=[]
            for i,s in enumerate(scores):
                if i in chosen or s<=0: continue
                gain=len(overlaps[i]-coverage)/max(1,len(terms))
                val=.78*s + (.12 if self.examples[i]['target'] not in labels else 0)+.16*gain
                options.append((val,i))
            if not options: break
            _,i=max(options); ex=self.examples[i]
            block=f"Related case: {clip(ex['case'])}\nVerified answer: {ex['target']}"
            if used+len(block)>self.budget:
                room=self.budget-used-len("Related case: \nVerified answer: ")+len(ex['target'])
                if room<100: break
                block=f"Related case: {clip(ex['case'],room)}\nVerified answer: {ex['target']}"
            parts.append(block); used+=len(block)+2; chosen.append(i); labels.add(ex['target']); coverage|=overlaps[i]
        self._last="\n\n".join(parts); return self._last or "No overlapping cases."
    def predict(self,input:str):
        ex=self._render(input); r=self.call_llm(PROMPT.format(examples=ex,query=input))
        return extract_json_field(r,"final_answer"),{"full_response":r,"retrieved_examples":len(self._last.split("\n\n")) if self._last else 0}
    def learn_from_batch(self,batch_results):
        seen={(x['case'],x['target']) for x in self.examples}
        for row in batch_results:
            c=focus(source(row)); t=str(row['ground_truth'])
            if c and (c,t) not in seen: self.examples.append({'case':c,'target':t}); seen.add((c,t))
    def get_context_length(self): return len(self._last)
    def get_state(self): return json.dumps({'examples':self.examples},ensure_ascii=False)
    def set_state(self,state): self.examples=json.loads(state).get('examples',[]); self._last=''
