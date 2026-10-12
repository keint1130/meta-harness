"""Build label-specific positive and competing negative symptom profiles."""
import json,re,math
from collections import Counter,defaultdict
from typing import Any
from ..llm import LLMCallable
from ..memory_system import MemorySystem,extract_json_field

PROMPT="""Compare the query with the category profiles below. Positive cues are recurring details among verified cases; contrast cues are details more characteristic of alternatives. A missing cue is weak evidence, while a direct distinctive match is strong evidence. Use the examples to resolve close profiles, and return one exact label in the requested format.

Profiles:
{profiles}

Problem:
{problem}

Return JSON only: {{"reasoning":"brief rationale","final_answer":"exact label"}}"""
STOP=set("a an and are as at be been but by for from had has have he her his i in into is it its me my of on or our she that the their them there they this to up was we were when which with you your please provide diagnosis symptoms patient problem based following answer final feeling feel really also been having have".split())
def _body(s):
 m=re.search(r"## Patient Symptoms\s*(.*?)(?:\n\nPlease provide|$)",s,re.I|re.S); return m.group(1).strip() if m else s
def _terms(s): return set(w for w in re.findall(r"[a-z0-9]+",_body(s).lower()) if len(w)>2 and w not in STOP)
class ContrastiveLabelProfile(MemorySystem):
 def __init__(self,llm:LLMCallable): super().__init__(llm); self.examples=[]; self.cap=500; self._last=0
 def _profiles(self,query):
  if not self.examples:return "No learned categories."
  docs=[_terms(x.get("raw_question",x["input"])) for x in self.examples]; labels=defaultdict(list)
  for i,x in enumerate(self.examples): labels[x["target"]].append(i)
  vocab=Counter(t for d in docs for t in d); total=len(docs); q=_terms(query); ranked=[]
  for label,inds in labels.items():
   pos=Counter(t for i in inds for t in docs[i]); npos=len(inds)
   neg=Counter(t for i,d in enumerate(docs) if i not in inds for t in d); nneg=max(1,total-npos)
   # Smoothed log odds compare presence rates; keep only query matches or strongly characteristic terms.
   values=[]
   for t in set(pos)|set(neg):
    pp=(sum(t in docs[i] for i in inds)+.5)/(npos+1)
    pn=(sum(t in docs[i] for i in range(total) if i not in inds)+.5)/(nneg+1)
    score=math.log(pp/pn)
    values.append((score,t))
   values.sort(reverse=True)
   present=sorted(((s,t) for s,t in values if t in q),reverse=True)[:3]
   distinct=[t for s,t in values if s>0.45 and t not in q][:4]
   # Choose a query-relevant label exemplar by overlap, with stable index tie breaking.
   best=max(inds,key=lambda i:(len(q&docs[i]),i))
   fit=sum(max(0,s) for s,t in present)-.12*sum(1 for s,t in values if s<-.6 and t in q)
   ranked.append((fit,label,present,distinct,best))
  ranked.sort(key=lambda z:(z[0],z[1]),reverse=True)
  rows=[]
  for score,label,present,distinct,i in ranked[:7]:
   matches=", ".join(t for _,t in present) or "none"
   rows.append(f"Label: {label}; query matches: {matches}; characteristic cues: {', '.join(distinct) or 'few learned cues'}\nVerified example: {_body(self.examples[i].get('raw_question',self.examples[i]['input']))[:200]}")
  return "\n\n".join(rows)
 def predict(self,input:str):
  profile=self._profiles(input); self._last=len(profile); out=self.call_llm(PROMPT.format(profiles=profile,problem=input))
  return extract_json_field(out,"final_answer"),{"num_examples":len(self.examples)}
 def learn_from_batch(self,batch_results):
  for r in batch_results:
   md=r.get("metadata") or {}; self.examples.append({"input":str(r["input"]),"raw_question":str(r.get("raw_question") or md.get("raw_question") or r["input"]),"target":str(r["ground_truth"])})
  self.examples=self.examples[-self.cap:]
 def get_state(self):return json.dumps({"examples":self.examples},ensure_ascii=False)
 def set_state(self,state):self.examples=json.loads(state).get("examples",[])[-self.cap:]
 def get_context_length(self):return self._last
