"""Learn pairwise decision rules from recurring confusion boundaries."""
import json,re,math
from collections import Counter,defaultdict
from typing import Any
from ..llm import LLMCallable
from ..memory_system import MemorySystem,extract_json_field

PROMPT="""Use the verified examples and learned confusion rules as calibration. A rule applies only when its trigger details match; it says which label to reconsider and which label corrected it. Compare both labels against the current details before deciding. Follow the requested format and return one exact label.

Relevant correction rules:
{rules}

Nearby verified cases:
{cases}

Problem:
{problem}

Return JSON only: {{"reasoning":"brief rationale","final_answer":"exact label"}}"""
STOP=set("a an and are as at be been but by for from had has have he her his i in into is it its me my of on or our she that the their them there they this to up was we were when which with you your please provide diagnosis symptoms patient problem based following answer final feeling feel really also been having have".split())
def _body(s):
 m=re.search(r"## Patient Symptoms\s*(.*?)(?:\n\nPlease provide|$)",s,re.I|re.S); return m.group(1).strip() if m else s
def _tokens(s):return [w for w in re.findall(r"[a-z0-9]+",_body(s).lower()) if len(w)>2 and w not in STOP]
def _label(answer,labels):
 s=" ".join(re.findall(r"[a-z0-9]+",str(answer).lower())); hits=[x for x in labels if re.search(r"\b"+re.escape(x.lower())+r"\b",s)]; return max(hits,key=len) if hits else ""
class PairwiseErrorRules(MemorySystem):
 def __init__(self,llm:LLMCallable):super().__init__(llm);self.examples=[];self.cap=500;self._last=0
 def _rules(self,query):
  labels={x["target"] for x in self.examples}; bad=[]
  for x in self.examples:
   source=_label(x.get("prediction",""),labels); target=x["target"]
   if not source or source==target or x.get("was_correct",False):continue
   bad.append((source,target,set(_tokens(x.get("raw_question",x["input"])))) )
  if not bad:return []
  # Pair-specific cues are terms frequent in this error direction and rare in opposite directions.
  pairs=defaultdict(list)
  for a,b,terms in bad:pairs[(a,b)].append(terms)
  rules=[]; q=set(_tokens(query))
  for (source,target),cases in pairs.items():
   other=[terms for (a,b),items in pairs.items() if a==source and b!=target for terms in items]
   counts=Counter(t for terms in cases for t in terms); other_counts=Counter(t for terms in other for t in terms)
   cues=[]
   for t,n in counts.items():
    rate=(n+.5)/(len(cases)+1); other_rate=(other_counts[t]+.5)/(len(other)+1)
    strength=math.log(rate/other_rate)
    if strength>.35:cues.append((strength*t.count('_')+strength,t))
   cues.sort(reverse=True); cues=[t for _,t in cues[:5]]
   overlap=q&set(cues)
   if overlap:rules.append((len(overlap),sum(v for v,t in cues if t in overlap),(source,target,cues,overlap)))
  rules.sort(reverse=True);return [r[2] for r in rules[:4]]
 def _nearby(self,query):
  q=set(_tokens(query)); ranked=sorted(self.examples,key=lambda x:(len(q&set(_tokens(x.get("raw_question",x["input"])))),x["target"]),reverse=True)
  out=[]; labels=Counter()
  for x in ranked:
   if not q&set(_tokens(x.get("raw_question",x["input"]))):continue
   if labels[x["target"]]>=2:continue
   out.append(x);labels[x["target"]]+=1
   if len(out)>=5:break
  return out
 def predict(self,input:str):
  rules=self._rules(input);near=self._nearby(input)
  rt="\n".join(f"When cues {', '.join(sorted(overlap))} occur, reconsider {source}; a similar case was corrected to {target}." for source,target,cues,overlap in rules) or "No pairwise correction trigger matches."
  ct="\n\n".join(f"Problem details: {_body(x.get('raw_question',x['input']))[:220]}\nVerified label: {x['target']}" for x in near) or "No similar verified cases."
  self._last=len(rt)+len(ct);out=self.call_llm(PROMPT.format(rules=rt,cases=ct,problem=input))
  return extract_json_field(out,"final_answer"),{"num_examples":len(self.examples),"num_rules":len(rules),"num_retrieved":len(near)}
 def learn_from_batch(self,batch_results):
  for r in batch_results:
   md=r.get("metadata") or {};self.examples.append({"input":str(r["input"]),"raw_question":str(r.get("raw_question") or md.get("raw_question") or r["input"]),"prediction":str(r.get("prediction","")),"target":str(r["ground_truth"]),"was_correct":bool(r.get("was_correct",False))})
  self.examples=self.examples[-self.cap:]
 def get_state(self):return json.dumps({"examples":self.examples},ensure_ascii=False)
 def set_state(self,state):self.examples=json.loads(state).get("examples",[])[-self.cap:]
 def get_context_length(self):return self._last
