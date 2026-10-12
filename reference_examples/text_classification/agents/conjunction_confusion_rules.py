import json, math, re
from collections import Counter, defaultdict
from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field
STOP=set("a an and are as at be been but by for from had has have he her his i in into is it its me my of on or our she that the their them there they this to up was we were when which with you your please provide diagnosis symptoms patient problem based following answer final feeling feel really also been having have".split())
def body(s):
 m=re.search(r"## Patient Symptoms\s*(.*?)(?:\n\nPlease provide|$)",str(s),re.I|re.S); return m.group(1).strip() if m else str(s)
def words(s): return [w for w in re.findall(r"[a-z0-9]+",body(s).lower()) if len(w)>2 and w not in STOP]
def feats(s):
 w=words(s); return Counter(w+[a+"_"+b for a,b in zip(w,w[1:])])
PROMPT="""Apply a correction bundle only when multiple details match together. Compare suggested labels against all details. Verified cases are evidence, not an answer key. Follow requested format; return one exact label.\n\nCorrection bundles:\n{rules}\n\nNearby verified cases:\n{cases}\n\nProblem:\n{problem}\n\nReturn JSON only: {{\"reasoning\":\"brief rationale\",\"final_answer\":\"exact label\"}}"""
def label(s,labels):
 hits=[x for x in labels if re.search(r"\b"+re.escape(x.lower())+r"\b",str(s).lower())]; return max(hits,key=len) if hits else ""
class Candidate(MemorySystem):
 def __init__(self,llm:LLMCallable): super().__init__(llm); self.examples=[]; self._last=0
 def predict(self,input:str):
  labels={x['target'] for x in self.examples}; groups=defaultdict(list)
  for x in self.examples:
   a=label(x.get('prediction',''),labels); b=x['target']
   if a and a!=b and not x.get('was_correct',False): groups[a,b].append(set(words(x.get('raw_question',x['input']))))
  q=set(words(input)); rules=[]
  for (a,b),cases in groups.items():
   freq=Counter(t for c in cases for t in c); other=[z for (u,v),cs in groups.items() if u==a and v!=b for z in cs]; of=Counter(t for c in other for t in c); scored=[]
   for t,n in freq.items():
    val=math.log(((n+.4)/(len(cases)+.8))/((of[t]+.4)/(len(other)+.8)))
    if t in q and val>.2: scored.append((val,t))
   scored=sorted(scored,reverse=True)[:6]
   if len(scored)>=2: rules.append((sum(v for v,_ in scored),a,b,[t for _,t in scored]))
  rules=sorted(rules,reverse=True)[:4]; rt='\n'.join(f"Matched {','.join(ts)}: reconsider {a}; similar errors were corrected to {b}." for _,a,b,ts in rules) or 'No multi-detail correction applies.'
  rank=sorted(self.examples,key=lambda x:(len(q&set(words(x.get('raw_question',x['input'])))),x['target']),reverse=True); picked=[]; count=Counter()
  for x in rank:
   if q&set(words(x.get('raw_question',x['input']))) and count[x['target']]<2: picked.append(x); count[x['target']]+=1
   if len(picked)>=5: break
  cases='\n\n'.join(f"Details: {body(x.get('raw_question',x['input']))[:220]}\nVerified label: {x['target']}" for x in picked) or 'No verified cases.'
  self._last=len(rt)+len(cases); out=self.call_llm(PROMPT.format(rules=rt,cases=cases,problem=input)); return extract_json_field(out,'final_answer'),{'num_rules':len(rules),'num_retrieved':len(picked)}
 def learn_from_batch(self,batch_results):
  for r in batch_results:
   md=r.get('metadata') or {}; self.examples.append({'input':str(r['input']),'raw_question':str(r.get('raw_question') or md.get('raw_question') or r['input']),'prediction':str(r.get('prediction','')),'target':str(r['ground_truth']),'was_correct':bool(r.get('was_correct',False))})
  self.examples=self.examples[-500:]
 def get_state(self): return json.dumps({'examples':self.examples},ensure_ascii=False)
 def set_state(self,state): self.examples=json.loads(state).get('examples',[])[-500:]
 def get_context_length(self): return self._last
