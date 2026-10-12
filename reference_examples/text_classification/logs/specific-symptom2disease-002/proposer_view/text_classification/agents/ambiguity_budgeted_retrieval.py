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
PROMPT="""Retrieved cases are evidence, not an answer key. Support summarizes nearby cases per label. Check distinctive details especially when leading labels are close. Follow requested format and return one exact label.\n\nSupport and cases:\n{evidence}\n\nProblem:\n{problem}\n\nReturn JSON only: {{\"reasoning\":\"brief rationale\",\"final_answer\":\"exact label\"}}"""
def cosine(a,b,idf):
 dot=sum(v*b.get(k,0)*idf.get(k,.2)**2 for k,v in a.items()); na=math.sqrt(sum(v*v*idf.get(k,.2)**2 for k,v in a.items())); nb=math.sqrt(sum(v*v*idf.get(k,.2)**2 for k,v in b.items())); return dot/(na*nb) if na and nb else 0
class Candidate(MemorySystem):
 def __init__(self,llm:LLMCallable): super().__init__(llm); self.examples=[]; self._last=0
 def predict(self,input:str):
  if not self.examples: evidence='No prior examples.'; picked=[]
  else:
   docs=[feats(x.get('raw_question',x['input'])) for x in self.examples]; df=Counter(k for d in docs for k in set(d)); n=len(docs); idf={k:max(.15,math.log((n+1)/(v+1))) for k,v in df.items()}; q=feats(input); sims=[cosine(q,d,idf) for d in docs]; groups=defaultdict(list)
   for i,(x,s) in enumerate(zip(self.examples,sims)): groups[x['target']].append((s,i))
   support={lab:sum(s*s for s,_ in arr)/math.sqrt(len(arr)) for lab,arr in groups.items()}; labs=sorted(support,key=support.get,reverse=True); margin=(support[labs[0]]-support[labs[1]])/(support[labs[0]]+.001) if len(labs)>1 else 1; budget=3 if margin>=.35 else (5 if margin>=.16 else 9); picked=[]; counts=Counter()
   for pos in range(max(map(len,groups.values()))):
    for lab in labs:
     arr=sorted(groups[lab],reverse=True)
     if pos<len(arr) and counts[lab]<2 and arr[pos][0]>0: picked.append((arr[pos][0],self.examples[arr[pos][1]])); counts[lab]+=1
    if len(picked)>=budget: break
   supportlines=[]
   for lab in labs[:5]:
    entries=[x for _,x in picked if x['target']==lab]; supportlines.append(f"Label {lab} | support {support[lab]:.3f}"+''.join(f"\n Case: {body(x.get('raw_question',x['input']))[:180]}" for x in entries))
   evidence='\n'.join(supportlines)
  self._last=len(evidence); out=self.call_llm(PROMPT.format(evidence=evidence,problem=input)); return extract_json_field(out,'final_answer'),{'num_retrieved':len(picked)}
 def learn_from_batch(self,batch_results):
  for r in batch_results:
   md=r.get('metadata') or {}; self.examples.append({'input':str(r['input']),'raw_question':str(r.get('raw_question') or md.get('raw_question') or r['input']),'target':str(r['ground_truth'])})
  self.examples=self.examples[-500:]
 def get_state(self): return json.dumps({'examples':self.examples},ensure_ascii=False)
 def set_state(self,state): self.examples=json.loads(state).get('examples',[])[-500:]
 def get_context_length(self): return self._last
