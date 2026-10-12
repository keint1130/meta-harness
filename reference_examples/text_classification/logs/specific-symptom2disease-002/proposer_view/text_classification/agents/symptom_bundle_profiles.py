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
PROMPT="""Label cards list co-occurring distinctive term pairs learned from verified examples, their match count in the current problem, and a representative case. A single common word is weak evidence; prefer a coherent bundle and compare competing labels. Follow requested format; return one exact label.\n\nCards:\n{cards}\n\nProblem:\n{problem}\n\nReturn JSON only: {{\"reasoning\":\"brief rationale\",\"final_answer\":\"exact label\"}}"""
def bundle(s):
 w=words(s); return Counter([a+'_'+b for a,b in zip(w,w[1:])]+[a+'__'+b for i,a in enumerate(w) for b in w[i+2:i+5]])
class Candidate(MemorySystem):
 def __init__(self,llm:LLMCallable): super().__init__(llm); self.examples=[]; self._last=0
 def predict(self,input:str):
  q=bundle(input); groups=defaultdict(list)
  for x in self.examples: groups[x['target']].append(x)
  n=len(self.examples); df=Counter()
  for x in self.examples: df.update(set(bundle(x.get('raw_question',x['input']))))
  cards=[]
  for lab,items in groups.items():
   freq=Counter(); other=Counter()
   for x in items: freq.update(set(bundle(x.get('raw_question',x['input']))))
   for ol,arr in groups.items():
    if ol!=lab:
     for x in arr: other.update(set(bundle(x.get('raw_question',x['input']))))
   ranked=[]
   for f,c in freq.items():
    if f in q:
     score=(1+math.log(c))*max(.05,math.log((n+1)/(df[f]+1)))*max(.1,1+math.log((c+.5)/(other[f]+.5))); ranked.append((score,f,c))
   ranked=sorted(ranked,reverse=True)[:4]
   if ranked:
    best=max(items,key=lambda x:sum(f in bundle(x.get('raw_question',x['input'])) for _,f,_ in ranked)); cards.append((sum(z[0] for z in ranked),lab,ranked,best))
  cards=sorted(cards,reverse=True)[:7]; text='\n'.join(f"Label {lab} | matched bundle weight {score:.2f}\n Matched pairs: "+', '.join(f"{f.replace('_',' ')} ({c} examples)" for _,f,c in fs)+f"\n Verified case: {body(best.get('raw_question',best['input']))[:180]}" for score,lab,fs,best in cards) or 'No learned label cards.'
  self._last=len(text); out=self.call_llm(PROMPT.format(cards=text,problem=input)); return extract_json_field(out,'final_answer'),{'num_cards':len(cards)}
 def learn_from_batch(self,batch_results):
  for r in batch_results:
   md=r.get('metadata') or {}; self.examples.append({'input':str(r['input']),'raw_question':str(r.get('raw_question') or md.get('raw_question') or r['input']),'target':str(r['ground_truth'])})
  self.examples=self.examples[-500:]
 def get_state(self): return json.dumps({'examples':self.examples},ensure_ascii=False)
 def set_state(self,state): self.examples=json.loads(state).get('examples',[])[-500:]
 def get_context_length(self): return self._last
