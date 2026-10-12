"""Generate a provisional answer from retrieved demonstrations, then verify it."""
import json, math, re
from collections import Counter
from typing import Any
from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field
RUNS=re.compile(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+")
CJK=re.compile(r"^[\u3400-\u9fff]+$")
DRAFT="""From the current problem and demonstrations, draft the complete answer in the requested format. Include all distinct supported labels or outputs when the task requires more than one. Return JSON with final_answer only.
{examples}
Current problem:
{query}
{{"final_answer":"answer in requested format"}}"""
CHECK="""Check the provisional answer against the current problem. Examples are analogies and may differ. Retain only supported answer components and add any missing component warranted by the input. Preserve the requested output format. Return JSON.
Provisional answer: {draft}
Relevant verified examples:
{examples}
Current problem:
{query}
{{"reasoning":"brief evidence check","final_answer":"answer"}}"""
def feats(s):
 out=Counter()
 for r in RUNS.findall(s.lower()):
  if CJK.match(r): out.update(r[i:i+2] for i in range(len(r)-1)); out.update(r[i:i+3] for i in range(len(r)-2))
  else: out[r]+=1
 return out
def source(row):
 m=row.get('metadata') if isinstance(row.get('metadata'),dict) else {}
 return str(row.get('raw_question') or m.get('raw_question') or m.get('question') or row.get('input',''))
class Candidate(MemorySystem):
 """Separate answer generation from evidence-based label-set verification."""
 def __init__(self,llm:LLMCallable): super().__init__(llm); self.examples=[]; self.max_chars=9000; self._last=''
 def _retrieve(self,q):
  if not self.examples:return []
  docs=[feats(x['input']) for x in self.examples]; df=Counter()
  for d in docs: df.update(d.keys())
  n=len(docs); idf={t:math.log(1+(n+1)/(v+1)) for t,v in df.items()}
  def vec(c):
   raw={t:(1+math.log(v))*idf.get(t,0) for t,v in c.items() if v}; z=math.sqrt(sum(v*v for v in raw.values())) or 1
   return {t:v/z for t,v in raw.items()}
  qv=vec(feats(q)); return [self.examples[i] for _,i in sorted(((sum(v*vec(d).get(t,0) for t,v in qv.items()),i) for i,d in enumerate(docs)),reverse=True) if _>0][:8]
 def _render(self,q):
  parts=[]; used=0
  for x in self._retrieve(q):
   s=x['input']; s=s if len(s)<1150 else s[:820]+' … '+s[-327:]
   b=f"Q: {s}\nA: {x['target']}"
   if used+len(b)>self.max_chars: break
   parts.append(b); used+=len(b)+2
  self._last="\n\n".join(parts); return self._last or 'No demonstrations have been learned yet.'
 def predict(self,input):
  ex=self._render(input)
  first=self.call_llm(DRAFT.format(examples=ex,query=input)); draft=extract_json_field(first,'final_answer')
  second=self.call_llm(CHECK.format(draft=draft,examples=ex,query=input))
  return extract_json_field(second,'final_answer'),{'full_response':second,'draft_answer':draft,'retrieved_examples':len(self._last.split('\n\n')) if self._last else 0}
 def learn_from_batch(self,batch_results):
  seen={(x['input'],x['target']) for x in self.examples}
  for row in batch_results:
   inp=source(row); t=str(row['ground_truth'])
   if (inp,t) not in seen:self.examples.append({'input':inp,'target':t});seen.add((inp,t))
 def get_context_length(self):return len(self._last)
 def get_state(self):return json.dumps({'examples':self.examples},ensure_ascii=False)
 def set_state(self,state):self.examples=json.loads(state).get('examples',[]);self._last=''
