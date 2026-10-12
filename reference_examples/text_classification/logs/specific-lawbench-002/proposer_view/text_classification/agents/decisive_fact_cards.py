"""Store compact, query-matchable cards of potentially operative facts."""
import json, math, re
from collections import Counter
from typing import Any
from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field
RUNS=re.compile(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+")
CJK=re.compile(r"^[\u3400-\u9fff]+$")
MARKERS=re.compile(r"利用|非法|骗取|侵占|收受|给予|盗窃|抢夺|销售|运输|携带|造成|致.*?伤|死亡|虚报|冒领|持有|制造|组织|强迫|敲诈|胁迫|烧毁|过火")
PROMPT="""Each card is a short excerpt selected from a different verified case. Excerpts omit context, so treat them as leads, not rules. Match the conduct and result to the current facts and preserve its answer format.
{cards}
Current problem:
{query}
Return JSON: {{"reasoning":"brief","final_answer":"answer"}}"""
def features(s):
 out=Counter()
 for r in RUNS.findall(s.lower()):
  if CJK.match(r): out.update(r[i:i+2] for i in range(len(r)-1)); out.update(r[i:i+3] for i in range(len(r)-2))
  else: out[r]+=1
 return out
def source(row):
 m=row.get('metadata') if isinstance(row.get('metadata'),dict) else {}
 return str(row.get('raw_question') or m.get('raw_question') or m.get('question') or row.get('input',''))
def make_cards(text):
 text=re.split(r"\*\*Instructions:\*\*",text)[0]
 sentences=[" ".join(x.split()) for x in re.split(r"(?<=[。！？.!?；;])\s*|\r?\n+",text)]
 scored=[]
 for i,s in enumerate(sentences):
  if len(s)<28: continue
  match=len(MARKERS.findall(s)); numbers=len(re.findall(r"\d+(?:[.，,]\d+)?",s))
  if match or numbers:
   # Blend operative conduct terms with amounts, actors, and nearby context.
   score=2.0*match+min(3,numbers*.65)+min(len(s),260)/260
   scored.append((score,i,s))
 scored.sort(reverse=True)
 chosen=[]
 for _,i,s in scored:
  if all(abs(i-j)>1 for j,_ in chosen): chosen.append((i,s))
  if len(chosen)>=4: break
 if not chosen: chosen=[(i,s) for i,s in enumerate(sentences) if len(s)>35][:3]
 return " ".join(s for _,s in sorted(chosen))[:1600]
class Candidate(MemorySystem):
 """Index concise conduct and outcome evidence cards instead of whole opinions."""
 def __init__(self,llm:LLMCallable): super().__init__(llm); self.cards=[]; self.budget=5200; self._last=''
 def _select(self,q):
  qv=features(q); docs=[features(x['card']) for x in self.cards]; df=Counter()
  for d in docs: df.update(d.keys())
  n=max(1,len(docs)); idf={t:math.log(1+(n+1)/(v+1)) for t,v in df.items()}
  def vec(c):
   raw={t:(1+math.log(v))*idf.get(t,0) for t,v in c.items() if v}; z=math.sqrt(sum(v*v for v in raw.values())) or 1
   return {t:v/z for t,v in raw.items()}
  a=vec(qv); ranked=[]
  for i,d in enumerate(docs): ranked.append((sum(v*vec(d).get(t,0) for t,v in a.items()),i))
  return [self.cards[i] for score,i in sorted(ranked,reverse=True) if score>0][:12]
 def _render(self,q):
  parts=[]; used=0
  for x in self._select(q):
   b=f"Operative facts: {x['card']}\nVerified answer: {x['target']}"
   if used+len(b)>self.budget: continue
   parts.append(b); used+=len(b)+2
  self._last="\n\n".join(parts); return self._last or "No fact cards overlap the query."
 def predict(self,input):
  r=self.call_llm(PROMPT.format(cards=self._render(input),query=input))
  return extract_json_field(r,'final_answer'),{'full_response':r,'retrieved_cards':len(self._last.split('\n\n')) if self._last else 0}
 def learn_from_batch(self,batch_results):
  seen={(x['card'],x['target']) for x in self.cards}
  for row in batch_results:
   card=make_cards(source(row)); target=str(row['ground_truth'])
   if card and (card,target) not in seen: self.cards.append({'card':card,'target':target}); seen.add((card,target))
 def get_context_length(self): return len(self._last)
 def get_state(self): return json.dumps({'cards':self.cards},ensure_ascii=False)
 def set_state(self,state): self.cards=json.loads(state).get('cards',[]); self._last=''
