#!/usr/bin/env python3
"""Label selected lean4 actions by comparing afterstate tactical values.

For each single MAIN decision, all legal options are applied with search_begin
on the same determinization. lean4's own _eval_state scores the afterstate.
Labels are relative to the best legal option; forced moves are neutral.
"""
from __future__ import annotations
import argparse,csv,json,sys,importlib.util,random
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor')); sys.path.insert(0,str(ROOT/'src'/'mazlum'))
from cg.api import to_observation_class, search_begin, search_end, search_step
import features as F

def load_noai():
 path=ROOT/'bots/dragapult_lean4_noai/main.py'
 spec=importlib.util.spec_from_file_location('lean4_noai_label',path)
 mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
 return mod

def sig(o):
 if isinstance(o,dict): return tuple(o.get(k) for k in ('type','attackId','area','index','inPlayArea','inPlayIndex','number','cardId'))
 return tuple(getattr(o,k,None) for k in ('type','attackId','area','index','inPlayArea','inPlayIndex','number','cardId'))
def map_opt(orig,opts):
 s=sig(orig)
 for i,o in enumerate(opts):
  if sig(o)==s: return i
 return None

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--limit',type=int,default=0); ap.add_argument('--out-dir',type=Path,default=ROOT/'experiments/20261104_state_value/quality_labels'); args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
 mod=load_noai(); deck=mod._load_deck(); mod._load_cards()
 paths=sorted((ROOT/'replays/lean4_noai_57961261').glob('*replay.json'))
 if args.limit: paths=paths[:args.limit]
 rows=[]; skipped=0; games=0
 for p in paths:
  ep=json.loads(p.read_text()); steps=ep.get('steps') or []
  # side from initial deck
  decks=None
  try: decks=ep['steps'][1][0]['visualize'][0]['action'] if False else None
  except Exception: pass
  # use master? find side by exact deck in step 1 action
  try:
   acts=ep['steps'][1]
   d0=[int(x) for x in acts[0]['action']]; d1=[int(x) for x in acts[1]['action']]
   side=0 if sorted(d0)==sorted(deck) else (1 if sorted(d1)==sorted(deck) else None)
  except Exception: side=None
  if side is None: skipped+=1; continue
  games+=1
  for d in F.iter_decisions(ep):
   if d.player!=side: continue
   sel=d.select or {}
   if int(sel.get('type',-1))!=0 or int(sel.get('maxCount',0))!=1 or not d.action: continue
   opts=sel.get('option') or []
   if len(opts)<1: continue
   try: typed=to_observation_class(d.observation); hidden=mod._sample_hidden(typed,side)
   except Exception as e: skipped+=1; continue
   scores=[]; mapped=[]; chosen_score=None
   for ci,o in enumerate(opts):
    try:
     st=search_begin(typed,*hidden); sid=st.searchId
     so=st.observation.select.option or []
     mi=map_opt(o,so)
     if mi is None: search_end(); continue
     leaf=search_step(sid,[mi])
     cur=leaf.observation.current
     if cur is None: search_end(); continue
     if int(cur.result)>=0:
      sc=1e9 if int(cur.result)==side else -1e9
     else:
      sc=float(mod._eval_state(leaf.observation,side))
     scores.append(sc); mapped.append(ci)
     if ci in d.action: chosen_score=sc
    except Exception:
     pass
    finally:
     try: search_end()
     except Exception: pass
   if not scores or chosen_score is None: skipped+=1; continue
   best=max(scores); adv=float(chosen_score-best)
   rel=adv/max(1.0,abs(best)) if abs(best)>1 else 0.0
   forced=1 if len(opts)<=1 else 0
   if forced: label='neutral'
   elif rel >= -0.01: label='good'
   elif rel <= -0.05: label='bad'
   else: label='neutral'
   rows.append({'episode_id':int(p.name.split('-')[1]),'step':int(d.step_index),'turn':int((d.observation.get('current') or {}).get('turn',0) or 0),'n_options':len(opts),'chosen':int(d.action[0]),'best_score':best,'chosen_score':chosen_score,'adv':adv,'rel_adv':rel,'label':label,'forced':forced})
  if games%20==0: print('games',games,'rows',len(rows),flush=True)
 with (args.out_dir/'quality_labels.csv').open('w',newline='',encoding='utf-8') as fh:
  w=csv.DictWriter(fh,fieldnames=list(rows[0]) if rows else ['episode_id']);w.writeheader();w.writerows(rows)
 print('games',games,'rows',len(rows),'skipped',skipped,'labels',Counter(r['label'] for r in rows),flush=True)
 print('rel_adv quantiles',np.quantile([r['rel_adv'] for r in rows],[0,.05,.25,.5,.75,.95,1]) if rows else None,flush=True)
if __name__=='__main__': main()
