#!/usr/bin/env python3
"""Phase + context hierarchical action trees; no matchup-specific trees."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT=Path(__file__).resolve().parents[1]
OPP=['Dragapult','Mega Kangaskhan','Mega Excadrill','Mega Lopunny','Banette','Alakazam','Hydrapple','Mega Abomasnow','Festival Lead','Mega Lucario','Archaludon','Slowking-Kangaskhan','Teal Mask Ogerpon',"Marnie's Grimmsnarl"]

def build_X(d,seq=True):
 n=len(d['label']); state=d['state'].astype(np.float32); option=d['option'].astype(np.float32)
 opp=np.zeros((n,len(OPP)+1),np.float32); opp[np.arange(n),d['opp_code']]=1
 parts=[state,option,d['context'][:,None].astype(np.float32),d['select_type'][:,None].astype(np.float32),d['min_count'][:,None].astype(np.float32),d['max_count'][:,None].astype(np.float32),d['turn'][:,None].astype(np.float32)/20,opp]
 if seq:
  lh=np.zeros((n,18),np.float32); lh[np.arange(n),d['last_type']]=1
  ph=np.zeros((n,18),np.float32); ph[np.arange(n),d['prev_type']]=1; parts += [lh,ph]
 return np.concatenate(parts,axis=1)

def recall(pred,va,d,topk=1):
 ep=d['episode_id']; step=d['step_index']; label=d['label']; groups=defaultdict(list)
 for idx in np.nonzero(va)[0]: groups[(int(ep[idx]),int(step[idx]))].append((float(pred[idx]),int(idx)))
 c=t=0
 for arr in groups.values():
  arr.sort(reverse=True); nsel=int(label[[i for _,i in arr]].sum()); k=max(1,min(topk,nsel if topk==1 else len(arr)))
  c+=len({i for _,i in arr[:k]}&{i for _,i in arr if label[i]==1}); t+=len({i for _,i in arr if label[i]==1})
 return c/max(1,t)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=ROOT/'experiments/20261104_state_value/context_tree_search'); args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
 d=np.load(ROOT/'data/lean4_action_tree_dataset.npz'); ep=d['episode_id']; ue=np.unique(ep); rng=np.random.default_rng(47); rng.shuffle(ue); nv=max(1,int(len(ue)*.2)); val=set(ue[:nv].tolist()); tr=np.array([int(e) not in val for e in ep]); va=~tr
 ph=d['phase']; ctx=d['context']; won=d['won']; pd=d['prize_diff']; base=np.where(won>0,2.0,.5)*(1+.05*np.abs(pd))
 X=build_X(d,True)
 out=[]; best=(-1,None)
 for depth in [8,10,12,14]:
  for leaf in [20,40,80,160]:
   # phase trees as fallback
   phase_pred=np.full(len(ep),-1.0,np.float32)
   for p in [0,1,2]:
    mtr=tr&(ph==p); mva=va&(ph==p)
    if mtr.sum()<300 or mva.sum()<50: continue
    c=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,class_weight='balanced',random_state=47)
    c.fit(X[mtr],d['label'][mtr],sample_weight=base[mtr]); phase_pred[np.nonzero(mva)[0]]=c.predict_proba(X[mva])[:,1]
   # context overrides
   pred=phase_pred.copy(); nctx=0
   for p in [0,1,2]:
    for cx in np.unique(ctx):
     mtr=tr&(ph==p)&(ctx==cx); mva=va&(ph==p)&(ctx==cx)
     if mtr.sum()<400 or mva.sum()<50: continue
     c=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,class_weight='balanced',random_state=47)
     c.fit(X[mtr],d['label'][mtr],sample_weight=base[mtr]); pred[np.nonzero(mva)[0]]=c.predict_proba(X[mva])[:,1]; nctx+=1
   r1=recall(pred,va,d,1); r3=recall(pred,va,d,3)
   rec={'depth':depth,'leaf':leaf,'contexts':nctx,'top1':r1,'top3':r3}; out.append(rec); print(rec,flush=True)
   if r1>best[0]:
    best=(r1,rec); (args.out/'best.json').write_text(json.dumps(rec,indent=2))
    # save representative rules for context 0 phase 1
    mtr=tr&(ph==1)&(ctx==0)
    if mtr.sum()>100:
     c=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,class_weight='balanced',random_state=47); c.fit(X[mtr],d['label'][mtr],sample_weight=base[mtr]); (args.out/'best_phase1_main.txt').write_text(export_text(c,max_depth=5,decimals=2))
 with (args.out/'runs.jsonl').open('w') as fh:
  for r in out: fh.write(json.dumps(r)+'\n')
 print('BEST',best,'runs',len(out),flush=True)
if __name__=='__main__': main()
