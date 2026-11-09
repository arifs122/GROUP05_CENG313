#!/usr/bin/env python3
"""Context tree with state-value feature; no matchup split."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT=Path(__file__).resolve().parents[1]
OPP=['Dragapult','Mega Kangaskhan','Mega Excadrill','Mega Lopunny','Banette','Alakazam','Hydrapple','Mega Abomasnow','Festival Lead','Mega Lucario','Archaludon','Slowking-Kangaskhan','Teal Mask Ogerpon',"Marnie's Grimmsnarl"]

def build_X(d,extra=None):
 n=len(d['label']); state=d['state'].astype(np.float32); option=d['option'].astype(np.float32)
 opp=np.zeros((n,len(OPP)+1),np.float32); opp[np.arange(n),d['opp_code']]=1
 parts=[state,option,d['context'][:,None].astype(np.float32),d['select_type'][:,None].astype(np.float32),d['min_count'][:,None].astype(np.float32),d['max_count'][:,None].astype(np.float32),d['turn'][:,None].astype(np.float32)/20,opp]
 lh=np.zeros((n,18),np.float32); lh[np.arange(n),d['last_type']]=1
 ph=np.zeros((n,18),np.float32); ph[np.arange(n),d['prev_type']]=1
 parts += [lh,ph]
 if extra is not None: parts.append(extra[:,None])
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
 ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=ROOT/'experiments/20261104_state_value/context_value_search'); args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
 d=np.load(ROOT/'data/lean4_action_tree_dataset.npz'); ep=d['episode_id']; ue=np.unique(ep); rng=np.random.default_rng(47); rng.shuffle(ue); nv=max(1,int(len(ue)*.2)); val=set(ue[:nv].tolist()); tr=np.array([int(e) not in val for e in ep]); va=~tr
 base=np.where(d['won']>0,2.,.5)*(1+.05*np.abs(d['prize_diff']))
 # state-only value model trained only on action-train rows
 vm=HistGradientBoostingClassifier(max_iter=120,learning_rate=.06,max_leaf_nodes=15,l2_regularization=.1,random_state=47)
 vm.fit(d['state'][tr].astype(np.float32),d['won'][tr])
 v=vm.predict_proba(d['state'].astype(np.float32))[:,1].astype(np.float32)
 print('value AUC proxy',flush=True)
 out=[]; best=(-1,None)
 for use_v in [False,True]:
  X=build_X(d, v if use_v else None)
  for depth in [14,16,18,20]:
   for leaf in [40,80,120]:
    ph=d['phase']; ctx=d['context']; pred=np.full(len(ep),-1.,np.float32); nctx=0
    for p in [0,1,2]:
     # phase fallback
     mtr=tr&(ph==p); mva=va&(ph==p)
     if mtr.sum()>=300 and mva.sum()>=50:
      c=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,class_weight='balanced',random_state=47); c.fit(X[mtr],d['label'][mtr],sample_weight=base[mtr]); pred[np.nonzero(mva)[0]]=c.predict_proba(X[mva])[:,1]
     for cx in np.unique(ctx):
      mtr=tr&(ph==p)&(ctx==cx); mva=va&(ph==p)&(ctx==cx)
      if mtr.sum()<400 or mva.sum()<50: continue
      c=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,class_weight='balanced',random_state=47); c.fit(X[mtr],d['label'][mtr],sample_weight=base[mtr]); pred[np.nonzero(mva)[0]]=c.predict_proba(X[mva])[:,1]; nctx+=1
    r1=recall(pred,va,d,1); r3=recall(pred,va,d,3)
    rec={'use_value':use_v,'depth':depth,'leaf':leaf,'contexts':nctx,'top1':r1,'top3':r3}; out.append(rec); print(rec,flush=True)
    if r1>best[0]:
     best=(r1,rec); (args.out/'best.json').write_text(json.dumps(rec,indent=2))
     mtr=tr&(ph==1)&(ctx==0)
     if mtr.sum()>100:
      c=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,class_weight='balanced',random_state=47); c.fit(X[mtr],d['label'][mtr],sample_weight=base[mtr]); (args.out/'best_phase1_main.txt').write_text(export_text(c,max_depth=5,decimals=2))
 with (args.out/'runs.jsonl').open('w') as fh:
  for r in out: fh.write(json.dumps(r)+'\n')
 print('BEST',best,'runs',len(out),flush=True)
if __name__=='__main__': main()
