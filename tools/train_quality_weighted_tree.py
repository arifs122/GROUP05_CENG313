#!/usr/bin/env python3
"""Context tree with quality-label sample weights from fresh noAI replays."""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
ROOT=Path(__file__).resolve().parents[1]
OPP=['Dragapult','Mega Kangaskhan','Mega Excadrill','Mega Lopunny','Banette','Alakazam','Hydrapple','Mega Abomasnow','Festival Lead','Mega Lucario','Archaludon','Slowking-Kangaskhan','Teal Mask Ogerpon',"Marnie's Grimmsnarl"]
def build(d,v):
 n=len(d['label']);opp=np.zeros((n,len(OPP)+1),np.float32);opp[np.arange(n),d['opp_code']]=1
 lh=np.zeros((n,18),np.float32);lh[np.arange(n),d['last_type']]=1;ph=np.zeros((n,18),np.float32);ph[np.arange(n),d['prev_type']]=1
 return np.concatenate([d['state'].astype(np.float32),d['option'].astype(np.float32),d['context'][:,None].astype(np.float32),d['select_type'][:,None].astype(np.float32),d['min_count'][:,None].astype(np.float32),d['max_count'][:,None].astype(np.float32),d['turn'][:,None].astype(np.float32)/20,opp,lh,ph,v[:,None]],axis=1)
def recall(pred,va,d,topk=1):
 ep=d['episode_id'];step=d['step_index'];y=d['label'];g=defaultdict(list)
 for i in np.nonzero(va)[0]:g[(int(ep[i]),int(step[i]))].append((float(pred[i]),int(i)))
 c=t=0
 for a in g.values():
  a.sort(reverse=True);nsel=int(y[[i for _,i in a]].sum());k=max(1,min(topk,nsel if topk==1 else len(a)));c+=len({i for _,i in a[:k]}&{i for _,i in a if y[i]==1});t+=len({i for _,i in a if y[i]==1})
 return c/max(1,t)
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'experiments/20261104_state_value/quality_weighted_tree');args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
d=np.load(ROOT/'data/lean4_action_tree_dataset.npz');ep=d['episode_id'];step=d['step_index'];u=np.unique(ep);rng=np.random.default_rng(47);rng.shuffle(u);nv=max(1,int(len(u)*.2));val=set(u[:nv].tolist());tr=np.array([int(e) not in val for e in ep]);va=~tr
base=np.where(d['won']>0,2.,.5)*(1+.05*np.abs(d['prize_diff']))
q={};
for r in csv.DictReader(open(ROOT/'experiments/20261104_state_value/quality_labels/quality_labels.csv',encoding='utf-8')):
 q[(int(r['episode_id']),int(r['step']))]=r['label']
vm=HistGradientBoostingClassifier(max_iter=120,learning_rate=.06,max_leaf_nodes=15,l2_regularization=.1,random_state=47);vm.fit(d['state'][tr].astype(np.float32),d['won'][tr]);v=vm.predict_proba(d['state'].astype(np.float32))[:,1].astype(np.float32)
X=build(d,v);out=[];best=(-1,None)
for good_f,bad_f,name in [(2.0,.2,'good2_bad.2'),(2.0,0.0,'good2_bad0'),(3.0,.2,'good3_bad.2'),(1.0,.2,'good1_bad.2')]:
 w=base.copy(); nq=0
 for i in range(len(w)):
  if d['label'][i]!=1: continue
  lab=q.get((int(ep[i]),int(step[i])))
  if lab == 'good': w[i]*=good_f; nq+=1
  elif lab == 'bad': w[i]*=bad_f; nq+=1
 for depth,leaf in [(16,120),(18,160)]:
  ph=d['phase'];ctx=d['context'];pred=np.full(len(ep),-1.,np.float32);nctx=0
  for p in [0,1,2]:
   for cx in np.unique(ctx):
    mtr=tr&(ph==p)&(ctx==cx);mva=va&(ph==p)&(ctx==cx)
    if mtr.sum()<400 or mva.sum()<50:continue
    c=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,class_weight='balanced',random_state=47);c.fit(X[mtr],d['label'][mtr],sample_weight=w[mtr]);pred[np.nonzero(mva)[0]]=c.predict_proba(X[mva])[:,1];nctx+=1
  r=recall(pred,va,d);rec={'name':name,'depth':depth,'leaf':leaf,'top1':r,'contexts':nctx,'labeled_selected':nq};out.append(rec);print(rec,flush=True)
  if r>best[0]:best=(r,rec);(args.out/'best.json').write_text(json.dumps(rec,indent=2))
with (args.out/'runs.jsonl').open('w') as f:
 for r in out:f.write(json.dumps(r)+'\n')
print('BEST',best)
