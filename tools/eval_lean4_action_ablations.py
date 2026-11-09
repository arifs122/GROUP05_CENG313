#!/usr/bin/env python3
"""Ablations for lean4 action tree: global vs phase, sequence vs no-sequence."""
from __future__ import annotations
import argparse, csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src/mazlum')); import features as F

OPP=['Dragapult','Mega Kangaskhan','Mega Excadrill','Mega Lopunny','Banette','Alakazam','Hydrapple','Mega Abomasnow','Festival Lead','Mega Lucario','Archaludon','Slowking-Kangaskhan','Teal Mask Ogerpon',"Marnie's Grimmsnarl"]

def build(seq=True, phase_feature=False):
 d=np.load(ROOT/'data/lean4_action_tree_dataset.npz')
 state=d['state'].astype(np.float32); option=d['option'].astype(np.float32); ctx=d['context']; st=d['select_type']; mn=d['min_count']; mx=d['max_count']; turn=d['turn']
 opp=d['opp_code']; last=d['last_type']; prev=d['prev_type']; ph=d['phase']
 n=len(d['label']); opp_oh=np.zeros((n,len(OPP)+1),np.float32); opp_oh[np.arange(n),opp]=1
 parts=[state,option,ctx[:,None].astype(np.float32),st[:,None].astype(np.float32),mn[:,None].astype(np.float32),mx[:,None].astype(np.float32),turn[:,None].astype(np.float32)/20,opp_oh]
 if seq:
  lh=np.zeros((n,18),np.float32); lh[np.arange(n),last]=1; phh=np.zeros((n,18),np.float32); phh[np.arange(n),prev]=1; parts += [lh,phh]
 if phase_feature:
  poh=np.zeros((n,3),np.float32); poh[np.arange(n),ph]=1; parts += [poh]
 return d,np.concatenate(parts,axis=1)

def recall_for(pred, va, d):
 ep=d['episode_id']; step=d['step_index']; label=d['label']
 groups=defaultdict(list)
 for idx,p in zip(np.nonzero(va)[0],pred): groups[(int(ep[idx]),int(step[idx]))].append((float(p),int(idx)))
 cor=tot=0
 for arr in groups.values():
  arr.sort(reverse=True); nsel=int(label[[i for _,i in arr]].sum()); k=max(1,nsel)
  cor+=len({i for _,i in arr[:k]}&{i for _,i in arr if label[i]==1}); tot+=len({i for _,i in arr if label[i]==1})
 return cor/max(1,tot)

def main():
 d,base=build(seq=True,phase_feature=True)
 ep=d['episode_id']; ph=d['phase']; label=d['label']; won=d['won']; pd=d['prize_diff']
 seed=44; ue=np.unique(ep); rng=np.random.default_rng(seed); rng.shuffle(ue); nv=max(1,int(len(ue)*.2)); val=set(ue[:nv].tolist()); tr=np.array([int(e) not in val for e in ep]); va=~tr
 w=np.where(won>0,2.0,.5)*(1.0+.05*np.abs(pd))
 configs=[('global_phase_seq',True,True),('global_phase_noseq',False,True),('phase_seq',True,False),('phase_noseq',False,False)]
 out=[]
 for name,seq,pf in configs:
  if pf:
   _,X=build(seq=seq,phase_feature=True)
   clf=DecisionTreeClassifier(max_depth=7,min_samples_leaf=80,class_weight='balanced',random_state=seed)
   clf.fit(X[tr],label[tr],sample_weight=w[tr]); pr=clf.predict_proba(X[va])[:,1]
   r=recall_for(pr,va,d); print(name,r,flush=True); out.append((name,'all',r))
  else:
   # phase-specific; build matrix for each phase (no phase one-hot needed)
   _,X=build(seq=seq,phase_feature=False)
   for p in sorted(set(ph.tolist())):
    mtr=tr&(ph==p); mva=va&(ph==p)
    if mtr.sum()<1000 or mva.sum()<200: continue
    clf=DecisionTreeClassifier(max_depth=7,min_samples_leaf=80,class_weight='balanced',random_state=seed)
    clf.fit(X[mtr],label[mtr],sample_weight=w[mtr]); pr=clf.predict_proba(X[mva])[:,1]
    r=recall_for(pr,mva,d); print(name,'phase',p,r,flush=True); out.append((name,str(p),r))
  # summary average weighted? 
 with (ROOT/'results/lean4_action_tree_ablations.csv').open('w',newline='') as f:
  wtr=csv.writer(f); wtr.writerow(['config','phase','recall']); wtr.writerows(out)
 print('\nSUMMARY')
 for name in ['global_phase_seq','global_phase_noseq','phase_seq','phase_noseq']:
  vals=[x[2] for x in out if x[0]==name]
  if vals: print(name,'mean',sum(vals)/len(vals),'n',len(vals))
if __name__=='__main__': main()
