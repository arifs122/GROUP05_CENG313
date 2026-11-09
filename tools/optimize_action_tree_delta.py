#!/usr/bin/env python3
"""Try delta-weighted action trees, phase/context splits, and hyperparameters."""
from __future__ import annotations
import argparse, csv, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np, joblib
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT=Path(__file__).resolve().parents[1]
OPP=['Dragapult','Mega Kangaskhan','Mega Excadrill','Mega Lopunny','Banette','Alakazam','Hydrapple','Mega Abomasnow','Festival Lead','Mega Lucario','Archaludon','Slowking-Kangaskhan','Teal Mask Ogerpon',"Marnie's Grimmsnarl"]

def build_X(d, seq=True):
    n=len(d['label']); state=d['state'].astype(np.float32); option=d['option'].astype(np.float32)
    ctx=d['context']; st=d['select_type']; mn=d['min_count']; mx=d['max_count']; turn=d['turn']
    opp=np.zeros((n,len(OPP)+1),np.float32); opp[np.arange(n),d['opp_code']]=1
    parts=[state,option,ctx[:,None].astype(np.float32),st[:,None].astype(np.float32),mn[:,None].astype(np.float32),mx[:,None].astype(np.float32),turn[:,None].astype(np.float32)/20,opp]
    if seq:
        lh=np.zeros((n,18),np.float32); lh[np.arange(n),d['last_type']]=1
        ph=np.zeros((n,18),np.float32); ph[np.arange(n),d['prev_type']]=1; parts += [lh,ph]
    return np.concatenate(parts,axis=1)

def recall(pred, va, d, topk=1):
    ep=d['episode_id']; step=d['step_index']; label=d['label']; groups=defaultdict(list)
    for idx in np.nonzero(va)[0]: groups[(int(ep[idx]),int(step[idx]))].append((float(pred[idx]),int(idx)))
    c=t=0
    for arr in groups.values():
        arr.sort(reverse=True); nsel=int(label[[i for _,i in arr]].sum()); k=max(1,min(topk,nsel if topk==1 else len(arr)))
        c+=len({i for _,i in arr[:k]}&{i for _,i in arr if label[i]==1}); t+=len({i for _,i in arr if label[i]==1})
    return c/max(1,t)

def fit_eval(X,tr,va,d,w,depth,leaf,phase=None,ctx=None):
    mtr=tr.copy(); mva=va.copy()
    if phase is not None: mtr &= d['phase']==phase; mva &= d['phase']==phase
    if ctx is not None: mtr &= d['context']==ctx; mva &= d['context']==ctx
    if mtr.sum()<300 or mva.sum()<50: return None,None
    clf=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,class_weight='balanced',random_state=47)
    clf.fit(X[mtr],d['label'][mtr],sample_weight=w[mtr]); pr=clf.predict_proba(X[mva])[:,1]
    pred=np.full(len(d['label']),-1.0,np.float32); pred[np.nonzero(mva)[0]]=pr
    return clf,pred

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=ROOT/'experiments/20261104_state_value/delta_tree_search'); args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    d=np.load(ROOT/'data/lean4_action_tree_dataset.npz')
    a=np.load(ROOT/'experiments/20261104_state_value/actual_afterstate_rows.npz')
    sv=np.load(ROOT/'experiments/20261104_state_value/state_value_dataset.npz')
    # Train a state-only value model on the same action-train episodes.
    ep0=sv['episode_id']; ue0=np.unique(ep0); rng0=np.random.default_rng(47); rng0.shuffle(ue0)
    nv0=max(1,int(len(ue0)*.2)); val0=set(ue0[:nv0].tolist()); tr0=np.array([int(e) not in val0 for e in ep0])
    from sklearn.ensemble import HistGradientBoostingClassifier
    vm=HistGradientBoostingClassifier(max_iter=120,learning_rate=.06,max_leaf_nodes=15,l2_regularization=.1,random_state=47)
    vm.fit(sv['state'][tr0].astype(np.float32),sv['won'][tr0])
    p0=vm.predict_proba(a['state'].astype(np.float32))[:,1]
    p1=vm.predict_proba(a['next_state'].astype(np.float32))[:,1]
    delta=p1-p0
    dmap={}
    for i in range(len(delta)):
        dmap[(int(a['episode_id'][i]),int(a['step_index'][i]))]=float(delta[i])
    ep=d['episode_id']; step=d['step_index']; label=d['label']; won=d['won']; pd=d['prize_diff']
    # selected-row delta and base weights
    delta_row=np.zeros(len(label),np.float32)
    for i in range(len(label)):
        if label[i]==1:
            delta_row[i]=dmap.get((int(ep[i]),int(step[i])),0.0)
    base=np.where(won>0,2.0,.5)*(1.0+.05*np.abs(pd))
    ue=np.unique(ep); rng=np.random.default_rng(47); rng.shuffle(ue); nv=max(1,int(len(ue)*.2)); val=set(ue[:nv].tolist()); tr=np.array([int(e) not in val for e in ep]); va=~tr
    Xseq=build_X(d,True); Xno=build_X(d,False)
    out=[]; best=(-1,None)
    for seq,X in [('seq',Xseq),('noseq',Xno)]:
      for alpha in [0.0,5.0,10.0]:
        w=base.copy()
        # only selected actions get delta factor
        fac=np.clip(1.0+alpha*delta_row,0.2,2.0)
        w=np.where(label==1,w*fac,w)
        for depth in [6,8]:
          for leaf in [80,160]:
            # global
            clf,pr=fit_eval(X,tr,va,d,w,depth,leaf)
            if pr is not None:
                r1=recall(pr,va,d,1); r3=recall(pr,va,d,3)
                rec={'mode':'global','seq':seq,'alpha':alpha,'depth':depth,'leaf':leaf,'top1':r1,'top3':r3}
                out.append(rec); print(rec,flush=True)
                if r1>best[0]: best=(r1,rec); (args.out/'best.json').write_text(json.dumps(rec,indent=2))
            # phase split
            pred=np.full(len(label),-1.0,np.float32); ok=True
            for p in [0,1,2]:
                _,pp=fit_eval(X,tr,va,d,w,depth,leaf,phase=p)
                if pp is None: continue
                pred[pp>=0]=pp[pp>=0]
            if (pred>=0).any():
                r1=recall(pred,va,d,1); r3=recall(pred,va,d,3)
                rec={'mode':'phase','seq':seq,'alpha':alpha,'depth':depth,'leaf':leaf,'top1':r1,'top3':r3}
                out.append(rec); print(rec,flush=True)
                if r1>best[0]:
                    best=(r1,rec); (args.out/'best.json').write_text(json.dumps(rec,indent=2))
                    # save best phase trees
                    for p in [0,1,2]:
                        c,pp=fit_eval(X,tr,va,d,w,depth,leaf,phase=p)
                        if c is not None: (args.out/f'best_phase_{p}.txt').write_text(export_text(c,max_depth=5,decimals=2))
    with (args.out/'runs.jsonl').open('w') as fh:
        for r in out: fh.write(json.dumps(r)+'\n')
    print('BEST',best,'runs',len(out),flush=True)

if __name__=='__main__': main()
