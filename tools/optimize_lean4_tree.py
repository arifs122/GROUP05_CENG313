#!/usr/bin/env python3
"""Continuous, checkpointed lean4 action-tree search.

Safety: no submission, no destructive operations. Writes only under the chosen
experiment directory. Validation is episode-grouped. Metric is action top-1
recall (and top-3 is logged too).
"""
from __future__ import annotations
import argparse, csv, json, random, shutil, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT=Path(__file__).resolve().parents[1]
OPP=['Dragapult','Mega Kangaskhan','Mega Excadrill','Mega Lopunny','Banette','Alakazam','Hydrapple','Mega Abomasnow','Festival Lead','Mega Lucario','Archaludon','Slowking-Kangaskhan','Teal Mask Ogerpon',"Marnie's Grimmsnarl"]

def build_X(d, seq=True, phase=False):
    n=len(d['label'])
    state=d['state'].astype(np.float32); option=d['option'].astype(np.float32)
    ctx=d['context']; st=d['select_type']; mn=d['min_count']; mx=d['max_count']; turn=d['turn']
    opp=np.zeros((n,len(OPP)+1),np.float32); opp[np.arange(n),d['opp_code']]=1
    parts=[state,option,ctx[:,None].astype(np.float32),st[:,None].astype(np.float32),mn[:,None].astype(np.float32),mx[:,None].astype(np.float32),turn[:,None].astype(np.float32)/20,opp]
    if seq:
        lh=np.zeros((n,18),np.float32); lh[np.arange(n),d['last_type']]=1
        ph=np.zeros((n,18),np.float32); ph[np.arange(n),d['prev_type']]=1
        parts += [lh,ph]
    if phase:
        po=np.zeros((n,3),np.float32); po[np.arange(n),d['phase']]=1
        parts += [po]
    return np.concatenate(parts,axis=1)

def recall(pred, va, d, topk=1):
    ep=d['episode_id']; step=d['step_index']; label=d['label']
    groups=defaultdict(list)
    for idx,p in zip(np.nonzero(va)[0],pred): groups[(int(ep[idx]),int(step[idx]))].append((float(p),int(idx)))
    cor=tot=0
    for arr in groups.values():
        arr.sort(reverse=True)
        nsel=int(label[[i for _,i in arr]].sum()); k=max(1,min(topk,nsel if topk==1 else len(arr)))
        chosen={i for _,i in arr[:k]}; true={i for _,i in arr if label[i]==1}
        cor += len(chosen & true); tot += len(true)
    return cor/max(1,tot)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=ROOT/'data/lean4_action_tree_dataset.npz')
    ap.add_argument('--hours',type=float,default=5.0)
    ap.add_argument('--out',type=Path,default=None)
    ap.add_argument('--seed',type=int,default=123)
    args=ap.parse_args()
    out=args.out or (ROOT/'experiments'/f'autoloop_{time.strftime("%Y%m%d_%H%M%S")}')
    out.mkdir(parents=True,exist_ok=True)
    d=np.load(args.data)
    ep=d['episode_id']; ue=np.unique(ep); rng=np.random.default_rng(args.seed); rng.shuffle(ue); nv=max(1,int(len(ue)*.2)); val=set(ue[:nv].tolist()); tr=np.array([int(e) not in val for e in ep]); va=~tr
    w=np.where(d['won']>0,2.0,.5)*(1.0+.05*np.abs(d['prize_diff']))
    print('rows',len(ep),'train',int(tr.sum()),'val',int(va.sum()),'out',out,flush=True)
    rng2=random.Random(args.seed)
    deadline=time.time()+max(60.0,args.hours*3600.0)
    runs=[]; best=(-1,None)
    # deterministic initial configs, then random search
    configs=[]
    for depth in [5,6,7,8,10]:
        for leaf in [40,80,120,200,400]:
            for seq in [True,False]:
                for phase in [True,False]:
                    configs.append({'depth':depth,'leaf':leaf,'seq':seq,'phase':phase})
    rng2.shuffle(configs); it=0
    while time.time()<deadline and it<len(configs):
        cfg=configs[it]; it+=1
        try:
            if cfg['phase']:
                # phase-specific trees: build without phase one-hot
                X=build_X(d,seq=cfg['seq'],phase=False)
                pred=np.full(len(ep),-1.0,dtype=np.float32)
                for p in sorted(set(d['phase'].tolist())):
                    mtr=tr&(d['phase']==p); mva=va&(d['phase']==p)
                    if mtr.sum()<1000 or mva.sum()<200: continue
                    clf=DecisionTreeClassifier(max_depth=cfg['depth'],min_samples_leaf=cfg['leaf'],class_weight='balanced',random_state=args.seed)
                    clf.fit(X[mtr],d['label'][mtr],sample_weight=w[mtr])
                    pred[np.nonzero(mva)[0]]=clf.predict_proba(X[mva])[:,1]
                mask=pred>=-0.5
                r1=recall(pred,va,d,1); r3=recall(pred,va,d,3)
            else:
                X=build_X(d,seq=cfg['seq'],phase=True)
                clf=DecisionTreeClassifier(max_depth=cfg['depth'],min_samples_leaf=cfg['leaf'],class_weight='balanced',random_state=args.seed)
                clf.fit(X[tr],d['label'][tr],sample_weight=w[tr])
                pr=clf.predict_proba(X[va])[:,1]
                pred=np.full(len(ep),-1.0,np.float32); pred[np.nonzero(va)[0]]=pr
                r1=recall(pred,va,d,1); r3=recall(pred,va,d,3)
            rec={'iter':it,'depth':cfg['depth'],'leaf':cfg['leaf'],'seq':cfg['seq'],'phase':cfg['phase'],'top1':r1,'top3':r3}
            runs.append(rec)
            print(rec,flush=True)
            with (out/'runs.jsonl').open('a') as fh: fh.write(json.dumps(rec)+'\n')
            if r1>best[0]:
                best=(r1,rec)
                # Save a compact best config + a tree for inspection (global only)
                (out/'best.json').write_text(json.dumps(best[1],indent=2))
                Xb=build_X(d,seq=cfg['seq'],phase=cfg['phase'])
                clf=DecisionTreeClassifier(max_depth=cfg['depth'],min_samples_leaf=cfg['leaf'],class_weight='balanced',random_state=args.seed)
                if cfg['phase']:
                    # save each phase rules
                    for p in sorted(set(d['phase'].tolist())):
                        m=tr&(d['phase']==p)
                        if m.sum()<100: continue
                        c=DecisionTreeClassifier(max_depth=cfg['depth'],min_samples_leaf=cfg['leaf'],class_weight='balanced',random_state=args.seed)
                        c.fit(Xb[m],d['label'][m],sample_weight=w[m])
                        (out/f'best_phase_{p}.txt').write_text(export_text(c,max_depth=5,decimals=2))
                else:
                    clf.fit(Xb[tr],d['label'][tr],sample_weight=w[tr])
                    (out/'best_global.txt').write_text(export_text(clf,max_depth=5,decimals=2))
                print('NEW BEST',best,flush=True)
        except Exception as e:
            print('ERR',cfg,type(e).__name__,str(e)[:200],flush=True)
            with (out/'errors.log').open('a') as fh: fh.write(json.dumps({'cfg':cfg,'error':repr(e)})+'\n')
        if it%5==0:
            time.sleep(1)
    print('DONE best',best,'runs',len(runs),flush=True)

if __name__=='__main__': main()
