#!/usr/bin/env python3
"""Convert lean4 option-level rows into decision-level BC arrays.

Input: data/lean4_action_tree_dataset.npz (one row per legal option)
Output: data/lean4_bc_dataset.npz
  state [D,50], options [D,40,48], mask [D,40], card_ids [D,40,2],
  label [D], episode_id [D], step_index [D], split [D]
"""
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'data/lean4_action_tree_dataset.npz'
OUT=ROOT/'data/lean4_bc_dataset.npz'
MAX_OPT=40

def main():
    z=np.load(SRC)
    state_all=np.asarray(z['state']); option_all=np.asarray(z['option']); card_all=np.asarray(z['card_ids']); label_all=np.asarray(z['label'])
    ep=np.asarray(z['episode_id'],dtype=np.int64); step=np.asarray(z['step_index'],dtype=np.int64)
    # Stable composite key; step < 2^31 and episode < 2^40 in practice.
    key=ep*(1<<32)+step
    order=np.argsort(key,kind='stable'); key=key[order]
    # group boundaries
    starts=np.flatnonzero(np.r_[True,key[1:]!=key[:-1]])
    ends=np.r_[starts[1:],len(key)]
    D=len(starts)
    S=np.empty((D,state_all.shape[1]),np.float16)
    O=np.zeros((D,MAX_OPT,option_all.shape[1]),np.float16)
    M=np.zeros((D,MAX_OPT),np.float32)
    C=np.zeros((D,MAX_OPT,2),np.int16)
    L=np.full(D,-1,np.int64)
    EP=np.empty(D,np.int64); ST=np.empty(D,np.int64)
    skipped=0
    for j,(a,b) in enumerate(zip(starts,ends)):
        idx=order[a:b]
        if len(idx)>MAX_OPT:
            skipped+=1; idx=idx[:MAX_OPT]
        S[j]=state_all[idx[0]]
        O[j,:len(idx)]=option_all[idx]
        M[j,:len(idx)]=1.0
        C[j,:len(idx)]=card_all[idx]
        y=label_all[idx].astype(np.int8)
        pos=np.flatnonzero(y==1)
        if len(pos)==0:
            skipped+=1; continue
        # multi-select rows are abbreviated to the first selected option
        L[j]=int(pos[0])
        EP[j]=int(ep[idx[0]]); ST[j]=int(step[idx[0]])
        if j%50000==0: print('group',j,'/',D,flush=True)
    # Drop unlabeled decisions
    keep=L>=0
    S,O,M,C,L,EP,ST=S[keep],O[keep],M[keep],C[keep],L[keep],EP[keep],ST[keep]
    # episode-level split
    ue=np.unique(EP); rng=np.random.default_rng(47); rng.shuffle(ue)
    val=set(ue[:max(1,int(len(ue)*0.2))].tolist())
    split=np.array([1 if int(e) in val else 0 for e in EP],dtype=np.int8)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT,state=S,options=O,mask=M,card_ids=C,label=L,
                        episode_id=EP,step_index=ST,split=split)
    print('decisions',len(L),'episodes',len(ue),'skipped',skipped,'wrote',OUT,OUT.stat().st_size/1e6,'MB',flush=True)
    print('max options used',int(M.sum(1).max()),'mean',float(M.sum(1).mean()),flush=True)
    print('train',int((split==0).sum()),'val',int((split==1).sum()),flush=True)

if __name__=='__main__': main()
