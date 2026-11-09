#!/usr/bin/env python3
"""Train a compact context-tree model and export arrays for the bot."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier

ROOT=Path(__file__).resolve().parents[1]
OPP=['Dragapult','Mega Kangaskhan','Mega Excadrill','Mega Lopunny','Banette','Alakazam','Hydrapple','Mega Abomasnow','Festival Lead','Mega Lucario','Archaludon','Slowking-Kangaskhan','Teal Mask Ogerpon',"Marnie's Grimmsnarl"]

def feature_matrix(d):
    n=len(d['label'])
    opp=np.zeros((n,len(OPP)+1),np.float32); opp[np.arange(n),d['opp_code']]=1
    # Deliberately no sequence/value features: deployment must be self-contained.
    return np.concatenate([
        d['state'].astype(np.float32), d['option'].astype(np.float32),
        d['context'][:,None].astype(np.float32), d['select_type'][:,None].astype(np.float32),
        d['min_count'][:,None].astype(np.float32), d['max_count'][:,None].astype(np.float32),
        d['turn'][:,None].astype(np.float32)/20.0, opp,
    ],axis=1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,default=ROOT/'data/lean4_action_tree_dataset.npz'); ap.add_argument('--out',type=Path,default=ROOT/'bots/dragapult_lean4_tree/tree_model.npz'); ap.add_argument('--depth',type=int,default=16); ap.add_argument('--leaf',type=int,default=120); args=ap.parse_args()
    d=np.load(args.data); X=feature_matrix(d); w=np.where(d['won']>0,2.,.5)*(1+.05*np.abs(d['prize_diff'])); ph=d['phase']; ctx=d['context']
    arrays={}
    for p in [0,1,2]:
        for cx in np.unique(ctx):
            m=(ph==p)&(ctx==cx)
            if m.sum()<300: continue
            clf=DecisionTreeClassifier(max_depth=args.depth,min_samples_leaf=args.leaf,class_weight='balanced',random_state=47)
            clf.fit(X[m],d['label'][m],sample_weight=w[m])
            t=clf.tree_
            key=f'p{int(p)}_c{int(cx)}'
            arrays[f'{key}_left']=t.children_left.astype(np.int16)
            arrays[f'{key}_right']=t.children_right.astype(np.int16)
            arrays[f'{key}_feat']=t.feature.astype(np.int16)
            arrays[f'{key}_thr']=t.threshold.astype(np.float32)
            # binary probability at each leaf
            val=t.value.astype(np.float32)
            arrays[f'{key}_prob']=(val[:,0,1]/np.maximum(1.0,val[:,0,:].sum(axis=1))).astype(np.float32)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out,**arrays)
    print('trees',len(arrays)//5,'wrote',args.out,args.out.stat().st_size/1024,'KB')
if __name__=='__main__': main()
