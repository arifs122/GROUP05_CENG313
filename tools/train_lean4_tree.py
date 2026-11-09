#!/usr/bin/env python3
"""Train interpretable per-context trees on lean4-deck decisions only.

No Munkidori-deck wins are used: the source dataset contains only the exact
lean4 60-card list.  Validation is episode-grouped.
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "mazlum"))
import features as F  # noqa: E402

TOP_OPP = ["Dragapult","Mega Kangaskhan","Mega Excadrill","Mega Lopunny","Banette",
           "Alakazam","Hydrapple","Mega Abomasnow","Festival Lead","Mega Lucario",
           "Archaludon","Slowking-Kangaskhan","Teal Mask Ogerpon","Marnie's Grimmsnarl"]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,default=ROOT/'data/lean4_tree_dataset.npz'); ap.add_argument('--master',type=Path,default=ROOT/'results/master_replays.csv'); ap.add_argument('--out-dir',type=Path,default=ROOT/'results'); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    d=np.load(args.data)
    state=d['state'].astype(np.float32); option=d['option'].astype(np.float32)
    label=d['label'].astype(np.int8); context=d['context'].astype(np.int16)
    stype=d['select_type'].astype(np.int16); mins=d['min_count'].astype(np.int16); maxs=d['max_count'].astype(np.int16)
    eps=d['episode_id']; won=d['won'].astype(np.float32); step=d['step_index']
    # opponent archetype per episode, from master (lean4 side)
    ep_opp={}
    for r in csv.DictReader(args.master.open(encoding='utf-8')):
        if r['source'] not in ('lean_57629056','ours'): continue
        # both sides; find lean4 side
        # master has deck lists; use arch fields + source; only one Dragapult side mostly
        eid=int(r['episode_id'])
        # opponent of lean4: if either side is not Dragapult? use opponent arch where current deck lean4 is Dragapult
        if r['arch0']=='Dragapult' and r['arch1']!='Dragapult': ep_opp[eid]=r['arch1']
        elif r['arch1']=='Dragapult' and r['arch0']!='Dragapult': ep_opp[eid]=r['arch0']
        elif r['arch0']=='Dragapult' and r['arch1']=='Dragapult': ep_opp.setdefault(eid,'Dragapult')
    # Build X: state + option + context + type + min/max + opponent one-hot
    n=len(label)
    opp_code=np.array([TOP_OPP.index(ep_opp.get(int(e),'unknown')) if ep_opp.get(int(e),'unknown') in TOP_OPP else len(TOP_OPP) for e in eps],dtype=np.int16)
    X=np.concatenate([state, option,
                      context[:,None].astype(np.float32), stype[:,None].astype(np.float32),
                      mins[:,None].astype(np.float32), maxs[:,None].astype(np.float32),
                      (opp_code[:,None]==np.arange(len(TOP_OPP)+1)[None,:]).astype(np.float32)],axis=1)
    names=list(F.STATE_FEATURE_NAMES)+list(F.OPTION_FEATURE_NAMES)+['context','select_type','min_count','max_count']+[f'opp_{x}' for x in TOP_OPP+['unknown']]
    # episode split
    rng=np.random.default_rng(args.seed); ue=np.unique(eps); rng.shuffle(ue); nval=max(1,int(len(ue)*0.2)); val_eps=set(ue[:nval].tolist())
    tr=np.array([int(e) not in val_eps for e in eps]); va=~tr
    print('rows',n,'episodes',len(ue),'train',tr.sum(),'val',va.sum(),'opp missing',int((opp_code==len(TOP_OPP)).sum()),flush=True)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    metrics=[]
    # contexts with enough validation/training rows
    for ctx in sorted(set(context.tolist())):
        mtr=tr & (context==ctx); mva=va & (context==ctx)
        if mtr.sum()<200 or mva.sum()<50: continue
        # per-decision grouping for top1
        clf=DecisionTreeClassifier(max_depth=6,min_samples_leaf=80,class_weight='balanced',random_state=args.seed)
        w=np.where(won[mtr]>0,1.5,0.5)
        clf.fit(X[mtr],label[mtr],sample_weight=w)
        pred=clf.predict_proba(X[mva])[:,1]
        # group validation decisions by episode/step
        groups=defaultdict(list)
        for idx,p in zip(np.nonzero(mva)[0],pred):
            groups[(int(eps[idx]),int(step[idx]))].append((float(p),int(idx)))
        correct=total=0; multi=0
        for key,arr in groups.items():
            arr.sort(reverse=True)
            # choose as many as selected count in this decision
            nsel=int(label[[i for _,i in arr]].sum())
            k=max(1,nsel)
            chosen={i for _,i in arr[:k]}
            true={i for _,i in arr if label[i]==1}
            correct+=len(chosen & true); total+=len(true); multi+= (nsel>1)
        acc=correct/max(1,total)
        metrics.append({'context':int(ctx),'train_rows':int(mtr.sum()),'val_rows':int(mva.sum()),'val_decisions':len(groups),'top1_recall':acc,'multi_decisions':multi,'depth':clf.get_depth(),'leaves':clf.get_n_leaves()})
        txt=export_text(clf,feature_names=names,max_depth=4,decimals=2)
        (args.out_dir/f'lean4_tree_context_{ctx}.txt').write_text(txt)
        print(f'ctx {ctx:3} train={mtr.sum():6} val={mva.sum():5} decisions={len(groups):5} recall={acc:.3f} depth={clf.get_depth()} leaves={clf.get_n_leaves()}',flush=True)
    with (args.out_dir/'lean4_tree_metrics.csv').open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=list(metrics[0]) if metrics else ['context']);w.writeheader();w.writerows(metrics)
    print('\nmain context rules (first 100 lines):')
    print((args.out_dir/'lean4_tree_context_0.txt').read_text()[:12000])

if __name__=='__main__': main()
