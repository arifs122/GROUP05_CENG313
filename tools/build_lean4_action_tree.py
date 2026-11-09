#!/usr/bin/env python3
"""Phase-hierarchical action tree for the lean4 deck family.

- lean4 + nearby no-Munkidori/no-Chi-Yu Dragapult decks only
- no matchup-specific trees: opponent archetype is a feature
- features: state, option, context, opponent last/prev action, phase
- label: selected option
- sample weight: wins heavier + prize-diff magnitude
- validation: episode-grouped
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/mazlum')); sys.path.insert(0,str(ROOT/'tools'))
import features as F  # noqa: E402
import replay_meta as R  # noqa: E402

LEAN=list(map(int,(ROOT/'bots/dragapult_lean4/deck.csv').read_text().split()))
LEAN5=list(map(int,(ROOT/'bots/dragapult_lean5/deck.csv').read_text().split()))
OPP=['Dragapult','Mega Kangaskhan','Mega Excadrill','Mega Lopunny','Banette','Alakazam','Hydrapple','Mega Abomasnow','Festival Lead','Mega Lucario','Archaludon','Slowking-Kangaskhan','Teal Mask Ogerpon',"Marnie's Grimmsnarl"]

def near(c):
    if c.get(112,0) or c.get(31,0): return False
    lc=Counter(LEAN)
    return sum(max(0,c[k]-lc.get(k,0)) for k in set(c)|set(lc))<=6

def phase_of(turn, myp, oppp, deck):
    if turn<=3: return 0
    if myp<=2 or oppp<=2 or deck<=10: return 2
    return 1

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--master',type=Path,default=ROOT/'results/master_replays.csv'); ap.add_argument('--out-data',type=Path,default=ROOT/'data/lean4_action_tree_dataset.npz'); ap.add_argument('--out-dir',type=Path,default=ROOT/'results'); ap.add_argument('--seed',type=int,default=44); args=ap.parse_args()
    rows=list(csv.DictReader(args.master.open(encoding='utf-8')))
    games=[]
    for r in rows:
        d0=list(map(int,r['deck0'].split())); d1=list(map(int,r['deck1'].split()))
        if near(Counter(d0)) or near(Counter(d1)):
            games.append(r)
    # prioritize lean exact, then lean5, then rest
    games.sort(key=lambda r:(r['source']!='lean_57629056',r['source']!='lean5_57668293',r['episode_id']))
    print('games',len(games),Counter(r['source'] for r in games),flush=True)
    S=[];O=[];C=[];LBL=[];CTX=[];ST=[];MN=[];MX=[];TURN=[];PH=[];OPPCODE=[];LAST=[];PREV=[];EP=[];STEP=[];WON=[];PD=[];SIDE=[]
    for gi,r in enumerate(games):
        ep=json.loads(Path(r['path']).read_text()); decks=R.extract_decks(ep)
        if not decks: continue
        d0=list(decks[0]);d1=list(decks[1]); side=0 if near(Counter(d0)) else (1 if near(Counter(d1)) else None)
        if side is None: continue
        opp_arch=r['arch1'] if side==0 else r['arch0']; oc=OPP.index(opp_arch) if opp_arch in OPP else len(OPP)
        won=1 if r['winner']!='' and int(r['winner'])==side else 0
        last={0:None,1:None}; prev={0:None,1:None}
        for d in F.iter_decisions(ep):
            s=d.player
            if s!=side:
                try:
                    if d.action:
                        oo=(d.select.get('option') or [])[d.action[0]]
                        last[s]=int(oo.get('type',-1))
                except Exception: pass
                continue
            if not d.action: continue
            sel=d.select or {}; ctx=int(sel.get('context',-1)); st=int(sel.get('type',-1)); mn=int(sel.get('minCount',0)); mx=int(sel.get('maxCount',0))
            cur=d.observation.get('current') or {}; turn=int(cur.get('turn',0) or 0)
            players=cur.get('players') or [{},{}]
            try:
                myp=len(players[side].get('prize') or []); oppp=len(players[1-side].get('prize') or []); deckc=int(players[side].get('deckCount',60) or 60)
            except Exception: myp=oppp=6; deckc=60
            ph=phase_of(turn,myp,oppp,deckc)
            state=F.state_vector(d.observation,LEAN); opts,mask=F.options_matrix(d.observation); ci=F.card_ids_matrix(d.observation)
            selected=set(int(x) for x in d.action if 0<=int(x)<F.MAX_OPTIONS)
            if not selected: continue
            lt=last[1-side] if last[1-side] is not None else -1
            pt=prev[1-side] if prev[1-side] is not None else -1
            pdiff=float(state[F.STATE_FEATURE_NAMES.index('prize_diff')])
            for i in range(int(mask.sum())):
                S.append(state); O.append(opts[i]); C.append(ci[i]); LBL.append(1 if i in selected else 0)
                CTX.append(ctx); ST.append(st); MN.append(mn); MX.append(mx); TURN.append(turn); PH.append(ph)
                OPPCODE.append(oc); LAST.append(max(0,min(17,lt))); PREV.append(max(0,min(17,pt))); EP.append(int(r['episode_id'])); STEP.append(int(d.step_index)); WON.append(won); PD.append(pdiff); SIDE.append(side)
            prev=last.copy()
        if gi%100==0: print('game',gi,'rows',len(S),flush=True)
    if not S: raise SystemExit('no rows')
    state=np.stack(S).astype(np.float16); option=np.stack(O).astype(np.float16); cids=np.stack(C).astype(np.int16)
    label=np.asarray(LBL,dtype=np.int8); ctx=np.asarray(CTX,dtype=np.int16); stype=np.asarray(ST,dtype=np.int16); mn=np.asarray(MN,dtype=np.int16); mx=np.asarray(MX,dtype=np.int16)
    turn=np.asarray(TURN,dtype=np.int16); ph=np.asarray(PH,dtype=np.int8); oc=np.asarray(OPPCODE,dtype=np.int16); last=np.asarray(LAST,dtype=np.int16); prev=np.asarray(PREV,dtype=np.int16)
    ep=np.asarray(EP,dtype=np.int64); step=np.asarray(STEP,dtype=np.int32); won=np.asarray(WON,dtype=np.int8); pd=np.asarray(PD,dtype=np.float32); side=np.asarray(SIDE,dtype=np.int8)
    args.out_data.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out_data,state=state,option=option,card_ids=cids,label=label,context=ctx,select_type=stype,min_count=mn,max_count=mx,turn=turn,phase=ph,opp_code=oc,last_type=last,prev_type=prev,episode_id=ep,step_index=step,won=won,prize_diff=pd,side=side)
    print('rows',len(label),'episodes',len(set(EP)),'selected',int(label.sum()),'phases',Counter(PH),flush=True)
    # features
    opp_oh=np.zeros((len(label),len(OPP)+1),dtype=np.float32); opp_oh[np.arange(len(label)),oc]=1
    last_oh=np.zeros((len(label),18),dtype=np.float32); last_oh[np.arange(len(label)),last]=1
    prev_oh=np.zeros((len(label),18),dtype=np.float32); prev_oh[np.arange(len(label)),prev]=1
    X=np.concatenate([state,option,ctx[:,None].astype(np.float32),stype[:,None].astype(np.float32),mn[:,None].astype(np.float32),mx[:,None].astype(np.float32),turn[:,None].astype(np.float32)/20,opp_oh,last_oh,prev_oh],axis=1)
    names=list(F.STATE_FEATURE_NAMES)+list(F.OPTION_FEATURE_NAMES)+['context','select_type','min_count','max_count','turn_norm']+[f'opp_{x}' for x in OPP+['other']]+[f'last_{i}' for i in range(18)]+[f'prev_{i}' for i in range(18)]
    # episode split
    ue=np.unique(ep); rng=np.random.default_rng(args.seed); rng.shuffle(ue); nv=max(1,int(len(ue)*0.2)); val=set(ue[:nv].tolist()); tr=np.array([int(e) not in val for e in ep]); va=~tr
    print('train',int(tr.sum()),'val',int(va.sum()),'episodes',len(ue),flush=True)
    # phase-specific trees; no matchup-specific split
    metrics=[]
    for phase in sorted(set(ph.tolist())):
        mtr=tr & (ph==phase); mva=va & (ph==phase)
        if mtr.sum()<1000 or mva.sum()<200: continue
        clf=DecisionTreeClassifier(max_depth=7,min_samples_leaf=80,class_weight='balanced',random_state=args.seed)
        w=np.where(won[mtr]>0,2.0,0.5) * (1.0+0.05*np.abs(pd[mtr]))
        clf.fit(X[mtr],label[mtr],sample_weight=w)
        prob=clf.predict_proba(X[mva])[:,1]
        groups=defaultdict(list)
        for idx,p in zip(np.nonzero(mva)[0],prob): groups[(int(ep[idx]),int(step[idx]))].append((float(p),int(idx)))
        correct=total=0
        for key,arr in groups.items():
            arr.sort(reverse=True); nsel=int(label[[i for _,i in arr]].sum()); k=max(1,nsel)
            chosen={i for _,i in arr[:k]}; true={i for _,i in arr if label[i]==1}
            correct+=len(chosen&true); total+=len(true)
        recall=correct/max(1,total)
        metrics.append({'phase':int(phase),'train_rows':int(mtr.sum()),'val_rows':int(mva.sum()),'decisions':len(groups),'top1_recall':recall,'depth':clf.get_depth(),'leaves':clf.get_n_leaves()})
        (args.out_dir/f'lean4_action_tree_phase_{phase}.txt').write_text(export_text(clf,feature_names=names,max_depth=5,decimals=2))
        print('phase',phase,'train',int(mtr.sum()),'val',int(mva.sum()),'decisions',len(groups),'recall',round(recall,4),'depth',clf.get_depth(),'leaves',clf.get_n_leaves(),flush=True)
    with (args.out_dir/'lean4_action_tree_metrics.csv').open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=list(metrics[0]) if metrics else ['phase']);w.writeheader();w.writerows(metrics)
if __name__=='__main__': main()
