#!/usr/bin/env python3
"""Lean4 family: state value + actual-afterstate advantage dataset.

No search/Q/prior.  Value is a lightweight tabular model trained from states.
For each real lean4 decision we compare V(before) with V(after the chosen
action); forced moves are kept neutral.  This is the first counterfactual-lite
step before generating alternative afterstates.
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score

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

def phase_of(turn,myp,oppp,deck):
    if turn<=3: return 0
    if myp<=2 or oppp<=2 or deck<=10: return 2
    return 1

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--master',type=Path,default=ROOT/'results/master_replays.csv'); ap.add_argument('--out-dir',type=Path,default=ROOT/'experiments/20261104_state_value'); ap.add_argument('--seed',type=int,default=45); args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(args.master.open(encoding='utf-8')))
    games=[]
    for r in rows:
        d0=list(map(int,r['deck0'].split())); d1=list(map(int,r['deck1'].split()))
        if near(Counter(d0)) or near(Counter(d1)): games.append(r)
    games.sort(key=lambda r:(r['source']!='lean_57629056',r['source']!='lean5_57668293',r['episode_id']))
    print('games',len(games),Counter(r['source'] for r in games),flush=True)
    VS=[]; VY=[]; VE=[]; VOPP=[]; VT=[]; VPH=[]; VSTEP=[]
    AS=[]; AO=[]; ACTX=[]; AST=[]; AMIN=[]; AMAX=[]; ADELTA=[]; AFORCED=[]; AEP=[]; ASTEP=[]; AOPP=[]; ATURN=[]; AACT=[]; AWON=[]
    for gi,r in enumerate(games):
        try: ep=json.loads(Path(r['path']).read_text())
        except Exception: continue
        decks=R.extract_decks(ep)
        if not decks: continue
        d0=list(decks[0]); d1=list(decks[1]); side=0 if near(Counter(d0)) else (1 if near(Counter(d1)) else None)
        if side is None: continue
        steps=ep.get('steps') or []
        opp_arch=r['arch1'] if side==0 else r['arch0']; oc=OPP.index(opp_arch) if opp_arch in OPP else len(OPP)
        won=1 if r['winner']!='' and int(r['winner'])==side else 0
        for d in F.iter_decisions(ep):
            if d.player!=side: continue
            cur=d.observation.get('current') or {}
            if not cur or cur.get('players') is None: continue
            state=F.state_vector(d.observation,LEAN)
            turn=int(cur.get('turn',0) or 0); players=cur.get('players') or [{},{}]
            myp=len(players[side].get('prize') or []); oppp=len(players[1-side].get('prize') or []); deckc=int(players[side].get('deckCount',60) or 60)
            ph=phase_of(turn,myp,oppp,deckc)
            VS.append(state); VY.append(won); VE.append(int(r['episode_id'])); VOPP.append(oc); VT.append(turn); VPH.append(ph); VSTEP.append(int(d.step_index))
            if not d.action: continue
            # actual afterstate from the replay: next step observation for this player
            nxt=None
            if d.step_index+1 < len(steps):
                try:
                    nobs=steps[d.step_index+1][side].get('observation') or {}
                    ncur=nobs.get('current') or {}
                    if ncur and ncur.get('result',-1)>=0:
                        nval=1.0 if int(ncur['result'])==side else 0.0
                    elif ncur:
                        nval=float(ncur and 0)  # placeholder; value model later
                        nxt=nobs
                except Exception: pass
            # value model will be trained first; store afterstate row if available
            if nxt is not None:
                nstate=F.state_vector(nxt,LEAN)
                sel=d.select or {}; ctx=int(sel.get('context',-1)); st=int(sel.get('type',-1)); mn=int(sel.get('minCount',0)); mx=int(sel.get('maxCount',0))
                try: o=(sel.get('option') or [])[d.action[0]]
                except Exception: o={}
                opts,mask=F.options_matrix(d.observation); opt=opts[d.action[0]] if d.action[0]<len(opts) else np.zeros(F.OPTION_FEATURE_SIZE,np.float32)
                AS.append(state); AO.append(opt); ACTX.append(ctx); AST.append(st); AMIN.append(mn); AMAX.append(mx); DELTA=np.nan
                ADELTA.append(DELTA); AFORCED.append(1 if int(mask.sum())<=1 else 0); AEP.append(int(r['episode_id'])); ASTEP.append(int(d.step_index)); AOPP.append(oc); ATURN.append(turn); AACT.append(int(d.action[0])); AWON.append(won)
                # store next state separately in a hidden sidecar? use V datasets later
                # append next state/options into arrays via temporary lists
                if 'NEXT' not in globals(): NEXT=[]
                NEXT.append(nstate)
        if gi%100==0: print('game',gi,'v',len(VS),'adv',len(AS),flush=True)
    # Save value dataset
    np.savez_compressed(args.out_dir/'state_value_dataset.npz',state=np.stack(VS).astype(np.float16),won=np.asarray(VY,np.int8),episode_id=np.asarray(VE,np.int64),opp_code=np.asarray(VOPP,np.int16),turn=np.asarray(VT,np.int16),phase=np.asarray(VPH,np.int8),step_index=np.asarray(VSTEP,np.int32))
    print('value rows',len(VY),'episodes',len(set(VE)),flush=True)
    # train value model
    ue=np.unique(VE); rng=np.random.default_rng(args.seed); rng.shuffle(ue); nv=max(1,int(len(ue)*.2)); val=set(ue[:nv].tolist()); tr=np.array([int(e) not in val for e in VE]); va=~tr
    opp_oh=np.zeros((len(VY),len(OPP)+1),np.float32); opp_oh[np.arange(len(VY)),np.asarray(VOPP)]=1
    X=np.concatenate([np.stack(VS).astype(np.float32),opp_oh,np.asarray(VT)[:,None]/20,np.asarray(VPH)[:,None]],axis=1)
    clf=HistGradientBoostingClassifier(max_iter=150,learning_rate=.06,max_leaf_nodes=15,l2_regularization=.1,random_state=args.seed)
    clf.fit(X[tr],np.asarray(VY)[tr])
    p=clf.predict_proba(X[va])[:,1]
    print('VALUE AUC',roc_auc_score(np.asarray(VY)[va],p),'brier',brier_score_loss(np.asarray(VY)[va],p),'acc',accuracy_score(np.asarray(VY)[va],p>=.5),flush=True)
    # Save model via joblib
    import joblib; joblib.dump(clf,args.out_dir/'value_model.joblib')
    # advantage rows need next states; we stored only if available but DELTA placeholder; reconstruct later? For now save selected action/state and next states as a separate npz.
    if AS:
        np.savez_compressed(args.out_dir/'actual_afterstate_rows.npz',state=np.stack(AS).astype(np.float16),next_state=np.stack(NEXT).astype(np.float16),option=np.stack(AO).astype(np.float16),context=np.asarray(ACTX,np.int16),select_type=np.asarray(AST,np.int16),min_count=np.asarray(AMIN,np.int16),max_count=np.asarray(AMAX,np.int16),forced=np.asarray(AFORCED,np.int8),episode_id=np.asarray(AEP,np.int64),step_index=np.asarray(ASTEP,np.int32),opp_code=np.asarray(AOPP,np.int16),turn=np.asarray(ATURN,np.int16),won=np.asarray(AWON,np.int8))
        print('afterstate rows',len(AS),flush=True)
    return 0

if __name__=='__main__': main()
