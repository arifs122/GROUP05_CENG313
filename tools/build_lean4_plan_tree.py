#!/usr/bin/env python3
"""Plan-level win-probability tree for lean4 deck only.

Each row is the action actually selected by a lean4 pilot. Features include our
state, the selected option, opponent archetype, and the opponent's previous two
actions. Target is the game result. No non-lean4 action labels are used.
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/mazlum')); sys.path.insert(0,str(ROOT/'tools'))
import features as F  # noqa: E402
import replay_meta as R  # noqa: E402

LEAN=sorted(int(x) for x in (ROOT/'bots/dragapult_lean4/deck.csv').read_text().split() if x.strip())
LEAN5=sorted(int(x) for x in (ROOT/'bots/dragapult_lean5/deck.csv').read_text().split() if x.strip())
OPP=['Dragapult','Mega Kangaskhan','Mega Excadrill','Mega Lopunny','Banette','Alakazam','Hydrapple','Mega Abomasnow','Festival Lead','Mega Lucario','Archaludon','Slowking-Kangaskhan','Teal Mask Ogerpon','Marnie\'s Grimmsnarl']

def plan_of(o, obs, side):
    typ=int(o.get('type',-1)); actor,_=F._option_cards(obs,side,o)
    if typ==13: return 'attack'
    if typ==12: return 'retreat'
    if typ==8: return 'energy'
    if typ==10: return 'ability'
    if typ==9: return 'evolve'
    if typ==7:
        # utility / setup / draw / disruption by actor role
        return 'play_util' if actor not in (119,120,121,140,235,1071,184,112,31) else 'play_pokemon'
    if typ in (3,4,5,6): return 'choose_card'
    if typ==14: return 'end'
    return 'other'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--master',type=Path,default=ROOT/'results/master_replays.csv'); ap.add_argument('--out-data',type=Path,default=ROOT/'data/lean4_plan_dataset.npz'); ap.add_argument('--out-dir',type=Path,default=ROOT/'results'); ap.add_argument('--seed',type=int,default=43); args=ap.parse_args()
    rows=list(csv.DictReader(args.master.open(encoding='utf-8')))
    lean_set=set(LEAN)
    def near_deck(d):
        c=Counter(d)
        # exclude Munkidori/Chi-Yu variants; allow the old lean4 family only
        if c.get(112,0) or c.get(31,0): return False
        dist=sum(max(0,c[k]-Counter(LEAN).get(k,0)) for k in set(c)|set(LEAN))
        return dist <= 6
    games=[]
    for r in rows:
        # any source is allowed, but near_deck rejects Munkidori/Chi-Yu lists
        d0=list(map(int,r['deck0'].split())); d1=list(map(int,r['deck1'].split()))
        if near_deck(d0) or near_deck(d1): games.append(r)
    games.sort(key=lambda r:(r['source']!='lean_57629056',r['episode_id']))
    print('games',len(games),Counter(r['source'] for r in games),flush=True)
    S=[]; O=[]; C=[]; M=[]; XC=[]; ST=[]; MIN=[]; MAX=[]; PLAN=[]; EP=[]; STEP=[]; WON=[]; OPPCODE=[]; LAST=[]; PREV=[]; ACTOR=[]; TARGET=[]; TURN=[]; SIDE=[]
    for gi,r in enumerate(games):
        ep=json.loads(Path(r['path']).read_text()); decks=R.extract_decks(ep)
        if not decks: continue
        d0=list(decks[0]); d1=list(decks[1]); side=0 if near_deck(d0) else (1 if near_deck(d1) else None)
        if side is None: continue
        opp_arch=r['arch1'] if side==0 else r['arch0']
        # lean5 is treated as the same strategy family; deck histogram uses lean4 for comparable features
        opp_code=OPP.index(opp_arch) if opp_arch in OPP else len(OPP)
        won=1 if r['winner']!='' and int(r['winner'])==side else 0
        last={0:None,1:None}; prev={0:None,1:None}
        for d in F.iter_decisions(ep):
            s=d.player
            # update opponent action history only for opponent decisions
            if s != side:
                try:
                    oo=(d.select.get('option') or [])[d.action[0]] if d.action else {}
                    last[s]=(int(oo.get('type',-1)), int((F._option_cards(d.observation,s,oo)[0] or 0))) if oo else None
                except Exception: pass
                continue
            if not d.action: continue
            sel=d.select or {}; ctx=int(sel.get('context',-1)); st=int(sel.get('type',-1)); mn=int(sel.get('minCount',0)); mx=int(sel.get('maxCount',0))
            state=F.state_vector(d.observation,LEAN); opts,mask=F.options_matrix(d.observation); ci=F.card_ids_matrix(d.observation)
            for ix in d.action:
                try: o=(sel.get('option') or [])[ix]
                except Exception: continue
                actor,target=F._option_cards(d.observation,side,o)
                S.append(state); O.append(opts[int(ix)]); C.append(ci[int(ix)])
                XC.append(ctx); ST.append(st); MIN.append(mn); MAX.append(mx); PLAN.append(plan_of(o,d.observation,side))
                EP.append(int(r['episode_id'])); STEP.append(int(d.step_index)); WON.append(won); OPPCODE.append(opp_code)
                lt=last[1-side] or (-1,0); pt=prev[1-side] or (-1,0)
                LAST.append(lt[0]); PREV.append(pt[0]); ACTOR.append(int(actor or 0)); TARGET.append(int(target or 0)); TURN.append(int((d.observation.get('current') or {}).get('turn',0) or 0)); SIDE.append(side)
            prev=last.copy()
        if gi%20==0: print('game',gi,'rows',len(S),flush=True)
    if not S: raise SystemExit('no rows')
    state=np.stack(S).astype(np.float16); option=np.stack(O).astype(np.float16); cids=np.stack(C).astype(np.int16)
    arr={k:np.asarray(v) for k,v in dict(label=WON,context=XC,select_type=ST,min_count=MIN,max_count=MAX,plan=PLAN,episode_id=EP,step_index=STEP,opp_code=OPPCODE,last_type=LAST,prev_type=PREV,actor=ACTOR,target=TARGET,turn=TURN,side=SIDE).items()}
    args.out_data.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out_data,state=state,option=option,card_ids=cids,**arr)
    print('rows',len(WON),'episodes',len(set(EP)),'plans',Counter(PLAN),'wins',sum(WON),flush=True)
    # one row per selected action; plan tree with outcome target
    XC=np.asarray(XC,dtype=np.int16); ST=np.asarray(ST,dtype=np.int16); MIN=np.asarray(MIN,dtype=np.int16); MAX=np.asarray(MAX,dtype=np.int16); TURN=np.asarray(TURN,dtype=np.int16); WO=np.asarray(WON,dtype=np.int8); OPPCODE=np.asarray(OPPCODE,dtype=np.int16); LAST=np.asarray(LAST,dtype=np.int16); PREV=np.asarray(PREV,dtype=np.int16)
    n=len(WON); ue=np.unique(EP); rng=np.random.default_rng(args.seed); rng.shuffle(ue); nv=max(1,int(len(ue)*0.2)); val=set(ue[:nv].tolist()); tr=np.array([int(e) not in val for e in EP]); va=~tr
    plan_names=sorted(set(PLAN)); plan_code=np.array([plan_names.index(p) for p in PLAN],dtype=np.int16)
    opp_oh=np.zeros((n,len(OPP)+1),dtype=np.float32); opp_oh[np.arange(n),OPPCODE]=1
    last_oh=np.zeros((n,18),dtype=np.float32); last_oh[np.arange(n),np.clip(LAST,0,17)]=1
    prev_oh=np.zeros((n,18),dtype=np.float32); prev_oh[np.arange(n),np.clip(PREV,0,17)]=1
    plan_oh=np.zeros((n,len(plan_names)),dtype=np.float32); plan_oh[np.arange(n),plan_code]=1
    # option features + state + opp + sequence + plan
    X=np.concatenate([state,option,
                      XC[:,None].astype(np.float32),ST[:,None].astype(np.float32),MIN[:,None].astype(np.float32),MAX[:,None].astype(np.float32),
                      TURN[:,None].astype(np.float32)/20, opp_oh,last_oh,prev_oh,plan_oh],axis=1)
    names=list(F.STATE_FEATURE_NAMES)+list(F.OPTION_FEATURE_NAMES)+['context','select_type','min_count','max_count','turn_norm']+[f'opp_{x}' for x in OPP+['other']]+[f'last_{i}' for i in range(18)]+[f'prev_{i}' for i in range(18)]+[f'plan_{x}' for x in plan_names]
    clf=DecisionTreeClassifier(max_depth=6,min_samples_leaf=80,class_weight='balanced',random_state=args.seed)
    clf.fit(X[tr],WO[tr])
    p=clf.predict_proba(X[va])[:,1]
    print('val AUC',roc_auc_score(WO[va],p),'nodes',clf.get_n_leaves(),'depth',clf.get_depth(),flush=True)
    txt=export_text(clf,feature_names=names,max_depth=5,decimals=2)
    (args.out_dir/'lean4_plan_tree.txt').write_text(txt)
    print(txt[:15000])

if __name__ == "__main__":
    main()
