"""Small NumPy-only context-tree fallback for lean4.

Only single MAIN decisions are overridden, and only at high confidence.
All other decisions stay with the original rule policy.
"""
from __future__ import annotations
from pathlib import Path
import os, json
import numpy as np
import re
import features

OPP = ['Dragapult','Mega Kangaskhan','Mega Excadrill','Mega Lopunny','Banette',
       'Alakazam','Hydrapple','Mega Abomasnow','Festival Lead','Mega Lucario',
       'Archaludon','Slowking-Kangaskhan','Teal Mask Ogerpon',"Marnie's Grimmsnarl"]
MODEL = Path(__file__).with_name('tree_model.npz')

class TreePolicy:
    def __init__(self, deck):
        d = np.load(MODEL)
        self.a = {k: d[k] for k in d.files}
        self.deck = list(deck)
        self.keys = sorted({k.rsplit('_',1)[0] for k in self.a if k.endswith('_left')})
        self.opp_map = {re.sub(r'[^a-z0-9]+','',x.lower()): i for i,x in enumerate(OPP)}
    def _phase(self, cur, me):
        turn = int(cur.get('turn',0) or 0)
        ps = cur.get('players') or [{},{}]
        myp = len((ps[me] or {}).get('prize') or [])
        opp = len((ps[1-me] or {}).get('prize') or [])
        deck = int((ps[me] or {}).get('deckCount',60) or 60)
        if turn <= 3: return 0
        if myp <= 2 or opp <= 2 or deck <= 10: return 2
        return 1
    def _pred(self, key, row):
        left=self.a[key+'_left']; right=self.a[key+'_right']; feat=self.a[key+'_feat']; thr=self.a[key+'_thr']; prob=self.a[key+'_prob']
        i=0
        while left[i] != right[i]:
            f=int(feat[i])
            if f < 0: break
            i = int(left[i]) if row[f] <= thr[i] else int(right[i])
        return float(prob[i])
    def choose(self, obs_dict, opp_arch=None, rule_action=None):
        select=obs_dict.get('select') or {}
        if int(select.get('type',-1)) != 0 or int(select.get('maxCount',0)) != 1 or int(select.get('minCount',0)) != 1:
            return None
        opts=select.get('option') or []
        if len(opts) < 2: return None
        cur=obs_dict.get('current') or {}
        me=int(cur.get('yourIndex',0) or 0)
        state=features.state_vector(obs_dict,self.deck)
        opt,mask=features.options_matrix(obs_dict)
        ctx=int(select.get('context',-1)); st=int(select.get('type',-1)); mn=int(select.get('minCount',0)); mx=int(select.get('maxCount',0))
        turn=int(cur.get('turn',0) or 0)
        key=f'p{self._phase(cur,me)}_c{ctx}'
        if key not in self.keys: return None
        code=self.opp_map.get(re.sub(r'[^a-z0-9]+','',str(opp_arch or '').lower()), len(OPP))
        oh=np.zeros(len(OPP)+1,dtype=np.float32); oh[code]=1.0
        sel=[]
        for i in range(int(mask.sum())):
            row=np.concatenate([state,opt[i],np.array([ctx,st,mn,mx,turn/20.0],dtype=np.float32),oh])
            sel.append((self._pred(key,row),i))
        if not sel: return None
        sel.sort(reverse=True)
        top=sel[0][1]
        if int((opts[top] or {}).get('type',-1))==14: return None
        if rule_action is not None:
            try:
                rt=int((opts[int(rule_action)] or {}).get('type',-1)); tt=int((opts[top] or {}).get('type',-1))
                if rt != tt: return None
            except Exception: return None
        # High-confidence gate: otherwise leave the rule policy alone.
        if os.environ.get('TREE_TRACE'):
            try:
                print('TREE_CAND', json.dumps({'turn':turn,'ctx':ctx,'key':key,'top':int(sel[0][1]),'p':round(sel[0][0],3),'margin':round(sel[0][0]-sel[1][0],3) if len(sel)>1 else 0}), flush=True)
            except Exception: pass
        if sel[0][0] < 0.40 or (len(sel)>1 and sel[0][0]-sel[1][0] < 0.08):
            return None
        if os.environ.get('TREE_TRACE'):
            try:
                cur=obs_dict.get('current') or {}; me=int(cur.get('yourIndex',0) or 0); ps=cur.get('players') or [{},{}]
                print('TREE', json.dumps({'turn':turn,'ctx':ctx,'key':key,'top':int(top),'p':round(sel[0][0],3),'margin':round(sel[0][0]-sel[1][0],3),'rule_opts':[(i,int((o or {}).get('type',-1))) for i,o in enumerate(opts)[:8]],'my_active':(ps[me].get('active') or [{}])[0].get('id'),'opp_active':(ps[1-me].get('active') or [{}])[0].get('id'),'prizes':[len(ps[me].get('prize') or []),len(ps[1-me].get('prize') or [])]}),flush=True)
            except Exception: pass
        return int(top)
_POLICY=None
def get_policy(deck):
    global _POLICY
    if _POLICY is None: _POLICY=TreePolicy(deck)
    return _POLICY
