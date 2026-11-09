from __future__ import annotations
import sys, torch, numpy as np
from pathlib import Path
from torch import nn
_HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(_HERE))
import features, rule_main

class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.emb=nn.Embedding(4096,32,padding_idx=0)
        self.s=nn.Sequential(nn.Linear(50,128),nn.ReLU(),nn.Linear(128,128))
        self.o=nn.Sequential(nn.Linear(48+64,128),nn.ReLU(),nn.Linear(128,128))
        self.h=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,s,o,i):
        a=self.o(torch.cat([o,self.emb(i[...,0]),self.emb(i[...,1])],-1)); b=self.s(s).unsqueeze(1)
        return self.h(torch.cat([a,b.expand(-1,a.size(1),-1)],-1)).squeeze(-1)
_NET=None
def _net():
    global _NET
    if _NET is None:
        m=Net(); raw=torch.load(Path(__file__).with_name('model.pt'),map_location='cpu',weights_only=True)
        mapped={}
        for k,v in raw.items():
            nk=k.replace('state.','s.').replace('opt.','o.').replace('head.','h.').replace('card.','emb.')
            mapped[nk]=v
        m.load_state_dict(mapped); m.eval(); _NET=m
    return _NET
def _deck():
    return [int(x) for x in (Path(__file__).with_name('deck.csv')).read_text().split() if x.strip()]
def agent(obs):
    sel=obs.get('select')
    if sel is None: return rule_main.agent(obs)
    if int(sel.get('type',-1))!=0:
        return rule_main.agent(obs)
    opts=sel.get('option') or []
    if not opts: return []
    try:
        s=torch.from_numpy(features.state_vector(obs,_deck()).astype(np.float32)).unsqueeze(0)
        o,mask=features.options_matrix(obs); c=features.card_ids_matrix(obs)
        o=torch.from_numpy(o.astype(np.float32)).unsqueeze(0); c=torch.from_numpy(c.astype(np.int64)).unsqueeze(0); mk=torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            raw=_net()(s,o,c).masked_fill(mk==0,-1e9)[0]
            probs=torch.softmax(raw, dim=-1)
            top=torch.topk(probs, min(2, len(probs)))
            if float(top.values[0]) < 0.55 or (len(top.values)>1 and float(top.values[0]-top.values[1]) < 0.15):
                return rule_main.agent(obs)
            logits=raw
        valid=int(mask.sum())
        if valid<=0: return rule_main.agent(obs)
        k=int(sel.get('maxCount',1) or 1); k=max(1,min(k,valid))
        order=torch.argsort(logits,descending=True).tolist(); take=[]
        for i in order:
            if i<len(opts) and mask[i]>0:
                take.append(int(i))
                if len(take)>=k: break
        return take or rule_main.agent(obs)
    except Exception:
        return rule_main.agent(obs)
