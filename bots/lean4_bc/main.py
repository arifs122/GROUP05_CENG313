from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import features

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(4096, 32, padding_idx=0)
        self.s = nn.Sequential(nn.Linear(50, 128), nn.ReLU(), nn.Linear(128, 128))
        self.o = nn.Sequential(nn.Linear(48 + 64, 128), nn.ReLU(), nn.Linear(128, 128))
        self.h = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, s, o, i):
        a = self.o(torch.cat([o, self.emb(i[..., 0]), self.emb(i[..., 1])], -1))
        b = self.s(s).unsqueeze(1)
        return self.h(torch.cat([a, b.expand(-1, a.size(1), -1)], -1)).squeeze(-1)

_DECK = None
_NET = None
def _deck():
    global _DECK
    if _DECK is None:
        _DECK = [int(x) for x in (Path(__file__).with_name('deck.csv')).read_text().split() if x.strip()]
    return _DECK
def _net():
    global _NET
    if _NET is None:
        m = Net(); m.load_state_dict(torch.load(Path(__file__).with_name('model.pt'), map_location='cpu', weights_only=True)); m.eval(); _NET = m
    return _NET

def agent(obs):
    sel = obs.get('select')
    if sel is None:
        return list(_deck())
    opts = sel.get('option') or []
    if not opts:
        return []
    try:
        s = torch.from_numpy(features.state_vector(obs, _deck()).astype(np.float32)).unsqueeze(0)
        o, mask = features.options_matrix(obs)
        c = features.card_ids_matrix(obs)
        o = torch.from_numpy(o.astype(np.float32)).unsqueeze(0)
        c = torch.from_numpy(c.astype(np.int64)).unsqueeze(0)
        mk = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            logits = _net()(s, o, c).masked_fill(mk == 0, -1e9)[0]
        valid = int(mask.sum())
        if valid <= 0:
            return []
        k = int(sel.get('maxCount', 1) or 1)
        k = max(1, min(k, valid))
        order = torch.argsort(logits, descending=True).tolist()
        take = []
        for i in order:
            if i < len(opts) and mask[i] > 0:
                take.append(int(i))
                if len(take) >= k:
                    break
        return take or [0]
    except Exception:
        # legal fallback: never forfeit because of an inference error
        return list(range(min(int(sel.get('maxCount', 1) or 1), len(opts))))
