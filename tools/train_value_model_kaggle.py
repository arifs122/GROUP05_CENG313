#!/usr/bin/env python3
"""Kaggle-ready Dragapult value-model training (self-contained).

Upload your replays as a Kaggle dataset, e.g.:

    ptcg-replays/
      episode-*.json

Then in a Kaggle notebook:

    !python train_value_model_kaggle.py \
        --replays /kaggle/input/ptcg-replays \
        --out /kaggle/working/value_model.py

Only NumPy is required (available on Kaggle).  No PyTorch, no engine, no repo
dependencies.  It reads only public state + own hand and produces a tiny linear
win-probability model that can be copied into the bot bundle.

The model deliberately uses only Dragapult-perspective states.  Hidden opponent
cards are never used.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np

FEATURE_NAMES = [
    "bias", "turn", "early", "late",
    "my_prizes", "opp_prizes", "prize_diff", "prize_ratio",
    "my_deck", "opp_deck", "deck_diff",
    "my_hand", "opp_hand", "hand_diff",
    "my_bench", "opp_bench", "bench_diff",
    "my_active_hp", "opp_active_hp", "hp_diff",
    "my_active_hp_frac", "opp_active_hp_frac",
    "my_active_energy", "opp_active_energy", "energy_diff",
    "my_total_hp", "opp_total_hp", "total_hp_diff",
    "my_total_energy", "opp_total_energy",
    "my_discard", "opp_discard", "discard_diff",
    "stadium", "first",
    "hand_energy", "hand_supporter_like", "hand_item_like",
    "hand_boss", "hand_lillie", "hand_judge", "hand_ultra",
    "hand_poffin", "hand_hammer", "hand_rare_candy",
    "hand_dreepy", "hand_drakloak", "hand_dragapult", "hand_budew",
    "active_dragapult", "active_budew", "active_meowth",
    "bench_pokemon", "bench_energy", "bench_tools",
    "first_player_flag", "opp_seen_total",
    "opp_seen_boss", "opp_seen_dragapult", "opp_seen_dwebble",
    "opp_seen_crustle", "opp_seen_lucario", "opp_seen_grimmsnarl",
    "opp_seen_hydrapple", "opp_seen_starmie", "opp_seen_archaludon",
]


def get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def cid(card):
    return int(get(card, "id", 0) or 0)


def hp(p):
    return int(get(p, "hp", 0) or 0)


def max_hp(p):
    return int(get(p, "maxHp", get(p, "hp", 0)) or 0)


def energy(p):
    return len(get(p, "energies", None) or get(p, "energyCards", None) or [])


def tools(p):
    return len(get(p, "tools", None) or [])


def extract_decks(e):
    try:
        action = e["steps"][0][0]["visualize"][0]["action"]
        if not isinstance(action, list) or len(action) != 2:
            return None
        return [list(map(int, d)) for d in action]
    except (KeyError, IndexError, TypeError):
        return None


def is_dragapult(deck):
    return Counter(deck).get(121, 0) >= 2


def public_opponent_ids(obs, me):
    cur = get(obs, "current")
    players = get(cur, "players", None) or []
    if len(players) < 2:
        return set()
    op = players[1 - me]
    ids = set()

    def add(card):
        if not card:
            return
        x = cid(card)
        if x:
            ids.add(x)
        for group in ("energyCards", "tools", "preEvolution"):
            for child in get(card, group, None) or []:
                add(child)

    for zone in (get(op, "active", None) or [], get(op, "bench", None) or [],
                 get(op, "discard", None) or []):
        for card in zone:
            add(card)
    for card in get(cur, "stadium", None) or []:
        if get(card, "playerIndex", -1) != me:
            add(card)
    for log in get(obs, "logs", None) or []:
        if int(get(log, "playerIndex", -1)) == me:
            continue
        x = get(log, "cardId", None)
        if x is not None:
            ids.add(int(x))
    return ids


def features(obs, me):
    cur = get(obs, "current")
    players = get(cur, "players", None) or []
    if len(players) < 2:
        return [0.0] * len(FEATURE_NAMES)
    my, op = players[me], players[1 - me]
    turn = int(get(cur, "turn", 0) or 0)
    myp = len(get(my, "prize", None) or [])
    opp = len(get(op, "prize", None) or [])
    myd = int(get(my, "deckCount", 0) or 0)
    opd = int(get(op, "deckCount", 0) or 0)
    myh = len(get(my, "hand", None) or [])
    oph = int(get(op, "handCount", 0) or 0)
    myb = len(get(my, "bench", None) or [])
    opb = len(get(op, "bench", None) or [])
    ma = (get(my, "active", None) or [None])[0]
    oa = (get(op, "active", None) or [None])[0]
    mh = hp(ma) if ma else 0
    oh = hp(oa) if oa else 0
    mmax = max(1, max_hp(ma)) if ma else 1
    omax = max(1, max_hp(oa)) if oa else 1
    men = energy(ma) if ma else 0
    oen = energy(oa) if oa else 0
    mymons = [p for p in (get(my, "active", None) or []) + (get(my, "bench", None) or []) if p]
    oomons = [p for p in (get(op, "active", None) or []) + (get(op, "bench", None) or []) if p]
    mthp = sum(hp(p) for p in mymons)
    othp = sum(hp(p) for p in oomons)
    mten = sum(energy(p) for p in mymons)
    oten = sum(energy(p) for p in oomons)
    mybench = get(my, "bench", None) or []
    bench_en = sum(energy(p) for p in mybench)
    bench_tools = sum(tools(p) for p in mybench)
    stadium = get(cur, "stadium", None) or []
    hids = [cid(c) for c in (get(my, "hand", None) or []) if c]
    hand = Counter(hids)
    active_id = cid(ma) if ma else 0
    seen = public_opponent_ids(obs, me)
    energy_ids = {2, 5, 7, 1, 3, 4, 6, 8, 9, 10, 11, 14, 18, 19}
    supporter_like = {1182, 1198, 1210, 1213, 1227, 1240, 1245, 1205}
    item_like = {1079, 1080, 1086, 1097, 1120, 1121, 1152, 1156, 1256}

    def count(ids):
        return sum(hand.get(x, 0) for x in ids)

    return [float(x) for x in [
        1.0, min(turn, 40) / 20.0,
        1.0 if turn <= 4 else 0.0,
        1.0 if (myp <= 2 or opp <= 2 or myd <= 10) else 0.0,
        myp / 6.0, opp / 6.0, (opp - myp) / 6.0, (6 - myp) / 6.0,
        myd / 60.0, opd / 60.0, (opd - myd) / 60.0,
        myh / 15.0, oph / 15.0, (myh - oph) / 15.0,
        myb / 5.0, opb / 5.0, (myb - opb) / 5.0,
        mh / 300.0, oh / 300.0, (oh - mh) / 300.0,
        mh / mmax, oh / omax,
        men / 3.0, oen / 3.0, (men - oen) / 3.0,
        mthp / 1000.0, othp / 1000.0, (othp - mthp) / 1000.0,
        mten / 6.0, oten / 6.0,
        len(get(my, "discard", None) or []) / 40.0,
        len(get(op, "discard", None) or []) / 40.0,
        (len(get(op, "discard", None) or []) - len(get(my, "discard", None) or [])) / 40.0,
        1.0 if stadium else 0.0, 0.0,
        count(energy_ids), count(supporter_like), count(item_like),
        hand.get(1182, 0), hand.get(1227, 0), hand.get(1213, 0), hand.get(1121, 0),
        hand.get(1086, 0), hand.get(1120, 0), hand.get(1079, 0),
        hand.get(119, 0), hand.get(120, 0), hand.get(121, 0), hand.get(235, 0),
        1.0 if active_id == 121 else 0.0,
        1.0 if active_id == 235 else 0.0,
        1.0 if active_id == 1071 else 0.0,
        float(myb), float(bench_en), float(bench_tools),
        1.0 if get(cur, "firstPlayer", None) == me else 0.0,
        len(seen),
        1.0 if 1182 in seen else 0.0,
        1.0 if 121 in seen else 0.0,
        1.0 if 344 in seen else 0.0,
        1.0 if 345 in seen else 0.0,
        1.0 if 678 in seen else 0.0,
        1.0 if 648 in seen else 0.0,
        1.0 if 150 in seen else 0.0,
        1.0 if 1031 in seen else 0.0,
        1.0 if 169 in seen else 0.0,
    ]]


def load_states(root: Path, stride: int, max_episodes: int = 0):
    """Load a bounded, deduplicated set of episode files one by one.

    The old version stored every episode JSON in a dict before applying
    --max-episodes, which could use many GB and trigger Kaggle OOM.  Now the
    file list is sampled first; only the selected files are loaded and each
    JSON is released immediately.
    """
    paths = sorted(root.rglob("episode-*.json"))
    seen_names = set()
    unique = []
    for path in paths:
        if path.name in seen_names:
            continue
        seen_names.add(path.name)
        unique.append(path)
    if max_episodes and len(unique) > max_episodes:
        random.Random(7).shuffle(unique)
        unique = unique[:max_episodes]
    print("selected episode files", len(unique))

    X, y, groups = [], [], []
    for path in unique:
        try:
            e = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        eid = get(get(e, "info", {}) or {}, "EpisodeId")
        decks = extract_decks(e)
        rewards = get(e, "rewards", None) or []
        if eid is None or not decks or len(rewards) != 2 or rewards[0] == rewards[1]:
            del e
            continue
        eid = int(eid)
        for me in (0, 1):
            if not is_dragapult(decks[me]):
                continue
            label = 1.0 if rewards[me] == 1 else 0.0
            n = 0
            for step in get(e, "steps", None) or []:
                if me >= len(step):
                    continue
                obs = get(step[me], "observation", None) or {}
                select = get(obs, "select", None)
                if not select or int(get(select, "context", -1)) != 0:
                    continue
                n += 1
                if n % stride:
                    continue
                f = features(obs, me)
                if len(f) != len(FEATURE_NAMES):
                    continue
                X.append(f)
                y.append(label)
                groups.append(eid)
        del e
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(groups)


def split(groups, seed=7):
    ids = sorted(set(int(x) for x in groups))
    random.Random(seed).shuffle(ids)
    cut = int(len(ids) * 0.8)
    train_ids = set(ids[:cut])
    mask = np.array([int(g) in train_ids for g in groups])
    return mask, ~mask


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def fit(X, y, epochs=60, batch_size=2048, lr=0.01, l2=0.001, seed=7):
    """Mini-batch Adam; never forms the full-dataset gradient."""
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1], dtype=np.float64)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    n = len(X)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb, yb = X[idx], y[idx]
            p = sigmoid(xb @ w)
            grad = xb.T @ (p - yb) / max(1, len(idx)) + l2 * w
            t += 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad * grad)
            mhat = m / (1 - beta1 ** t)
            vhat = v / (1 - beta2 ** t)
            w -= lr * mhat / (np.sqrt(vhat) + eps)
    return w


def auc(y, p):
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if not npos or not nneg:
        return 0.0
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def threshold_metrics(y, p, threshold):
    pred = p >= threshold
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = {
        "n": int(len(y)),
        "base": round(float(y.mean()), 4),
        "logloss": round(float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), 4),
        "brier": round(float(((p - y) ** 2).mean()), 4),
        "accuracy": round(float(((p >= 0.5) == y).mean()), 4),
        "auc": round(auc(y, p), 4),
        "at_0.5": threshold_metrics(y, p, 0.5),
    }
    best = max((threshold_metrics(y, p, t) for t in (0.25, 0.35, 0.45, 0.55, 0.65)), key=lambda r: r["f1"])
    out["best_f1"] = best
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays", type=Path, default=Path("/kaggle/input/ptcg-replays"))
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--max-states", type=int, default=400000)
    args = parser.parse_args()
    X, y, groups = load_states(args.replays, max(1, args.stride), args.max_episodes)
    if not len(X):
        raise SystemExit("no Dragapult states found")
    if args.max_states and len(X) > args.max_states:
        idx = random.Random(7).sample(range(len(X)), args.max_states)
        X = [X[i] for i in idx]
        y = [y[i] for i in idx]
        groups = [groups[i] for i in idx]
    print("states", X.shape, "episodes", len(set(groups)), "positive", round(float(y.mean()), 4))
    tr, te = split(groups)
    mean = X[tr].mean(0); std = X[tr].std(0); std[std < 1e-8] = 1.0
    Xs = (X - mean) / std
    w = fit(Xs[tr], y[tr], epochs=args.epochs, batch_size=args.batch_size,
             lr=args.lr, l2=args.l2)
    print("train", metrics(y[tr], sigmoid(Xs[tr] @ w)))
    print("test ", metrics(y[te], sigmoid(Xs[te] @ w)))

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "config": vars(args),
        "states": int(len(y)),
        "episodes": int(len(set(groups))),
        "train": metrics(y[tr], sigmoid(Xs[tr] @ w)),
        "test": metrics(y[te], sigmoid(Xs[te] @ w)),
    }
    # Path objects are not JSON serializable.
    report["config"]["replays"] = str(report["config"]["replays"])
    report["config"]["out"] = str(report["config"]["out"])
    (args.out / "value_model_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    payload = {
        "feature_names": FEATURE_NAMES,
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "weights": [float(x) for x in w],
    }
    (args.out / "value_model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        '"""Auto-generated value model. Copy next to main.py."""',
        "from __future__ import annotations",
        "import math",
        f"FEATURE_NAMES = {FEATURE_NAMES!r}",
        f"MEAN = {[round(float(x), 8) for x in mean]!r}",
        f"STD = {[round(float(x), 8) for x in std]!r}",
        f"WEIGHTS = {[round(float(x), 8) for x in w]!r}",
        "",
        "def predict(features):",
        "    z = 0.0",
        "    for value, mean, std, weight in zip(features, MEAN, STD, WEIGHTS):",
        "        z += ((value - mean) / std) * weight",
        "    z = max(-40.0, min(40.0, z))",
        "    return 1.0 / (1.0 + math.exp(-z))",
        "",
    ]
    (args.out / "value_model.py").write_text("\n".join(lines), encoding="utf-8")
    print("written", args.out / "value_model.py", args.out / "value_model.json")


if __name__ == "__main__":
    main()
