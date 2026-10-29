#!/usr/bin/env python3
"""Train a small pure-Python value model from episode replays.

No NumPy/PyTorch required.  Only Dragapult-perspective MAIN decisions are used.
Features contain no hidden opponent cards.

Usage:
    python3 tools/train_value_model.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "bots" / "dragapult_ai"))

import replay_meta as rm  # noqa: E402
import value_features  # noqa: E402

OUT = ROOT / "bots" / "dragapult_ai" / "value_model.py"


def sigmoid(z: float) -> float:
    z = max(-40.0, min(40.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def load_states(paths: list[Path], stride: int = 2):
    by_id: dict[int, dict] = {}
    for path in paths:
        try:
            e = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        eid = (e.get("info") or {}).get("EpisodeId")
        if eid is not None:
            by_id.setdefault(int(eid), e)

    X, y, groups = [], [], []
    for eid, e in by_id.items():
        decks = rm.extract_decks(e)
        winner = rm.winner_index(e)
        if not decks or winner not in (0, 1):
            continue
        first = rm.first_player(e)
        for me in (0, 1):
            if rm.archetype(decks[me]) != "Dragapult":
                continue
            label = 1.0 if winner == me else 0.0
            n = 0
            for player, _step, obs, select, _action in rm.iter_decisions(e):
                if player != me:
                    continue
                if int(select.get("context", -1)) != 0:
                    continue
                n += 1
                if n % stride:
                    continue
                feat = value_features.extract(obs, me, first)
                if len(feat) != len(value_features.FEATURE_NAMES):
                    continue
                X.append(feat)
                y.append(label)
                groups.append(eid)
    return X, y, groups


def split_by_episode(groups, seed: int = 7):
    ids = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(ids)
    cut = int(len(ids) * 0.8)
    train_ids = set(ids[:cut])
    train = [i for i, g in enumerate(groups) if g in train_ids]
    test = [i for i, g in enumerate(groups) if g not in train_ids]
    return train, test


def standardize(X, train_idx):
    n = len(train_idx)
    m = len(X[0])
    mean = [0.0] * m
    for i in train_idx:
        row = X[i]
        for j in range(m):
            mean[j] += row[j]
    mean = [v / max(1, n) for v in mean]
    var = [0.0] * m
    for i in train_idx:
        row = X[i]
        for j in range(m):
            d = row[j] - mean[j]
            var[j] += d * d
    std = [math.sqrt(v / max(1, n)) or 1.0 for v in var]
    return mean, std


def fit_sgd(Xs, y, idx, epochs=80, lr=0.08, l2=0.002, seed=7):
    rng = random.Random(seed)
    theta = [0.0] * len(Xs[0])
    order = list(idx)
    n = max(1, len(order))
    for _epoch in range(epochs):
        rng.shuffle(order)
        for i in order:
            row = Xs[i]
            z = sum(w * v for w, v in zip(theta, row))
            err = sigmoid(z) - y[i]
            for j in range(len(theta)):
                theta[j] -= lr * (err * row[j] + l2 * theta[j])
    return theta


def auc_score(y, p):
    pairs = sorted(zip(p, y))
    n_pos = sum(1 for _, yy in pairs if yy == 1)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return None
    rank_sum = 0.0
    for rank, (_, yy) in enumerate(pairs, start=1):
        if yy == 1:
            rank_sum += rank
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def metrics(y, p):
    n = len(y)
    eps = 1e-6
    ll = brier = acc = 0.0
    for yy, pp in zip(y, p):
        pp = min(1 - eps, max(eps, pp))
        ll -= yy * math.log(pp) + (1 - yy) * math.log(1 - pp)
        brier += (pp - yy) ** 2
        acc += int((pp >= 0.5) == (yy == 1))
    return {
        "n": n,
        "base_rate": round(sum(y) / max(1, n), 4),
        "logloss": round(ll / max(1, n), 4),
        "brier": round(brier / max(1, n), 4),
        "accuracy": round(acc / max(1, n), 4),
        "auc": round(auc_score(y, p) or 0.0, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Dragapult value model")
    parser.add_argument("--replays", type=Path, default=ROOT / "replays")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    paths = sorted(args.replays.rglob("episode-*.json"))
    X, y, groups = load_states(paths, stride=max(1, args.stride))
    if not X:
        raise SystemExit("no states")
    print("replay files", len(paths), "states", len(X), "positives", round(sum(y) / len(y), 4))
    train_idx, test_idx = split_by_episode(groups)
    mean, std = standardize(X, train_idx)
    Xs = [[(v - mean[j]) / std[j] for j, v in enumerate(row)] for row in X]
    theta = fit_sgd(Xs, y, train_idx, epochs=args.epochs)
    train_p = [sigmoid(sum(w * v for w, v in zip(theta, Xs[i]))) for i in train_idx]
    test_p = [sigmoid(sum(w * v for w, v in zip(theta, Xs[i]))) for i in test_idx]
    print("train", metrics([y[i] for i in train_idx], train_p))
    print("test ", metrics([y[i] for i in test_idx], test_p))
    # calibration on holdout
    bins = []
    for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
        idx = [i for i in range(len(test_idx)) if test_p[i] >= lo and (test_p[i] < lo + 0.2 if lo < 0.8 else True)]
        if idx:
            bins.append((lo, len(idx), round(sum(test_p[i] for i in idx) / len(idx), 3),
                         round(sum(y[test_idx[i]] for i in idx) / len(idx), 3)))
    print("calibration", bins)

    lines = [
        '"""Auto-generated by tools/train_value_model.py. Do not edit by hand."""',
        "from __future__ import annotations",
        "import math",
        f"FEATURE_NAMES = {value_features.FEATURE_NAMES!r}",
        f"MEAN = {[round(x, 8) for x in mean]!r}",
        f"STD = {[round(x, 8) for x in std]!r}",
        f"WEIGHTS = {[round(x, 8) for x in theta]!r}",
        "",
        "def predict(features):",
        "    z = 0.0",
        "    for value, mean, std, weight in zip(features, MEAN, STD, WEIGHTS):",
        "        z += ((value - mean) / std) * weight",
        "    z = max(-40.0, min(40.0, z))",
        "    return 1.0 / (1.0 + math.exp(-z))",
        "",
    ]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("written", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
