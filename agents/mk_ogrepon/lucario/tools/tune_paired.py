#!/usr/bin/env python3
"""Paired, fork-cloned weight racing for the Lucario heuristic.

Only a handful of weights are touched: ko, attack, boss_setup, lillie,
retreat, attach.  The engine has no public seed, so `pair_arena` starts one
deal and forks it for baseline/candidate; coin RNG is forced onto the copied
C++ rng by clearing `GameConfig.deviceRand`.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
LUCARIO = HERE.parent
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(LUCARIO))
os.environ.setdefault("PTCG_BELIEF_LOG", "0")

import lucario_agent as L  # noqa: E402
import paired_arena as P  # noqa: E402

TRAIN = [
    "bots/_template",
    "bots/banette",
    "benchmarks/meta_decks/mega_abomasnow",
    "benchmarks/external_agents/bellibolt",
]
HOLDOUT = [
    "bots/dragapult",
    "benchmarks/external_agents/alakazam",
    "benchmarks/external_agents/lucario_mega_hariyama",
]
KEYS = ["ko", "attack", "boss_setup", "lillie", "retreat", "attach"]


def paired_diffs(series):
    diffs = []
    for a, b in zip(series[0], series[1]):
        if a == 2 or b == 2:
            continue
        diffs.append((1 if b == 0 else 0) - (1 if a == 0 else 0))
    return diffs


def summarize_diffs(diffs):
    n = len(diffs)
    if n == 0:
        return {"n": 0, "mean": 0.0, "se": 0.0, "z": 0.0}
    m = statistics.mean(diffs)
    se = statistics.stdev(diffs) / (n ** 0.5) if n > 1 else 0.0
    return {"n": n, "mean": round(m, 4), "se": round(se, 4),
            "z": round(m / se, 3) if se else 0.0}


def eval_pair(base_w, cand_w, pool, games, deck):
    all_diffs = []
    per_opp = {}
    for opp in pool:
        per = P.paired_series(
            [{"label": "baseline", "weights": base_w},
             {"label": "candidate", "weights": cand_w}],
            opp, games=games, deck_path=deck,
        )
        d = paired_diffs(per)
        all_diffs.extend(d)
        per_opp[opp] = summarize_diffs(d)
    total = summarize_diffs(all_diffs)
    return {"total": total, "per_opponent": per_opp}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", type=Path, default=LUCARIO / "deck_974_pokepad.csv")
    parser.add_argument("--train-games", type=int, default=40)
    parser.add_argument("--holdout-games", type=int, default=200)
    parser.add_argument("--out", type=Path, default=LUCARIO / "reports" / "paired_tuning.json")
    parser.add_argument("--weights-out", type=Path, default=LUCARIO / "weights_paired.json")
    args = parser.parse_args()

    L._DECK = [int(x) for x in args.deck.read_text(encoding="utf-8").split() if x.strip()]
    base = dict(L.W)
    candidates = []
    for key in KEYS:
        for factor in (0.8, 1.25):
            w = dict(base)
            w[key] = float(w[key]) * factor
            candidates.append((f"{key}*{factor}", w))

    results = []
    promising = []
    for label, w in candidates:
        # Quick 40-deal/opponent race.
        res = eval_pair(base, w, TRAIN, args.train_games, args.deck)
        results.append({"label": label, "train": res})
        t = res["total"]
        print(f"{label:20s} train n={t['n']} d={t['mean']:+.3f} se={t['se']:.3f} z={t['z']:+.2f}")
        if t["z"] >= 1.0:
            promising.append((label, w, res))

    selected = None
    if promising:
        promising.sort(key=lambda x: x[2]["total"]["mean"], reverse=True)
        label, w, train = promising[0]
        hold = eval_pair(base, w, HOLDOUT, args.holdout_games, args.deck)
        print(f"holdout {label}: n={hold['total']['n']} d={hold['total']['mean']:+.3f} "
              f"se={hold['total']['se']:.3f} z={hold['total']['z']:+.2f}")
        if hold["total"]["z"] >= 1.96 and hold["total"]["mean"] > 0:
            args.weights_out.write_text(json.dumps(w, indent=2), encoding="utf-8")
            selected = {"label": label, "weights": w, "train": train, "holdout": hold}
            print(f"wrote {args.weights_out}")
        else:
            print("holdout not significant; weights.json not written")

    report = {
        "deck": str(args.deck),
        "baseline": base,
        "keys": KEYS,
        "train_pool": TRAIN,
        "holdout_pool": HOLDOUT,
        "candidates": results,
        "selected": selected,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
