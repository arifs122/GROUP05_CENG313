#!/usr/bin/env python3
"""Small CMA-ES tuner for the Lucario heuristic weights.

The competition engine does not expose a deal seed, so every candidate is
evaluated on a fixed number of games against a fixed opponent pool; the
selection is done on a train pool and the final number is reported on a
held-out pool.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
LUCARIO = HERE.parent
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(LUCARIO))

os.environ.setdefault("PTCG_BELIEF_LOG", "0")
import lucario_agent as L  # noqa: E402
import eval_pool as E  # noqa: E402

BOUNDS = {
    "ko": (500, 2000),
    "attack": (50, 800),
    "basic": (200, 1400),
    "evolve": (200, 1500),
    "mega_evolve": (200, 1600),
    "attach": (100, 1000),
    "poffin": (0, 1000),
    "ultra": (100, 1000),
    "gong": (100, 1000),
    "lillie": (100, 1200),
    "lillie_low_hand": (0, 200),
    "boss_setup": (100, 1300),
    "boss_ko": (300, 2200),
    "lunatone": (100, 1200),
    "retreat": (50, 1200),
    "ppp": (50, 1000),
    "cape": (50, 1000),
    "night_stretcher": (0, 700),
    "switch": (0, 900),
    "bench_target": (0, 8),
}
KEYS = list(BOUNDS)
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


def clip(x):
    lo = np.array([BOUNDS[k][0] for k in KEYS], dtype=float)
    hi = np.array([BOUNDS[k][1] for k in KEYS], dtype=float)
    return np.minimum(np.maximum(x, lo), hi)


def to_weights(x):
    return {k: float(v) for k, v in zip(KEYS, x)}


class CMAES:
    def __init__(self, x0, sigma=0.22, population=12, seed=7):
        self.n = len(x0)
        self.mean = np.asarray(x0, dtype=float)
        self.sigma = float(sigma)
        self.lam = int(population)
        self.mu = self.lam // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        self.mueff = 1.0 / np.sum(self.weights ** 2)
        self.cc = (4 + self.mueff / self.n) / (self.n + 4 + 2 * self.mueff / self.n)
        self.cs = (self.mueff + 2) / (self.n + self.mueff + 5)
        self.c1 = 2 / ((self.n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (self.n + 1)) - 1) + self.cs
        self.chi_n = math.sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (21 * self.n ** 2))
        self.pc = np.zeros(self.n)
        self.ps = np.zeros(self.n)
        self.C = np.eye(self.n)
        self.rng = np.random.default_rng(seed)

    def ask(self):
        self.eval_C, self.eval_B = np.linalg.eigh(self.C)
        self.eval_D = np.sqrt(np.maximum(self.eval_C, 1e-20))
        z = self.rng.standard_normal((self.lam, self.n))
        y = (z * self.eval_D) @ self.eval_B.T
        return clip(self.mean + self.sigma * y)

    def tell(self, candidates, fitness):
        order = np.argsort(fitness)
        x_sel = candidates[order[: self.mu]]
        old_mean = self.mean.copy()
        self.mean = np.sum(x_sel * self.weights[:, None], axis=0)
        y = (self.mean - old_mean) / self.sigma
        invsqrt = self.eval_B @ np.diag(1 / self.eval_D) @ self.eval_B.T
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * (invsqrt @ y)
        hsig = (np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * 1)) / self.chi_n) < (1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        artmp = (x_sel - old_mean) / self.sigma
        rank_mu = sum(w * np.outer(v, v) for w, v in zip(self.weights, artmp))
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * rank_mu)
        self.C = (self.C + self.C.T) / 2
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chi_n - 1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", type=Path, default=LUCARIO / "deck.csv")
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--games", type=int, default=10, help="games per train opponent")
    parser.add_argument("--holdout-games", type=int, default=20)
    parser.add_argument("--out", type=Path, default=LUCARIO / "weights.json")
    parser.add_argument("--report", type=Path, default=LUCARIO / "reports" / "cma_es.json")
    args = parser.parse_args()

    L._DECK = [int(x) for x in args.deck.read_text(encoding="utf-8").split() if x.strip()]
    x0 = np.array([L.W.get(k, BOUNDS[k][0]) for k in KEYS], dtype=float)
    baseline = E.evaluate(L.W, games_per=args.games, pool_paths=TRAIN,
                          deck_path=args.deck, verbose=False)
    print(f"baseline train mean={baseline['mean_win_rate']:.3f} min={baseline['min_win_rate']:.3f}")
    cma = CMAES(x0, sigma=90.0, population=args.population)
    best = (1e9, x0.copy(), None)
    for gen in range(1, args.generations + 1):
        cand = cma.ask()
        fit = []
        scores = []
        for i, x in enumerate(cand):
            w = to_weights(x)
            res = E.evaluate(w, games_per=args.games, pool_paths=TRAIN,
                             deck_path=args.deck, verbose=False)
            # Worst matchup is the primary objective.
            objective = -(0.7 * res["min_win_rate"] + 0.3 * res["mean_win_rate"])
            fit.append(objective)
            scores.append((res["min_win_rate"], res["mean_win_rate"]))
            print(f" gen{gen:02d} cand{i:02d} min={res['min_win_rate']:.3f} mean={res['mean_win_rate']:.3f}")
        cma.tell(cand, np.array(fit))
        idx = int(np.argmin(fit))
        if fit[idx] < best[0]:
            best = (fit[idx], cand[idx].copy(), scores[idx])
        print(f"gen {gen}: best min/mean={scores[idx]} sigma={cma.sigma:.3f}")

    best_weights = to_weights(best[1])
    # Report train and held-out pools for the chosen candidate.
    train = E.evaluate(best_weights, games_per=args.games, pool_paths=TRAIN,
                       deck_path=args.deck, verbose=False)
    hold = E.evaluate(best_weights, games_per=args.holdout_games, pool_paths=HOLDOUT,
                      deck_path=args.deck, verbose=False)
    args.out.write_text(json.dumps(best_weights, indent=2), encoding="utf-8")
    report = {
        "deck": str(args.deck),
        "keys": KEYS,
        "baseline_train": baseline,
        "train": train,
        "holdout": hold,
        "best_objective": best[0],
        "best_weights": best_weights,
        "generations": args.generations,
        "population": args.population,
        "games_per_train_opponent": args.games,
        "holdout_games": args.holdout_games,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"written {args.out} and {args.report}")
    print(f"train min={train['min_win_rate']:.3f} mean={train['mean_win_rate']:.3f} "
          f"holdout min={hold['min_win_rate']:.3f} mean={hold['mean_win_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
