#!/usr/bin/env python3
"""Evaluate the Lucario heuristic against a mixed opponent pool."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
LUCARIO = ROOT / "agents" / "mk_ogrepon" / "lucario"
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(LUCARIO))

import lucario_agent as L  # noqa: E402
from arena import load_bot, make_agent, play_match  # noqa: E402

DEFAULT_POOL = [
    "bots/_template",
    "bots/dragapult",
    "benchmarks/external_agents/alakazam",
    "bots/banette",
    "benchmarks/external_agents/lucario_mega_hariyama",
    "benchmarks/external_agents/bellibolt",
    "benchmarks/meta_decks/mega_abomasnow",
    "noisy:bots/dragapult",
]


def load_pool(paths: list[str]):
    pool = []
    for path in paths:
        noisy = None
        if path.startswith("noisy:"):
            noisy = 0.07
            path = path.split(":", 1)[1]
        if path == "random":
            pool.append(("random", make_agent("random", random.Random(7)), load_deck(ROOT / "bots/_template/deck.csv")))
            continue
        try:
            bot = load_bot(ROOT / path)
        except Exception as exc:
            print(f"skip {path}: {exc}", file=sys.stderr)
            continue
        agent = bot.agent
        name = bot.name
        if noisy is not None:
            base_agent = agent
            rng = random.Random(97)

            def agent(obs, _base=base_agent, _rng=rng):
                if _rng.random() < noisy:
                    sel = obs["select"] if isinstance(obs, dict) else obs.select
                    n = len(sel["option"] if isinstance(sel, dict) else sel.option)
                    mn = sel["minCount"] if isinstance(sel, dict) else sel.minCount
                    mx = sel["maxCount"] if isinstance(sel, dict) else sel.maxCount
                    count = max(mn, min(mx, 1))
                    return list(range(count))
                return _base(obs)

            name = "noisy_" + name
        pool.append((name, agent, bot.deck))
    return pool


def load_deck(path: Path) -> list[int]:
    return [int(x) for x in path.read_text(encoding="utf-8").split() if x.strip()]


def evaluate(weights: dict | None = None, games_per: int = 20,
             pool_paths: list[str] | None = None, deck_path: Path | None = None,
             verbose: bool = True) -> dict:
    if weights:
        L.W.update({k: float(v) for k, v in weights.items()})
    deck = load_deck(deck_path or (LUCARIO / "deck.csv"))
    L._DECK = list(deck)
    pool = load_pool(pool_paths or DEFAULT_POOL)
    results = {}
    for name, opp_agent, opp_deck in pool:
        wins = losses = draws = errors = 0
        for game in range(games_per):
            # Seat-swapped: half the games candidate is P0, half P1.
            candidate_is_p0 = (game % 2 == 0)
            reverse_first = ((game // 2) % 2 == 1)
            if candidate_is_p0:
                r = play_match(
                    L.agent, opp_agent, deck, opp_deck,
                    bo3=False, reverse_first=reverse_first, max_steps=5000,
                )
                winner = r["winner"]
                ours = 0
            else:
                r = play_match(
                    opp_agent, L.agent, opp_deck, deck,
                    bo3=False, reverse_first=reverse_first, max_steps=5000,
                )
                winner = r["winner"]
                ours = 1
            if winner is None:
                errors += 1
            elif winner == ours:
                wins += 1
            elif winner == (1 - ours):
                losses += 1
            else:
                draws += 1
        played = max(1, wins + losses)
        results[name] = {
            "wins": wins, "losses": losses, "draws": draws, "errors": errors,
            "win_rate": wins / played,
        }
        if verbose:
            print(f"{name:45s} {wins:3d}-{losses:3d}  {wins/played:6.1%}  errors={errors}")
    rates = [v["win_rate"] for v in results.values()]
    out = {
        "games_per_opponent": games_per,
        "opponents": results,
        "mean_win_rate": sum(rates) / max(1, len(rates)),
        "min_win_rate": min(rates) if rates else 0.0,
        "worst": min(results, key=lambda k: results[k]["win_rate"]) if results else None,
    }
    if verbose:
        print(f"mean={out['mean_win_rate']:.3f} min={out['min_win_rate']:.3f} worst={out['worst']}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20, help="games per opponent")
    parser.add_argument("--deck", type=Path, default=LUCARIO / "deck.csv")
    parser.add_argument("--out", type=Path, default=LUCARIO / "reports" / "pool_baseline.json")
    parser.add_argument("--weights", type=Path, default=None)
    args = parser.parse_args()
    if args.weights and args.weights.is_file():
        L.W.update(json.loads(args.weights.read_text()))
    started = time.perf_counter()
    result = evaluate(games_per=args.games, deck_path=args.deck)
    result["deck"] = str(args.deck)
    result["seconds"] = round(time.perf_counter() - started, 2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"written {args.out} in {result['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
