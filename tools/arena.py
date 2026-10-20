#!/usr/bin/env python3
"""Local arena: run games between baseline policies.

Examples:
    python3 tools/arena.py --games 20
    python3 tools/arena.py --p0 heuristic --p1 random --games 50
    python3 tools/arena.py --deck0 decks/mega_abomasnow.csv --deck1 decks/mega_abomasnow.csv

Results are printed and optionally written to a JSON file.  This is our
main measurement tool: every policy change should be compared with it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src" / "agent"))

from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from policy import act as heuristic_act  # noqa: E402


def load_deck(path: Path) -> list[int]:
    cards = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(cards) != 60:
        raise ValueError(f"{path}: expected 60 cards, found {len(cards)}")
    return cards


def random_agent(rng: random.Random):
    def _agent(obs: dict) -> list[int]:
        select = obs["select"]
        count = int(select["maxCount"])
        return rng.sample(range(len(select["option"])), count)

    return _agent


def first_agent(obs: dict) -> list[int]:
    return list(range(int(obs["select"]["maxCount"])))


def make_agent(name: str, rng: random.Random):
    if name == "heuristic":
        return heuristic_act
    if name == "random":
        return random_agent(rng)
    if name == "first":
        return first_agent
    raise ValueError(f"unknown agent: {name}")


def is_valid(action, select: dict) -> bool:
    if not isinstance(action, list) or not all(isinstance(x, int) for x in action):
        return False
    if len(set(action)) != len(action):
        return False
    if not (int(select["minCount"]) <= len(action) <= int(select["maxCount"])):
        return False
    return all(0 <= x < len(select["option"]) for x in action)


def play_game(agent0, agent1, deck0: list[int], deck1: list[int], reverse: bool, max_steps: int):
    start = time.perf_counter()
    obs, start_data = battle_start(deck0, deck1, reverse)
    if obs is None:
        return {
            "winner": None,
            "error": f"battle_start failed (player={start_data.errorPlayer}, type={start_data.errorType})",
            "steps": 0,
            "seconds": time.perf_counter() - start,
        }

    steps = 0
    try:
        while steps < max_steps:
            current = obs["current"]
            if current["result"] >= 0:
                return {
                    "winner": int(current["result"]),
                    "error": None,
                    "steps": steps,
                    "seconds": time.perf_counter() - start,
                }

            select = obs["select"]
            if select is None:
                return {
                    "winner": None,
                    "error": "unexpected initial select during a game",
                    "steps": steps,
                    "seconds": time.perf_counter() - start,
                }

            player = int(current["yourIndex"])
            agent = agent0 if player == 0 else agent1
            action = agent(obs)
            if not is_valid(action, select):
                return {
                    "winner": 1 - player,
                    "error": f"player {player} returned an invalid action: {action!r}",
                    "steps": steps,
                    "seconds": time.perf_counter() - start,
                }

            obs = battle_select(action)
            steps += 1

        return {
            "winner": 2,
            "error": "max_steps reached",
            "steps": steps,
            "seconds": time.perf_counter() - start,
        }
    finally:
        battle_finish()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local PTCG games")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--p0", default="heuristic", choices=["heuristic", "random", "first"])
    parser.add_argument("--p1", default="random", choices=["heuristic", "random", "first"])
    parser.add_argument("--deck0", type=Path, default=ROOT / "decks" / "mega_abomasnow.csv")
    parser.add_argument("--deck1", type=Path, default=ROOT / "decks" / "mega_abomasnow.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--reverse", choices=["auto", "p0", "p1"], default="auto",
                        help="who chooses whether to go first (auto alternates)")
    parser.add_argument("--out", type=Path, default=None, help="write results as JSON")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    deck0 = load_deck(args.deck0)
    deck1 = load_deck(args.deck1)
    rng = random.Random(args.seed)
    agent0 = make_agent(args.p0, rng)
    agent1 = make_agent(args.p1, rng)

    results = []
    wins = [0, 0, 0]
    errors = 0

    for game in range(args.games):
        if args.reverse == "auto":
            reverse = game % 2 == 1
        else:
            reverse = args.reverse == "p1"
        result = play_game(agent0, agent1, deck0, deck1, reverse, args.max_steps)
        results.append(result)

        winner = result["winner"]
        if winner is None:
            errors += 1
            label = "ERROR"
        else:
            wins[winner] += 1
            label = "DRAW" if winner == 2 else f"P{winner} win"
            if result["error"]:
                errors += 1

        if not args.quiet:
            print(
                f"game {game + 1:3d}: {label:<8} "
                f"steps={result['steps']:4d} "
                f"time={result['seconds']:.2f}s"
                + (f"  [{result['error']}]" if result["error"] else "")
            )

    played = max(1, args.games)
    print()
    print(f"P0={args.p0} vs P1={args.p1}  games={args.games}  seed={args.seed}")
    print(f"P0 wins: {wins[0]} ({wins[0] / played:.1%})")
    print(f"P1 wins: {wins[1]} ({wins[1] / played:.1%})")
    print(f"draws:   {wins[2]}")
    print(f"errors:  {errors}")
    print(f"avg decision steps: {sum(r['steps'] for r in results) / played:.1f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "p0": args.p0,
            "p1": args.p1,
            "games": args.games,
            "seed": args.seed,
            "wins": wins,
            "errors": errors,
            "results": results,
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"results written to {args.out}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
