#!/usr/bin/env python3
"""Local arena: run games between baselines or bots in ``bots/``.

Examples:
    python3 tools/arena.py --games 20
    python3 tools/arena.py --p0 heuristic --p1 random --games 50
    python3 tools/arena.py --bot0 bots/a --bot1 bots/b --games 50
    python3 tools/arena.py --bot0 bots/a --bot1 bots/b --games 30 --bo3
    python3 tools/arena.py --bot0 bots/a --bot1 bots/b --deck0 decks/x.csv --deck1 decks/y.csv

Each bot folder must contain ``main.py`` (with an ``agent`` function) and
``deck.csv`` (60 cards).  Results are printed and can be written to JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

DEFAULT_DECK = ROOT / "decks" / "mega_abomasnow.csv"


@dataclass
class Bot:
    name: str
    path: Path
    deck_path: Path
    deck: list[int]
    agent: Callable[[dict], list[int]]
    meta: dict = field(default_factory=dict)


def load_deck(path: Path) -> list[int]:
    cards = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(cards) != 60:
        raise ValueError(f"{path}: expected 60 cards, found {len(cards)}")
    return cards


def load_bot(path: Path) -> Bot:
    """Load a bot folder (or its main.py) with a unique module name.

    Helper modules imported at the top of main.py are allowed.  They are
    removed from ``sys.modules`` after loading so that another bot can use a
    helper with the same name.  Keep all imports at the top of main.py.
    """
    path = path.resolve()
    bot_dir = path if path.is_dir() else path.parent
    main_path = bot_dir / "main.py"
    deck_path = bot_dir / "deck.csv"

    if not main_path.is_file():
        raise FileNotFoundError(f"bot entry point not found: {main_path}")
    if not deck_path.is_file():
        raise FileNotFoundError(f"bot deck not found: {deck_path}")

    deck = load_deck(deck_path)
    meta: dict = {}
    meta_path = bot_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid meta.json in {bot_dir}: {exc}") from exc

    digest = hashlib.sha1(str(main_path).encode("utf-8")).hexdigest()[:8]
    safe_name = re.sub(r"\W+", "_", bot_dir.name)
    module_name = f"ptcg_bot_{safe_name}_{digest}"

    spec = importlib.util.spec_from_file_location(module_name, main_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {main_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    before = set(sys.modules)
    sys.path.insert(0, str(bot_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(bot_dir))
        except ValueError:
            pass
        # Drop helper modules loaded from this bot folder, but keep main.
        for loaded_name in set(sys.modules) - before:
            if loaded_name == module_name:
                continue
            loaded = sys.modules.get(loaded_name)
            loaded_file = getattr(loaded, "__file__", None)
            if not loaded_file:
                continue
            try:
                if Path(loaded_file).resolve().is_relative_to(bot_dir):
                    del sys.modules[loaded_name]
            except OSError:
                pass

    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise AttributeError(f"{main_path} does not define a callable agent(obs_dict)")

    name = str(meta.get("name") or bot_dir.name)
    return Bot(name=name, path=bot_dir, deck_path=deck_path, deck=deck, agent=agent, meta=meta)


def random_agent(rng: random.Random):
    def _agent(obs: dict) -> list[int]:
        select = obs["select"]
        count = int(select["maxCount"])
        return rng.sample(range(len(select["option"])), count)

    return _agent


def first_agent(obs: dict) -> list[int]:
    return list(range(int(obs["select"]["maxCount"])))


def _heuristic_agent():
    """Load src/agent/policy.py without reserving the generic name ``policy``."""
    module_name = "ptcg_heuristic_policy"
    module = sys.modules.get(module_name)
    if module is None:
        path = ROOT / "src" / "agent" / "policy.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.act


def make_agent(name: str, rng: random.Random):
    if name == "heuristic":
        return _heuristic_agent()
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


def play_game(
    agent0,
    agent1,
    deck0: list[int],
    deck1: list[int],
    reverse: bool,
    max_steps: int,
) -> dict:
    """Play one game and return its result."""
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
            try:
                action = agent(obs)
            except Exception as exc:  # noqa: BLE001
                return {
                    "winner": 1 - player,
                    "error": f"player {player} raised {type(exc).__name__}: {exc}",
                    "steps": steps,
                    "seconds": time.perf_counter() - start,
                }
            if not is_valid(action, select):
                return {
                    "winner": 1 - player,
                    "error": f"player {player} returned an invalid action: {action!r}",
                    "steps": steps,
                    "seconds": time.perf_counter() - start,
                }

            try:
                obs = battle_select(action)
            except Exception as exc:  # noqa: BLE001
                return {
                    "winner": 1 - player,
                    "error": f"battle_select failed for player {player}: {exc}",
                    "steps": steps,
                    "seconds": time.perf_counter() - start,
                }
            steps += 1

        return {
            "winner": 2,
            "error": "max_steps reached",
            "steps": steps,
            "seconds": time.perf_counter() - start,
        }
    finally:
        battle_finish()


def play_match(
    agent0,
    agent1,
    deck0: list[int],
    deck1: list[int],
    *,
    bo3: bool = False,
    reverse_first: bool = False,
    max_steps: int = 5000,
) -> dict:
    """Play a single game or a BO3 match.

    In BO3, round 1 uses ``reverse_first``, round 2 flips it and round 3 uses
    ``reverse_first`` again.  The first player to two round wins takes the
    match; a 1-1-1 score is a draw.
    """
    if not bo3:
        result = play_game(agent0, agent1, deck0, deck1, reverse_first, max_steps)
        winner = result["winner"]
        wins = [int(winner == 0), int(winner == 1), int(winner == 2)]
        return {"winner": winner, "wins": wins, "rounds": [result], "error": result["error"]}

    wins = [0, 0]
    draws = 0
    rounds = []
    for round_index in range(3):
        reverse = reverse_first ^ (round_index % 2 == 1)
        result = play_game(agent0, agent1, deck0, deck1, reverse, max_steps)
        rounds.append(result)
        winner = result["winner"]
        if winner is None:
            return {
                "winner": None,
                "wins": [wins[0], wins[1], draws],
                "rounds": rounds,
                "error": result["error"],
            }
        if winner in (0, 1):
            wins[winner] += 1
        else:
            draws += 1
        if wins[0] == 2 or wins[1] == 2:
            break

    if wins[0] == wins[1]:
        winner = 2
    else:
        winner = 0 if wins[0] > wins[1] else 1
    return {
        "winner": winner,
        "wins": [wins[0], wins[1], draws],
        "rounds": rounds,
        "error": None,
    }


def resolve_side(args, side: int, rng: random.Random):
    bot_path = getattr(args, f"bot{side}")
    deck_arg = getattr(args, f"deck{side}")
    policy_name = getattr(args, f"p{side}")

    if bot_path is not None:
        bot = load_bot(bot_path)
        deck = load_deck(deck_arg) if deck_arg is not None else bot.deck
        return bot.name, bot.agent, deck, bot

    deck = load_deck(deck_arg if deck_arg is not None else DEFAULT_DECK)
    return policy_name, make_agent(policy_name, rng), deck, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local PTCG games or matches")
    parser.add_argument("--games", type=int, default=10, help="number of games/matches")
    parser.add_argument("--p0", default="heuristic", choices=["heuristic", "random", "first"])
    parser.add_argument("--p1", default="random", choices=["heuristic", "random", "first"])
    parser.add_argument("--bot0", type=Path, default=None, help="bot folder (overrides --p0)")
    parser.add_argument("--bot1", type=Path, default=None, help="bot folder (overrides --p1)")
    parser.add_argument("--deck0", type=Path, default=None, help="override P0 deck")
    parser.add_argument("--deck1", type=Path, default=None, help="override P1 deck")
    parser.add_argument("--bo3", action="store_true", help="play best-of-three matches")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--reverse", choices=["auto", "p0", "p1"], default="auto",
                        help="who chooses whether to go first (auto alternates)")
    parser.add_argument("--out", type=Path, default=None, help="write results as JSON")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.games <= 0:
        raise SystemExit("--games must be positive")

    rng = random.Random(args.seed)
    label0, agent0, deck0, bot0 = resolve_side(args, 0, rng)
    label1, agent1, deck1, bot1 = resolve_side(args, 1, rng)

    results = []
    wins = [0, 0, 0]
    errors = 0
    total_steps = 0

    for game in range(args.games):
        if args.reverse == "auto":
            reverse_first = game % 2 == 1
        else:
            reverse_first = args.reverse == "p1"

        result = play_match(
            agent0,
            agent1,
            deck0,
            deck1,
            bo3=args.bo3,
            reverse_first=reverse_first,
            max_steps=args.max_steps,
        )
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

        round_steps = sum(round_["steps"] for round_ in result["rounds"])
        total_steps += round_steps
        round_seconds = sum(round_["seconds"] for round_ in result["rounds"])

        if not args.quiet:
            score = ""
            if args.bo3:
                score = f" ({result['wins'][0]}-{result['wins'][1]})"
            print(
                f"match {game + 1:3d}: {label:<8}{score} "
                f"steps={round_steps:4d} time={round_seconds:.2f}s"
                + (f"  [{result['error']}]" if result["error"] else "")
            )

    played = max(1, args.games)
    print()
    print(f"P0={label0} vs P1={label1}  matches={args.games}  "
          f"format={'BO3' if args.bo3 else 'single'}  seed={args.seed}")
    print(f"P0 wins: {wins[0]} ({wins[0] / played:.1%})")
    print(f"P1 wins: {wins[1]} ({wins[1] / played:.1%})")
    print(f"draws:   {wins[2]}")
    print(f"errors:  {errors}")
    print(f"avg decision steps: {total_steps / played:.1f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "bo3" if args.bo3 else "single",
            "p0": {"label": label0, "path": str(bot0.path) if bot0 else None,
                   "meta": bot0.meta if bot0 else None},
            "p1": {"label": label1, "path": str(bot1.path) if bot1 else None,
                   "meta": bot1.meta if bot1 else None},
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
