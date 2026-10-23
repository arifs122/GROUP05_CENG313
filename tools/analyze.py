#!/usr/bin/env python3
"""Per-matchup anomaly report for a bot.

Win rates in cabt are noisy: the same policy can swing several points over a
few dozen games.  Behavioural anomalies are deterministic, so this tool counts
them alongside the score.  The taxonomy follows the one described in
``wmh/ptcg-abc``'s ``docs/optimization_protocol.md`` (ideas only, original
implementation):

* ``error_games``       - our agent raised or returned an illegal action
* ``attack_no_damage``  - our attack was used and no HP change followed
* ``no_offense_loss``   - we lost while landing at most one damaging attack
* ``deckout_loss``      - we lost with an empty deck

Usage:
    python3 tools/analyze.py --bot bots/dragapult --all --games 200
    python3 tools/analyze.py --bot bots/dragapult \
        --opponents benchmarks/external_agents/alakazam --games 200
    python3 tools/analyze.py --bot bots/dragapult --all --games 100 --json results/anomalies.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "tools"))

from arena import is_valid, load_bot, play_match  # noqa: E402

ATTACK_LOG = 15
HP_CHANGE_LOG = 16
TURN_START_LOG = 2
TURN_END_LOG = 3


def attack_info() -> tuple[dict[int, str], set[int]]:
    """Return (id -> name, ids of printed 0-damage utility attacks)."""
    from cg.api import all_attack

    names: dict[int, str] = {}
    utility: set[int] = set()
    for attack in all_attack():
        attack_id = int(attack.attackId)
        names[attack_id] = attack.name
        if not attack.damage:
            utility.add(attack_id)
    return names, utility


def deck_attack_ids(deck: list[int]) -> set[int]:
    from cg.api import all_card_data

    cards = {int(c.cardId): c for c in all_card_data()}
    out: set[int] = set()
    for card_id in set(deck):
        card = cards.get(int(card_id))
        if card is None:
            continue
        for attack_id in getattr(card, "attacks", None) or []:
            out.add(int(attack_id))
    return out


@dataclass
class GameStats:
    invalid: list[dict] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)
    attacks: Counter = field(default_factory=Counter)
    damaging: Counter = field(default_factory=Counter)
    damage_events: int = 0
    deckout_seen: bool = False
    decisions: int = 0


class Tracker:
    """Wraps our agent and records deterministic anomalies per game."""

    def __init__(self, agent, deck: list[int]):
        self.agent = agent
        self.our_attacks = deck_attack_ids(deck)
        self.stats = GameStats()

    def new_game(self) -> None:
        self.stats = GameStats()

    def __call__(self, obs: dict) -> list[int]:
        select = obs.get("select")
        if select is None:
            return self.agent(obs)

        self.stats.decisions += 1
        state = obs["current"]
        me = int(state["yourIndex"])

        if int(state["players"][me].get("deckCount", 1)) <= 0:
            self.stats.deckout_seen = True

        pending: int | None = None
        for log in obs.get("logs") or []:
            kind = log.get("type")
            if kind == ATTACK_LOG:
                attack_id = log.get("attackId")
                if attack_id is not None and int(attack_id) in self.our_attacks:
                    self.stats.attacks[int(attack_id)] += 1
                    pending = int(attack_id)
                else:
                    pending = None
            elif kind == HP_CHANGE_LOG and int(log.get("value") or 0) < 0:
                self.stats.damage_events += 1
                if pending is not None:
                    self.stats.damaging[pending] += 1
                    pending = None
            elif kind in (TURN_START_LOG, TURN_END_LOG):
                pending = None

        try:
            action = self.agent(obs)
        except Exception as exc:  # noqa: BLE001
            self.stats.exceptions.append(f"{type(exc).__name__}: {exc}")
            raise

        if not is_valid(action, select):
            self.stats.invalid.append(
                {
                    "context": select.get("context"),
                    "type": select.get("type"),
                    "effect": (select.get("effect") or {}).get("id")
                    if isinstance(select.get("effect"), dict)
                    else None,
                    "min": select.get("minCount"),
                    "max": select.get("maxCount"),
                    "action": action,
                    "turn": state.get("turn"),
                }
            )
        return action


@dataclass
class MatchupStats:
    name: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    error_games: int = 0
    opponent_errors: int = 0
    invalid: int = 0
    exceptions: int = 0
    deckout_loss: int = 0
    no_offense_loss: int = 0
    attacks: Counter = field(default_factory=Counter)
    damaging: Counter = field(default_factory=Counter)
    decisions: int = 0
    seconds: float = 0.0
    invalid_samples: list[dict] = field(default_factory=list)
    error_samples: list[str] = field(default_factory=list)

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float | None:
        decided = self.wins + self.losses
        return self.wins / decided if decided else None

    def zero_damage(self, utility: set[int] | None = None) -> dict[int, int]:
        utility = utility or set()
        return {
            attack_id: count - self.damaging.get(attack_id, 0)
            for attack_id, count in self.attacks.items()
            if attack_id not in utility
        }


def discover_opponents(bot_path: Path) -> list[Path]:
    found: list[Path] = []
    for base in (ROOT / "benchmarks" / "meta_decks", ROOT / "benchmarks" / "external_agents"):
        if not base.is_dir():
            continue
        for path in sorted(base.iterdir()):
            if not path.is_dir() or path.name.startswith("_"):
                continue
            if not (path / "main.py").is_file() or not (path / "deck.csv").is_file():
                continue
            if path.resolve() == bot_path.resolve():
                continue
            found.append(path)
    return found


def run_matchup(bot, opponent_path: Path, games: int, bo3: bool, max_steps: int,
                quiet: bool) -> MatchupStats:
    opponent = load_bot(opponent_path)
    tracker = Tracker(bot.agent, bot.deck)
    stats = MatchupStats(name=opponent.name)
    started = time.perf_counter()

    for game in range(games):
        tracker.new_game()
        result = play_match(
            tracker,
            opponent.agent,
            bot.deck,
            opponent.deck,
            bo3=bo3,
            reverse_first=(game % 2 == 1),
            max_steps=max_steps,
        )
        winner = result["winner"]
        if winner is None:
            stats.draws += 1
        elif winner == 0:
            stats.wins += 1
        else:
            stats.losses += 1

        game_stats = tracker.stats
        our_fault = bool(game_stats.invalid or game_stats.exceptions)
        if our_fault:
            stats.error_games += 1
        elif result.get("error"):
            stats.opponent_errors += 1
        stats.invalid += len(game_stats.invalid)
        stats.exceptions += len(game_stats.exceptions)
        stats.invalid_samples.extend(game_stats.invalid[:2])
        if game_stats.exceptions:
            stats.error_samples.extend(game_stats.exceptions[:2])
        if result.get("error"):
            stats.error_samples.append(result["error"][:160])

        if winner == 1:
            if game_stats.deckout_seen:
                stats.deckout_loss += 1
            if sum(game_stats.damaging.values()) <= 1:
                stats.no_offense_loss += 1

        stats.attacks.update(game_stats.attacks)
        stats.damaging.update(game_stats.damaging)
        stats.decisions += game_stats.decisions

        if not quiet and (game + 1) % 50 == 0:
            print(f"  {opponent.name}: {game + 1}/{games}", file=sys.stderr)

    stats.seconds = time.perf_counter() - started
    return stats


def print_report(rows: list[MatchupStats], names: dict[int, str], utility: set[int]) -> None:
    header = (
        f"{'opponent':26s} {'W-L-D':>10s} {'win%':>6s} {'err':>4s} {'opp':>4s} {'inv':>4s} "
        f"{'deck':>5s} {'no-off':>7s} {'atk':>5s} {'0dmg':>5s} {'ms/g':>6s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        rate = "-" if row.win_rate is None else f"{row.win_rate:.1%}"
        zero = sum(row.zero_damage(utility).values())
        attacks = sum(row.attacks.values())
        ms = row.seconds / max(1, row.games) * 1000
        print(
            f"{row.name:26s} {row.wins:3d}-{row.losses:3d}-{row.draws:<2d} "
            f"{rate:>6s} {row.error_games:4d} {row.opponent_errors:4d} {row.invalid:4d} "
            f"{row.deckout_loss:5d} {row.no_offense_loss:7d} {attacks:5d} {zero:5d} {ms:6.0f}"
        )

    detailed = []
    for row in rows:
        zero_map = row.zero_damage(utility)
        if any(count for count in zero_map.values()):
            detailed.append((row, zero_map))
    if detailed:
        print("\nattacks that never changed HP (attack: zero/total)")
        for row, zero_map in detailed:
            parts = []
            for attack_id, zero in sorted(zero_map.items(), key=lambda item: -item[1]):
                total = row.attacks[attack_id]
                if zero:
                    parts.append(f"{names.get(attack_id, attack_id)} {zero}/{total}")
            if parts:
                print(f"  {row.name}: " + ", ".join(parts))

    samples = [row for row in rows if row.invalid_samples]
    if samples:
        print("\ninvalid-action samples")
        for row in samples:
            for sample in row.invalid_samples[:3]:
                print(f"  {row.name}: {json.dumps(sample, default=str)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run matchups and count deterministic anomalies")
    parser.add_argument("--bot", type=Path, default=ROOT / "bots" / "dragapult")
    parser.add_argument("--opponents", type=Path, nargs="*", default=None)
    parser.add_argument("--all", action="store_true", help="use every benchmark opponent")
    parser.add_argument("--games", type=int, default=200, help="games per opponent")
    parser.add_argument("--bo3", action="store_true")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.games <= 0:
        raise SystemExit("--games must be positive")

    if not args.all and not args.opponents:
        raise SystemExit("pass --all or --opponents PATH [PATH ...]")

    bot = load_bot(args.bot)
    opponents = list(args.opponents or []) if not args.all else []
    if args.all:
        opponents.extend(discover_opponents(args.bot))
    if not opponents:
        raise SystemExit("no opponents found")

    names, utility = attack_info()
    rows = []
    for path in opponents:
        if not args.quiet:
            print(f"running {path} ...", file=sys.stderr)
        rows.append(run_matchup(bot, path, args.games, args.bo3, args.max_steps, args.quiet))

    rows.sort(key=lambda row: (row.win_rate if row.win_rate is not None else -1.0, row.name))
    print(f"\nbot: {bot.name}  games/opponent: {args.games}  format: "
          f"{'BO3' if args.bo3 else 'single'}\n")
    print_report(rows, names, utility)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bot": bot.name,
            "games": args.games,
            "format": "bo3" if args.bo3 else "single",
            "matchups": [
                {
                    "name": row.name,
                    "wins": row.wins,
                    "losses": row.losses,
                    "draws": row.draws,
                    "win_rate": row.win_rate,
                    "error_games": row.error_games,
                    "opponent_errors": row.opponent_errors,
                    "invalid": row.invalid,
                    "exceptions": row.exceptions,
                    "deckout_loss": row.deckout_loss,
                    "no_offense_loss": row.no_offense_loss,
                    "zero_damage": {
                        names.get(attack_id, str(attack_id)): count
                        for attack_id, count in row.zero_damage(utility).items()
                        if count
                    },
                    "invalid_samples": row.invalid_samples[:3],
                    "error_samples": row.error_samples[:3],
                }
                for row in rows
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
