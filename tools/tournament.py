#!/usr/bin/env python3
"""Round-robin tournament for bots in ``bots/``.

Examples:
    python3 tools/tournament.py --games 50
    python3 tools/tournament.py --games 30 --bo3
    python3 tools/tournament.py --deck decks/mega_abomasnow.csv --games 100

The tournament plays every pair of bots, alternating the player who chooses
whether to go first.  It writes a Markdown report and a JSON file under
``results/`` and prints the report to stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from arena import load_bot, load_deck, play_match  # noqa: E402


def discover_bots(bots_dir: Path):
    bots = []
    for path in sorted(bots_dir.iterdir()):
        if not path.is_dir() or path.name.startswith("_") or path.name == "__pycache__":
            continue
        if not (path / "main.py").is_file() or not (path / "deck.csv").is_file():
            continue
        try:
            bots.append(load_bot(path))
        except Exception as exc:  # noqa: BLE001
            print(f"skip {path}: {exc}", file=sys.stderr)
    return bots


def ci95(score: float, n: int) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    standard_error = math.sqrt(max(score * (1.0 - score), 0.0) / n)
    margin = 1.96 * standard_error
    return max(0.0, score - margin), min(1.0, score + margin)


def update_elo(rating_a: float, rating_b: float, score_a: float, k: float = 24.0):
    expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    delta = k * (score_a - expected_a)
    return rating_a + delta, rating_b - delta


def pct(wins: int, losses: int) -> float | None:
    total = wins + losses
    if total == 0:
        return None
    return wins / total


def fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def build_markdown(config: dict, bots, stats: dict, pairwise: dict) -> str:
    lines = []
    lines.append("# Tournament results")
    lines.append("")
    lines.append(f"- Date: {config['generated_at']}")
    lines.append(f"- Format: {config['format']}")
    lines.append(f"- Matches per pair: {config['games_per_pair']}")
    lines.append(f"- Bots: {len(bots)}")
    lines.append(f"- Total matches: {config['total_matches']}")
    if config.get("deck_override"):
        lines.append(f"- Deck override: `{config['deck_override']}`")
    lines.append("")

    ranking = sorted(
        stats.items(),
        key=lambda item: (item[1]["elo"], item[1]["score"], item[0]),
        reverse=True,
    )
    lines.append("## Overall")
    lines.append("")
    lines.append("| # | Bot | W-L-D | Score | 95% CI | Elo | Worst matchup |")
    lines.append("|---:|---|---:|---:|---:|---:|---|")
    for rank, (name, row) in enumerate(ranking, start=1):
        low, high = ci95(row["score"], row["matches"])
        worst = row["worst_matchup"]
        worst_text = "-" if worst is None else f"vs {worst[0]}: {worst[1]:.0%}"
        lines.append(
            f"| {rank} | {name} | {row['wins']}-{row['losses']}-{row['draws']} "
            f"| {row['score']:.1%} | {low:.0%}-{high:.0%} "
            f"| {row['elo']:.0f} | {worst_text} |"
        )
    lines.append("")

    lines.append("## Pairwise win rate (row vs column)")
    lines.append("")
    header = "| Bot | " + " | ".join(bot.name for bot in bots) + " |"
    separator = "|---|" + "|".join("---:" for _ in bots) + "|"
    lines.append(header)
    lines.append(separator)
    index = {bot.name: i for i, bot in enumerate(bots)}
    for bot in bots:
        cells = []
        for other in bots:
            if bot.name == other.name:
                cells.append("—")
                continue
            key = (min(index[bot.name], index[other.name]), max(index[bot.name], index[other.name]))
            pair = pairwise[key]
            first_is_row = index[bot.name] < index[other.name]
            wins = pair["first_wins"] if first_is_row else pair["second_wins"]
            losses = pair["second_wins"] if first_is_row else pair["first_wins"]
            cells.append(fmt_pct(pct(wins, losses)))
        lines.append(f"| {bot.name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Draws are excluded from the pairwise percentages.")
    lines.append("")

    errors = sum(row["errors"] for row in stats.values())
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- Errors: {errors} (counted on both bots for a failed match)")
    lines.append("- Local simulator only; official episodes are BO3 and use the ladder.")
    lines.append("- `results/` is git-ignored; paste the summary into the PR.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a round-robin bot tournament")
    parser.add_argument("--bots-dir", type=Path, default=ROOT / "bots")
    parser.add_argument("--games", type=int, default=20, help="matches per pair")
    parser.add_argument("--bo3", action="store_true", help="play BO3 matches")
    parser.add_argument("--deck", type=Path, default=None, help="override every bot's battle deck")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.games <= 0:
        raise SystemExit("--games must be positive")

    bots = discover_bots(args.bots_dir)
    if len(bots) < 2:
        print(
            f"Need at least 2 bots in {args.bots_dir} (found {len(bots)}).\n"
            "Create one with: cp -r bots/_template bots/<your_bot>",
            file=sys.stderr,
        )
        return 1

    names = [bot.name for bot in bots]
    if len(set(names)) != len(names):
        raise SystemExit("Bot names must be unique; check meta.json files")

    deck_override = load_deck(args.deck) if args.deck else None
    stats = {
        bot.name: {
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "errors": 0,
            "matches": 0,
            "score": 0.0,
            "elo": 1000.0,
            "worst_matchup": None,
        }
        for bot in bots
    }

    pairwise = {}
    for i, j in combinations(range(len(bots)), 2):
        pairwise[(i, j)] = {"first_wins": 0, "second_wins": 0, "draws": 0}

    matches: list[dict] = []
    total = 0
    started = time.perf_counter()

    pairs = list(combinations(range(len(bots)), 2))
    for i, j in pairs:
        bot_a, bot_b = bots[i], bots[j]
        deck_a = deck_override if deck_override is not None else bot_a.deck
        deck_b = deck_override if deck_override is not None else bot_b.deck
        for game in range(args.games):
            reverse_first = (total % 2 == 1)
            try:
                result = play_match(
                    bot_a.agent,
                    bot_b.agent,
                    deck_a,
                    deck_b,
                    bo3=args.bo3,
                    reverse_first=reverse_first,
                    max_steps=args.max_steps,
                )
            except Exception as exc:  # noqa: BLE001
                result = {
                    "winner": None,
                    "wins": [0, 0, 0],
                    "rounds": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }

            total += 1
            winner = result["winner"]
            pair = pairwise[(i, j)]

            if winner is None:
                stats[bot_a.name]["errors"] += 1
                stats[bot_b.name]["errors"] += 1
            else:
                stats[bot_a.name]["matches"] += 1
                stats[bot_b.name]["matches"] += 1
                if winner == 0:
                    stats[bot_a.name]["wins"] += 1
                    stats[bot_b.name]["losses"] += 1
                    pair["first_wins"] += 1
                    score_a = 1.0
                elif winner == 1:
                    stats[bot_a.name]["losses"] += 1
                    stats[bot_b.name]["wins"] += 1
                    pair["second_wins"] += 1
                    score_a = 0.0
                else:
                    stats[bot_a.name]["draws"] += 1
                    stats[bot_b.name]["draws"] += 1
                    pair["draws"] += 1
                    score_a = 0.5

                stats[bot_a.name]["elo"], stats[bot_b.name]["elo"] = update_elo(
                    stats[bot_a.name]["elo"], stats[bot_b.name]["elo"], score_a
                )

            matches.append(
                {
                    "a": bot_a.name,
                    "b": bot_b.name,
                    "winner": winner,
                    "wins": result["wins"],
                    "error": result["error"],
                    "rounds": [
                        {
                            "winner": round_["winner"],
                            "steps": round_["steps"],
                            "seconds": round_["seconds"],
                            "error": round_["error"],
                        }
                        for round_ in result["rounds"]
                    ],
                }
            )
            if not args.quiet and (game + 1) % 10 == 0:
                print(f"{bot_a.name} vs {bot_b.name}: {game + 1}/{args.games}")

    for row in stats.values():
        if row["matches"]:
            row["score"] = (row["wins"] + 0.5 * row["draws"]) / row["matches"]

    # Worst matchup = lowest win rate against a single opponent.
    for i, bot in enumerate(bots):
        matchups = []
        for j, other in enumerate(bots):
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            pair = pairwise[key]
            wins = pair["first_wins"] if i < j else pair["second_wins"]
            losses = pair["second_wins"] if i < j else pair["first_wins"]
            rate = pct(wins, losses)
            if rate is not None:
                matchups.append((other.name, rate))
        if matchups:
            stats[bot.name]["worst_matchup"] = min(matchups, key=lambda item: item[1])

    elapsed = time.perf_counter() - started
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config = {
        "generated_at": generated_at,
        "format": "BO3" if args.bo3 else "single game",
        "games_per_pair": args.games,
        "total_matches": total,
        "elapsed_seconds": round(elapsed, 2),
        "deck_override": str(args.deck) if args.deck else None,
    }
    markdown = build_markdown(config, bots, stats, pairwise)

    json_path = args.out_dir / f"tournament_{stamp}.json"
    md_path = args.out_dir / f"tournament_{stamp}.md"
    payload = {
        "config": config,
        "bots": [
            {"name": bot.name, "path": str(bot.path), "meta": bot.meta}
            for bot in bots
        ],
        "stats": stats,
        "pairwise": [
            {
                "a": bots[i].name,
                "b": bots[j].name,
                **pairwise[(i, j)],
            }
            for i, j in pairs
        ],
        "matches": matches,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")

    print()
    print(markdown)
    print(f"report: {md_path}")
    print(f"json:   {json_path}")
    print(f"elapsed: {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
