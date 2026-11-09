#!/usr/bin/env python3
"""Crash / legality / time stress test for the MK/Ogerpon agent."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BOT = ROOT / "agents" / "mk_ogrepon" / "bot"
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(BOT))

import mk_policy  # noqa: E402
import mk_state as S  # noqa: E402
from arena import is_valid, load_bot  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

OPPONENTS = [
    "bots/_template",
    "bots/banette",
    "bots/dragapult",
    "benchmarks/external_agents/alakazam",
    "benchmarks/meta_decks/mega_abomasnow",
]


def main() -> int:
    games_per = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    deck = mk_policy.load_deck()
    result = mk_policy.agent({"select": None})
    assert len(result) == 60, f"initial deck length {len(result)}"

    total_games = 0
    invalid = 0
    errors = 0
    exceptions = 0
    times: list[float] = []
    types: Counter = Counter()
    contexts: Counter = Counter()
    outcomes: Counter = Counter()

    for opponent_path in OPPONENTS:
        try:
            opponent = load_bot(ROOT / opponent_path)
        except Exception as exc:
            print(f"skip {opponent_path}: {exc}")
            continue
        for g in range(games_per):
            reverse = g % 2 == 1
            obs, start = battle_start(deck, opponent.deck, reverse)
            if obs is None:
                errors += 2
                continue
            try:
                for step in range(6000):
                    cur = obs["current"]
                    if cur["result"] >= 0:
                        outcomes[cur["result"]] += 1
                        break
                    sel = obs["select"]
                    if sel is None:
                        break
                    player = int(cur["yourIndex"])
                    types[int(sel["type"])] += 1
                    contexts[int(sel["context"])] += 1
                    if player == 0:
                        started = time.perf_counter()
                        try:
                            action = mk_policy.agent(obs)
                        except Exception:
                            exceptions += 1
                            action = list(range(min(1, int(sel["maxCount"]))))
                        times.append(time.perf_counter() - started)
                        if not is_valid(action, sel):
                            invalid += 1
                            action = list(range(min(1, int(sel["maxCount"]))))
                    else:
                        action = opponent.agent(obs)
                        if not is_valid(action, sel):
                            invalid += 1
                            action = list(range(min(1, int(sel["maxCount"]))))
                    obs = battle_select(action)
                else:
                    errors += 1
            except Exception as exc:
                exceptions += 1
                print(f"game error {opponent_path} {g}: {type(exc).__name__}: {exc}")
            finally:
                battle_finish()
            total_games += 1

    stats = mk_policy.stats()
    rules = mk_policy.get_rules()
    searcher = rules.searcher
    times_sorted = sorted(times)
    payload = {
        "games": total_games,
        "opponents": OPPONENTS,
        "games_per_opponent": games_per,
        "invalid_actions": invalid,
        "engine_errors": errors,
        "exceptions": exceptions,
        "policy_fallbacks": stats["fallbacks"],
        "decisions": stats["calls"],
        "avg_decision_seconds": round(sum(times) / max(1, len(times)), 6),
        "p95_decision_seconds": round(times_sorted[int(len(times_sorted) * 0.95)], 6) if times else 0,
        "max_decision_seconds": round(max(times), 6) if times else 0,
        "select_types": dict(types),
        "select_contexts": dict(contexts),
        "outcomes": dict(outcomes),
        "search": {
            "enabled": searcher is not None,
            "overrides": getattr(searcher, "overrides", 0),
            "total_seconds": round(getattr(searcher, "total_seconds", 0.0), 4),
        },
    }
    out = ROOT / "agents" / "mk_ogrepon" / "reports" / "robustness.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if invalid == 0 and exceptions == 0 and errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
