#!/usr/bin/env python3
"""2x2 pilot/deck experiment for the Mega Lucario line."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
LUCARIO = ROOT / "agents" / "mk_ogrepon" / "lucario"
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(LUCARIO))
os.environ.setdefault("PTCG_BELIEF_LOG", "0")

import lucario_agent as L  # noqa: E402
from arena import load_bot, load_deck, play_match  # noqa: E402

EXTERNAL = ROOT / "benchmarks" / "external_agents" / "lucario_mega_hariyama"


def external_agent():
    name = "external_lucario_eval"
    spec = importlib.util.spec_from_file_location(name, EXTERNAL / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    # For paired/fast local measurement: keep the same policy, disable its
    # expensive forward search.
    module.USE_SEARCH = False
    module.SEARCH_BUDGET = 0.0
    return module.agent


def read_deck(path: Path):
    return [int(x) for x in path.read_text(encoding="utf-8").split() if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--out", type=Path,
                        default=LUCARIO / "reports" / "pilot_deck_2x2.json")
    parser.add_argument("--candidate", default="all")
    parser.add_argument("--opponent", default="all")
    args = parser.parse_args()

    ext = external_agent()
    drag = load_bot(ROOT / "bots" / "dragapult")
    ext_bot = load_bot(EXTERNAL)
    # Use the fast external instance as opponent too.
    ext_opp = (ext, read_deck(EXTERNAL / "deck.csv"))

    candidates = [
        ("ours_petit", L.agent, read_deck(LUCARIO / "deck_real_petit.csv")),
        ("ours_pokepad", L.agent, read_deck(LUCARIO / "deck_pokepad.csv")),
        ("external_pokepad", ext, read_deck(LUCARIO / "deck_pokepad.csv")),
        ("external_own", ext, read_deck(EXTERNAL / "deck.csv")),
    ]
    opponents = [
        ("dragapult_proxy", drag.agent, drag.deck),
        ("external_lucario", ext_opp[0], ext_opp[1]),
    ]
    if args.candidate != "all":
        candidates = [x for x in candidates if x[0] == args.candidate]
    if args.opponent != "all":
        opponents = [x for x in opponents if x[0] == args.opponent]

    result = {}
    for cname, cagent, cdeck in candidates:
        L._DECK = list(cdeck)
        row = {}
        for oname, oagent, odeck in opponents:
            wins = losses = draws = errors = 0
            for g in range(args.games):
                candidate_p0 = g % 2 == 0
                reverse = (g // 2) % 2 == 1
                if candidate_p0:
                    r = play_match(cagent, oagent, cdeck, odeck,
                                   bo3=False, reverse_first=reverse)
                    ours = 0
                else:
                    r = play_match(oagent, cagent, odeck, cdeck,
                                   bo3=False, reverse_first=reverse)
                    ours = 1
                w = r["winner"]
                if w is None:
                    errors += 1
                elif w == ours:
                    wins += 1
                elif w == 1 - ours:
                    losses += 1
                else:
                    draws += 1
            n = max(1, wins + losses)
            row[oname] = {"wins": wins, "losses": losses, "draws": draws,
                          "errors": errors, "win_rate": wins / n}
            print(f"{cname:18s} vs {oname:18s} {wins:3d}-{losses:3d} "
                  f"{wins/n:6.1%} errors={errors}")
        result[cname] = row
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"games": args.games, "result": result}, indent=2), encoding="utf-8")
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
