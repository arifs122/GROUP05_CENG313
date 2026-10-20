#!/usr/bin/env python3
"""Print the observations our agent sees, one decision at a time.

Useful for learning the API and for debugging a policy:

    python3 tools/inspect_obs.py --steps 15
    python3 tools/inspect_obs.py --steps 40 --deck decks/mega_abomasnow.csv --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src" / "agent"))

from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from policy import act  # noqa: E402

OPTION_FIELDS = (
    "type",
    "number",
    "cardId",
    "attackId",
    "area",
    "index",
    "playerIndex",
    "inPlayArea",
    "inPlayIndex",
)


def describe_option(option: dict) -> str:
    fields = [f"{key}={option[key]}" for key in OPTION_FIELDS if key in option]
    return ", ".join(fields) if fields else "{}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", type=Path, default=ROOT / "decks" / "mega_abomasnow.csv")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--json", action="store_true", help="dump the full observation")
    args = parser.parse_args()

    deck = [int(x) for x in args.deck.read_text(encoding="utf-8").split() if x.strip()]
    obs, _ = battle_start(deck, deck)
    if obs is None:
        print("battle_start failed")
        return 1

    try:
        for step in range(args.steps):
            current = obs["current"]
            if current["result"] >= 0:
                print(f"\nresult={current['result']} after {step} decisions")
                break

            select = obs["select"]
            print(
                f"\n--- decision {step + 1} | turn={current['turn']} "
                f"player={current['yourIndex']} | "
                f"type={select['type']} context={select['context']} "
                f"min={select['minCount']} max={select['maxCount']} "
                f"options={len(select['option'])}"
            )
            for i, option in enumerate(select["option"][:12]):
                print(f"  [{i}] {describe_option(option)}")
            if len(select["option"]) > 12:
                print(f"  ... {len(select['option']) - 12} more")

            if args.json:
                print(json.dumps(obs, indent=2, ensure_ascii=False))

            action = act(obs)
            print(f"  -> action {action}")
            obs = battle_select(action)
    finally:
        battle_finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
