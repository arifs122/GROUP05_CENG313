#!/usr/bin/env python3
"""Check that the local simulator is usable.

Run after scripts/setup_assets.sh:

    python3 scripts/check_env.py
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src" / "agent"))


def main() -> int:
    print(f"python: {sys.version.split()[0]} ({platform.system()} {platform.machine()})")

    try:
        from cg.api import all_attack, all_card_data
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: could not import cg: {exc}")
        print("Run: bash scripts/setup_assets.sh /path/to/sample_submission/cg")
        return 1

    cards = all_card_data()
    attacks = all_attack()
    print(f"engine: ok ({len(cards)} cards, {len(attacks)} attacks)")

    try:
        from policy import load_deck
        deck = load_deck()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: could not load deck: {exc}")
        return 1
    print(f"deck: ok ({len(deck)} cards, {len(set(deck))} unique)")

    from cg.game import battle_finish, battle_select, battle_start

    obs, start_data = battle_start(deck, deck)
    if obs is None:
        print(f"FAIL: battle_start error player={start_data.errorPlayer} type={start_data.errorType}")
        return 1

    select = obs["select"]
    print(
        "first select:",
        f"type={select['type']}",
        f"context={select['context']}",
        f"options={len(select['option'])}",
    )
    # One legal selection to exercise the loop, then stop.
    obs = battle_select(list(range(int(select["maxCount"]))))
    print(f"after one action: turn={obs['current']['turn']} result={obs['current']['result']}")
    battle_finish()

    print("\nOK: local environment is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
