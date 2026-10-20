"""Smoke tests for the local setup.

Run with:
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src" / "agent"))

DECKS = [ROOT / "decks" / "mega_abomasnow.csv", ROOT / "decks" / "current.csv"]


class SmokeTest(unittest.TestCase):
    def test_deck_file(self) -> None:
        path = next((p for p in DECKS if p.is_file()), None)
        self.assertIsNotNone(path, "no deck file found")
        cards = [int(x) for x in path.read_text(encoding="utf-8").split()]
        self.assertEqual(len(cards), 60)

    def test_engine_and_short_game(self) -> None:
        try:
            from cg.api import all_card_data
            from cg.game import battle_finish, battle_select, battle_start
            from policy import act, load_deck
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"engine not set up: {exc}")

        self.assertGreater(len(all_card_data()), 0)
        deck = load_deck()
        obs, start_data = battle_start(deck, deck)
        self.assertIsNotNone(obs, f"battle_start failed: {start_data.errorPlayer}")

        try:
            for _ in range(300):
                current = obs["current"]
                if current["result"] >= 0:
                    break
                action = act(obs)
                select = obs["select"]
                self.assertGreaterEqual(len(action), select["minCount"])
                self.assertLessEqual(len(action), select["maxCount"])
                self.assertEqual(len(action), len(set(action)))
                obs = battle_select(action)
        finally:
            battle_finish()


if __name__ == "__main__":
    unittest.main()
