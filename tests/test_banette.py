"""Smoke tests for the first Banette bot."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "tools"))


class BanetteTest(unittest.TestCase):
    def test_deck_is_legal_shape(self) -> None:
        deck = [
            int(line)
            for line in (ROOT / "bots" / "banette" / "deck.csv").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(deck), 60)
        self.assertTrue(all(1 <= card_id <= 1431 for card_id in deck))

    def test_short_self_play(self) -> None:
        try:
            from arena import load_bot, play_match
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"engine not set up: {exc}")

        bot = load_bot(ROOT / "bots" / "banette")
        self.assertEqual(len(bot.deck), 60)
        self.assertEqual(len(bot.agent({"select": None})), 60)

        result = play_match(
            bot.agent,
            bot.agent,
            bot.deck,
            bot.deck,
            bo3=False,
            reverse_first=False,
            max_steps=300,
        )
        self.assertIn(result["winner"], (0, 1, 2, None))


if __name__ == "__main__":
    unittest.main()
