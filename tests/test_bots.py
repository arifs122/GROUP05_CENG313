"""Tests for bot loading and the tournament building blocks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "tools"))


class BotTest(unittest.TestCase):
    def test_template_bot_loads(self) -> None:
        try:
            from arena import load_bot
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"engine not set up: {exc}")

        bot = load_bot(ROOT / "bots" / "_template")
        self.assertEqual(len(bot.deck), 60)
        self.assertTrue(callable(bot.agent))
        self.assertEqual(len(bot.agent({"select": None})), 60)

    def test_template_short_match(self) -> None:
        try:
            from arena import load_bot, play_match
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"engine not set up: {exc}")

        bot = load_bot(ROOT / "bots" / "_template")
        result = play_match(
            bot.agent,
            bot.agent,
            bot.deck,
            bot.deck,
            bo3=False,
            reverse_first=False,
            max_steps=200,
        )
        self.assertIn(result["winner"], (0, 1, 2, None))


if __name__ == "__main__":
    unittest.main()
