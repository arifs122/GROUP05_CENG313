"""Tests for the observation-only runtime belief tracker."""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots" / "dragapult"))


class BeliefRuntimeTest(unittest.TestCase):
    def test_deck_library(self) -> None:
        from belief_data import ARCHETYPE_DECKS

        self.assertGreaterEqual(len(ARCHETYPE_DECKS), 10)
        for name, deck in ARCHETYPE_DECKS.items():
            self.assertEqual(len(deck), 60, name)

    def test_hypergeometric(self) -> None:
        from belief_runtime import _p_at_least_one

        self.assertGreater(_p_at_least_one(60, 4, 7), 0.0)
        self.assertLess(_p_at_least_one(60, 4, 7), 1.0)
        self.assertEqual(_p_at_least_one(60, 0, 7), 0.0)

    def test_observer_and_one_log_per_turn(self) -> None:
        from belief_runtime import BeliefObserver

        deck = [int(x) for x in (ROOT / "bots" / "dragapult" / "deck.csv").read_text().split()]
        obs = {
            "current": {
                "turn": 1,
                "players": [
                    {
                        "hand": [{"id": 119, "serial": 1}],
                        "discard": [], "active": [], "bench": [],
                        "prize": [None] * 6, "deckCount": 53, "handCount": 1,
                    },
                    {
                        "hand": None,
                        "discard": [{"id": 121, "serial": 2}],
                        "active": [], "bench": [],
                        "prize": [None] * 6, "deckCount": 53, "handCount": 6,
                    },
                ],
                "stadium": [],
            },
            "logs": [],
        }
        belief = BeliefObserver(deck)
        snapshot = belief.update(obs, 0)
        self.assertIsNotNone(snapshot)
        self.assertAlmostEqual(sum(belief.posterior.values()), 1.0)
        self.assertEqual(belief.best_deck()[0], "dragapult")

        out = io.StringIO()
        with redirect_stdout(out):
            belief.emit()
            belief.emit()  # same turn: should not produce a second line
        lines = [line for line in out.getvalue().splitlines() if line.startswith("BELIEF ")]
        self.assertEqual(len(lines), 1)

        obs["current"]["turn"] = 2
        belief.update(obs, 0)
        out = io.StringIO()
        with redirect_stdout(out):
            belief.emit()
        self.assertEqual(out.getvalue().count("BELIEF "), 1)


if __name__ == "__main__":
    unittest.main()
