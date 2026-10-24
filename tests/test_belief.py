"""Tests for the public-information belief tracker."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


class BeliefTest(unittest.TestCase):
    def test_hypergeometric(self) -> None:
        from belief import hypergeom_at_least_one

        self.assertGreater(hypergeom_at_least_one(60, 4, 7), 0.0)
        self.assertLess(hypergeom_at_least_one(60, 4, 7), 1.0)
        self.assertEqual(hypergeom_at_least_one(60, 0, 7), 0.0)

    def test_seen_cards_and_posterior(self) -> None:
        from belief import Belief

        deck = [1182] * 4 + [1] * 56
        obs = {
            "current": {
                "turn": 5,
                "players": [
                    {
                        "active": [], "bench": [], "deckCount": 40, "discard": [],
                        "prize": [None] * 6, "handCount": 5, "hand": [],
                    },
                    {
                        "active": [], "bench": [], "deckCount": 40,
                        "discard": [{"id": 1182, "serial": 1, "playerIndex": 1}],
                        "prize": [None] * 6, "handCount": 6, "hand": None,
                    },
                ],
                "stadium": [],
                "looking": None,
            },
            "logs": [],
        }
        belief = Belief(opponent_index=1, deck_library={"synthetic": deck})
        belief.update(obs)
        self.assertEqual(belief.seen_counts[1182], 1)
        self.assertAlmostEqual(belief.posterior["synthetic"], 1.0)
        self.assertGreater(belief.p_in_hand(1182), 0.0)
        self.assertGreater(belief.p_next_draw(1182), 0.0)


if __name__ == "__main__":
    unittest.main()
