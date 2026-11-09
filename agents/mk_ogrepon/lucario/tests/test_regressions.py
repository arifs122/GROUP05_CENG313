"""Small regression tests for the Lucario heuristic rules.

These are pure state/action tests; they do not need the simulator.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[2] / "vendor"))
sys.path.insert(0, str(HERE))

import lucario_agent as L


class LucarioRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        L._POFFIN_CACHE = None

    def test_cape_only_on_mega_or_hariyama(self) -> None:
        s = L.State(active="Mega Lucario ex", bench=["Lunatone"], energy={"Mega Lucario ex": 1})
        good = L.Act(raw=[0], kind="item", card="Hero’s Cape", target="Mega Lucario ex")
        bad = L.Act(raw=[0], kind="item", card="Hero’s Cape", target="Lunatone")
        self.assertGreater(L.score(good, s), 0)
        self.assertLess(L.score(bad, s), 0)

    def test_boss_without_setup_or_ko_is_not_played(self) -> None:
        s = L.State(active="Riolu", bench=["Makuhita"], energy={"Riolu": 1})
        a = L.Act(raw=[0], kind="supporter", card="Boss’s Orders", target="Solrock")
        self.assertLess(L.score(a, s), 0)

    def test_retreat_without_charged_bench_is_penalized(self) -> None:
        s = L.State(active="Mega Lucario ex", bench=["Riolu"], energy={"Mega Lucario ex": 2})
        a = L.Act(raw=[0], kind="retreat")
        self.assertLess(L.score(a, s), 0)

    def test_attach_to_attacker_beats_non_attacker(self) -> None:
        s = L.State(active="Riolu", bench=["Lunatone"], energy={"Riolu": 0, "Lunatone": 0})
        good = L.Act(raw=[0], kind="attach", card="Basic {F} Energy", target="Riolu")
        bad = L.Act(raw=[0], kind="attach", card="Basic {F} Energy", target="Lunatone")
        self.assertGreater(L.score(good, s), L.score(bad, s))

    def test_poffin_deck_awareness(self) -> None:
        # 677 Riolu is 80 HP, so Poffin can only fetch Budew.
        L._DECK = [677] * 4 + [673] * 4 + [235]
        L._POFFIN_CACHE = None
        s = L.State(active="Riolu", bench=["Budew"], energy={})
        a = L.Act(raw=[0], kind="item", card="Buddy-Buddy Poffin")
        low = L.score(a, s)
        # 974 Riolu is 70 HP and is a legal Poffin target.
        L._DECK = [974] * 4 + [673] * 4 + [235]
        L._POFFIN_CACHE = None
        high = L.score(a, s)
        self.assertGreater(high, low)

    def test_mega_evolve_prioritized_when_in_hand(self) -> None:
        s = L.State(active="Riolu", bench=[], hand_cards=["Mega Lucario ex"],
                    energy={"Riolu": 1})
        evolve = L.Act(raw=[0], kind="evolve", card="Mega Lucario ex", target="Riolu")
        basic = L.Act(raw=[1], kind="basic", card="Makuhita")
        self.assertGreater(L.score(evolve, s), L.score(basic, s))

    def test_premium_power_pro_needs_attacker(self) -> None:
        s = L.State(active="Solrock", bench=[], energy={"Solrock": 1})
        a = L.Act(raw=[0], kind="item", card="Premium Power Pro")
        self.assertLess(L.score(a, s), 0)


if __name__ == "__main__":
    unittest.main()
