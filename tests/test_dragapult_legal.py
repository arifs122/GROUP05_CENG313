"""Regression tests for the Dragapult illegal-action fix.

The sample pilot returned ``[]`` when every legal option scored UNNECESSARY
(Drakloak's Recon Directive, Crispin's attach choice).  ``[]`` is illegal
whenever ``minCount`` is positive, and an illegal action forfeits the game.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "tools"))


class DragapultLegalTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from arena import load_bot
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"engine not set up: {exc}")
        try:
            self.bot = load_bot(ROOT / "bots" / "dragapult")
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"could not load bot: {exc}")
        # Keep the unit tests fast; search is exercised by the arena runs.
        self.bot.agent.__globals__["SEARCH_ENABLED"] = False

    def test_reset_hook_is_exposed(self) -> None:
        self.assertTrue(callable(getattr(self.bot.agent, "reset", None)))

    def test_empty_policy_is_legalized(self) -> None:
        """A policy that returns [] must still produce a legal action."""
        from arena import is_valid
        from cg.game import battle_finish, battle_select, battle_start

        obs, _ = battle_start(self.bot.deck, self.bot.deck)
        self.assertIsNotNone(obs)
        try:
            for _ in range(60):
                select = obs["select"]
                if select["context"] == 0:
                    break
                obs = battle_select(list(range(int(select["maxCount"]))))
            else:
                self.skipTest("no MAIN decision reached")

            select = obs["select"]
            self.assertGreater(int(select["minCount"]), 0)

            module = self.bot.agent.__globals__
            original = module["_policy"]
            module["_policy"] = lambda _obs: []
            try:
                action = self.bot.agent(obs)
            finally:
                module["_policy"] = original

            self.assertTrue(is_valid(action, select), f"illegal action: {action!r}")
            self.assertGreaterEqual(len(action), int(select["minCount"]))
            self.assertLessEqual(len(action), int(select["maxCount"]))
        finally:
            battle_finish()

    def test_typed_observation_matches_dict(self) -> None:
        """Kaggle can hand the agent a typed Observation instead of a dict."""
        from cg.api import to_observation_class
        from cg.game import battle_finish, battle_start

        obs, _ = battle_start(self.bot.deck, self.bot.deck)
        self.assertIsNotNone(obs)
        try:
            typed = to_observation_class(obs)
            self.assertEqual(self.bot.agent(obs), self.bot.agent(typed))
        finally:
            battle_finish()

    def test_typed_deck_selection_returns_deck(self) -> None:
        """The validation episode failed here: a typed select=None must return 60 cards."""
        from cg.api import to_observation_class
        from cg.game import battle_finish, battle_start

        obs, _ = battle_start(self.bot.deck, self.bot.deck)
        self.assertIsNotNone(obs)
        try:
            deck_obs = dict(obs)
            deck_obs["select"] = None
            typed = to_observation_class(deck_obs)
            self.assertIsNone(typed.select)
            action = self.bot.agent(typed)
            self.assertEqual(len(action), 60)
            self.assertEqual(action, self.bot.deck)
        finally:
            battle_finish()

    def test_short_games_have_no_invalid_actions(self) -> None:
        from arena import load_bot, play_match

        opponent = load_bot(ROOT / "bots" / "_template")
        for game in range(20):
            result = play_match(
                self.bot.agent,
                opponent.agent,
                self.bot.deck,
                opponent.deck,
                bo3=False,
                reverse_first=(game % 2 == 1),
                max_steps=4000,
            )
            self.assertIsNone(result["error"], f"game {game}: {result['error']}")


if __name__ == "__main__":
    unittest.main()
