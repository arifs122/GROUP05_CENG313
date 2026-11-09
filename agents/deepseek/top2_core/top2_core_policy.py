"""Policy orchestration: rules first, search as a conservative reranker."""

from __future__ import annotations

import os
from pathlib import Path

import top2_core_data as D
from top2_core_rules import RuleEngine

BOT_DIR = Path(__file__).resolve().parent
_DECK: list[int] | None = None
_RULES: RuleEngine | None = None
_SEARCH = None


def load_deck(path: Path | None = None) -> list[int]:
    global _DECK
    if _DECK is not None:
        return list(_DECK)
    candidates = []
    env_path = os.environ.get("TOP2_DECK")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([BOT_DIR / "deck.csv", Path.cwd() / "deck.csv",
                       Path("/kaggle_simulations/agent/deck.csv")])
    for candidate in candidates:
        if candidate.is_file():
            cards = [int(x) for x in candidate.read_text(encoding="utf-8").split() if x.strip()]
            if len(cards) == 60:
                _DECK = cards
                return list(_DECK)
    raise FileNotFoundError("top2_core deck.csv not found (expected 60 cards)")


def get_rules() -> RuleEngine:
    global _RULES, _SEARCH
    if _RULES is None:
        deck = load_deck()
        search_enabled = os.environ.get("TOP2_SEARCH", "1") not in ("0", "false", "False")
        if search_enabled:
            try:
                from top2_core_search import SearchEngine

                try:
                    from top2_core_value import value_logit

                    value_fn = value_logit
                except Exception:
                    value_fn = None
                try:
                    from top2_core_policy_net import get_prior

                    prior = get_prior()
                except Exception:
                    prior = None
                _SEARCH = SearchEngine(deck, value_fn=value_fn)
                _RULES = RuleEngine(deck, searcher=_SEARCH, prior=prior)
                _SEARCH.rules = _RULES
            except Exception:
                _SEARCH = None
                _RULES = RuleEngine(deck)
        else:
            _RULES = RuleEngine(deck)
    return _RULES


def agent(obs_dict: dict) -> list[int]:
    return get_rules().choose(obs_dict)
