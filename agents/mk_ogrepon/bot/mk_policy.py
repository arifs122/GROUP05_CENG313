"""Policy orchestration and Kaggle entry point helpers."""

from __future__ import annotations

import os
import time
from pathlib import Path

import mk_state as S
from mk_rules import RuleEngine

BOT_DIR = Path(__file__).resolve().parent
_DECK: list[int] | None = None
_RULES: RuleEngine | None = None
_SEARCH = None
_LAST = {"calls": 0, "seconds": 0.0, "fallbacks": 0}

# The competition gives 600 s of banked overage.  Stay far below it; the rule
# layer is fast and search uses only the remaining per-decision budget.
DEFAULT_BUDGET = float(os.environ.get("MK_BUDGET", "0.18"))


def load_deck() -> list[int]:
    global _DECK
    if _DECK is not None:
        return list(_DECK)
    candidates: list[Path] = []
    env_path = os.environ.get("MK_DECK")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([
        BOT_DIR / "deck.csv",
        Path.cwd() / "deck.csv",
        Path("/kaggle_simulations/agent/deck.csv"),
    ])
    for path in candidates:
        if not path.is_file():
            continue
        try:
            cards = [int(x) for x in path.read_text(encoding="utf-8").split() if x.strip()]
        except (OSError, ValueError):
            continue
        if len(cards) == 60:
            _DECK = cards
            return list(_DECK)
    raise FileNotFoundError("mk_ogrepon deck.csv not found (expected 60 cards)")


def get_rules() -> RuleEngine:
    global _RULES, _SEARCH
    if _RULES is None:
        deck = load_deck()
        search_enabled = os.environ.get("MK_SEARCH", "1") not in ("0", "false", "False", "no")
        prior = None
        prior_weight = float(os.environ.get("MK_PRIOR_WEIGHT", "60"))
        if os.environ.get("MK_PRIOR", "1") not in ("0", "false", "False", "no"):
            try:
                from mk_model import PolicyPrior

                prior = PolicyPrior()
            except Exception:
                prior = None
        if search_enabled:
            try:
                from mk_search import SearchEngine

                value_model = None
                value_weight = float(os.environ.get("MK_VALUE_WEIGHT", "0"))
                if os.environ.get("MK_VALUE", "1") not in ("0", "false", "False", "no"):
                    try:
                        from mk_value import ValueModel

                        value_model = ValueModel()
                    except Exception:
                        value_model = None
                rollouter = RuleEngine(deck, prior=prior, prior_weight=prior_weight)
                _SEARCH = SearchEngine(deck, budget=DEFAULT_BUDGET, rollouter=rollouter,
                                       value_model=value_model, value_weight=value_weight)
                _RULES = RuleEngine(deck, searcher=_SEARCH, prior=prior,
                                    prior_weight=prior_weight)
            except Exception:
                _SEARCH = None
                _RULES = RuleEngine(deck, prior=prior, prior_weight=prior_weight)
        else:
            _RULES = RuleEngine(deck, prior=prior, prior_weight=prior_weight)
    return _RULES


def _legal_fallback(obs: dict, deck: list[int]) -> list[int]:
    sel = S.select(obs)
    if sel is None:
        return list(deck)
    opts = S.options(obs)
    if not opts:
        return []
    mn = min(S.min_count(obs), len(opts))
    mx = min(S.max_count(obs), len(opts))
    if mx <= 0:
        return []
    count = max(mn, min(mx, 1))
    return list(range(count))


def agent(obs_dict: dict) -> list[int]:
    """Kaggle/local arena entry point.  Never raises, never illegal."""
    started = time.perf_counter()
    try:
        rules = get_rules()
        # The competition gives 600 s of banked overage.  Keep a large safety
        # margin and fall back to the rule layer if we are approaching it.
        if _LAST["seconds"] > 450.0 and getattr(rules, "searcher", None) is not None:
            try:
                rules.searcher.budget = 0.0
            except Exception:
                pass
        action = rules.choose(obs_dict)
        sel = S.select(obs_dict)
        if sel is None:
            if isinstance(action, list) and len(action) == 60:
                return action
            return load_deck()
        opts = S.options(obs_dict)
        mn = S.min_count(obs_dict)
        mx = min(S.max_count(obs_dict), len(opts))
        if (not isinstance(action, list) or len(action) < min(mn, len(opts))
                or len(action) > mx or len(set(action)) != len(action)
                or any(not isinstance(x, int) or x < 0 or x >= len(opts) for x in action)):
            _LAST["fallbacks"] += 1
            return _legal_fallback(obs_dict, load_deck())
        return [int(x) for x in action]
    except Exception:
        _LAST["fallbacks"] += 1
        try:
            return _legal_fallback(obs_dict, load_deck())
        except Exception:
            return []
    finally:
        _LAST["calls"] += 1
        _LAST["seconds"] += time.perf_counter() - started


def stats() -> dict:
    return dict(_LAST)
