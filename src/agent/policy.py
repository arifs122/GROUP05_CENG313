"""Baseline policy for the PTCG AI Battle Challenge.

This module intentionally works on the raw observation dict that the
simulator passes to the agent.  Keeping the baseline dependency-free makes
it easy to swap in a smarter policy later.

The policy is deliberately simple:

* MAIN selections: evolve / use ability / attach energy / play a card /
  attack / retreat / discard / end (first legal option in that order).
* ATTACK selections: pick the legal attack with the highest printed damage.
* YES/NO: answer "yes" when possible.
* COUNT: pick the largest number for draw/damage-counter effects, otherwise
  the smallest legal number.
* Card selections: take the first legal options (minimum count when the
  effect asks us to discard something).

It is a starting point, not a strong agent.
"""

from __future__ import annotations

import os
from pathlib import Path

from cg.api import OptionType, SelectContext, SelectType

# Printed damage per attack id, loaded lazily from the engine.
_ATTACK_DAMAGE: dict[int, int] | None = None

# Lower number = higher priority during the MAIN selection.
_MAIN_PRIORITY = {
    int(OptionType.EVOLVE): 0,
    int(OptionType.ABILITY): 1,
    int(OptionType.ATTACH): 2,
    int(OptionType.PLAY): 3,
    int(OptionType.ATTACK): 4,
    int(OptionType.RETREAT): 5,
    int(OptionType.DISCARD): 6,
    int(OptionType.END): 7,
}

# Contexts where the effect asks us to get rid of something.
_DISCARD_CONTEXTS = {
    int(SelectContext.DISCARD),
    int(SelectContext.DISCARD_ENERGY_CARD),
    int(SelectContext.DISCARD_TOOL_CARD),
    int(SelectContext.DISCARD_CARD_OR_ATTACHED_CARD),
    int(SelectContext.DISCARD_ENERGY),
}

# Contexts where "more" is better.
_MORE_IS_BETTER_COUNT_CONTEXTS = {
    int(SelectContext.DRAW_COUNT),
    int(SelectContext.DAMAGE_COUNTER_COUNT),
    int(SelectContext.REMOVE_DAMAGE_COUNTER_COUNT),
}


def _get(obj, key: str, default=None):
    """Read a field from either a dict or a dataclass-like object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _attack_damage(attack_id: int | None) -> int:
    global _ATTACK_DAMAGE
    if attack_id is None:
        return -1
    if _ATTACK_DAMAGE is None:
        from cg.api import all_attack

        _ATTACK_DAMAGE = {int(a.attackId): int(a.damage or 0) for a in all_attack()}
    return _ATTACK_DAMAGE.get(int(attack_id), 0)


def _best_main(options: list[dict]) -> int:
    return min(
        range(len(options)),
        key=lambda i: (_MAIN_PRIORITY.get(int(_get(options[i], "type", -1)), 99), i),
    )


def _best_attack(options: list[dict]) -> int:
    return max(
        range(len(options)),
        key=lambda i: (_attack_damage(_get(options[i], "attackId")), -i),
    )


def _best_count(options: list[dict], context: int) -> int:
    numbers = [int(_get(o, "number", 0)) for o in options]
    if context in _MORE_IS_BETTER_COUNT_CONTEXTS:
        return max(range(len(numbers)), key=lambda i: (numbers[i], -i))
    return min(range(len(numbers)), key=lambda i: (numbers[i], i))


def _yes_index(options: list[dict]) -> int | None:
    for i, option in enumerate(options):
        if int(_get(option, "type", -1)) == int(OptionType.YES):
            return i
    return None


def act(obs: dict) -> list[int]:
    """Return a list of legal option indices for the current observation."""
    select = obs.get("select")
    if select is None:
        # Initial deck selection.
        return load_deck()

    options = select.get("option") or []
    if not options:
        return []

    min_count = max(0, int(select.get("minCount", 0)))
    max_count = min(int(select.get("maxCount", 0)), len(options))
    if max_count <= 0:
        return []

    select_type = int(select.get("type", -1))
    context = int(select.get("context", -1))

    if select_type == int(SelectType.MAIN):
        return [_best_main(options)]

    if select_type == int(SelectType.ATTACK):
        return [_best_attack(options)]

    if select_type == int(SelectType.YES_NO):
        yes = _yes_index(options)
        if yes is not None:
            return [yes]
        return list(range(min(1, max_count)))

    if select_type == int(SelectType.COUNT):
        return [_best_count(options, context)]

    # Card / energy / tool / skill selections.
    if context in _DISCARD_CONTEXTS:
        count = min_count
    else:
        count = max_count
    count = max(min_count, min(count, max_count))
    return list(range(count))


# --------------------------------------------------------------------------
# Deck loading
# --------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_DECK: list[int] | None = None


def _deck_candidates() -> list[Path]:
    candidates = []
    env_path = os.environ.get("PTCG_DECK")
    if env_path:
        candidates.append(Path(env_path))

    # Inside a submission bundle deck.csv sits next to main.py.
    candidates.append(_HERE / "deck.csv")
    candidates.append(_HERE.parents[1] / "decks" / "current.csv")
    candidates.append(_HERE.parents[1] / "decks" / "mega_abomasnow.csv")
    candidates.append(Path.cwd() / "deck.csv")
    candidates.append(Path("/kaggle_simulations/agent/deck.csv"))
    return candidates


def load_deck() -> list[int]:
    """Load and cache the 60-card deck from the first existing candidate."""
    global _DECK
    if _DECK is not None:
        return list(_DECK)

    for path in _deck_candidates():
        if path.is_file():
            cards = [
                int(line.strip())
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(cards) != 60:
                raise ValueError(f"{path} contains {len(cards)} cards, expected 60")
            _DECK = cards
            return list(_DECK)

    searched = "\n  ".join(str(p) for p in _deck_candidates())
    raise FileNotFoundError(
        "deck.csv not found. Set PTCG_DECK or create decks/current.csv.\n"
        f"Searched:\n  {searched}"
    )
