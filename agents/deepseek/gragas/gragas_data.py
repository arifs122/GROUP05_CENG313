"""Static card knowledge for the Top2 Dragapult agent.

This module is deliberately small and dependency-light.  The engine is only
used for card/attack metadata; the policy never reads the hidden replay fields.
"""

from __future__ import annotations

from functools import lru_cache

# --- our Top2 list ---------------------------------------------------------
DREEPY = 119
DRAKLOAK = 120
DRAGAPULT = 121
MUNKIDORI = 112
BUDEW = 235
FEZANDIPITI = 140
MEOWTH = 1071
CHI_YU = 31

FIRE = 2
PSYCHIC = 5
DARK = 7

POFFIN = 1086
POKE_PAD = 1152
ULTRA_BALL = 1121
CRUSHING_HAMMER = 1120
NIGHT_STRETCHER = 1097
UNFAIR_STAMP = 1080
WATCHTOWER = 1256
RISKY_RUINS = 1260
BOSS = 1182
CRISPIN = 1198
JUDGE = 1213
LILLIE = 1227

PHANTOM_DIVE = 154
JET_HEADBUTT = 153
DRAGON_HEADBUTT = 152
MIND_BEND = 141
ITCHY_POLLEN = 323
CRUEL_ARROW = 183
TUCK_TAIL = 1546
GROUND_MELTER = 20

# Roles used by the rule scorer.
SETUP_IDS = {DREEPY, DRAKLOAK, DRAGAPULT, BUDEW, POFFIN, POKE_PAD, ULTRA_BALL}
DRAW_IDS = {LILLIE, JUDGE, DRAKLOAK}
SEARCH_IDS = {POFFIN, POKE_PAD, ULTRA_BALL}
ENERGY_IDS = {FIRE, PSYCHIC, DARK}
ACE_SPEC_IDS = {UNFAIR_STAMP}

# Top2 deck card IDs, used as the default opponent model when no meta deck
# matches the visible cards.
TOP2_DECK = [
    2, 2, 2, 2, 5, 5, 5, 5, 7, 7, 31, 112, 112, 119, 119, 119, 119,
    120, 120, 120, 120, 121, 121, 121, 140, 235, 235, 1071, 1080, 1086,
    1086, 1086, 1086, 1097, 1097, 1120, 1120, 1120, 1120, 1121, 1121,
    1121, 1152, 1152, 1152, 1152, 1182, 1182, 1182, 1198, 1198, 1198,
    1213, 1213, 1227, 1227, 1227, 1227, 1256, 1260,
]

# Meta deck library. Each list is the opponent's *public archetype list*; it
# is only used to sample hidden cards during search determinisation.
try:
    from top2_core_meta import META_DECKS as _META_DECKS
except Exception:  # pragma: no cover - bundle/path fallback
    _META_DECKS = {}

META_DECKS: dict[str, list[int]] = {
    name: list(cards) for name, cards in _META_DECKS.items() if len(cards) == 60
}
if not META_DECKS:
    META_DECKS = {"dragapult": list(TOP2_DECK)}


@lru_cache(maxsize=1)
def _cards() -> dict[int, object]:
    try:
        from cg.api import all_card_data

        return {int(c.cardId): c for c in all_card_data()}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _attacks() -> dict[int, object]:
    try:
        from cg.api import all_attack

        return {int(a.attackId): a for a in all_attack()}
    except Exception:
        return {}


def card(card_id: int | None):
    if card_id is None:
        return None
    return _cards().get(int(card_id))


def attack(attack_id: int | None):
    if attack_id is None:
        return None
    return _attacks().get(int(attack_id))


def card_name(card_id: int | None) -> str:
    c = card(card_id)
    return str(getattr(c, "name", card_id))


def energy_type(card_id: int | None) -> int:
    c = card(card_id)
    return int(getattr(c, "energyType", -1)) if c is not None else -1


def prize_value(card_id: int | None) -> int:
    c = card(card_id)
    if c is None:
        return 1
    if bool(getattr(c, "megaEx", False)):
        return 3
    if bool(getattr(c, "ex", False)):
        return 2
    return 1


def is_basic(card_id: int | None) -> bool:
    c = card(card_id)
    return bool(getattr(c, "basic", False)) if c is not None else False


def is_pokemon(card_id: int | None) -> bool:
    c = card(card_id)
    return getattr(c, "cardType", None) == 0 if c is not None else False
