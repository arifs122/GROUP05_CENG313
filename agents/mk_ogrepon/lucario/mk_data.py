"""Static card knowledge and small policy constants for the MK/Ogerpon agent."""

from __future__ import annotations

from functools import lru_cache

# --- deck ---------------------------------------------------------------
GRASS = 1
PRISM = 16
CHI_YU = 31
OGERPON_TEAL = 96
OGERPON_WELL = 108
MUNKIDORI = 112
FEZANDIPITI = 140
LATIAS = 184
PECHARUNT = 230
CLEFAIRY = 272
KANGASKHAN = 756
MEOWTH = 1071
UNFAIR_STAMP = 1080
BUG_CATCHING = 1094
NIGHT_STRETCHER = 1097
ENERGY_SWITCH = 1116
ULTRA_BALL = 1121
TERA_ORB = 1127
BOSS = 1182
CIPHERMANIAC = 1188
N_PLAN = 1221
LILLIE = 1227
AREA_ZERO = 1250

# attack ids
MYRIAD_LEAF_SHOWER = 120
SOB = 135
TORRENTIAL_PUMP = 136
MIND_BEND = 141
CRUEL_ARROW = 183
EON_BLADE = 243
POISON_CHAIN = 315
FULL_MOON_RONDO = 371
ALLURE = 19
GROUND_MELTER = 20
RAPID_FIRE_COMBO = 1092
TUCK_TAIL = 1546

# CardData CardType
POKEMON = 0
ITEM = 1
TOOL = 2
SUPPORTER = 3
STADIUM = 4
BASIC_ENERGY = 5
SPECIAL_ENERGY = 6

# EnergyType
COLORLESS = 0
GRASS_T = 1
FIRE_T = 2
WATER_T = 3
LIGHTNING_T = 4
PSYCHIC_T = 5
FIGHTING_T = 6
DARKNESS_T = 7
METAL_T = 8
DRAGON_T = 9

DECK = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    16, 16, 16, 16,
    31,
    96, 96, 96, 96,
    108,
    112,
    140,
    184, 184,
    230,
    272, 272,
    756, 756,
    1071, 1071,
    1080,
    1094, 1094, 1094,
    1097, 1097,
    1116, 1116, 1116, 1116,
    1121, 1121, 1121, 1121,
    1127, 1127,
    1182, 1182, 1182,
    1188,
    1221,
    1227, 1227, 1227, 1227,
    1250, 1250, 1250,
]

CORE_POKEMON = {
    OGERPON_TEAL, OGERPON_WELL, LATIAS, CLEFAIRY, KANGASKHAN,
    MEOWTH, FEZANDIPITI, MUNKIDORI, CHI_YU, PECHARUNT,
}
TERA_IDS = {OGERPON_TEAL, OGERPON_WELL}
ATTACKERS = {OGERPON_TEAL, OGERPON_WELL, KANGASKHAN, LATIAS, CLEFAIRY,
             FEZANDIPITI, MUNKIDORI, CHI_YU, PECHARUNT, MEOWTH}
SUPPORT_POKEMON = {LATIAS, CLEFAIRY, MEOWTH, FEZANDIPITI, MUNKIDORI, CHI_YU, PECHARUNT}
BENCH_ORDER = {
    OGERPON_TEAL: 150,
    OGERPON_WELL: 138,
    KANGASKHAN: 90,
    CLEFAIRY: 86,
    LATIAS: 84,
    MEOWTH: 78,
    FEZANDIPITI: 72,
    MUNKIDORI: 62,
    CHI_YU: 70,
    PECHARUNT: 60,
}
ACTIVE_ORDER = {
    KANGASKHAN: 100,
    OGERPON_TEAL: 96,
    OGERPON_WELL: 94,
    CLEFAIRY: 90,
    CHI_YU: 76,
    LATIAS: 72,
    PECHARUNT: 64,
    FEZANDIPITI: 56,
    MUNKIDORI: 54,
    MEOWTH: 42,
}

# Static value used when a card must be discarded or picked from a pile.
BASE_VALUE = {
    OGERPON_TEAL: 150,
    OGERPON_WELL: 138,
    KANGASKHAN: 150,
    LATIAS: 140,
    CLEFAIRY: 140,
    MEOWTH: 120,
    FEZANDIPITI: 115,
    MUNKIDORI: 90,
    CHI_YU: 70,
    PECHARUNT: 60,
    GRASS: 30,
    PRISM: 55,
    AREA_ZERO: 105,
    LILLIE: 125,
    BOSS: 125,
    ULTRA_BALL: 120,
    BUG_CATCHING: 110,
    TERA_ORB: 95,
    ENERGY_SWITCH: 110,
    N_PLAN: 100,
    NIGHT_STRETCHER: 90,
    CIPHERMANIAC: 85,
    UNFAIR_STAMP: 100,
}

# Roles for search/selection heuristics.
SETUP_IDS = {OGERPON_TEAL, OGERPON_WELL, KANGASKHAN, LATIAS, CLEFAIRY,
             MEOWTH, FEZANDIPITI, MUNKIDORI, CHI_YU, PECHARUNT}
DRAW_IDS = {LILLIE, BUG_CATCHING, CIPHERMANIAC, CHI_YU, MEOWTH, KANGASKHAN,
            FEZANDIPITI, OGERPON_TEAL}
SEARCH_IDS = {ULTRA_BALL, BUG_CATCHING, TERA_ORB, NIGHT_STRETCHER}
ENERGY_IDS = {GRASS, PRISM}

# Public archetype fingerprints.  Only visible cards are used.
ARCHETYPE_KEYS: list[tuple[str, tuple[int, ...]]] = [
    ("Dragapult", (121, 120, 119)),
    ("Alakazam", (743,)),
    ("Mega Kangaskhan", (756,)),
    ("Mega Lucario", (678, 677)),
    ("Hydrapple", (150, 149)),
    ("Archaludon", (190,)),
    ("Banette", (1375, 1374)),
    ("Festival Lead", (1245, 90)),
    ("Mega Excadrill", (1406, 547)),
    ("Mega Abomasnow", (723,)),
    ("Ceruledge", (320, 319)),
    ("Mega Starmie", (1031, 1030)),
    ("Mega Froslass", (861,)),
    ("Marnie's Grimmsnarl", (648,)),
    ("Cynthia's Garchomp", (381,)),
    ("Teal Mask Ogerpon", (96, 108)),
    ("N's Zorua", (292, 291)),
    ("Slowking", (163,)),
    ("Crustle", (345, 344)),
    ("Munkidori-Crustle", (112, 345)),
    ("Chandelure", (98, 495)),
    ("Mega Charizard", (790, 928)),
    ("Mega Gardevoir", (747,)),
    ("Mega Gengar", (772,)),
    ("Mega Absol", (687,)),
    ("Mega Blaziken", (326,)),
    ("Raging Bolt", (269, 270)),
    ("Unknown", ()),
]


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
    return str(getattr(c, "name", card_id)) if c is not None else str(card_id)


def card_type(card_id: int | None) -> int:
    c = card(card_id)
    return int(getattr(c, "cardType", -1)) if c is not None else -1


def is_pokemon(card_id: int | None) -> bool:
    return card_type(card_id) == POKEMON


def is_basic(card_id: int | None) -> bool:
    c = card(card_id)
    return bool(getattr(c, "basic", False)) if c is not None else False


def is_tera(card_id: int | None) -> bool:
    c = card(card_id)
    return bool(getattr(c, "tera", False)) if c is not None else False


def is_ex(card_id: int | None) -> bool:
    c = card(card_id)
    return bool(getattr(c, "ex", False)) if c is not None else False


def is_mega(card_id: int | None) -> bool:
    c = card(card_id)
    return bool(getattr(c, "megaEx", False)) if c is not None else False


def prize_value(card_id: int | None) -> int:
    if is_mega(card_id):
        return 3
    if is_ex(card_id):
        return 2
    return 1


def weakness(card_id: int | None) -> int | None:
    c = card(card_id)
    if c is None:
        return None
    w = getattr(c, "weakness", None)
    return int(w) if w is not None else None


def resistance(card_id: int | None) -> int | None:
    c = card(card_id)
    if c is None:
        return None
    r = getattr(c, "resistance", None)
    return int(r) if r is not None else None


def energy_type(card_id: int | None) -> int:
    c = card(card_id)
    return int(getattr(c, "energyType", -1)) if c is not None else -1


def attacks_of(card_id: int | None) -> list[int]:
    c = card(card_id)
    return [int(a) for a in (getattr(c, "attacks", []) or [])] if c is not None else []


def base_damage(attack_id: int | None) -> int:
    a = attack(attack_id)
    return int(getattr(a, "damage", 0) or 0) if a is not None else 0


def role_value(card_id: int | None) -> int:
    return BASE_VALUE.get(int(card_id), 20) if card_id is not None else 0
