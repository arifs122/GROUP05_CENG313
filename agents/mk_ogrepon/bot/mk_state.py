"""Observation helpers for the Mega Kangaskhan / Ogerpon agent.

The competition observation is a plain dict locally and may be a dataclass-like
object in other runners, so every accessor accepts both.  Nothing here reads
hidden information: only public zones and the player's own hand are exposed.
"""

from __future__ import annotations

from typing import Any

# AreaType
DECK = 1
HAND = 2
DISCARD = 3
ACTIVE = 4
BENCH = 5
PRIZE = 6
STADIUM = 7
ENERGY = 8
TOOL = 9
PRE_EVOLUTION = 10
PLAYER = 11
LOOKING = 12

# SelectType
S_MAIN = 0
S_CARD = 1
S_ATTACHED_CARD = 2
S_CARD_OR_ATTACHED = 3
S_ENERGY = 4
S_SKILL = 5
S_ATTACK = 6
S_EVOLVE = 7
S_COUNT = 8
S_YES_NO = 9
S_SPECIAL_CONDITION = 10

# SelectContext
C_MAIN = 0
C_SETUP_ACTIVE = 1
C_SETUP_BENCH = 2
C_SWITCH = 3
C_TO_ACTIVE = 4
C_TO_BENCH = 5
C_TO_FIELD = 6
C_TO_HAND = 7
C_DISCARD = 8
C_TO_DECK = 9
C_TO_DECK_BOTTOM = 10
C_TO_PRIZE = 11
C_NOT_MOVE = 12
C_DAMAGE_COUNTER = 13
C_DAMAGE_COUNTER_ANY = 14
C_DAMAGE = 15
C_REMOVE_DAMAGE_COUNTER = 16
C_HEAL = 17
C_EVOLVES_FROM = 18
C_EVOLVES_TO = 19
C_DEVOLVE = 20
C_ATTACH_FROM = 21
C_ATTACH_TO = 22
C_DETACH_FROM = 23
C_LOOK = 24
C_EFFECT_TARGET = 25
C_DISCARD_ENERGY_CARD = 26
C_DISCARD_TOOL_CARD = 27
C_SWITCH_ENERGY_CARD = 28
C_DISCARD_CARD_OR_ATTACHED = 29
C_DISCARD_ENERGY = 30
C_TO_HAND_ENERGY = 31
C_TO_DECK_ENERGY = 32
C_SWITCH_ENERGY = 33
C_SKILL_ORDER = 34
C_ATTACK = 35
C_DISABLE_ATTACK = 36
C_EVOLVE = 37
C_DRAW_COUNT = 38
C_DAMAGE_COUNTER_COUNT = 39
C_REMOVE_DAMAGE_COUNTER_COUNT = 40
C_IS_FIRST = 41
C_MULLIGAN = 42
C_ACTIVATE = 43
C_FIRST_EFFECT = 44
C_MORE_DEVOLVE = 45
C_COIN_HEAD = 46
C_AFFECT_SPECIAL_CONDITION = 47
C_RECOVER_SPECIAL_CONDITION = 48

# OptionType
O_NUMBER = 0
O_YES = 1
O_NO = 2
O_CARD = 3
O_TOOL_CARD = 4
O_ENERGY_CARD = 5
O_ENERGY = 6
O_PLAY = 7
O_ATTACH = 8
O_EVOLVE = 9
O_ABILITY = 10
O_DISCARD = 11
O_RETREAT = 12
O_ATTACK = 13
O_END = 14
O_SKILL = 15
O_SPECIAL_CONDITION = 16


def get(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def current(obs: Any) -> Any:
    return get(obs, "current")


def select(obs: Any) -> Any:
    return get(obs, "select")


def select_type(obs: Any) -> int:
    return as_int(get(select(obs), "type", -1), -1)


def select_context(obs: Any) -> int:
    return as_int(get(select(obs), "context", -1), -1)


def min_count(obs: Any) -> int:
    return max(0, as_int(get(select(obs), "minCount", 0), 0) or 0)


def max_count(obs: Any) -> int:
    return max(0, as_int(get(select(obs), "maxCount", 0), 0) or 0)


def options(obs: Any) -> list:
    return list(get(select(obs), "option", []) or [])


def you(obs: Any) -> int:
    return as_int(get(current(obs), "yourIndex", 0), 0)


def players(obs: Any) -> list:
    return list(get(current(obs), "players", []) or [])


def player(obs: Any, index: int) -> Any:
    ps = players(obs)
    return ps[index] if 0 <= index < len(ps) else None


def my_state(obs: Any) -> Any:
    return player(obs, you(obs))


def opp_state(obs: Any) -> Any:
    return player(obs, 1 - you(obs))


def active(obs: Any, owner: int) -> Any:
    st = player(obs, owner)
    arr = list(get(st, "active", []) or [])
    return arr[0] if arr else None


def bench(obs: Any, owner: int) -> list:
    st = player(obs, owner)
    return list(get(st, "bench", []) or [])


def field(obs: Any, owner: int) -> list:
    out = []
    a = active(obs, owner)
    if a is not None:
        out.append(a)
    out.extend(bench(obs, owner))
    return out


def hand(obs: Any, owner: int) -> list | None:
    st = player(obs, owner)
    h = get(st, "hand", None)
    return None if h is None else list(h)


def hand_ids(obs: Any, owner: int) -> list[int]:
    out = []
    for card in hand(obs, owner) or []:
        cid = card_id(card)
        if cid is not None:
            out.append(cid)
    return out


def hand_count(obs: Any, owner: int) -> int:
    return as_int(get(player(obs, owner), "handCount", 0), 0) or 0


def discard(obs: Any, owner: int) -> list:
    return list(get(player(obs, owner), "discard", []) or [])


def discard_ids(obs: Any, owner: int) -> list[int]:
    out = []
    for card in discard(obs, owner):
        cid = card_id(card)
        if cid is not None:
            out.append(cid)
    return out


def prize(obs: Any, owner: int) -> list:
    return list(get(player(obs, owner), "prize", []) or [])


def prize_count(obs: Any, owner: int) -> int:
    return len(prize(obs, owner))


def deck_count(obs: Any, owner: int) -> int:
    return as_int(get(player(obs, owner), "deckCount", 0), 0) or 0


def stadium(obs: Any) -> list:
    return list(get(current(obs), "stadium", []) or [])


def stadium_ids(obs: Any) -> list[int]:
    out = []
    for card in stadium(obs):
        cid = card_id(card)
        if cid is not None:
            out.append(cid)
    return out


def looking(obs: Any) -> list:
    return list(get(current(obs), "looking", []) or [])


def deck_cards(obs: Any) -> list:
    return list(get(select(obs), "deck", []) or [])


def card_id(card: Any) -> int | None:
    if card is None:
        return None
    return as_int(get(card, "id"), None)


def serial(card: Any) -> int | None:
    return as_int(get(card, "serial"), None)


def pokemon_id(pokemon: Any) -> int | None:
    return card_id(pokemon)


def energies(pokemon: Any) -> list[int]:
    return [int(e) for e in (get(pokemon, "energies", []) or [])]


def energy_count(pokemon: Any) -> int:
    return len(energies(pokemon))


def energy_cards(pokemon: Any) -> list[int]:
    out = []
    for card in get(pokemon, "energyCards", []) or []:
        cid = card_id(card)
        if cid is not None:
            out.append(cid)
    return out


def tools(pokemon: Any) -> list[int]:
    out = []
    for card in get(pokemon, "tools", []) or []:
        cid = card_id(card)
        if cid is not None:
            out.append(cid)
    return out


def pre_evolution(pokemon: Any) -> list[int]:
    out = []
    for card in get(pokemon, "preEvolution", []) or []:
        cid = card_id(card)
        if cid is not None:
            out.append(cid)
    return out


def hp(pokemon: Any) -> int:
    return as_int(get(pokemon, "hp", 0), 0) or 0


def max_hp(pokemon: Any) -> int:
    return as_int(get(pokemon, "maxHp", 0), 0) or 0


def damaged(pokemon: Any) -> bool:
    return max_hp(pokemon) > 0 and hp(pokemon) < max_hp(pokemon)


def turn(obs: Any) -> int:
    return as_int(get(current(obs), "turn", 0), 0) or 0


def result(obs: Any) -> int:
    return as_int(get(current(obs), "result", -1), -1)


def supporter_played(obs: Any) -> bool:
    return bool(get(current(obs), "supporterPlayed", False))


def stadium_played(obs: Any) -> bool:
    return bool(get(current(obs), "stadiumPlayed", False))


def energy_attached(obs: Any) -> bool:
    return bool(get(current(obs), "energyAttached", False))


def retreated(obs: Any) -> bool:
    return bool(get(current(obs), "retreated", False))


def option_type(option: Any) -> int:
    return as_int(get(option, "type", -1), -1)


def option_int(option: Any, key: str, default=None):
    value = get(option, key, None)
    if value is None:
        return default
    return as_int(value, default)


def zone_cards(obs: Any, owner: int, area: int) -> list:
    """Visible cards in a zone.  Deck/looking are read from select/current."""
    area = int(area)
    if area == HAND:
        return list(hand(obs, owner) or [])
    if area == DISCARD:
        return discard(obs, owner)
    if area == ACTIVE:
        return list(get(player(obs, owner), "active", []) or [])
    if area == BENCH:
        return bench(obs, owner)
    if area == PRIZE:
        return prize(obs, owner)
    if area == STADIUM:
        return stadium(obs)
    if area == DECK:
        return deck_cards(obs)
    if area == LOOKING:
        return looking(obs)
    return []


def card_at(obs: Any, owner: int, area: int | None, index: int | None) -> int | None:
    if area is None or index is None:
        return None
    try:
        cards = zone_cards(obs, owner, int(area))
        idx = int(index)
    except (TypeError, ValueError):
        return None
    if 0 <= idx < len(cards):
        return card_id(cards[idx])
    return None


def pokemon_at(obs: Any, owner: int, area: int | None, index: int | None) -> Any:
    if area is None or index is None:
        return None
    try:
        idx = int(index)
    except (TypeError, ValueError):
        return None
    if int(area) == ACTIVE:
        arr = list(get(player(obs, owner), "active", []) or [])
        return arr[idx] if 0 <= idx < len(arr) else None
    if int(area) == BENCH:
        arr = bench(obs, owner)
        return arr[idx] if 0 <= idx < len(arr) else None
    return None


def option_owner(obs: Any, option: Any) -> int:
    """Best guess at the owner of an option's card."""
    pi = option_int(option, "playerIndex", None)
    if pi is not None:
        return int(pi)
    area = option_int(option, "area", None)
    inplay = option_int(option, "inPlayArea", None)
    if inplay is not None:
        return you(obs)
    if area in (ACTIVE, BENCH):
        return you(obs)
    return you(obs)


def option_cards(obs: Any, option: Any) -> tuple[int | None, int | None]:
    """Return ``(actor_card_id, target_card_id)`` for an option.

    actor is the card being played/attached/selected; target is the Pokémon or
    card the option points at.  This covers MAIN and targeted selections.
    """
    otype = option_type(option)
    me = you(obs)
    owner = option_owner(obs, option)
    area = option_int(option, "area", None)
    index = option_int(option, "index", None)

    actor: int | None = None
    target: int | None = None

    if otype in (O_TOOL_CARD, O_ENERGY_CARD, O_ENERGY):
        parent = pokemon_at(obs, owner, area, index)
        if parent is not None:
            if otype == O_TOOL_CARD:
                arr = list(get(parent, "tools", []) or [])
                idx = option_int(option, "toolIndex", 0) or 0
            else:
                arr = list(get(parent, "energyCards", []) or [])
                idx = option_int(option, "energyIndex", 0) or 0
            if 0 <= idx < len(arr):
                actor = card_id(arr[idx])
            if otype == O_ENERGY and actor is None:
                actor = None
        if otype == O_ENERGY and actor is None:
            # Energy unit options can expose only the attached Pokémon.
            target = card_id(parent)
    elif otype == O_PLAY:
        actor = card_at(obs, me, HAND, index)
    elif otype in (O_CARD, O_SKILL, O_ABILITY, O_DISCARD, O_ATTACH, O_EVOLVE):
        actor = card_at(obs, owner, area, index)
    elif otype in (O_RETREAT, O_ATTACK):
        actor = pokemon_id(active(obs, me))
    elif otype == O_NUMBER:
        actor = None

    inplay_area = option_int(option, "inPlayArea", None)
    inplay_index = option_int(option, "inPlayIndex", None)
    if inplay_area is not None and inplay_index is not None:
        target = card_at(obs, owner, inplay_area, inplay_index)
    elif area in (ACTIVE, BENCH) and index is not None:
        target = card_at(obs, owner, area, index)
        if actor is None:
            actor = target
    elif otype == O_ATTACK:
        target = pokemon_id(active(obs, 1 - me))

    return actor, target


def known_counts(obs: Any, owner: int) -> dict[int, int]:
    """Counts of all visible cards belonging to ``owner`` plus shared stadium."""
    counts: dict[int, int] = {}

    def add(cid):
        if cid is not None:
            counts[int(cid)] = counts.get(int(cid), 0) + 1

    for card in hand(obs, owner) or []:
        add(card_id(card))
    for card in discard(obs, owner):
        add(card_id(card))
    for p in field(obs, owner):
        add(pokemon_id(p))
        for cid in pre_evolution(p):
            add(cid)
        for cid in energy_cards(p):
            add(cid)
        for cid in tools(p):
            add(cid)
    for card in stadium(obs):
        if as_int(get(card, "playerIndex", -1), -1) in (-1, owner):
            add(card_id(card))
    if owner == you(obs):
        for card in looking(obs):
            add(card_id(card))
        for card in deck_cards(obs):
            add(card_id(card))
    return counts


def all_visible_ids(obs: Any) -> list[int]:
    """Publicly visible card ids, excluding our own hand and hidden zones."""
    me = you(obs)
    out: list[int] = []
    for owner in (0, 1):
        st = player(obs, owner)
        if st is None:
            continue
        for p in field(obs, owner):
            cid = pokemon_id(p)
            if cid is not None:
                out.append(cid)
            out.extend(pre_evolution(p))
            out.extend(energy_cards(p))
            out.extend(tools(p))
        if owner != me:
            out.extend(discard_ids(obs, owner))
    for card in stadium(obs):
        cid = card_id(card)
        if cid is not None:
            out.append(cid)
    return out
