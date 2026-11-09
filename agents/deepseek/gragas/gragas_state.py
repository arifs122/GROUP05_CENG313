"""Observation helpers that work with both the raw dict and cg dataclasses."""

from __future__ import annotations

from typing import Any

AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6
AREA_STADIUM = 7
AREA_ENERGY = 8
AREA_TOOL = 9
AREA_PRE_EVOLUTION = 10
AREA_PLAYER = 11
AREA_LOOKING = 12


def get(obj: Any, key: str, default=None):
    """Read a field from a dict or a dataclass-like object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def current(obs: Any) -> Any:
    return get(obs, "current")


def select(obs: Any) -> Any:
    return get(obs, "select")


def current_player(obs: Any) -> int:
    return int(get(current(obs), "yourIndex", 0) or 0)


def players(obs: Any) -> list:
    return list(get(current(obs), "players", []) or [])


def player_state(obs: Any, player: int) -> Any:
    ps = players(obs)
    return ps[player] if 0 <= player < len(ps) else None


def card_id(card: Any) -> int | None:
    if card is None:
        return None
    cid = get(card, "id")
    try:
        return int(cid) if cid is not None else None
    except (TypeError, ValueError):
        return None


def zone(obs: Any, player: int, area: int) -> list:
    """Visible cards in a zone; prize/deck may contain None entries."""
    state = player_state(obs, player)
    if state is None:
        return []
    area = int(area)
    if area == AREA_HAND:
        return list(get(state, "hand", []) or [])
    if area == AREA_DISCARD:
        return list(get(state, "discard", []) or [])
    if area == AREA_ACTIVE:
        return list(get(state, "active", []) or [])
    if area == AREA_BENCH:
        return list(get(state, "bench", []) or [])
    if area == AREA_PRIZE:
        return list(get(state, "prize", []) or [])
    if area == AREA_DECK:
        return list(get(select(obs), "deck", []) or [])
    if area == AREA_LOOKING:
        return list(get(current(obs), "looking", []) or [])
    if area == AREA_STADIUM:
        return list(get(current(obs), "stadium", []) or [])
    return []


def card_at(obs: Any, player: int, area: int | None, index: int | None) -> int | None:
    if area is None or index is None:
        return None
    try:
        cards = zone(obs, player, int(area))
        idx = int(index)
    except (TypeError, ValueError):
        return None
    if 0 <= idx < len(cards):
        return card_id(cards[idx])
    return None


def pokemon_at(obs: Any, player: int, area: int | None, index: int | None) -> Any:
    """Return the Pokémon object at (area, index), not its card id."""
    if area is None or index is None:
        return None
    try:
        idx = int(index)
    except (TypeError, ValueError):
        return None
    if int(area) == AREA_ACTIVE:
        arr = zone(obs, player, AREA_ACTIVE)
        return arr[idx] if 0 <= idx < len(arr) else None
    if int(area) == AREA_BENCH:
        arr = zone(obs, player, AREA_BENCH)
        return arr[idx] if 0 <= idx < len(arr) else None
    return None


def active(obs: Any, player: int) -> Any:
    arr = zone(obs, player, AREA_ACTIVE)
    return arr[0] if arr else None


def active_id(obs: Any, player: int) -> int | None:
    return card_id(active(obs, player))


def bench(obs: Any, player: int) -> list:
    return zone(obs, player, AREA_BENCH)


def bench_ids(obs: Any, player: int) -> list[int]:
    out = []
    for p in bench(obs, player):
        cid = card_id(p)
        if cid is not None:
            out.append(cid)
    return out


def all_field(obs: Any, player: int) -> list:
    out = []
    a = active(obs, player)
    if a is not None:
        out.append(a)
    out.extend(bench(obs, player))
    return out


def hand(obs: Any, player: int) -> list:
    return zone(obs, player, AREA_HAND)


def hand_count(obs: Any, player: int) -> int:
    state = player_state(obs, player)
    return int(get(state, "handCount", 0) or 0)


def discard(obs: Any, player: int) -> list:
    return zone(obs, player, AREA_DISCARD)


def discard_ids(obs: Any, player: int) -> list[int]:
    return [cid for cid in (card_id(c) for c in discard(obs, player)) if cid is not None]


def prize(obs: Any, player: int) -> list:
    return zone(obs, player, AREA_PRIZE)


def prize_count(obs: Any, player: int) -> int:
    return len(prize(obs, player))


def deck_count(obs: Any, player: int) -> int:
    state = player_state(obs, player)
    return int(get(state, "deckCount", 0) or 0)


def stadium(obs: Any) -> list:
    return list(get(current(obs), "stadium", []) or [])


def stadium_ids(obs: Any) -> list[int]:
    return [cid for cid in (card_id(c) for c in stadium(obs)) if cid is not None]


def energies(pokemon: Any) -> list[int]:
    return [int(e) for e in (get(pokemon, "energies", []) or [])]


def energy_cards(pokemon: Any) -> list[int]:
    out = []
    for c in get(pokemon, "energyCards", []) or []:
        cid = card_id(c)
        if cid is not None:
            out.append(cid)
    return out


def tools(pokemon: Any) -> list[int]:
    out = []
    for c in get(pokemon, "tools", []) or []:
        cid = card_id(c)
        if cid is not None:
            out.append(cid)
    return out


def pre_evolution(pokemon: Any) -> list[int]:
    out = []
    for c in get(pokemon, "preEvolution", []) or []:
        cid = card_id(c)
        if cid is not None:
            out.append(cid)
    return out


def hp(pokemon: Any) -> int:
    return int(get(pokemon, "hp", 0) or 0)


def max_hp(pokemon: Any) -> int:
    return int(get(pokemon, "maxHp", 0) or 0)


def is_damaged(pokemon: Any) -> bool:
    return max_hp(pokemon) > 0 and hp(pokemon) < max_hp(pokemon)


def select_type(obs: Any) -> int:
    try:
        return int(get(select(obs), "type", -1))
    except (TypeError, ValueError):
        return -1


def select_context(obs: Any) -> int:
    try:
        return int(get(select(obs), "context", -1))
    except (TypeError, ValueError):
        return -1


def min_count(obs: Any) -> int:
    return max(0, int(get(select(obs), "minCount", 0) or 0))


def max_count(obs: Any) -> int:
    return max(0, int(get(select(obs), "maxCount", 0) or 0))


def option_type(option: Any) -> int:
    try:
        return int(get(option, "type", -1))
    except (TypeError, ValueError):
        return -1


def option_int(option: Any, key: str, default=None):
    value = get(option, key)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def option_cards(obs: Any, option: Any) -> tuple[int | None, int | None]:
    """Return ``(actor_card_id, target_card_id)`` for an option.

    The engine uses two different shapes:
    * MAIN PLAY/ATTACH/EVOLVE: ``area/index`` = source card, ``inPlayArea`` /
      ``inPlayIndex`` = target Pokémon.
    * CARD selections (damage counters, switch, ...): ``area``, ``index`` and
      ``playerIndex`` identify the card directly; there is no ``inPlayArea``.
    """
    otype = option_type(option)
    me = current_player(obs)
    player_index = option_int(option, "playerIndex", None)

    actor: int | None = None
    target: int | None = None

    area = option_int(option, "area", None)
    index = option_int(option, "index", None)

    if otype in (4, 5, 6):  # attached tool / energy card / energy unit
        owner = player_index if player_index is not None else me
        pokemon = pokemon_at(obs, owner, area, index)
        if pokemon is not None:
            if otype == 4:
                arr = list(get(pokemon, "tools", []) or [])
                idx = option_int(option, "toolIndex", None)
            else:
                arr = list(get(pokemon, "energyCards", []) or [])
                idx = option_int(option, "energyIndex", None)
            if idx is None:
                idx = 0
            if 0 <= int(idx) < len(arr):
                actor = card_id(arr[int(idx)])
        if actor is None and otype == 6:
            # Some effects expose only the energy type, not the card.
            actor = None
    elif otype == 7:  # PLAY from hand
        actor = card_at(obs, me, AREA_HAND, index)
    elif otype in (8, 9, 10, 3, 11, 15, 16):
        owner = player_index if player_index is not None else me
        actor = card_at(obs, owner, area, index)
    elif otype in (12, 13):  # RETREAT / ATTACK
        actor = active_id(obs, me)
    elif otype == 0:  # NUMBER / count option
        actor = None

    inplay_area = option_int(option, "inPlayArea", None)
    inplay_index = option_int(option, "inPlayIndex", None)
    if inplay_area is not None and inplay_index is not None:
        owner = player_index if player_index is not None else me
        target = card_at(obs, owner, inplay_area, inplay_index)
    elif area in (AREA_ACTIVE, AREA_BENCH) and index is not None:
        owner = player_index if player_index is not None else me
        target = card_at(obs, owner, area, index)
        if actor is None:
            actor = target
    elif otype == 13:  # attack targets the opponent's Active
        target = active_id(obs, 1 - me)

    return actor, target


def option_target_player(obs: Any, option: Any) -> int | None:
    """Which player owns the card this option points at, when known."""
    pi = option_int(option, "playerIndex", None)
    if pi is not None:
        return pi
    inplay_area = option_int(option, "inPlayArea", None)
    if inplay_area is not None:
        # MAIN attach/evolve target is our own field.
        return current_player(obs)
    area = option_int(option, "area", None)
    if area in (AREA_ACTIVE, AREA_BENCH):
        return current_player(obs)
    return None


def known_cards(obs: Any, player: int) -> list[int]:
    """All card IDs visible to this player in public/private zones."""
    out: list[int] = []

    def add_card(card):
        cid = card_id(card)
        if cid is not None:
            out.append(cid)

    for c in hand(obs, player):
        add_card(c)
    for c in discard(obs, player):
        add_card(c)
    for p in all_field(obs, player):
        add_card(p)
        for cid in pre_evolution(p):
            out.append(cid)
        for cid in energy_cards(p):
            out.append(cid)
        for cid in tools(p):
            out.append(cid)
    for c in stadium(obs):
        if int(get(c, "playerIndex", -1) or -1) in (-1, player):
            add_card(c)
    for c in zone(obs, player, AREA_LOOKING):
        add_card(c)
    return out


def result(obs: Any) -> int:
    try:
        return int(get(current(obs), "result", -1) or -1)
    except (TypeError, ValueError):
        return -1
