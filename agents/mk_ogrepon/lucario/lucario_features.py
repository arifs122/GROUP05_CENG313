"""Small public-state / option featurizer used by the numpy prior.

The vector is intentionally compact and deterministic: it is shared by the
PyTorch trainer and the numpy inference path, so parity is exact.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import mk_data as D
import mk_state as S

STATE_DIM = 112
OPTION_DIM = 192

# 23 unique cards in our list + unknown slot.
ROLE_IDS = [
    6, 235, 673, 674, 675, 676, 677, 678, 974, 1086, 1097, 1102,
    1121, 1123, 1141, 1142, 1152, 1159, 1182, 1192, 1227, 1252,
]
ROLE_INDEX = {cid: i + 1 for i, cid in enumerate(ROLE_IDS)}
ROLE_COUNT = len(ROLE_IDS) + 1

OPTION_TYPE_COUNT = 17
OPTION_CONTEXT_COUNT = 49
AREA_COUNT = 13
ATTACK_IDS = [
    D.ALLURE, D.GROUND_MELTER, D.MYRIAD_LEAF_SHOWER, D.SOB,
    D.TORRENTIAL_PUMP, D.MIND_BEND, D.CRUEL_ARROW, D.EON_BLADE,
    D.POISON_CHAIN, D.FULL_MOON_RONDO, D.RAPID_FIRE_COMBO, D.TUCK_TAIL,
]
ATTACK_INDEX = {aid: i + 1 for i, aid in enumerate(ATTACK_IDS)}
ATTACK_COUNT = len(ATTACK_IDS) + 1

ARCHETYPES = [
    "Dragapult", "Alakazam", "Mega Kangaskhan", "Mega Lucario",
    "Hydrapple", "Archaludon", "Banette", "Festival Lead",
    "Mega Excadrill", "Mega Abomasnow", "Ceruledge", "Mega Starmie",
    "Mega Froslass", "Marnie's Grimmsnarl", "Cynthia's Garchomp",
    "Teal Mask Ogerpon", "N's Zorua", "Slowking", "Crustle", "Unknown",
]
ARCH_INDEX = {name: i for i, name in enumerate(ARCHETYPES)}

OPP_KEYS = [
    121, 120, 119, 743, 742, 741, 756, 678, 677, 150, 149, 190,
    1375, 1374, 1245, 90, 1406, 547, 723, 320, 319, 1031, 861, 648, 381,
]


def _add_scaled(out: list[float], value: float, scale: float) -> None:
    try:
        out.append(float(value) / scale if scale else 0.0)
    except (TypeError, ValueError):
        out.append(0.0)


def _onehot(out: list[float], value: int | None, count: int) -> None:
    idx = int(value) if value is not None else -1
    for i in range(count):
        out.append(1.0 if i == idx else 0.0)


def _card_role(cid: int | None) -> int:
    return ROLE_INDEX.get(int(cid), 0) if cid is not None else 0


def _counts_for(obs: dict, owner: int) -> Counter[int]:
    c: Counter[int] = Counter()
    for card in S.hand(obs, owner) or []:
        cid = S.card_id(card)
        if cid is not None:
            c[cid] += 1
    for p in S.field(obs, owner):
        cid = S.pokemon_id(p)
        if cid is not None:
            c[cid] += 1
        for x in S.pre_evolution(p):
            c[x] += 1
        for x in S.energy_cards(p):
            c[x] += 1
        for x in S.tools(p):
            c[x] += 1
    for card in S.discard(obs, owner):
        cid = S.card_id(card)
        if cid is not None:
            c[cid] += 1
    return c


def _opp_visible_counts(obs: dict) -> Counter[int]:
    opp = 1 - S.you(obs)
    c: Counter[int] = Counter()
    for p in S.field(obs, opp):
        cid = S.pokemon_id(p)
        if cid is not None:
            c[cid] += 1
        for x in S.pre_evolution(p):
            c[x] += 1
        for x in S.energy_cards(p):
            c[x] += 1
    for cid in S.discard_ids(obs, opp):
        c[cid] += 1
    for card in S.stadium(obs):
        if S.as_int(S.get(card, "playerIndex", -1), -1) == opp:
            cid = S.card_id(card)
            if cid is not None:
                c[cid] += 1
    return c


def _archetype(obs: dict) -> int:
    counts = _opp_visible_counts(obs)
    best = len(ARCHETYPES) - 1
    best_score = 0
    for name, keys in D.ARCHETYPE_KEYS:
        if not keys:
            continue
        score = sum(min(2, counts[k]) for k in keys)
        if score > best_score:
            best_score = score
            best = ARCH_INDEX.get(name, best)
    return best


def _max_potential(obs: dict, pokemon: Any) -> int:
    if pokemon is None:
        return 0
    cid = S.pokemon_id(pokemon)
    if cid is None:
        return 0
    have = S.energy_count(pokemon)
    defender = S.active(obs, 1 - S.you(obs))
    best = 0
    for aid in D.attacks_of(cid):
        a = D.attack(aid)
        if a is None:
            continue
        if len(getattr(a, "energies", []) or []) > have:
            continue
        dmg = int(getattr(a, "damage", 0) or 0)
        if aid == D.CRUEL_ARROW:
            dmg = 100
        best = max(best, dmg)
    return best


def state_features(obs: dict) -> list[float]:
    me = S.you(obs)
    opp = 1 - me
    my = S.my_state(obs)
    ps = S.players(obs)
    out: list[float] = []

    _add_scaled(out, S.turn(obs), 20.0)
    _add_scaled(out, S.as_int(S.get(S.current(obs), "turnActionCount", 0), 0) or 0, 10.0)
    out.append(float(me))
    out.append(1.0 if S.as_int(S.get(S.current(obs), "firstPlayer", -1), -1) == me else 0.0)
    out.append(1.0 if S.supporter_played(obs) else 0.0)
    out.append(1.0 if S.stadium_played(obs) else 0.0)
    out.append(1.0 if S.energy_attached(obs) else 0.0)
    out.append(1.0 if S.retreated(obs) else 0.0)

    _add_scaled(out, S.prize_count(obs, me), 6.0)
    _add_scaled(out, S.prize_count(obs, opp), 6.0)
    _add_scaled(out, S.prize_count(obs, opp) - S.prize_count(obs, me), 6.0)
    _add_scaled(out, S.deck_count(obs, me), 60.0)
    _add_scaled(out, S.deck_count(obs, opp), 60.0)
    _add_scaled(out, S.hand_count(obs, me), 10.0)
    _add_scaled(out, S.hand_count(obs, opp), 10.0)
    _add_scaled(out, len(S.bench(obs, me)), 8.0)
    _add_scaled(out, len(S.bench(obs, opp)), 8.0)
    _add_scaled(out, S.as_int(S.get(my, "benchMax", 5), 5) or 5, 8.0)

    a = S.active(obs, me)
    b = S.active(obs, opp)
    for p in (a, b):
        if p is None:
            out.extend([0.0, 0.0, 0.0, 0.0])
        else:
            _add_scaled(out, S.hp(p) / max(1, S.max_hp(p)), 1.0)
            _add_scaled(out, S.energy_count(p), 5.0)
            _add_scaled(out, D.prize_value(S.pokemon_id(p)), 3.0)
            out.append(1.0 if S.damaged(p) else 0.0)

    _add_scaled(out, sum(S.energy_count(p) for p in S.field(obs, me)), 12.0)
    _add_scaled(out, sum(S.energy_count(p) for p in S.field(obs, opp)), 12.0)
    _add_scaled(out, sum(S.hp(p) for p in S.field(obs, me)), 1200.0)
    _add_scaled(out, sum(S.hp(p) for p in S.field(obs, opp)), 1200.0)
    _add_scaled(out, sum(max(0, (S.max_hp(p) - S.hp(p)) // 10) for p in S.field(obs, me)), 12.0)
    _add_scaled(out, sum(max(0, (S.max_hp(p) - S.hp(p)) // 10) for p in S.field(obs, opp)), 12.0)

    _add_scaled(out, _max_potential(obs, a), 350.0)
    _add_scaled(out, _max_potential(obs, b), 350.0)
    our_deck = S.deck_count(obs, me)
    prizes = S.prize_count(obs, me)
    out.append(1.0 if our_deck <= max(5, prizes * 2 + 2) else 0.0)
    out.append(1.0 if our_deck > max(2, prizes * 2) + 4 else 0.0)

    # Stadium one-hot: none / Area Zero / other.
    sids = S.stadium_ids(obs)
    out.append(1.0 if not sids else 0.0)
    out.append(1.0 if D.AREA_ZERO in sids else 0.0)
    out.append(1.0 if sids and D.AREA_ZERO not in sids else 0.0)

    _onehot(out, _archetype(obs), len(ARCHETYPES))

    mine = _counts_for(obs, me)
    for cid in ROLE_IDS:
        _add_scaled(out, min(4, mine.get(cid, 0)), 4.0)

    oppc = _opp_visible_counts(obs)
    for cid in OPP_KEYS:
        _add_scaled(out, min(4, oppc.get(cid, 0)), 4.0)

    if len(out) < STATE_DIM:
        out.extend([0.0] * (STATE_DIM - len(out)))
    return out[:STATE_DIM]


def _option_base_damage(cid: int | None, attack_id: int | None) -> int:
    if attack_id is None:
        return 0
    a = D.attack(attack_id)
    dmg = int(getattr(a, "damage", 0) or 0) if a is not None else 0
    if attack_id == D.CRUEL_ARROW:
        dmg = 100
    return dmg


def option_features(obs: dict, option: Any) -> list[float]:
    me = S.you(obs)
    owner = S.option_owner(obs, option)
    actor, target = S.option_cards(obs, option)
    out: list[float] = []

    _onehot(out, S.option_type(option), OPTION_TYPE_COUNT)
    ctx = S.select_context(obs)
    _onehot(out, ctx if 0 <= ctx < OPTION_CONTEXT_COUNT else -1, OPTION_CONTEXT_COUNT)
    _onehot(out, S.option_int(option, "area", -1) if S.option_int(option, "area", None) is not None else -1, AREA_COUNT)
    _onehot(out, S.option_int(option, "inPlayArea", -1) if S.option_int(option, "inPlayArea", None) is not None else -1, AREA_COUNT)

    _add_scaled(out, S.option_int(option, "index", 0) or 0, 10.0)
    _add_scaled(out, S.option_int(option, "inPlayIndex", 0) or 0, 8.0)
    out.append(float(owner == me))
    _add_scaled(out, S.option_int(option, "number", 0) or 0, 10.0)

    _onehot(out, ATTACK_INDEX.get(S.option_int(option, "attackId", None), 0), ATTACK_COUNT)
    _onehot(out, _card_role(actor), ROLE_COUNT)
    _onehot(out, _card_role(target), ROLE_COUNT)
    _add_scaled(out, _option_base_damage(actor, S.option_int(option, "attackId", None)), 350.0)
    _add_scaled(out, D.role_value(actor), 150.0)
    _add_scaled(out, D.role_value(target), 150.0)

    target_p = None
    area = S.option_int(option, "inPlayArea", None)
    idx = S.option_int(option, "inPlayIndex", None)
    if area is not None and idx is not None:
        target_p = S.pokemon_at(obs, owner, area, idx)
    elif target is not None:
        target_p = None
    if target_p is not None:
        _add_scaled(out, S.hp(target_p), 350.0)
        _add_scaled(out, S.energy_count(target_p), 5.0)
        _add_scaled(out, D.prize_value(S.pokemon_id(target_p)), 3.0)
        out.append(1.0 if S.damaged(target_p) else 0.0)
    else:
        out.extend([0.0, 0.0, 0.0, 0.0])

    attack_id = S.option_int(option, "attackId", None)
    a = D.attack(attack_id) if attack_id is not None else None
    _add_scaled(out, len(getattr(a, "energies", []) or []) if a is not None else 0, 5.0)

    if len(out) < OPTION_DIM:
        out.extend([0.0] * (OPTION_DIM - len(out)))
    return out[:OPTION_DIM]


def decision_features(obs: dict) -> tuple[list[float], list[list[float]]]:
    state = state_features(obs)
    opts = [option_features(obs, o) for o in S.options(obs)]
    return state, opts
