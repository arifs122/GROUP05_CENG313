"""Public-state and legal-option features for the Top2 from-scratch agent.

The same module is used by the offline dataset builder, the PyTorch trainers
and the NumPy runtime.  It only reads information that is present in the
agent's own observation; replay-only fields (visualize, search_begin_input,
opponent hand/prize identities, logs) are never touched.
"""

from __future__ import annotations

import numpy as np

import gragas_data as D
import gragas_state as S

MAX_OPTIONS = 40
CARD_ID_LIMIT = 2048
OPTION_TYPE_COUNT = 17
ENERGY_TYPE_COUNT = 12

SETUP_IDS = {D.DREEPY, D.DRAKLOAK, D.DRAGAPULT, D.BUDEW,
             D.POFFIN, D.POKE_PAD, D.ULTRA_BALL}
TECH_IDS = {D.MUNKIDORI, D.BOSS, D.JUDGE, D.BUDEW, D.MEOWTH, D.FEZANDIPITI, D.CHI_YU}
TARGET_CONTEXTS = {3, 4, 5, 13, 14, 16, 22, 25, 40}

KEY_IDS = (D.DREEPY, D.DRAKLOAK, D.DRAGAPULT, D.MUNKIDORI,
           D.FIRE, D.PSYCHIC, D.DARK)

# Role slots used by the 78-dim state vector.  Each entry is
# ((card ids counted together), denominator).  The top2 profile reproduces the
# original Dragapult feature values exactly; the grass profile is for the
# #2 ladder team's Ogerpon/Kangaskhan shell.
OGERPON_IDS = (96, 108)       # Teal Mask / Wellspring Mask Ogerpon ex
KANGASKHAN_ID = 756          # Mega Kangaskhan ex
LATIAS_ID = 184              # Latias ex
CLEFAIRY_ID = 272            # Lillie's Clefairy ex
GRASS_ENERGY_ID = 1
PRISM_ENERGY_ID = 16
BUG_CATCHING_ID = 1094
ULTRA_BALL_ID = 1121

PROFILES = {
    "top2": {
        "field": [
            ((D.DREEPY,), 4.0),
            ((D.DRAKLOAK,), 4.0),
            ((D.DRAGAPULT,), 3.0),
            ((D.MUNKIDORI,), 2.0),
            ((D.BUDEW,), 2.0),
        ],
        "remaining": [
            ((D.DREEPY,), 4.0),
            ((D.DRAKLOAK,), 4.0),
            ((D.DRAGAPULT,), 3.0),
            ((D.MUNKIDORI,), 2.0),
            ((D.FIRE,), 4.0),
            ((D.PSYCHIC,), 4.0),
            ((D.DARK,), 2.0),
        ],
    },
    "grass": {
        "field": [
            (OGERPON_IDS, 5.0),
            ((KANGASKHAN_ID,), 2.0),
            ((LATIAS_ID,), 2.0),
            ((D.MUNKIDORI,), 1.0),
            ((CLEFAIRY_ID,), 2.0),
        ],
        "remaining": [
            (OGERPON_IDS, 5.0),
            ((KANGASKHAN_ID,), 2.0),
            ((GRASS_ENERGY_ID,), 11.0),
            ((PRISM_ENERGY_ID,), 4.0),
            ((BUG_CATCHING_ID,), 3.0),
            ((ULTRA_BALL_ID,), 4.0),
            ((D.BOSS,), 3.0),
        ],
    },
}

STATE_FEATURE_NAMES = (
    "turn_norm", "turn_action_norm", "prize_diff", "my_prizes", "opp_prizes",
    "my_prize_progress", "opp_prize_progress", "my_deck_ratio", "opp_deck_ratio",
    "my_hand_norm", "opp_hand_norm", "my_bench_ratio", "opp_bench_ratio",
    "my_active_hp", "my_active_maxhp", "my_active_hp_ratio",
    "opp_active_hp", "opp_active_maxhp", "opp_active_hp_ratio",
    "my_active_id", "opp_active_id", "my_active_energy", "opp_active_energy",
    "my_active_damage", "opp_active_damage",
    "my_bench_hp", "opp_bench_hp", "my_bench_damaged", "opp_bench_damaged",
    "my_discard_norm", "opp_discard_norm", "my_discard_energy", "opp_discard_energy",
    "my_discard_supporters", "opp_discard_supporters",
    "stadium_present", "stadium_id",
    "my_active_can_attack", "opp_active_can_attack",
    "my_best_damage", "opp_best_damage",
    "my_active_threatened", "opp_active_threatened",
    "my_dreepy", "my_drakloak", "my_dragapult", "my_munkidori", "my_budew",
    "my_dreepy_left", "my_drakloak_left", "my_dragapult_left",
    "my_munkidori_left", "my_fire_left", "my_psychic_left", "my_dark_left",
    "my_hand_pokemon", "my_hand_supporters", "my_hand_energy",
    "my_discard_boss", "opp_discard_boss",
    "my_bench_energy", "opp_bench_energy",
    *(f"deck_bucket_{i}" for i in range(16)),
)

OPTION_FEATURE_NAMES = (
    *(f"type_{i}" for i in range(OPTION_TYPE_COUNT)),
    "has_area", "has_inplay_area", "target_active", "target_bench",
    "index_norm", "area_norm", "inplay_area_norm", "number_norm",
    "attack_id_norm", "attack_damage", "attack_energy_cost",
    *(f"attack_energy_type_{i}" for i in range(ENERGY_TYPE_COUNT)),
    "attack_lethal", "context_norm", "is_target_context",
    "is_play", "is_attach", "is_evolve", "is_ability", "is_attack",
    "is_retreat", "is_end",
    "actor_setup", "actor_tech", "actor_prize", "target_prize",
    "target_hp_ratio", "target_hp_norm", "target_maxhp_norm",
    "target_energy", "target_fire", "target_psychic", "target_dark",
    "target_is_own", "target_is_opponent",
    "actor_energy_fire", "actor_energy_psychic", "actor_energy_dark",
    "attack_kos_target", "target_damage_norm",
)

STATE_FEATURE_SIZE = len(STATE_FEATURE_NAMES)
OPTION_FEATURE_SIZE = len(OPTION_FEATURE_NAMES)


def _id_norm(card_id: int | None) -> float:
    return float(card_id or 0) / 2000.0


def _count_ids(ids: list[int], target: int | None) -> int:
    if target is None:
        return 0
    return sum(1 for x in ids if int(x) == int(target))


def _count_group(ids: list[int], group: tuple[int, ...]) -> int:
    group_set = set(int(x) for x in group)
    return sum(1 for x in ids if int(x) in group_set)


def _field_ids(obs: dict, player: int) -> list[int]:
    out = []
    for p in S.all_field(obs, player):
        cid = S.card_id(p)
        if cid is not None:
            out.append(cid)
    return out


def _can_attack(pokemon) -> int:
    if pokemon is None:
        return 0
    cid = S.card_id(pokemon)
    c = D.card(cid)
    if c is None:
        return 0
    energy_count = len(S.energies(pokemon))
    for aid in getattr(c, "attacks", []) or []:
        atk = D.attack(aid)
        if atk is not None and len(getattr(atk, "energies", []) or []) <= energy_count:
            return 1
    return 0


def _best_damage(pokemon, defender) -> int:
    if pokemon is None:
        return 0
    cid = S.card_id(pokemon)
    c = D.card(cid)
    if c is None:
        return 0
    energy_count = len(S.energies(pokemon))
    best = 0
    for aid in getattr(c, "attacks", []) or []:
        atk = D.attack(aid)
        if atk is None:
            continue
        if len(getattr(atk, "energies", []) or []) <= energy_count:
            damage = int(getattr(atk, "damage", 0) or 0)
            if defender is not None:
                dc = D.card(S.card_id(defender))
                ac = D.card(cid)
                if dc is not None and ac is not None:
                    if getattr(dc, "weakness", None) == getattr(ac, "energyType", None):
                        damage *= 2
                    if getattr(dc, "resistance", None) == getattr(ac, "energyType", None):
                        damage = max(0, damage - 30)
            best = max(best, damage)
    return best


def _card_type(card_id: int | None) -> int:
    c = D.card(card_id)
    return int(getattr(c, "cardType", -1)) if c is not None else -1


def _is_energy(card_id: int | None) -> bool:
    return _card_type(card_id) in (5, 6)


def _is_supporter(card_id: int | None) -> bool:
    return _card_type(card_id) == 3


def _is_pokemon(card_id: int | None) -> bool:
    return _card_type(card_id) == 0


def state_vector(observation: dict, deck: list[int] | None = None,
                 profile: str = "top2") -> np.ndarray:
    prof = PROFILES.get(profile, PROFILES["top2"])
    me = S.current_player(observation)
    opp = 1 - me
    cur = S.current(observation) or {}
    my = S.player_state(observation, me) or {}
    op = S.player_state(observation, opp) or {}
    my_active = S.active(observation, me)
    op_active = S.active(observation, opp)
    my_prizes = S.prize_count(observation, me)
    op_prizes = S.prize_count(observation, opp)
    my_deck = S.deck_count(observation, me)
    op_deck = S.deck_count(observation, opp)

    my_bench = S.bench(observation, me)
    op_bench = S.bench(observation, opp)
    my_field_ids = _field_ids(observation, me)
    op_discard_ids = S.discard_ids(observation, opp)

    my_discard_ids = S.discard_ids(observation, me)
    my_discard_energy = sum(1 for cid in my_discard_ids if _is_energy(cid))
    op_discard_energy = sum(1 for cid in op_discard_ids if _is_energy(cid))
    my_discard_supporters = sum(1 for cid in my_discard_ids if _is_supporter(cid))
    op_discard_supporters = sum(1 for cid in op_discard_ids if _is_supporter(cid))

    my_best = _best_damage(my_active, op_active)
    op_best = _best_damage(op_active, my_active)
    my_hp = S.hp(my_active)
    op_hp = S.hp(op_active)

    # Remaining copies in our known deck list (public to us, hidden prize
    # identities are deliberately not modelled as known).
    if deck:
        pool = list(deck)
        for cid in S.known_cards(observation, me):
            try:
                pool.remove(int(cid))
            except ValueError:
                pass
        remaining_counts = [
            _count_group(pool, group) / denom
            for group, denom in prof["remaining"]
        ]
    else:
        remaining_counts = [0.0 for _ in prof["remaining"]]

    field_counts = [
        _count_group(my_field_ids, group) / denom
        for group, denom in prof["field"]
    ]

    hand_ids = [S.card_id(c) for c in S.hand(observation, me)]
    hand_pokemon = sum(1 for cid in hand_ids if _is_pokemon(cid))
    hand_supporters = sum(1 for cid in hand_ids if _is_supporter(cid))
    hand_energy = sum(1 for cid in hand_ids if _is_energy(cid))

    stadium = S.stadium(observation)
    stadium_id = S.card_id(stadium[0]) if stadium else 0

    base = [
        float(S.get(cur, "turn", 0) or 0) / 20.0,
        float(S.get(cur, "turnActionCount", 0) or 0) / 10.0,
        (op_prizes - my_prizes) / 6.0,
        my_prizes / 6.0,
        op_prizes / 6.0,
        (6.0 - my_prizes) / 5.0,
        (6.0 - op_prizes) / 5.0,
        my_deck / 60.0,
        op_deck / 60.0,
        S.hand_count(observation, me) / 10.0,
        S.hand_count(observation, opp) / 10.0,
        len(my_bench) / 5.0,
        len(op_bench) / 5.0,
        my_hp / 350.0,
        S.max_hp(my_active) / 350.0,
        (my_hp / S.max_hp(my_active)) if S.max_hp(my_active) > 0 else 0.0,
        op_hp / 350.0,
        S.max_hp(op_active) / 350.0,
        (op_hp / S.max_hp(op_active)) if S.max_hp(op_active) > 0 else 0.0,
        _id_norm(S.card_id(my_active)),
        _id_norm(S.card_id(op_active)),
        len(S.energies(my_active)) / 5.0,
        len(S.energies(op_active)) / 5.0,
        max(0, S.max_hp(my_active) - my_hp) / 100.0,
        max(0, S.max_hp(op_active) - op_hp) / 100.0,
        sum(S.hp(p) for p in my_bench) / 1000.0,
        sum(S.hp(p) for p in op_bench) / 1000.0,
        sum(1 for p in my_bench if S.is_damaged(p)) / 5.0,
        sum(1 for p in op_bench if S.is_damaged(p)) / 5.0,
        len(my_discard_ids) / 20.0,
        len(op_discard_ids) / 20.0,
        my_discard_energy / 10.0,
        op_discard_energy / 10.0,
        my_discard_supporters / 10.0,
        op_discard_supporters / 10.0,
        1.0 if stadium else 0.0,
        _id_norm(stadium_id),
        float(_can_attack(my_active)),
        float(_can_attack(op_active)),
        my_best / 300.0,
        op_best / 300.0,
        1.0 if my_active is not None and op_best >= my_hp > 0 else 0.0,
        1.0 if op_active is not None and my_best >= op_hp > 0 else 0.0,
        *field_counts,
        *remaining_counts,
        hand_pokemon / 10.0,
        hand_supporters / 5.0,
        hand_energy / 10.0,
        1.0 if D.BOSS in my_discard_ids else 0.0,
        1.0 if D.BOSS in op_discard_ids else 0.0,
        sum(len(S.energies(p)) for p in my_bench) / 10.0,
        sum(len(S.energies(p)) for p in op_bench) / 10.0,
    ]

    buckets = np.zeros(16, dtype=np.float32)
    if deck:
        for cid in deck:
            buckets[int(cid) % 16] += 1.0
        buckets /= max(1.0, float(len(deck)))
    return np.concatenate([np.asarray(base, dtype=np.float32), buckets])


def _attack_info(observation: dict, option: dict, me: int):
    attack_id = S.option_int(option, "attackId", None)
    atk = D.attack(attack_id)
    actor = S.active(observation, me)
    defender = S.active(observation, 1 - me)
    damage = int(getattr(atk, "damage", 0) or 0) if atk is not None else 0
    energies = [int(e) for e in (getattr(atk, "energies", []) or [])] if atk is not None else []
    if atk is not None and actor is not None and defender is not None:
        ac = D.card(S.card_id(actor))
        dc = D.card(S.card_id(defender))
        if ac is not None and dc is not None:
            if getattr(dc, "weakness", None) == getattr(ac, "energyType", None):
                damage *= 2
            if getattr(dc, "resistance", None) == getattr(ac, "energyType", None):
                damage = max(0, damage - 30)
    return attack_id, damage, energies


def _option_features(observation: dict, option: dict) -> np.ndarray:
    me = S.current_player(observation)
    opp = 1 - me
    otype = S.option_type(option)
    ctx = S.select_context(observation)
    actor, target = S.option_cards(observation, option)
    owner = S.option_target_player(observation, option)
    target_pokemon = None
    area = S.option_int(option, "area", None)
    index = S.option_int(option, "index", None)
    inplay_area = S.option_int(option, "inPlayArea", None)
    inplay_index = S.option_int(option, "inPlayIndex", None)
    if inplay_area is not None and inplay_index is not None:
        target_pokemon = S.pokemon_at(observation, owner if owner is not None else me,
                                      inplay_area, inplay_index)
    elif area in (S.AREA_ACTIVE, S.AREA_BENCH):
        target_pokemon = S.pokemon_at(observation, owner if owner is not None else me, area, index)
    elif otype == 13:  # MAIN attack targets the opponent's Active
        target_pokemon = S.active(observation, opp)
        owner = opp

    attack_id, damage, energies = _attack_info(observation, option, me)
    target_hp = S.hp(target_pokemon)
    target_maxhp = S.max_hp(target_pokemon)
    target_energy = S.energies(target_pokemon)
    target_id = S.card_id(target_pokemon) if target_pokemon is not None else target

    vec = np.zeros(OPTION_FEATURE_SIZE, dtype=np.float32)
    if 0 <= otype < OPTION_TYPE_COUNT:
        vec[otype] = 1.0
    off = OPTION_TYPE_COUNT
    vec[off + 0] = 1.0 if area is not None else 0.0
    vec[off + 1] = 1.0 if inplay_area is not None else 0.0
    vec[off + 2] = 1.0 if inplay_area == S.AREA_ACTIVE else 0.0
    vec[off + 3] = 1.0 if inplay_area == S.AREA_BENCH else 0.0
    vec[off + 4] = float(index or 0) / 60.0
    vec[off + 5] = float(area or 0) / 12.0
    vec[off + 6] = float(inplay_area or 0) / 12.0
    vec[off + 7] = float(S.option_int(option, "number", 0) or 0) / 10.0
    vec[off + 8] = float(attack_id or 0) / 2000.0
    vec[off + 9] = damage / 300.0
    vec[off + 10] = len(energies) / 5.0
    eoff = off + 11
    for energy in energies:
        if 0 <= energy < ENERGY_TYPE_COUNT:
            vec[eoff + energy] += 1.0 / 5.0
    vec[off + 11 + ENERGY_TYPE_COUNT] = 1.0 if damage >= target_hp > 0 else 0.0
    base = off + 12 + ENERGY_TYPE_COUNT
    vec[base + 0] = float(ctx) / 48.0 if ctx >= 0 else 0.0
    vec[base + 1] = 1.0 if ctx in TARGET_CONTEXTS else 0.0
    vec[base + 2] = 1.0 if otype == 7 else 0.0
    vec[base + 3] = 1.0 if otype == 8 else 0.0
    vec[base + 4] = 1.0 if otype == 9 else 0.0
    vec[base + 5] = 1.0 if otype == 10 else 0.0
    vec[base + 6] = 1.0 if otype == 13 else 0.0
    vec[base + 7] = 1.0 if otype == 12 else 0.0
    vec[base + 8] = 1.0 if otype == 14 else 0.0
    vec[base + 9] = 1.0 if actor in SETUP_IDS else 0.0
    vec[base + 10] = 1.0 if actor in TECH_IDS else 0.0
    vec[base + 11] = D.prize_value(actor) / 3.0
    vec[base + 12] = D.prize_value(target_id) / 3.0
    vec[base + 13] = target_hp / target_maxhp if target_maxhp > 0 else 0.0
    vec[base + 14] = target_hp / 350.0
    vec[base + 15] = target_maxhp / 350.0
    vec[base + 16] = len(target_energy) / 5.0
    vec[base + 17] = 1.0 if D.FIRE in target_energy else 0.0
    vec[base + 18] = 1.0 if D.PSYCHIC in target_energy else 0.0
    vec[base + 19] = 1.0 if D.DARK in target_energy else 0.0
    vec[base + 20] = 1.0 if owner == me else 0.0
    vec[base + 21] = 1.0 if owner is not None and owner != me else 0.0
    actor_energy = D.energy_type(actor)
    vec[base + 22] = 1.0 if actor_energy == D.FIRE else 0.0
    vec[base + 23] = 1.0 if actor_energy == D.PSYCHIC else 0.0
    vec[base + 24] = 1.0 if actor_energy == D.DARK else 0.0
    vec[base + 25] = 1.0 if target is not None and damage >= target_hp > 0 else 0.0
    vec[base + 26] = max(0, target_maxhp - target_hp) / 100.0 if target_maxhp > 0 else 0.0
    return vec


def options_matrix(observation: dict) -> tuple[np.ndarray, np.ndarray]:
    select = S.select(observation) or {}
    raw = list(S.get(select, "option", []) or [])
    options = np.zeros((MAX_OPTIONS, OPTION_FEATURE_SIZE), dtype=np.float32)
    mask = np.zeros(MAX_OPTIONS, dtype=np.float32)
    for i, option in enumerate(raw[:MAX_OPTIONS]):
        options[i] = _option_features(observation, option)
        mask[i] = 1.0
    return options, mask


def card_ids_matrix(observation: dict) -> np.ndarray:
    select = S.select(observation) or {}
    raw = list(S.get(select, "option", []) or [])
    ids = np.zeros((MAX_OPTIONS, 2), dtype=np.int16)
    for i, option in enumerate(raw[:MAX_OPTIONS]):
        actor, target = S.option_cards(observation, option)
        ids[i, 0] = min(max(int(actor or 0), 0), CARD_ID_LIMIT - 1)
        ids[i, 1] = min(max(int(target or 0), 0), CARD_ID_LIMIT - 1)
    return ids
