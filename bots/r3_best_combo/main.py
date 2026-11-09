from __future__ import annotations

import math
import os
import random
import sys
import time
from collections import defaultdict

from cg.api import (
    AreaType, CardType, Log, LogType, Observation, SelectContext, SelectType,
    OptionType, Card, Pokemon, State, all_attack, all_card_data,
    to_observation_class,
)

try:
    from cg.api import search_begin, search_end, search_step
    _SEARCH_OK = True
except Exception:  # noqa: BLE001
    _SEARCH_OK = False

try:
    from belief_runtime import BeliefObserver
    _BELIEF_OK = True
except Exception:  # noqa: BLE001
    _BELIEF_OK = False

try:
    from profile_policy import infer_profile, PROFILE_WEIGHTS
    _PROFILE_OK = True
except Exception:  # noqa: BLE001
    _PROFILE_OK = False

# Observation-only logging.  The observer never changes the action returned by
# the policy; it writes compact JSON lines to stdout for Kaggle agent logs.
BELIEF_LOG = os.environ.get("PTCG_BELIEF_LOG", "1") != "0"
_BELIEF = None

"""
Dragapult ex — Phantom Dive spread + Crushing Hammer disruption.

PILOT: the official "rule-based agent — Dragapult ex deck" sample, which is a competent,
well-tuned pilot for THIS exact consensus top-tier list (it beat a hand-rolled policy 13-1
in cabt). We keep its logic verbatim and only wrap it in our robust scaffolding:
  - cwd-independent deck.csv loader (works under cabt + the Kaggle ladder),
  - a try/except guard so a single mis-handled SelectContext can never forfeit the game
    (the raw sample has NO error handling — any exception = a lost game),
  - lightweight diagnostics.
Source: docs/official/models/a-sample-rule-based-agent-dragapult-ex-deck.ipynb.
"""

# ── robust deck loader (cwd-independent) ───────────────────────────────────────
def _resolve_deck_path():
    cands = []
    if "__file__" in globals():
        cands.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv"))
    cands += ["deck.csv", "/kaggle_simulations/agent/deck.csv"]
    cands += [os.path.join(p, "deck.csv") for p in sys.path if p]
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("deck.csv not found")


my_deck: list[int] = []


def _load_deck() -> list[int]:
    """Load deck.csv on first use, never at import time."""
    global my_deck
    if not my_deck:
        with open(_resolve_deck_path(), "r") as _f:
            cards = [int(x) for x in _f.read().splitlines() if x.strip()][:60]
        if len(cards) != 60:
            raise ValueError(f"deck.csv must have 60 ids, got {len(cards)}")
        my_deck = cards
    return my_deck


def read_deck_csv() -> list[int]:
    """Deck for the initial selection (also used by the validation harness)."""
    return list(_load_deck())


card_table: dict[int, object] = {}
attack_table: dict[int, object] = {}


def _load_cards():
    """Populate card/attack metadata on first use, never at import time."""
    global card_table, attack_table
    if not card_table:
        card_table.update({c.cardId: c for c in all_card_data()})
        attack_table.update({a.attackId: a for a in all_attack()})
    return card_table

# Decklist ids
Dreepy = 119
Drakloak = 120
Dragapult_ex = 121
Fezandipiti_ex = 140
Latias_ex = 184
Budew = 235
Meowth_ex = 1071
Rare_Candy = 1079
Unfair_Stamp = 1080
Buddy_Buddy_Poffin = 1086
Night_Stretcher = 1097
Crushing_Hammer = 1120
Ultra_Ball = 1121
Poke_Pad = 1152
Lucky_Helmet = 1156
Boss_Orders = 1182
Crispin = 1198
Brock_Scouting = 1210
Lillie_Determination = 1227
Team_Rocket_Watchtower = 1256
Nighttime_Mine = 1266
Munkidori = 112
Chi_Yu = 31
Dunsparce = 65
Dudunsparce = 66
Risky_Ruins = 1260
Judge = 1213
Rosa_Encouragement = 1240
Basic_Dark_Energy = 7

# 26/26 top-team Dragapult replays (Oct 24 sample) chose first, and local
# A/B with this list is clearly better going first.
GO_FIRST = True

# Forward search: 1-ply lookahead over the rule policy's top candidates, then
# a greedy rollout of the rest of our turn using the same policy.
SEARCH_ENABLED = False
SEARCH_K = 4
SEARCH_TIME_BUDGET = float(os.environ.get("PTCG_SEARCH_BUDGET", "0.25"))
SEARCH_NODE_BUDGET = int(os.environ.get("PTCG_SEARCH_NODES", "300"))
SEARCH_ROLLOUT_MAX = 50
Basic_Fire_Energy = 2
Basic_Psychic_Energy = 5

# Coarse "how useful is this card in hand" weights.  Unlike hand_score(), these
# do not depend on the current board, so they can be used to compare a hand
# before and after a shuffle-and-draw effect without recursion.
_HAND_QUALITY = {
    Dreepy: 8, Drakloak: 6, Dragapult_ex: 12,
    Fezandipiti_ex: 5, Latias_ex: 4, Budew: 6, Meowth_ex: 5,
    Munkidori: 4, Dunsparce: 3, Dudunsparce: 5,
    Rare_Candy: 10, Unfair_Stamp: 7, Buddy_Buddy_Poffin: 6,
    Night_Stretcher: 4, Crushing_Hammer: 2, Ultra_Ball: 5,
    Poke_Pad: 5, Lucky_Helmet: 2, Boss_Orders: 8, Crispin: 6,
    Brock_Scouting: 5, Lillie_Determination: 6, Judge: 4,
    Rosa_Encouragement: 5, Team_Rocket_Watchtower: 3,
    Nighttime_Mine: 2, Risky_Ruins: 2,
    Basic_Dark_Energy: 2, Basic_Fire_Energy: 3, Basic_Psychic_Energy: 3,
}


def _hypergeom_at_least_one(unseen: int, copies: int, draws: int) -> float:
    """P(at least one of ``copies`` in ``unseen`` appears in ``draws``)."""
    if copies <= 0 or draws <= 0 or unseen <= 0:
        return 0.0
    copies = min(int(copies), int(unseen))
    draws = min(int(draws), int(unseen))
    if unseen - copies < draws:
        return 1.0
    miss = math.comb(unseen - copies, draws) / math.comb(unseen, draws)
    return max(0.0, min(1.0, 1.0 - miss))


def _hand_quality(cards) -> float:
    return float(sum(_HAND_QUALITY.get(card.id, 1) for card in cards if card is not None))


def _expected_draw_quality(card_counts, draws: int) -> float:
    total = sum(max(0, int(count)) for count in card_counts.values())
    if total <= 0:
        return 0.0
    average = sum(
        _HAND_QUALITY.get(card_id, 1) * max(0, int(count))
        for card_id, count in card_counts.items()
    ) / total
    return average * draws


UNNECESSARY = -10000000


class AttackPlan:
    attack: int = 0
    counter: list[int] = []


can_switch = False
can_attack = False
can_main_attack = False
can_energy_attach = False
use_support = 0
bench_attacker = False
pre_turn_log: list = []
current_turn_log: list = []

prize: list[int] = []
card_counts = defaultdict(int)
serial_set = set()
opp_seen_counts = defaultdict(int)
opp_seen_serials = set()
plan_a = AttackPlan()
plan_b = AttackPlan()

# ── diagnostics ────────────────────────────────────────────────────────────────
_DIAG = {"decisions": 0, "policy_ok": 0, "policy_fallback": 0,
         "obs_fallback": 0, "deck_returns": 0, "errors": {}}

# Option scores from the most recent rule-policy call, aligned with
# select.option.  The search uses them to choose candidate actions.
_LAST_SCORES: list[float] = []


def _diag_record_error(exc):
    k = type(exc).__name__ + ": " + str(exc)[:160]
    _DIAG["errors"][k] = _DIAG["errors"].get(k, 0) + 1


def diag_reset():
    _DIAG.update({"decisions": 0, "policy_ok": 0, "policy_fallback": 0,
                  "obs_fallback": 0, "deck_returns": 0, "errors": {}})


def diag_snapshot():
    s = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _DIAG.items()}
    s["fallback_rate"] = (s.get("policy_fallback", 0) + s.get("obs_fallback", 0)) / max(1, s["decisions"])
    return s


def _legal_fallback(select):
    try:
        n = len(select.option)
        return list(range(min(max(0, select.minCount), n)))
    except Exception:
        return []


def _legal_fallback_from_dict(obs_dict):
    try:
        sel = obs_dict.get("select") or {}
        return list(range(min(max(0, sel.get("minCount", 0)), len(sel.get("option") or []))))
    except Exception:
        return []


def _legal_fallback_select(select):
    """Minimum legal selection from either a dict or a typed select."""
    if isinstance(select, dict):
        n = len(select.get("option") or [])
        k = int(select.get("minCount", 0))
    else:
        n = len(getattr(select, "option", []) or [])
        k = int(getattr(select, "minCount", 0))
    return list(range(min(max(0, k), n)))


def _legalize(action, select):
    """Clamp an action to the legal shape for this selection.

    Accepts either the dict-shaped select from the simulator or the typed
    Observation.select, and never touches a deck return (select is None).
    """
    if select is None:
        return list(action or [])
    if isinstance(select, dict):
        options = select.get("option") or []
        min_count = int(select.get("minCount", 0))
        max_count = int(select.get("maxCount", 0))
    else:
        options = getattr(select, "option", None)
        if options is None:
            return list(action or [])
        min_count = int(getattr(select, "minCount", 0))
        max_count = int(getattr(select, "maxCount", 0))

    n = len(options)
    min_count = max(0, min(min_count, n))
    max_count = max(min_count, min(max_count, n))

    out: list[int] = []
    seen: set[int] = set()
    for index in action or []:
        if len(out) >= max_count:
            break
        if isinstance(index, int) and 0 <= index < n and index not in seen:
            out.append(index)
            seen.add(index)
    for index in range(n):
        if len(out) >= min_count:
            break
        if index not in seen:
            out.append(index)
            seen.add(index)
    return out


# ── sample helpers (verbatim) ──────────────────────────────────────────────────
def no_damage_dex(id: int) -> bool:
    """Defending Pokémon with innate immunity to Dragapult ex's attack."""
    return id == 158 or id == 207 or id == 330 or id == 345


def no_damage_counter(pokemon: Pokemon) -> bool:
    """Target that prevents Phantom Dive's bench damage counters (abilities/energy)."""
    if pokemon.id == 28 or pokemon.id == 199 or pokemon.id == 203 or pokemon.id == 207 or pokemon.id == 362 or pokemon.id == 1136:
        return True
    for card in pokemon.energyCards:
        if card.id == 11 or card.id == 20:  # Mist Energy, Rock Fighting Energy
            return True
    return False


def prize_count(pokemon: Pokemon, is_attack_damage: bool) -> int:
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    if is_attack_damage:
        for card in pokemon.energyCards:
            if card.id == 12:  # Legacy Energy
                count -= 1
        for card in pokemon.tools:
            if card.id == 1172 and "Lillie" in data.name:  # Lillie’s Pearl
                count -= 1
    return max(0, count)


def pokemon_score(pokemon: Pokemon, is_attack_damage: bool) -> int:
    data = card_table[pokemon.id]
    score = prize_count(pokemon, is_attack_damage) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130
    id = pokemon.id
    if id == 144 or id == 322 or id == 323 or id == 337:  # Squawkabilly ex, Noctowl, Fan Rotom, Archaludon ex
        score -= 200
    if id == 112 and len(pokemon.energies) >= 1:  # Munkidori
        score += 300
    score += pokemon.hp
    return score


def add_card_count(card, my_index: int):
    if card is None:
        return
    if isinstance(card, Pokemon) or card.playerIndex == my_index:
        if card.serial not in serial_set:
            card_counts[card.id] -= 1
            serial_set.add(card.serial)
    if isinstance(card, Pokemon):
        for c in card.energyCards:
            add_card_count(c, my_index)
        for c in card.tools:
            add_card_count(c, my_index)
        for c in card.preEvolution:
            add_card_count(c, my_index)


def _record_opp_id_serial(card_id, serial):
    """Record one uniquely identified opponent card exactly once."""
    if card_id is None or serial is None:
        return
    serial = int(serial)
    if serial in opp_seen_serials:
        return
    opp_seen_serials.add(serial)
    opp_seen_counts[int(card_id)] += 1


def _record_opp_card(card):
    if card is None:
        return
    _record_opp_id_serial(getattr(card, "id", None), getattr(card, "serial", None))
    for group in ("energyCards", "tools", "preEvolution"):
        for child in getattr(card, group, None) or []:
            _record_opp_card(child)


def _update_opp_seen(obs: Observation, my_index: int):
    """Update public opponent card counts from this observation.

    Serials make this idempotent even though the simulator repeats zones and
    logs across selections.  Hidden hand/prize cards are never counted.
    """
    state = obs.current
    op_state = state.players[1 - my_index]
    for zone in (op_state.active, op_state.bench, op_state.discard):
        for card in zone or []:
            _record_opp_card(card)
    for card in state.stadium or []:
        if getattr(card, "playerIndex", -1) != my_index:
            _record_opp_card(card)
    for log in obs.logs or []:
        if getattr(log, "playerIndex", -1) == my_index:
            continue
        card_id = getattr(log, "cardId", None)
        serial = getattr(log, "serial", None)
        if card_id is not None and serial is not None:
            _record_opp_id_serial(card_id, serial)


def _opp_unseen_total() -> int:
    return max(0, 60 - sum(opp_seen_counts.values()))


def _p_opp_card(obs: Observation, my_index: int, card_id: int, copies: int) -> float:
    op_state = obs.current.players[1 - my_index]
    remaining = max(0, int(copies) - int(opp_seen_counts.get(card_id, 0)))
    return _hypergeom_at_least_one(
        _opp_unseen_total(), remaining, int(op_state.handCount)
    )


def _p_self_any(card_counts, card_ids, turns: int) -> float:
    unseen = sum(max(0, int(count)) for count in card_counts.values())
    copies = sum(max(0, int(card_counts.get(card_id, 0))) for card_id in card_ids)
    return _hypergeom_at_least_one(unseen, copies, turns)


def set_card_counts(obs: Observation, my_index: int):
    card_counts.clear()
    serial_set.clear()
    for id in my_deck:
        card_counts[id] += 1
    state = obs.current
    my_state = state.players[my_index]
    for card in my_state.hand:
        add_card_count(card, my_index)
    for card in my_state.discard:
        add_card_count(card, my_index)
    for card in my_state.bench:
        add_card_count(card, my_index)
    for card in my_state.active:
        add_card_count(card, my_index)
    for card in state.stadium:
        add_card_count(card, my_index)
    if state.looking is not None:
        for card in state.looking:
            add_card_count(card, my_index)
    add_card_count(obs.select.effect, my_index)


def get_card(obs: Observation, area: AreaType, index: int, player_index: int):
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


def main_option_proc(obs: Observation, damage: int):
    state = obs.current
    select = obs.select
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    global can_switch
    global can_attack
    global can_main_attack
    global can_energy_attach

    can_switch = False
    can_attack = False
    can_main_attack = False
    can_energy_attach = False
    for o in select.option:
        if o.type == OptionType.RETREAT:
            can_switch = True
        elif o.type == OptionType.ATTACK:
            can_attack = True
            if o.attackId == 154:  # Phantom Dive
                can_main_attack = True

    plan_a.attack = -1
    plan_b.attack = -1
    if not can_main_attack and not (bench_attacker and can_switch):
        return

    cards = [op_state.active[0]]
    for pokemon in op_state.bench:
        cards.append(pokemon)
    counter_indices = []
    ci = []
    ci.append(0)
    remain_damage = 60
    while ci:
        index = ci[-1]
        hp = cards[index].hp
        if remain_damage >= hp:
            counter_indices.append(ci.copy())
            if index < len(cards) - 1:
                remain_damage -= hp
                ci.append(index + 1)
                continue
        if index == len(cards) - 1:
            ci.pop()
            if ci:
                remain_damage += cards[ci[-1]].hp
        if ci:
            ci[-1] += 1
    counter_indices.append([])

    remain_prize = len(my_state.prize)
    plan_score = 0
    for i, pokemon in enumerate(cards):
        base_prize_count = 0
        base_score = pokemon_score(pokemon, True)
        active_damage = 0 if no_damage_dex(pokemon.id) else damage
        if pokemon.hp <= active_damage:
            base_prize_count += prize_count(pokemon, True)
        else:
            base_score *= active_damage / pokemon.hp
        ci = []
        max_score = base_score
        if remain_prize <= base_prize_count:
            max_score = 50000
        else:
            for indices in counter_indices:
                if i in indices:
                    continue
                prize = base_prize_count
                score = base_score
                for index in indices:
                    prize += prize_count(cards[index], False)
                    score += pokemon_score(cards[index], False)
                if remain_prize <= prize:
                    score = 50000
                else:
                    if prize >= 2:
                        if remain_prize <= 4:
                            score -= 1200
                    elif prize == 1:
                        score -= 300
                    else:
                        score += 1200
                if max_score < score:
                    max_score = score
                    ci = indices
        if plan_score < max_score:
            plan_score = max_score
            plan_a.attack = i
            plan_a.counter = ci
        if i == 0:
            plan_b.attack = plan_a.attack
            plan_b.counter = plan_a.counter


# ── the sample policy (verbatim logic), renamed; called inside a guard ──────────
def _policy(obs_dict: dict) -> list[int]:
    # The simulator normally hands us a dict, but the initial deck-selection
    # observation arrives as a typed Observation.  Accept both instead of
    # re-converting an object through to_observation_class (which loses select).
    global _LAST_SCORES
    _LAST_SCORES = []
    _load_cards()
    obs = obs_dict if isinstance(obs_dict, Observation) else to_observation_class(obs_dict)
    if obs.select is None:
        _DIAG["deck_returns"] += 1
        return my_deck

    global pre_turn_log
    global current_turn_log

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    if state.turn == 0:
        prize.clear()
        pre_turn_log.clear()
        current_turn_log.clear()
    else:
        for log in obs.logs:
            current_turn_log.append(log)
            if log.type == LogType.TURN_END:
                pre_turn_log = current_turn_log
                current_turn_log = []

    pre_ko = False
    no_item = False
    for log in pre_turn_log:
        if log.type == LogType.ATTACK:
            if log.attackId == 323:  # Itchy Pollen
                no_item = True
        elif log.type == LogType.MOVE_CARD:
            if (log.playerIndex == my_index
                    and (log.fromArea == AreaType.BENCH or log.fromArea == AreaType.ACTIVE)
                    and log.toArea == AreaType.DISCARD):
                pre_ko = True

    # `card_counts` after set_card_counts is (deck + prizes), i.e. every card
    # we have not seen.  That is the right quantity for "can this search card
    # still find a Dreepy?"  The previous code tried to subtract an inferred
    # prize pool, but the inference actually produced the cards revealed by
    # the last search, so deck_counts ended up holding only those and search
    # cards were undervalued for the rest of the game.
    set_card_counts(obs, my_index)
    if select.deck is not None:
        for card in select.deck:
            card_counts[card.id] -= 1
    deck_counts = card_counts

    # ── public-information belief (no learning) ────────────────────────────
    # Opponent: count every revealed card and use a generic 2-3 copy prior for
    # Boss's Orders.  We deliberately keep this conservative: it is only a
    # risk signal, not a hard rule.
    _update_opp_seen(obs, my_index)
    boss_copies = 3 if opp_seen_counts.get(Boss_Orders, 0) else 2
    boss_risk = _p_opp_card(obs, my_index, Boss_Orders, boss_copies)
    my_hand_value = _hand_quality(my_state.hand)
    expected_4 = _expected_draw_quality(card_counts, 4)
    late_game = (
        len(my_state.prize) <= 2
        or len(op_state.prize) <= 2
        or my_state.deckCount <= 10
    )
    turns_left = max(1, min(len(my_state.prize), len(op_state.prize)))
    setup_ids = (Dreepy, Drakloak, Dragapult_ex, Rare_Candy)
    p_setup = _p_self_any(card_counts, setup_ids, turns_left)

    profile = "default"
    profile_w = {}
    if _PROFILE_OK:
        try:
            if _BELIEF is not None and _BELIEF.snapshot_data:
                profile = _BELIEF.snapshot_data.get("profile", "default")
                profile_w = PROFILE_WEIGHTS.get(profile, {})
        except Exception:
            pass
    # Small deck-aware tuning: the lean list plays 4 Poke Pad and 2 Judge,
    # which are situational in multiples.  Lower their priority, do not remove
    # them from consideration.
    poke_pad_weight = 0.80 if my_deck.count(Poke_Pad) >= 4 else 1.0
    judge_weight = 0.65 if my_deck.count(Judge) >= 2 else 1.0

    prize_diff = len(my_state.prize) - len(op_state.prize)

    global bench_attacker

    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    discard_counts = defaultdict(int)

    active_id = 0
    bench_attacker = False
    can_evolve_dreepy = False
    evolve_dreepy_count = 0
    can_evolve_drakloak = False
    damage = 200
    night_mine = any(card.id == Nighttime_Mine for card in state.stadium)
    for card in my_state.active:
        if card is None:
            continue
        active_id = card.id
        field_counts[card.id] += 1
        if not card.appearThisTurn:
            if card.id == Dreepy:
                can_evolve_dreepy = True
                evolve_dreepy_count += 1
            elif card.id == Drakloak:
                can_evolve_drakloak = True
    for card in my_state.bench:
        field_counts[card.id] += 1
        if not card.appearThisTurn:
            if card.id == Dreepy:
                can_evolve_dreepy = True
                evolve_dreepy_count += 1
            elif card.id == Drakloak:
                can_evolve_drakloak = True
        if card.id == Dragapult_ex and len(card.energies) >= (3 if night_mine else 2):
            bench_attacker = True
    main_pokemon_count = field_counts[Dreepy] + field_counts[Drakloak] + field_counts[Dragapult_ex]
    no_more_dex = (field_counts[Dragapult_ex] * 2 >= len(op_state.prize))

    # Late-game feasibility.  If we already have a fuelled attacker, do not keep
    # investing in a new Dreepy -> Drakloak -> Dragapult line unless the missing
    # piece is likely to arrive before the prize race ends.  This is the
    # "stop waiting for the bench setup" rule.
    active_can_attack = any(
        p is not None
        and p.id == Dragapult_ex
        and len(p.energies) >= (3 if night_mine else 2)
        for p in my_state.active
    )
    has_attacker = bench_attacker or active_can_attack
    hand_ids = {card.id for card in my_state.hand}
    can_setup_now = (
        can_evolve_dreepy
        or can_evolve_drakloak
        or (Rare_Candy in hand_ids and Dragapult_ex in hand_ids)
        or (Dreepy in hand_ids and Drakloak in hand_ids)
    )
    late_no_setup = (
        late_game
        and has_attacker
        and not can_setup_now
        and turns_left <= 2
        and p_setup < (0.20 + profile_w.get("late", 0.0))
    )

    stadium_id = 0
    for card in state.stadium:
        stadium_id = card.id

    support_count = 0

    for card in my_state.discard:
        discard_counts[card.id] += 1

    def attach_score(attach_id: int, pokemon: Pokemon, active: bool) -> int:
        energy_count = len(pokemon.energies)
        # Dark Energy exists only for Munkidori's Adrena-Brain.
        if attach_id == Basic_Dark_Energy:
            has_dark = any(
                card_table.get(e.id) is not None and int(card_table[e.id].energyType) == 7
                for e in pokemon.energyCards
            )
            if pokemon.id == Munkidori and not has_dark:
                return 28000
            return UNNECESSARY
        # Nighttime Mine taxes Tera attacks by one {C}, and Dragapult ex is
        # Tera.  Without this the pilot caps at two energy and can never use
        # Phantom Dive while the opponent's stadium is up.
        needed = 3 if (pokemon.id == Dragapult_ex and night_mine) else 2
        if card_table[attach_id].cardType == CardType.TOOL:
            score = 60000
            if active:
                score += 1000
            return score
        if pokemon.id == Budew:
            return UNNECESSARY
        elif pokemon.id == Meowth_ex or pokemon.id == Fezandipiti_ex or pokemon.id == Latias_ex:
            if active and not can_switch and not my_state.asleep and not my_state.paralyzed:
                if bench_attacker or field_counts[Budew] >= 1:
                    return 22000
                else:
                    return 18000
            else:
                return UNNECESSARY
        if active and can_main_attack:
            return UNNECESSARY
        score = 20000
        if pokemon.id == Drakloak:
            score += 2000
        elif pokemon.id == Dreepy:
            score += 1000
        elif pokemon.id == Dragapult_ex:
            score += 0
        elif pokemon.id == Dragapult_ex:
            score += 0
        elif pokemon.id == Dragapult_ex:
            score += 0
        if energy_count >= needed:
            return UNNECESSARY          # already fueled up, don't over-fill
        elif energy_count == 1:
            if attach_id == pokemon.energyCards[0].id:
                return UNNECESSARY
            if pokemon.id == Dragapult_ex:
                score += 250
            elif pokemon.id == Dreepy:
                score -= 150
            else:
                score -= 200
            if active:
                score += 200
        else:  # energy_count == 0
            if active:
                if bench_attacker:
                    score += 400
            else:
                if pokemon.id == Dragapult_ex:
                    score += 150
                elif pokemon.id == Dreepy:
                    score += 100
                else:
                    score += 50
                if bench_attacker:
                    score -= 200
        if no_more_dex and (pokemon.id == Dreepy or pokemon.id == Drakloak):
            score -= 500
        return score

    def hand_score(id: int, ignore_count: bool):
        score = 0
        if id == Dreepy:
            if main_pokemon_count >= 3:
                score = 1000
            else:
                score = 18000
        elif id == Drakloak:
            if can_evolve_dreepy:
                score = 20000
            else:
                score = 3000
        elif id == Dragapult_ex:
            if no_more_dex:
                score = UNNECESSARY
            elif can_evolve_dreepy and hand_counts[Rare_Candy] >= 1 and not no_item:
                score = 40000
            elif can_evolve_drakloak:
                if field_counts[id] == 0:
                    score = 30000
                elif field_counts[id] == 1:
                    score = 10000
                else:
                    score = 50
            else:
                if field_counts[id] >= 2:
                    score = 50
                else:
                    score = 2000
        elif id == Fezandipiti_ex:
            if pre_ko:
                score = 15000
            elif prize_diff <= -2:
                score = 5
            elif len(op_state.prize) == 1:
                score = UNNECESSARY
        elif id == Latias_ex:
            if active_id == Fezandipiti_ex or active_id == Meowth_ex or active_id == Dreepy:
                if field_counts[Drakloak] + field_counts[Dragapult_ex] == 0:
                    score = 28000
                else:
                    score = 15000
            else:
                score = 10
        elif id == Budew:
            if field_counts[id] + field_counts[Drakloak] + field_counts[Dragapult_ex] >= 1:
                score = UNNECESSARY
            elif state.turn >= 2:
                score = 30000 + int(5000 * profile_w.get("budew", 0.0))
        elif id == Meowth_ex:
            if support_count > hand_counts[Boss_Orders] or stadium_id == Team_Rocket_Watchtower:
                score = 5
            elif state.supporterPlayed:
                score = 40
            else:
                score = 35000
        elif id == Chi_Yu:
            score = 12000 if field_counts[Chi_Yu] == 0 else 500
        elif id == Munkidori:
            score = 15000 if field_counts[Munkidori] == 0 else 500
        elif id == Dunsparce:
            score = 12000 if field_counts[Dunsparce] + field_counts[Dudunsparce] == 0 else 500
        elif id == Dudunsparce:
            score = 30000 if field_counts[Dunsparce] >= 1 else 5000
        elif id == Rare_Candy:
            if no_more_dex:
                score = UNNECESSARY
            elif can_evolve_dreepy and hand_counts[Dragapult_ex] >= 1:
                score = 40000
        elif id == Unfair_Stamp:
            if pre_ko:
                score = 80000
            elif len(op_state.prize) == 1:
                score = UNNECESSARY
            else:
                score = 80
        elif id == Buddy_Buddy_Poffin:
            count = deck_counts[Dreepy]
            if count == 0:
                score = UNNECESSARY
            else:
                if state.turn <= 2 and field_counts[Budew] == 0 and deck_counts[Budew] >= 1:
                    count += 1
                if count >= 2:
                    score = 35000 + int(4000 * profile_w.get("search", 0.0))
        elif id == Night_Stretcher:
            for i in discard_counts:
                if discard_counts[i] >= 1:
                    card_type = card_table[i].cardType
                    if card_type == CardType.POKEMON or card_type == CardType.BASIC_ENERGY:
                        score = max(score, hand_score(i, ignore_count))
        elif id == Crushing_Hammer:
            score = 20 + int(6000 * profile_w.get("hammer", 0.0))
        elif id == Ultra_Ball:
            if main_pokemon_count <= 2 or field_counts[Dreepy] >= 1:
                score = 70
            else:
                score = 5
            score += int(4000 * profile_w.get("search", 0.0))
        elif id == Poke_Pad:
            score = max(hand_score(Dreepy, ignore_count), hand_score(Drakloak, ignore_count))
            score += int(4000 * profile_w.get("search", 0.0))
            score = int(score * poke_pad_weight)
        elif id == Lucky_Helmet:
            score = 15
        elif id == Boss_Orders:
            if plan_a.attack > 0:
                score = 60000 + int(8000 * profile_w.get("boss", 0.0))
        elif id == Crispin:
            if not ignore_count or support_count == 0:
                if deck_counts[Basic_Fire_Energy] == 0 or deck_counts[Basic_Psychic_Energy] == 0:
                    score = 10
                if not can_main_attack and not bench_attacker and field_counts[Dragapult_ex] >= 1:
                    score = 55000
                else:
                    score = 25000
        elif id == Brock_Scouting:
            if not ignore_count or support_count == 0:
                if state.turn == 2 and field_counts[Budew] + field_counts[Latias_ex] == 0:
                    score = 50000
                else:
                    score = 30000
        elif id == Lillie_Determination:
            if not ignore_count or support_count == 0:
                draw_n = 8 if len(my_state.prize) == 6 else 6
                expected_lillie = _expected_draw_quality(card_counts, draw_n)
                net_lillie = my_hand_value - expected_lillie
                my_hand_count = len(my_state.hand)
                if my_state.deckCount <= 8:
                    # Lillie is the anti-deck-out card: it shuffles the hand
                    # back in before drawing.  With a large hand it refills the
                    # deck; prioritise it over Boss so a gust cannot consume
                    # the supporter slot while the deck is empty.
                    if my_hand_count >= 5 and len(op_state.prize) > 1:
                        score = 90000
                    else:
                        score = 25000
                elif my_hand_count <= 3 or net_lillie <= -4:
                    score = 45000
                elif my_hand_count >= 7 and net_lillie >= 3:
                    score = 8000
                elif len(my_state.prize) == 6:
                    score = 42000
                else:
                    score = 25000
        elif id == Judge:
            if not ignore_count or support_count == 0:
                net_judge = my_hand_value - expected_4
                opp_hand = int(op_state.handCount)
                my_hand_count = len(my_state.hand)
                if (my_state.deckCount <= 8 and my_hand_count >= 5
                        and len(op_state.prize) > 1):
                    # Deck-safety reset: refill before the draw step.  Still
                    # leave Boss available when the opponent is on one prize.
                    score = 70000
                elif my_state.deckCount <= 8 and my_hand_count >= 4:
                    score = 30000
                elif opp_hand >= 8 and net_judge <= 6:
                    score = 55000
                elif opp_hand >= 6 and net_judge <= 3:
                    score = 45000
                elif opp_hand - my_hand_count >= 3 and net_judge <= 4:
                    score = 38000
                elif my_hand_count <= 3 and opp_hand >= 4:
                    score = 32000
                elif net_judge > 4:
                    score = 4000
                else:
                    score = 7000
                if score > 0:
                    score = int(score * judge_weight)
                    score += int(6000 * profile_w.get("judge", 0.0))
        elif id == Rosa_Encouragement:
            if not ignore_count or support_count == 0:
                has_stage2 = any(
                    p is not None and p.id == Dragapult_ex
                    for p in (my_state.active + my_state.bench)
                )
                has_energy = any(
                    discard_counts[c] > 0
                    and c in card_table
                    and card_table[c].cardType == CardType.BASIC_ENERGY
                    for c in discard_counts
                )
                if len(my_state.prize) > len(op_state.prize) and has_stage2 and has_energy:
                    score = 46000
                else:
                    score = UNNECESSARY
        elif id == Risky_Ruins:
            if stadium_id == Risky_Ruins:
                score = UNNECESSARY
            else:
                score = 5000 if stadium_id else 3000
        elif id == Team_Rocket_Watchtower:
            if stadium_id != 0 and stadium_id != Team_Rocket_Watchtower:
                score = 4000
        elif id == Basic_Fire_Energy or id == Basic_Psychic_Energy:
            if can_main_attack and (len(op_state.prize) <= 2
                                    or (bench_attacker and len(op_state.prize) <= 4)):
                score = UNNECESSARY
            else:
                max_score = -10000
                for pokemon in my_state.active:
                    if pokemon is None:
                        continue
                    max_score = max(max_score, attach_score(id, pokemon, True))
                for pokemon in my_state.bench:
                    max_score = max(max_score, attach_score(id, pokemon, False))
                score = max_score - 5000
                if can_main_attack or bench_attacker:
                    score /= 10
        elif id == Basic_Dark_Energy:
            max_score = -10000
            for pokemon in my_state.active + my_state.bench:
                if pokemon is None:
                    continue
                max_score = max(max_score, attach_score(id, pokemon, pokemon in my_state.active))
            score = max_score - 5000

        if late_no_setup:
            # Only stop spending search resources on a setup line that cannot
            # arrive in time.  Cards already in hand are still playable: the
            # policy should not throw away a piece it already has.
            if id == Rare_Candy:
                score = min(score, 25000)
            elif id == Buddy_Buddy_Poffin or id == Poke_Pad:
                score = min(score, 12000)
            elif id == Ultra_Ball:
                score = min(score, 5000)

        # A likely Boss's Orders makes another low-value bench body a liability.
        # Keep the effect small: Dragapult still needs a bench for Phantom Dive.
        if (score > 0 and boss_risk >= 0.60 and len(my_state.bench) >= 4
                and id in (Fezandipiti_ex, Meowth_ex, Latias_ex, Budew,
                           Munkidori, Dunsparce, Dudunsparce)):
            # Deliberately weak signal: the card-count estimate is noisy.
            score -= int(2500 * max(0.0, boss_risk - 0.50))

        if not ignore_count and hand_counts[id] > 0:
            if id == Drakloak and hand_counts[id] < evolve_dreepy_count:
                score -= 10
            elif id == Dreepy:
                score -= 100
            else:
                score -= 100000
        return score

    global use_support
    if context == SelectContext.MAIN:
        main_option_proc(obs, damage)

        use_support = 0
        if not state.supporterPlayed:
            support_score = 0
            for o in select.option:
                if o.type == OptionType.PLAY:
                    card = get_card(obs, AreaType.HAND, o.index, state.yourIndex)
                    if card_table[card.id].cardType == CardType.SUPPORTER:
                        score = hand_score(card.id, True)
                        if support_score < score:
                            support_score = score
                            use_support = card.id

    hand_scores = []
    negative_hand_count = 0
    for card in my_state.hand:
        score = hand_score(card.id, False)
        hand_scores.append(score)
        if score < 0:
            negative_hand_count += 1
        hand_counts[card.id] += 1
        if card_table[card.id].cardType == CardType.SUPPORTER and card.id != Boss_Orders:
            support_count += 1

    no_draw = (my_state.deckCount <= 8)
    do_switch = (not can_main_attack and (bench_attacker or (active_id != Budew and field_counts[Budew] >= 1 and state.turn >= 2)))
    effect_card_id = 0 if select.effect is None else select.effect.id
    context_card_id = 0 if select.contextCard is None else select.contextCard.id

    scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            if context == SelectContext.IS_FIRST:
                score = 1 if GO_FIRST else -1
            else:
                score = 1
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is not None:
                energy_count = 0
                hp = 0
                if isinstance(card, Pokemon):
                    energy_count = len(card.energies)
                    hp = card.hp
                if (context == SelectContext.SWITCH
                        or context == SelectContext.TO_ACTIVE
                        or context == SelectContext.SETUP_ACTIVE_POKEMON):
                    if o.playerIndex == my_index:
                        if card.id == Dreepy:
                            score += 10000
                        elif card.id == Drakloak:
                            if energy_count >= 1:
                                score += 20000
                            else:
                                score -= 10000
                        elif card.id == Dragapult_ex:
                            score += 50000
                        elif card.id == Budew:
                            if context != SelectContext.SWITCH:
                                score += 100000
                            elif not bench_attacker:
                                score += 30000
                        elif card.id == Fezandipiti_ex:
                            score -= 1000
                        elif card.id == Meowth_ex:
                            score -= 2000
                    else:
                        if plan_a.attack == o.index + 1:
                            score += 100000
                    score += energy_count * 1000
                    score += hp
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    if my_index == state.firstPlayer or card.id != Dreepy:
                        score = -1
                elif context == SelectContext.TO_BENCH or context == SelectContext.TO_HAND:
                    score = hand_score(card.id, False)
                    hand_counts[card.id] += 1
                    if effect_card_id == Crispin:
                        score = 100000 - hand_score(card.id, True)
                elif context == SelectContext.DISCARD:
                    hand_counts[card.id] -= 1
                    if card_table[card.id].cardType == CardType.SUPPORTER:
                        support_count -= 1
                    score = -hand_score(card.id, False)
                elif context == SelectContext.DAMAGE_COUNTER or context == SelectContext.DAMAGE_COUNTER_ANY:
                    if effect_card_id == Munkidori and hp > 0:
                        # Adrena-Brain moves up to 3 counters (30 damage).
                        score = 100000 - 10 * hp + pokemon_score(card, False)
                        if hp <= 30:
                            score += 50000
                        elif hp <= 60:
                            score += 20000
                        if o.area == AreaType.ACTIVE:
                            score += 25000
                        elif o.area == AreaType.BENCH:
                            score += 20000
                    elif hp > 0:
                        score = 100000 - 10 * hp + pokemon_score(card, False)
                        if context == SelectContext.DAMAGE_COUNTER:
                            if 210 <= hp <= 230:
                                score += 20000 + hp * 20
                                if o.area == AreaType.ACTIVE:
                                    score += 10000
                            elif 40 <= hp <= 90:
                                score += 10000 + hp * 20
                            elif hp <= 30:
                                score += -10000 + hp * 20
                            if card.id == 133 or card.id == 351:
                                score += 30000
                        else:
                            index = o.index + 1
                            if index in plan_b.counter:
                                score += 100000
                            else:
                                remain_damage = select.remainDamageCounter * 10
                                if 210 <= hp <= 200 + remain_damage:
                                    score += 30000
                                elif 20 <= hp <= 60 + remain_damage:
                                    score += 10000
                                elif hp == 10:
                                    score -= 100000
                                if hp <= remain_damage:
                                    score += 25000
                            if no_damage_counter(card):
                                score = -1
                elif context == SelectContext.REMOVE_DAMAGE_COUNTER:
                    # Adrena-Brain source: heal the most damaged body, and
                    # prefer Dragapult ex (2 prizes) over a support Pokémon.
                    if isinstance(card, Pokemon):
                        max_hp = getattr(card, "maxHp", hp) or hp
                        missing = max(0, max_hp - hp)
                        score = (
                            missing * 10
                            + (5000 if card.id == Dragapult_ex else 0)
                            + pokemon_score(card, False) * 0.05
                        )
                elif context == SelectContext.ATTACH_FROM:
                    score = attach_score(context_card_id, card, o.area == AreaType.ACTIVE)
                    if card.id == Dragapult_ex:
                        score += 200
                elif context == SelectContext.ATTACH_TO:
                    if isinstance(card, Pokemon):
                        score = attach_score(context_card_id, card, o.area == AreaType.ACTIVE)
                    else:
                        score = 0
        elif o.type == OptionType.ENERGY_CARD or o.type == OptionType.ENERGY:
            if o.playerIndex != state.yourIndex:
                if o.area == AreaType.BENCH:
                    score = 20
                else:
                    score = 10
                card = get_card(obs, o.area, o.index, o.playerIndex)
                if card_table[card.id].cardType == CardType.SPECIAL_ENERGY:
                    score += 1
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            card_score = hand_scores[o.index]
            if card.id == Dreepy:
                score = 54000
            elif card.id == Fezandipiti_ex:
                if card_score > 0:
                    score = 35000
                else:
                    score = -1
            elif card.id == Latias_ex:
                if active_id != Drakloak and active_id != Dragapult_ex:
                    score = 51000
                else:
                    score = -1
            elif card.id == Budew:
                if field_counts[Budew] == 0 and field_counts[Dragapult_ex] == 0:
                    score = 52000 + int(5000 * profile_w.get("budew", 0.0))
                else:
                    score = -1
            elif card.id == Meowth_ex:
                if state.supporterPlayed or stadium_id == Team_Rocket_Watchtower:
                    score = -1
                elif support_count == 0:
                    score = 50000
                elif support_count == hand_counts[Boss_Orders] and not plan_a.attack <= 0:
                    score = 50000
                else:
                    score = -1
            elif card.id == Chi_Yu:
                score = 24000 if field_counts[Chi_Yu] == 0 and len(my_state.bench) < 5 else -1
            elif card.id == Munkidori:
                score = 26000 if field_counts[Munkidori] == 0 else -1
            elif card.id == Dunsparce:
                score = 24000 if field_counts[Dunsparce] + field_counts[Dudunsparce] == 0 else -1
            elif card.id == Rare_Candy:
                if no_more_dex:
                    score = -1
                else:
                    score = 30000 if late_no_setup else 75000
            elif card.id == Unfair_Stamp:
                score = 15000 + int(10000 * profile_w.get("stamp", 0.0))
            elif card.id == Night_Stretcher:
                if card_score >= 18000:
                    score = 42000
                else:
                    score = -1
            elif card.id == Crushing_Hammer:
                score = 40000 + int(10000 * profile_w.get("hammer", 0.0))
            elif card.id == Boss_Orders:
                if card.id == use_support:
                    score = 35000
                else:
                    score = -1
            elif card.id == Lillie_Determination:
                if card.id == use_support:
                    score = 14000
                else:
                    score = -1
            elif card.id == Judge:
                score = 30000 if card.id == use_support else -1
                if score > 0:
                    score = int(score * judge_weight)
            elif card.id == Rosa_Encouragement:
                score = 32000 if card.id == use_support else -1
            elif card.id == Team_Rocket_Watchtower:
                if stadium_id > 0 or state.turn == 1:
                    score = 80000
                else:
                    score = -1
            elif card.id == Risky_Ruins:
                if stadium_id == Risky_Ruins:
                    score = -1
                elif stadium_id > 0:
                    score = 8000
                else:
                    score = 4000
            elif no_draw:
                score = -1
            elif card.id == Buddy_Buddy_Poffin:
                if deck_counts[Dreepy] > 0:
                    score = (12000 if late_no_setup else 46000) + int(4000 * profile_w.get("search", 0.0))
                else:
                    score = -1
            elif card.id == Ultra_Ball:
                if negative_hand_count >= 2:
                    score = (8000 if late_no_setup else 44000) + int(4000 * profile_w.get("search", 0.0))
                else:
                    score = -1
            elif card.id == Poke_Pad:
                if deck_counts[Dreepy] + deck_counts[Drakloak] > 0:
                    score = (12000 if late_no_setup else 45000) + int(4000 * profile_w.get("search", 0.0))
                    score = int(score * poke_pad_weight)
                else:
                    score = -1
            elif card.id == Crispin or card.id == Brock_Scouting:
                if card.id == use_support:
                    score = 35000
                else:
                    score = -1
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, o.area, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = attach_score(card.id, pokemon, o.inPlayArea == AreaType.ACTIVE)
            if pokemon.id == Dragapult_ex and len(pokemon.energies) < 2:
                score += 25000
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score += len(pokemon.energies)
            if pokemon.id == Dreepy:
                score += 30000
            elif pokemon.id == Dunsparce:
                score += 40000
            elif pokemon.id == Drakloak and (
                field_counts[Dragapult_ex] >= 2
                or (field_counts[Dragapult_ex] == 1 and len(op_state.prize) <= 2)
            ):
                score = -1
            else:
                score += 70000
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card.id == Munkidori:
                damaged = any(p is not None and int(getattr(p,"hp",0) or 0) < int(getattr(p,"maxHp",0) or 0) for p in (my_state.active + my_state.bench))
                opp_low = any(p is not None and int(getattr(p,"hp",0) or 0) <= 30 for p in (op_state.active + op_state.bench))
                score = (60000 if opp_low else 30000) if damaged else -1
            elif no_draw:
                score = -1
            elif card.id == 1267:  # Lumiose City
                score = 1
            else:
                score = 40000
        elif o.type == OptionType.RETREAT:
            if do_switch:
                score = 10000
            else:
                score = -1
        elif o.type == OptionType.ATTACK:
            score = o.attackId

        scores.append(score)

    output = []
    sorted_scores = []
    if len(scores) >= 1:
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        for i in range(select.maxCount):
            if sorted_scores[i][1] > UNNECESSARY and (sorted_scores[i][1] >= 0
                    or select.minCount > i
                    or (context != SelectContext.TO_BENCH and context != SelectContext.SETUP_BENCH_POKEMON)):
                output.append(sorted_scores[i][0])

    # Every option can score UNNECESSARY, leaving a short (or empty) selection.
    # Fill up to minCount from the ranking; an UNNECESSARY option is still a
    # legal choice, and returning too few indices is a forfeit.
    if len(output) < select.minCount:
        for index, _score in sorted_scores:
            if len(output) >= select.minCount:
                break
            if index not in output:
                output.append(index)

    _LAST_SCORES = list(scores)
    return _legalize(output, select)


def reset():
    """Clear cross-game state.  The arena calls this before every battle.

    Several globals (turn logs, inferred prize pool, plan) survive between
    games otherwise, and a match that ends on the same turn number as the
    next one starts can inherit stale data.
    """
    global pre_turn_log, current_turn_log, bench_attacker, use_support
    global can_switch, can_attack, can_main_attack, can_energy_attach

    pre_turn_log = []
    current_turn_log = []
    prize.clear()
    card_counts.clear()
    serial_set.clear()
    opp_seen_counts.clear()
    opp_seen_serials.clear()
    plan_a.attack = -1
    plan_a.counter = []
    plan_b.attack = -1
    plan_b.counter = []
    bench_attacker = False
    use_support = 0
    can_switch = False
    can_attack = False
    can_main_attack = False
    can_energy_attach = False
    diag_reset()
    if _BELIEF is not None:
        _BELIEF.reset()


def _sel_get(select, key, default=None):
    if isinstance(select, dict):
        return select.get(key, default)
    return getattr(select, key, default)


def _snapshot_state() -> dict:
    return {
        "pre": list(pre_turn_log),
        "cur": list(current_turn_log),
        "prize": list(prize),
        "cards": dict(card_counts),
        "serials": set(serial_set),
        "bench": bench_attacker,
        "support": use_support,
        "can_switch": can_switch,
        "can_attack": can_attack,
        "can_main_attack": can_main_attack,
        "can_energy_attach": can_energy_attach,
        "plan_a": (plan_a.attack, list(plan_a.counter)),
        "plan_b": (plan_b.attack, list(plan_b.counter)),
        "scores": list(_LAST_SCORES),
    }


def _restore_state(state: dict) -> None:
    global pre_turn_log, current_turn_log, prize, card_counts, serial_set
    global bench_attacker, use_support, can_switch, can_attack, can_main_attack
    global can_energy_attach, _LAST_SCORES

    pre_turn_log[:] = state["pre"]
    current_turn_log[:] = state["cur"]
    prize[:] = state["prize"]
    card_counts.clear()
    card_counts.update(state["cards"])
    serial_set.clear()
    serial_set.update(state["serials"])
    bench_attacker = state["bench"]
    use_support = state["support"]
    can_switch = state["can_switch"]
    can_attack = state["can_attack"]
    can_main_attack = state["can_main_attack"]
    can_energy_attach = state["can_energy_attach"]
    plan_a.attack, plan_a.counter = state["plan_a"][0], list(state["plan_a"][1])
    plan_b.attack, plan_b.counter = state["plan_b"][0], list(state["plan_b"][1])
    _LAST_SCORES = list(state["scores"])


def _sample_hidden(obs, me):
    """Determinize the hidden zones for search_begin.

    Our own deck/prizes are sampled from the cards the policy has not seen.
    The opponent's list is unknown, so we use our own list as a plausible
    stand-in (the same shortcut Bellibolt's reference agent uses).
    """
    st = obs.current
    my = st.players[me]
    op = st.players[1 - me]
    deck_list = _load_deck()

    set_card_counts(obs, me)
    pool = []
    for cid, count in card_counts.items():
        if count > 0:
            pool.extend([int(cid)] * int(count))
    sel = obs.select
    if sel is not None and getattr(sel, "deck", None):
        for card in sel.deck:
            if int(card.id) in pool:
                pool.remove(int(card.id))

    need_my = int(my.deckCount) + len(my.prize)
    if len(pool) < need_my:
        pool.extend((deck_list * ((need_my - len(pool)) // max(1, len(deck_list)) + 1)))
    random.shuffle(pool)
    my_prize = pool[:len(my.prize)]
    my_deck = pool[len(my.prize):len(my.prize) + int(my.deckCount)]

    op_pool = deck_list * 8
    random.shuffle(op_pool)
    op_prize = op_pool[:len(op.prize)]
    op_hand = op_pool[len(op.prize):len(op.prize) + int(op.handCount)]
    op_deck = op_pool[len(op.prize) + int(op.handCount):
                     len(op.prize) + int(op.handCount) + int(op.deckCount)]
    while len(op_deck) < int(op.deckCount):
        op_deck.append(deck_list[0])

    opp_active = []
    if op.active and op.active[0] is None:
        opp_active = [Dreepy]
    return my_deck, my_prize, op_deck, op_prize, op_hand, opp_active


def _eval_state(obs, me) -> float:
    """Leaf evaluation from our perspective (end of our turn)."""
    st = obs.current
    if st is None:
        return 0.0
    if int(st.result) >= 0:
        return 1e9 if int(st.result) == me else -1e9
    my = st.players[me]
    op = st.players[1 - me]

    value = (len(op.prize) - len(my.prize)) * 12000.0
    for sign, ps in ((1.0, my), (-1.0, op)):
        for pokemon in (ps.active + ps.bench):
            if pokemon is None:
                continue
            value += sign * (pokemon.hp * 2.0 + len(pokemon.energies) * 150.0)
    value += len(my.bench) * 200.0 + len(my.hand) * 15.0

    my_active = (my.active or [None])[0]
    op_active = (op.active or [None])[0]
    if my_active is not None and op_active is not None:
        if _max_damage(op_active, my_active) >= my_active.hp:
            data = card_table.get(my_active.id)
            prizes = 3 if getattr(data, "megaEx", False) else 2 if getattr(data, "ex", False) else 1
            value -= prizes * 5000.0
        if _max_damage(my_active, op_active) >= op_active.hp:
            data = card_table.get(op_active.id)
            prizes = 3 if getattr(data, "megaEx", False) else 2 if getattr(data, "ex", False) else 1
            value += prizes * 4000.0
    return value


def _max_damage(attacker, defender) -> int:
    if attacker is None or defender is None:
        return 0
    data = card_table.get(attacker.id)
    if data is None:
        return 0
    # The opponent gets one more manual attachment before attacking.
    energy = len(attacker.energies) + 1
    best = 0
    for attack_id in getattr(data, "attacks", None) or []:
        attack = attack_table.get(int(attack_id))
        if attack is None:
            continue
        if len(getattr(attack, "energies", None) or []) > energy:
            continue
        damage = int(getattr(attack, "damage", 0) or 0)
        defender_data = card_table.get(defender.id)
        if defender_data is not None:
            if (getattr(defender_data, "weakness", None) is not None
                    and int(defender_data.weakness) == int(getattr(data, "energyType", -1))):
                damage *= 2
            if (getattr(defender_data, "resistance", None) is not None
                    and int(defender_data.resistance) == int(getattr(data, "energyType", -1))):
                damage = max(0, damage - 30)
        best = max(best, damage)
    return best


def _search_main(obs, select, scores):
    """1-ply search over the rule policy's top MAIN candidates."""
    if not (_SEARCH_OK and SEARCH_ENABLED):
        return None
    if select is None or int(_sel_get(select, "type", -1)) != int(SelectType.MAIN):
        return None
    if int(_sel_get(select, "maxCount", 0)) != 1:
        return None
    options = _sel_get(select, "option", None) or []
    if len(options) < 2 or len(scores) != len(options):
        return None

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    candidates = [i for i in order if scores[i] > UNNECESSARY][:SEARCH_K]
    if len(candidates) < 2:
        return None

    me = int(obs.current.yourIndex)
    snapshot = _snapshot_state()
    best_index = None
    best_value = float("-inf")
    started = time.perf_counter()
    nodes = 0
    try:
        for candidate in candidates:
            if time.perf_counter() - started > SEARCH_TIME_BUDGET:
                break
            try:
                hidden = _sample_hidden(obs, me)
                state = search_begin(obs, *hidden)
                search_id = state.searchId
            except Exception:
                break
            try:
                current = search_step(search_id, [candidate])
                for _ in range(SEARCH_ROLLOUT_MAX):
                    if (time.perf_counter() - started > SEARCH_TIME_BUDGET
                            or nodes >= SEARCH_NODE_BUDGET):
                        break
                    leaf = current.observation
                    if leaf is None or leaf.select is None or leaf.current is None:
                        break
                    if int(leaf.current.result) >= 0:
                        break
                    if int(leaf.current.yourIndex) != me:
                        break
                    action = _policy(leaf)
                    if not action:
                        break
                    current = search_step(search_id, action)
                    nodes += 1
                value = _eval_state(current.observation, me)
                if value > best_value:
                    best_value = value
                    best_index = candidate
            except Exception:
                pass
            finally:
                try:
                    search_end()
                except Exception:
                    pass
    finally:
        _restore_state(snapshot)
    return best_index


def _obs_select(obs):
    """Read select from either the simulator dict or a typed Observation."""
    if isinstance(obs, dict):
        return obs.get("select")
    return getattr(obs, "select", None)


def _observe(obs_dict):
    """Log belief features for this real MAIN decision (never for search leaves).

    Keeping this in agent(), rather than in _policy(), prevents the 1-ply
    search rollouts from spamming the Kaggle agent log.
    """
    global _BELIEF
    if not _BELIEF_OK:
        return
    try:
        select = _obs_select(obs_dict)
        if select is None:
            return
        if int(_sel_get(select, "type", -1)) != int(SelectType.MAIN):
            return
        obs = obs_dict if isinstance(obs_dict, Observation) else to_observation_class(obs_dict)
        if obs.current is None:
            return
        if _BELIEF is None:
            _BELIEF = BeliefObserver(_load_deck())
        snap = _BELIEF.update(obs, int(obs.current.yourIndex))
        if snap is not None:
            if _PROFILE_OK:
                try:
                    snap["profile"] = infer_profile(snap)
                except Exception:
                    snap["profile"] = "default"
            if BELIEF_LOG:
                _BELIEF.emit()
    except Exception:
        # Telemetry must never forfeit a game.
        pass


# NOTE: agent must be the LAST function defined in this module.  The Kaggle
# validation runner calls the last callable it finds; defining reset() after
# agent made it call reset() and fail with "Validation Episode failed".
def agent(obs_dict: dict) -> list[int]:
    """Robust wrapper: deck-return shortcut + try/except guard around the sample pilot."""
    try:
        if _obs_select(obs_dict) is None:
            _DIAG["deck_returns"] += 1
            return list(_load_deck())
    except Exception:
        pass
    _observe(obs_dict)
    _DIAG["decisions"] += 1
    try:
        out = _policy(obs_dict)
        _DIAG["policy_ok"] += 1
        select = _obs_select(obs_dict)
        try:
            if (select is not None
                    and int(_sel_get(select, "type", -1)) == int(SelectType.MAIN)):
                obs_obj = obs_dict if isinstance(obs_dict, Observation) else to_observation_class(obs_dict)
                better = _search_main(obs_obj, select, _LAST_SCORES)
                if better is not None:
                    out = [better]
        except Exception as _search_exc:  # noqa: BLE001
            _diag_record_error(_search_exc)
        return _legalize(out, select)
    except Exception as exc:
        _diag_record_error(exc)
        _DIAG["policy_fallback"] += 1
        try:
            sel = _obs_select(obs_dict)
            if sel is None:
                _DIAG["deck_returns"] += 1
                return my_deck
            return _legalize(_legal_fallback_select(sel), sel)
        except Exception as exc2:
            _diag_record_error(exc2)
            _DIAG["obs_fallback"] += 1
            return _legal_fallback_from_dict(obs_dict if isinstance(obs_dict, dict) else {})
