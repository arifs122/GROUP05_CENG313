"""Public-state features shared by the value-model trainer and the bot.

Only information available to the acting player is used: own hand/field, public
opponent field/discard, prize/deck/hand counts.  No hidden opponent cards.
"""

from __future__ import annotations

from collections import Counter

FEATURE_NAMES = [
    "bias", "turn", "early", "late",
    "my_prizes", "opp_prizes", "prize_diff", "prize_ratio",
    "my_deck", "opp_deck", "deck_diff",
    "my_hand", "opp_hand", "hand_diff",
    "my_bench", "opp_bench", "bench_diff",
    "my_active_hp", "opp_active_hp", "hp_diff",
    "my_active_hp_frac", "opp_active_hp_frac",
    "my_active_energy", "opp_active_energy", "energy_diff",
    "my_total_hp", "opp_total_hp", "total_hp_diff",
    "my_total_energy", "opp_total_energy",
    "my_discard", "opp_discard", "discard_diff",
    "stadium", "first",
    "hand_energy", "hand_supporter_like", "hand_item_like",
    "hand_boss", "hand_lillie", "hand_judge", "hand_ultra",
    "hand_poffin", "hand_hammer", "hand_rare_candy",
    "hand_dreepy", "hand_drakloak", "hand_dragapult", "hand_budew",
    "active_dragapult", "active_budew", "active_meowth",
    "bench_pokemon", "bench_energy", "bench_tools",
    "first_player_flag", "opp_seen_total",
    "opp_seen_boss", "opp_seen_dragapult", "opp_seen_dwebble",
    "opp_seen_crustle", "opp_seen_lucario", "opp_seen_grimmsnarl",
    "opp_seen_hydrapple", "opp_seen_starmie", "opp_seen_archaludon",
]


def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _card_id(card):
    return int(_get(card, "id", 0) or 0)


def _pokemon_hp(pokemon):
    return int(_get(pokemon, "hp", 0) or 0)


def _pokemon_max_hp(pokemon):
    return int(_get(pokemon, "maxHp", _get(pokemon, "hp", 0)) or 0)


def _pokemon_energy(pokemon):
    return len(_get(pokemon, "energies", None) or _get(pokemon, "energyCards", None) or [])


def _pokemon_tools(pokemon):
    return len(_get(pokemon, "tools", None) or [])


def _all_pokemon(player):
    return [p for p in (_get(player, "active", None) or []) + (_get(player, "bench", None) or []) if p]


def _public_opponent_ids(obs, me: int) -> set[int]:
    current = _get(obs, "current")
    players = _get(current, "players", None) or []
    if len(players) < 2:
        return set()
    op = players[1 - me]
    ids: set[int] = set()

    def add(card):
        if not card:
            return
        cid = _card_id(card)
        if cid:
            ids.add(cid)
        for group in ("energyCards", "tools", "preEvolution"):
            for child in _get(card, group, None) or []:
                add(child)

    for zone in (_get(op, "active", None) or [], _get(op, "bench", None) or [],
                 _get(op, "discard", None) or []):
        for card in zone:
            add(card)
    for card in _get(current, "stadium", None) or []:
        if _get(card, "playerIndex", -1) != me:
            add(card)
    for log in _get(obs, "logs", None) or []:
        if int(_get(log, "playerIndex", -1)) == me:
            continue
        cid = _get(log, "cardId", None)
        if cid is not None:
            ids.add(int(cid))
    return ids


def extract(obs, me: int, first_player: int | None = None) -> list[float]:
    current = _get(obs, "current")
    players = _get(current, "players", None) or []
    if len(players) < 2:
        return [0.0] * len(FEATURE_NAMES)
    my = players[me]
    op = players[1 - me]
    turn = int(_get(current, "turn", 0) or 0)
    myp = len(_get(my, "prize", None) or [])
    opp = len(_get(op, "prize", None) or [])
    myd = int(_get(my, "deckCount", 0) or 0)
    opd = int(_get(op, "deckCount", 0) or 0)
    myh = len(_get(my, "hand", None) or [])
    oph = int(_get(op, "handCount", 0) or 0)
    myb = len(_get(my, "bench", None) or [])
    opb = len(_get(op, "bench", None) or [])
    my_active = ((_get(my, "active", None) or [None])[0])
    op_active = ((_get(op, "active", None) or [None])[0])
    my_hp = _pokemon_hp(my_active) if my_active else 0
    op_hp = _pokemon_hp(op_active) if op_active else 0
    my_max = max(1, _pokemon_max_hp(my_active)) if my_active else 1
    op_max = max(1, _pokemon_max_hp(op_active)) if op_active else 1
    my_en = _pokemon_energy(my_active) if my_active else 0
    op_en = _pokemon_energy(op_active) if op_active else 0
    my_mon = _all_pokemon(my)
    op_mon = _all_pokemon(op)
    my_total_hp = sum(_pokemon_hp(p) for p in my_mon)
    op_total_hp = sum(_pokemon_hp(p) for p in op_mon)
    my_total_en = sum(_pokemon_energy(p) for p in my_mon)
    op_total_en = sum(_pokemon_energy(p) for p in op_mon)
    my_bench_energy = sum(_pokemon_energy(p) for p in (_get(my, "bench", None) or []))
    my_bench_tools = sum(_pokemon_tools(p) for p in (_get(my, "bench", None) or []))
    stadium = _get(current, "stadium", None) or []
    hand_ids = [_card_id(c) for c in (_get(my, "hand", None) or []) if c]
    hand = Counter(hand_ids)
    active_id = _card_id(my_active) if my_active else 0
    seen = _public_opponent_ids(obs, me)

    # Small, stable groupings.  Exact card counts are deliberately avoided for
    # most cards to prevent overfitting to one deck list.
    energy_ids = {2, 5, 7, 1, 3, 4, 6, 8, 9, 10, 11, 14, 18, 19}
    supporter_like = {1182, 1198, 1210, 1213, 1227, 1240, 1245, 1205}
    item_like = {1079, 1080, 1086, 1097, 1120, 1121, 1152, 1156, 1256}

    def count(ids):
        return sum(hand.get(cid, 0) for cid in ids)

    feat = [
        1.0,
        min(turn, 40) / 20.0,
        1.0 if turn <= 4 else 0.0,
        1.0 if (myp <= 2 or opp <= 2 or myd <= 10) else 0.0,
        myp / 6.0, opp / 6.0, (opp - myp) / 6.0, (6 - myp) / 6.0,
        myd / 60.0, opd / 60.0, (opd - myd) / 60.0,
        myh / 15.0, oph / 15.0, (myh - oph) / 15.0,
        myb / 5.0, opb / 5.0, (myb - opb) / 5.0,
        my_hp / 300.0, op_hp / 300.0, (op_hp - my_hp) / 300.0,
        my_hp / my_max, op_hp / op_max,
        my_en / 3.0, op_en / 3.0, (my_en - op_en) / 3.0,
        my_total_hp / 1000.0, op_total_hp / 1000.0, (op_total_hp - my_total_hp) / 1000.0,
        my_total_en / 6.0, op_total_en / 6.0,
        len(_get(my, "discard", None) or []) / 40.0,
        len(_get(op, "discard", None) or []) / 40.0,
        (len(_get(op, "discard", None) or []) - len(_get(my, "discard", None) or [])) / 40.0,
        1.0 if stadium else 0.0,
        1.0 if first_player == me else 0.0,
        count(energy_ids), count(supporter_like), count(item_like),
        hand.get(1182, 0), hand.get(1227, 0), hand.get(1213, 0), hand.get(1121, 0),
        hand.get(1086, 0), hand.get(1120, 0), hand.get(1079, 0),
        hand.get(119, 0), hand.get(120, 0), hand.get(121, 0), hand.get(235, 0),
        1.0 if active_id == 121 else 0.0,
        1.0 if active_id == 235 else 0.0,
        1.0 if active_id == 1071 else 0.0,
        float(len(_get(my, "bench", None) or [])),
        float(my_bench_energy),
        float(my_bench_tools),
        1.0 if first_player == me else 0.0,
        len(seen),
        1.0 if 1182 in seen else 0.0,
        1.0 if 121 in seen else 0.0,
        1.0 if 344 in seen else 0.0,
        1.0 if 345 in seen else 0.0,
        1.0 if 678 in seen else 0.0,
        1.0 if 648 in seen else 0.0,
        1.0 if 150 in seen else 0.0,
        1.0 if 1031 in seen else 0.0,
        1.0 if 169 in seen else 0.0,
    ]
    return [float(x) for x in feat]
