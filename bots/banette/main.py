"""Banette policy for the PTCG AI Battle Challenge.

The deck is a single-prize, discard-scaling control deck:

* Shuppet / Banette / Poltchageist / Sinistcha have ``Hide 'n' Sneak``,
  which prevents attack/ability *effects* (Alakzam's counters, Dragapult's
  bench counters).
* Dhelmise does 30 + 140 when at least 4 Hide 'n' Sneak Pokémon are in the
  discard pile.
* Sinistcha spreads 4 damage counters to every opposing Pokémon with 6+.
* Gwynn discards two Pokémon without a Rule Box and draws 6.
* Bloodmoon Ursaluna ex / Kieran are the finishers into 270-300 HP ex Pokémon.

The policy is intentionally explicit: it scores legal options instead of
using a generic type priority.  It must always return a legal selection.
"""

from __future__ import annotations

from pathlib import Path

from cg.api import OptionType, SelectContext, SelectType, all_attack, all_card_data

# ---------------------------------------------------------------------------
# Card constants
# ---------------------------------------------------------------------------
SHUPPET = 1374
BANETTE = 1375
DHELMISE = 1380
POLTCHAGEIST = 1348
SINISTCHA = 1349
DUNSPARCE = 305
DUDUNSPARCE = 66
BLOODMOON = 44
FLUTTER_MANE = 56
LILLIES_CLEFAIRY = 272
PATRAT = 1340

POKE_PAD = 1152
ULTRA_BALL = 1121
POFFIN = 1086
LILLIES = 1227
GWYNN = 1428
POKEGEAR = 1122
BOSS = 1182
NIGHT_STRETCHER = 1097
KIERAN = 1191
PRISM_TOWER = 1430
NEUTRALIZATION_ZONE = 1247
AIR_BALLOON = 1174
LEGACY_ENERGY = 12
TELEPATH_ENERGY = 19
BASIC_PSYCHIC = 5

HIDE_SNEAK_IDS = {SHUPPET, BANETTE, POLTCHAGEIST, SINISTCHA}

CARDS = {card.cardId: card for card in all_card_data()}
ATTACKS = {attack.attackId: attack for attack in all_attack()}


def _hide_sneak(card_id: int | None) -> bool:
    if card_id is None:
        return False
    if card_id in HIDE_SNEAK_IDS:
        return True
    card = CARDS.get(card_id)
    return bool(card and any("Hide ’n’ Sneak" in skill.name for skill in card.skills))


def _card(card_id: int | None):
    return CARDS.get(card_id) if card_id is not None else None


def _prize_value(card_id: int | None) -> int:
    card = _card(card_id)
    if card is None:
        return 1
    if card.megaEx:
        return 3
    if card.ex:
        return 2
    return 1


def _is_dragon(card_id: int | None) -> bool:
    card = _card(card_id)
    return bool(card and int(card.energyType) == 9)


def _attack_name(attack_id: int | None) -> str:
    attack = ATTACKS.get(int(attack_id)) if attack_id is not None else None
    return attack.name if attack else ""


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _player_state(obs: dict, player: int) -> dict:
    return obs["current"]["players"][player]


def _card_from_area(obs: dict, player: int, area, index) -> int | None:
    if area is None or index is None:
        return None
    current = obs["current"]
    state = current["players"][player]
    area = int(area)
    index = int(index)
    try:
        if area == 2:
            return state["hand"][index]["id"]
        if area == 3:
            return state["discard"][index]["id"]
        if area == 4:
            return state["active"][index]["id"]
        if area == 5:
            return state["bench"][index]["id"]
        if area == 7:
            return current["stadium"][index]["id"]
        if area == 12:
            return current["looking"][index]["id"]
    except (KeyError, IndexError, TypeError):
        return None
    return None


def _option_card(obs: dict, player: int, option: dict, select: dict | None = None) -> int | None:
    kind = int(option.get("type", -1))
    if kind == int(OptionType.PLAY):
        return _card_from_area(obs, player, 2, option.get("index"))
    if kind in (int(OptionType.ATTACH), int(OptionType.EVOLVE), int(OptionType.ABILITY)):
        return _card_from_area(obs, player, option.get("area"), option.get("index"))
    if kind in (int(OptionType.CARD), int(OptionType.DISCARD)):
        area = option.get("area")
        if select is not None and area is not None and int(area) == 1:
            deck = select.get("deck") or []
            index = option.get("index")
            if index is not None and 0 <= int(index) < len(deck):
                return deck[int(index)]["id"]
        return _card_from_area(obs, player, area, option.get("index"))
    return None


def _counts(cards) -> dict[int, int]:
    out: dict[int, int] = {}
    for card in cards or []:
        if card is None:
            continue
        cid = card["id"]
        out[cid] = out.get(cid, 0) + 1
    return out


def _field_ids(obs: dict, player: int) -> list[int]:
    state = _player_state(obs, player)
    ids = []
    for pokemon in state.get("active") or []:
        if pokemon:
            ids.append(pokemon["id"])
    for pokemon in state.get("bench") or []:
        ids.append(pokemon["id"])
    return ids


def _hide_sneak_discard(obs: dict, player: int) -> int:
    state = _player_state(obs, player)
    return sum(1 for card in state.get("discard") or [] if _hide_sneak(card["id"]))


def _hand_supporter(obs: dict, player: int) -> bool:
    state = _player_state(obs, player)
    for card in state.get("hand") or []:
        data = _card(card["id"])
        if data and int(data.cardType) == int(3):  # Supporter
            return True
    return False


def _opponent_active(obs: dict, player: int) -> dict | None:
    active = _player_state(obs, 1 - player).get("active") or []
    return active[0] if active else None


def _bench_targets(obs: dict, player: int) -> list[dict]:
    return _player_state(obs, 1 - player).get("bench") or []


KEY_TARGET_BONUS = {
    163: 400,   # Slowking
    741: 350, 742: 350,  # Abra / Kadabra
    120: 300,   # Drakloak
    646: 250, 647: 250,  # Marnie's Impidimp / Morgrem
    305: 200, 66: 200,   # Dunsparce / Dudunsparce
    140: 150,   # Fezandipiti ex
    666: 100,   # Cinderace
}


def _target_score(obs: dict, player: int, option: dict) -> float:
    card_id = _option_card(obs, player, option)
    pokemon = None
    area = option.get("area")
    index = option.get("index")
    opp = 1 - player
    try:
        if int(area) == 4:
            pokemon = _player_state(obs, opp)["active"][int(index)]
        elif int(area) == 5:
            pokemon = _player_state(obs, opp)["bench"][int(index)]
    except (TypeError, KeyError, IndexError):
        pokemon = None
    hp = pokemon.get("hp", 0) if pokemon else 0
    return _prize_value(card_id) * 100 + max(0, 300 - hp) + KEY_TARGET_BONUS.get(card_id, 0)


def _can_ko(obs: dict, player: int, damage: int) -> bool:
    active = _opponent_active(obs, player)
    return bool(active and damage >= active.get("hp", 0))


def _attack_damage(obs: dict, player: int, attack_id: int | None) -> int:
    if attack_id is None:
        return 0
    name = _attack_name(attack_id)
    mine = _player_state(obs, player).get("active") or []
    attacker = mine[0] if mine else None
    defender = _opponent_active(obs, player)
    if not attacker or not defender:
        return 0
    discard_hide = _hide_sneak_discard(obs, player)
    my_bench = len(_player_state(obs, player).get("bench") or [])
    op_bench = len(_player_state(obs, 1 - player).get("bench") or [])

    damage = 0
    if name == "Vengeful Anchor":
        damage = 30 + (140 if discard_hide >= 4 else 0)
    elif name == "Puppet Pull":
        damage = 80
    elif name == "Blood Moon":
        damage = 240
    elif name == "Full Moon Rondo":
        damage = 20 + 20 * (my_bench + op_bench)
    elif name == "Matcha Spin":
        damage = 0
    elif name == "Hex Hurl":
        damage = 90
    elif name == "Hang Down":
        damage = 10
    elif name == "Furtive Drop":
        damage = 0
    else:
        attack = ATTACKS.get(int(attack_id))
        damage = int(attack.damage) if attack and attack.damage else 0

    # Weakness (Lillie's Clefairy changes Dragon weakness to Psychic).
    attacker_data = _card(attacker["id"])
    defender_data = _card(defender["id"])
    if attacker_data and defender_data and defender_data.weakness is not None:
        weakness = int(defender_data.weakness)
        if _is_dragon(defender["id"]) and int(attacker_data.energyType) == 5:
            weakness = 5
        if int(attacker_data.energyType) == weakness:
            damage *= 2
    return damage


def _attack_score(obs: dict, player: int, attack_id: int | None) -> float:
    name = _attack_name(attack_id)
    defender = _opponent_active(obs, player)
    if defender is None:
        return 1
    damage = _attack_damage(obs, player, attack_id)
    if name == "Matcha Spin":
        # Spread: 4 counters to every opposing Pokémon.
        if _hide_sneak_discard(obs, player) >= 6:
            bench = len(_player_state(obs, 1 - player).get("bench") or [])
            return 800 + 250 * bench
        return 50
    if damage >= defender.get("hp", 0):
        return 10000 + _prize_value(defender["id"]) * 1000
    score = damage * 10 + 100
    if name == "Puppet Pull":
        score += 500  # search any card
    if name == "Blood Moon":
        score += 300
    if name == "Vengeful Anchor" and _hide_sneak_discard(obs, player) >= 4:
        score += 250
    if name == "Hex Hurl":
        score += 200
    return score


def _play_score(obs: dict, player: int, card_id: int | None) -> float:
    if card_id is None:
        return 0
    state = _player_state(obs, player)
    hand_count = len(state.get("hand") or [])
    bench_count = len(state.get("bench") or [])
    deck_count = int(state.get("deckCount", 0))
    if card_id == POKE_PAD:
        return 4500 if deck_count > 5 else 500
    if card_id == ULTRA_BALL:
        return 4300 if hand_count >= 3 else 1000
    if card_id == POFFIN:
        return 4200 if bench_count < 5 else 1000
    if card_id == LILLIES:
        return 3800 if hand_count <= 5 else 1500
    if card_id == GWYNN:
        fuel = _hide_sneak_discard(obs, player)
        return 4000 if hand_count >= 2 and fuel < 6 else 1200
    if card_id == POKEGEAR:
        return 3000 if not _hand_supporter(obs, player) else 800
    if card_id == BOSS:
        targets = _bench_targets(obs, player)
        if not targets:
            return 1000
        active = _opponent_active(obs, player)
        can_hit = bool(active and _hide_sneak_discard(obs, player) >= 4)
        if can_hit and any(int(target.get("hp", 999)) <= 170 for target in targets):
            return 5000
        if any(target.get("id") in KEY_TARGET_BONUS for target in targets):
            return 3500
        return 1200
    if card_id == NIGHT_STRETCHER:
        return 2500
    if card_id == KIERAN:
        return 3200
    if card_id == PRISM_TOWER:
        return 2500 if not obs["current"].get("stadium") else 800
    if card_id == NEUTRALIZATION_ZONE:
        active = _opponent_active(obs, player)
        return 6000 if active and _prize_value(active["id"]) >= 2 else 2000
    if card_id == BLOODMOON:
        return 3000 if bench_count < 5 else 1200
    if card_id == LILLIES_CLEFAIRY:
        active = _opponent_active(obs, player)
        return 2500 if active and _is_dragon(active["id"]) else 1200
    if card_id == FLUTTER_MANE:
        active = _opponent_active(obs, player)
        data = _card(active["id"]) if active else None
        return 2000 if data and data.skills else 700
    if card_id == PATRAT:
        return 1200
    if card_id == AIR_BALLOON:
        return 1200
    return 1000


def _main_score(obs: dict, player: int, option: dict) -> float:
    kind = int(option.get("type", -1))
    if kind == int(OptionType.ATTACK):
        return _attack_score(obs, player, option.get("attackId"))
    if kind == int(OptionType.ABILITY):
        cid = _option_card(obs, player, option)
        if cid == DUDUNSPARCE:
            return 6000 if int(_player_state(obs, player).get("deckCount", 0)) > 5 else 500
        if cid == PRISM_TOWER:
            return 3500 if len(_player_state(obs, player).get("hand") or []) >= 4 else 800
        if cid == FLUTTER_MANE:
            return 1800
        if cid == LILLIES_CLEFAIRY:
            active = _opponent_active(obs, player)
            return 1500 if active and _is_dragon(active["id"]) else 300
        if cid == PATRAT:
            return 700
        return 500
    if kind == int(OptionType.EVOLVE):
        cid = _option_card(obs, player, option)
        if cid == DUDUNSPARCE:
            return 5500
        if cid == BANETTE:
            return 4000
        if cid == SINISTCHA:
            return 2000
        return 1000
    if kind == int(OptionType.PLAY):
        cid = _option_card(obs, player, option)
        return _play_score(obs, player, cid)
    if kind == int(OptionType.ATTACH):
        cid = _option_card(obs, player, option)
        target = _card_from_area(obs, player, option.get("inPlayArea"), option.get("inPlayIndex"))
        base = 0
        if cid == TELEPATH_ENERGY:
            base = 4200
        elif cid == BASIC_PSYCHIC:
            base = 3800
        elif cid == LEGACY_ENERGY:
            base = 3200
        elif cid == AIR_BALLOON:
            base = 1500
        else:
            base = 1800
        if target in (DHELMISE, BANETTE, BLOODMOON, LILLIES_CLEFAIRY):
            base += 400
        return base
    if kind == int(OptionType.RETREAT):
        state = _player_state(obs, player)
        active = (state.get("active") or [None])[0]
        if active and active.get("hp", 0) <= active.get("maxHp", 0) // 2:
            return 1200
        return -1
    if kind == int(OptionType.DISCARD):
        return -1
    if kind == int(OptionType.END):
        return 0
    return 100


def _pick_top(scores: list[tuple[float, int]], min_count: int, max_count: int) -> list[int]:
    count = min_count
    if count == 0 and scores and max(s for s, _ in scores) > 0:
        count = 1
    count = max(min_count, min(count, max_count, len(scores)))
    selected = [index for _, index in sorted(scores, reverse=True)[:count]]
    return selected


def _search_score(card_id: int | None) -> float:
    if card_id is None:
        return 0
    pokemon = {
        DHELMISE: 100, BANETTE: 95, SHUPPET: 90, DUNSPARCE: 85,
        BLOODMOON: 80, LILLIES_CLEFAIRY: 75, SINISTCHA: 70,
        POLTCHAGEIST: 65, DUDUNSPARCE: 60, FLUTTER_MANE: 40, PATRAT: 35,
    }
    trainer = {
        GWYNN: 100, BOSS: 90, LILLIES: 85, KIERAN: 60,
        POKE_PAD: 55, ULTRA_BALL: 50, NIGHT_STRETCHER: 45,
    }
    energy = {TELEPATH_ENERGY: 80, BASIC_PSYCHIC: 70}
    return pokemon.get(card_id, trainer.get(card_id, energy.get(card_id, 20)))


def _discard_score(obs: dict, player: int, option: dict, effect_id: int | None) -> float:
    cid = _option_card(obs, player, option)
    if cid is None:
        return 0
    fuel = _hide_sneak_discard(obs, player)
    field_hide = sum(1 for x in _field_ids(obs, player) if _hide_sneak(x))

    if effect_id == GWYNN:
        # Gwynn wants Hide 'n' Sneak bodies in the discard, but keep one body.
        if _hide_sneak(cid):
            bonus = 120 if fuel < 6 else 80
            if field_hide <= 1:
                bonus -= 60
            return bonus
        if cid in (DUNSPARCE, PATRAT, FLUTTER_MANE):
            return 50
        return -60

    # Ultra Ball / Prism Tower / generic discard: dump the least useful card.
    keep = _search_score(cid)
    if _hide_sneak(cid) and fuel < 6:
        return 80 - keep / 2
    if cid in (DHELMISE, BLOODMOON, LILLIES_CLEFAIRY):
        return -100
    return 40 - keep / 3


def _setup_active_score(card_id: int | None) -> float:
    return {
        SHUPPET: 100, DHELMISE: 95, DUNSPARCE: 85, POLTCHAGEIST: 75,
        BLOODMOON: 60, LILLIES_CLEFAIRY: 50, FLUTTER_MANE: 40, PATRAT: 30,
    }.get(card_id, 20)


def _setup_bench_score(card_id: int | None) -> float:
    return {
        DUNSPARCE: 100, SHUPPET: 90, POLTCHAGEIST: 80,
        DHELMISE: 70, BLOODMOON: 50, LILLIES_CLEFAIRY: 40,
    }.get(card_id, 10)


def _own_switch_score(obs: dict, player: int, option: dict) -> float:
    cid = _option_card(obs, player, option)
    if cid is None:
        return 0
    if cid == DHELMISE:
        return 100 if _hide_sneak_discard(obs, player) >= 4 else 65
    if cid == BANETTE:
        return 90
    if cid == SHUPPET:
        return 55
    if cid == BLOODMOON:
        return 80
    if cid == LILLIES_CLEFAIRY:
        active = _opponent_active(obs, player)
        return 85 if active and _is_dragon(active["id"]) else 55
    if cid == SINISTCHA:
        return 60
    return 30


def _choose(obs: dict, select: dict, player: int) -> list[int]:
    kind = int(select.get("type", -1))
    context = int(select.get("context", -1))
    options = select.get("option") or []
    mini = int(select.get("minCount", 0))
    maxi = min(int(select.get("maxCount", 0)), len(options))
    if maxi <= 0:
        return []

    effect_id = None
    effect = select.get("effect")
    if isinstance(effect, dict):
        effect_id = effect.get("id")
    elif effect is not None:
        effect_id = getattr(effect, "id", None)

    if kind == int(SelectType.MAIN):
        scores = [(_main_score(obs, player, option), i) for i, option in enumerate(options)]
        return _pick_top(scores, mini, maxi)

    if kind == int(SelectType.ATTACK):
        scores = [(_attack_score(obs, player, option.get("attackId")), i) for i, option in enumerate(options)]
        return _pick_top(scores, mini, maxi)

    if kind == int(SelectType.YES_NO):
        for i, option in enumerate(options):
            if int(option.get("type", -1)) == int(OptionType.YES):
                return [i]
        return [0]

    if kind == int(SelectType.COUNT):
        numbers = [(int(option.get("number", 0)), i) for i, option in enumerate(options)]
        if context in (38, 39, 40):  # draw / damage counters: more is better
            return [max(numbers)[1]]
        return [min(numbers)[1]]

    if context == int(SelectContext.SETUP_ACTIVE_POKEMON):
        scores = [(_setup_active_score(_option_card(obs, player, option)), i) for i, option in enumerate(options)]
        return _pick_top(scores, max(1, mini), maxi)

    if context == int(SelectContext.SETUP_BENCH_POKEMON):
        scores = [(_setup_bench_score(_option_card(obs, player, option)), i) for i, option in enumerate(options)]
        # Do not bench a random body if no Dunsparce/Shuppet is available.
        if scores and max(s for s, _ in scores) <= 10:
            return []
        return _pick_top(scores, mini, maxi)

    if context == int(SelectContext.DISCARD):
        scores = [(_discard_score(obs, player, option, effect_id), i) for i, option in enumerate(options)]
        return _pick_top(scores, mini, maxi)

    if context in (
        int(SelectContext.DISCARD_ENERGY_CARD),
        int(SelectContext.DISCARD_TOOL_CARD),
        int(SelectContext.DISCARD_CARD_OR_ATTACHED_CARD),
        int(SelectContext.DISCARD_ENERGY),
    ):
        scores = [(_discard_score(obs, player, option, effect_id), i) for i, option in enumerate(options)]
        return _pick_top(scores, mini, maxi)

    if context in (int(SelectContext.TO_HAND), int(SelectContext.LOOK), int(SelectContext.TO_DECK)):
        scores = [(_search_score(_option_card(obs, player, option, select)), i) for i, option in enumerate(options)]
        return _pick_top(scores, mini, maxi)

    if context == int(SelectContext.TO_BENCH):
        scores = [(_setup_bench_score(_option_card(obs, player, option)), i) for i, option in enumerate(options)]
        return _pick_top(scores, mini, maxi)

    if context in (int(SelectContext.SWITCH), int(SelectContext.TO_ACTIVE), int(SelectContext.TO_FIELD)):
        # Boss's Orders: the effect is the Boss card, options are the opponent's bench.
        if effect_id == BOSS:
            scores = [(_target_score(obs, player, option), i) for i, option in enumerate(options)]
        else:
            scores = [(_own_switch_score(obs, player, option), i) for i, option in enumerate(options)]
        return _pick_top(scores, max(1, mini), maxi)

    if context in (int(SelectContext.DAMAGE_COUNTER), int(SelectContext.DAMAGE_COUNTER_ANY), int(SelectContext.DAMAGE)):
        scores = [(_target_score(obs, player, option), i) for i, option in enumerate(options)]
        return _pick_top(scores, max(1, mini), maxi)

    if context in (int(SelectContext.EVOLVES_TO), int(SelectContext.EVOLVE)):
        scores = []
        for i, option in enumerate(options):
            cid = _option_card(obs, player, option)
            scores.append(({DUDUNSPARCE: 100, BANETTE: 90, SINISTCHA: 70}.get(cid, 10), i))
        return _pick_top(scores, max(1, mini), maxi)

    if context in (int(SelectContext.ATTACH_FROM), int(SelectContext.ATTACH_TO)):
        scores = []
        for i, option in enumerate(options):
            cid = _option_card(obs, player, option)
            scores.append(({TELEPATH_ENERGY: 100, BASIC_PSYCHIC: 90, LEGACY_ENERGY: 80}.get(cid, 20), i))
        return _pick_top(scores, max(1, mini), maxi)

    if kind == int(SelectType.ENERGY):
        scores = []
        for i, option in enumerate(options):
            area = option.get("area")
            index = option.get("index")
            cid = _card_from_area(obs, player, area, index)
            scores.append(({TELEPATH_ENERGY: 100, BASIC_PSYCHIC: 90, LEGACY_ENERGY: 80}.get(cid, 20), i))
        return _pick_top(scores, mini, maxi)

    if context == int(SelectContext.HEAL) or context == int(SelectContext.REMOVE_DAMAGE_COUNTER):
        scores = [(_own_switch_score(obs, player, option), i) for i, option in enumerate(options)]
        return _pick_top(scores, mini, maxi)

    # Safe fallback: take the minimum legal selection.
    return list(range(mini)) if mini > 0 else []


def _read_deck() -> list[int]:
    candidates = []
    if "__file__" in globals():
        candidates.append(Path(__file__).with_name("deck.csv"))
    candidates.extend([Path("deck.csv"), Path("/kaggle_simulations/agent/deck.csv")])
    for path in candidates:
        if not path.is_file():
            continue
        cards = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(cards) != 60:
            raise ValueError(f"{path}: expected 60 cards, found {len(cards)}")
        return cards
    raise FileNotFoundError("deck.csv not found next to main.py, in cwd, or in /kaggle_simulations/agent")


_DECK: list[int] | None = None


def agent(obs_dict: dict) -> list[int]:
    global _DECK
    select = obs_dict.get("select")
    if select is None:
        if _DECK is None:
            _DECK = _read_deck()
        return list(_DECK)

    player = int(obs_dict["current"]["yourIndex"])
    try:
        action = _choose(obs_dict, select, player)
    except Exception:  # noqa: BLE001
        action = []

    options = select.get("option") or []
    mini = int(select.get("minCount", 0))
    maxi = min(int(select.get("maxCount", 0)), len(options))
    action = [i for i in action if isinstance(i, int) and 0 <= i < len(options)]
    action = list(dict.fromkeys(action))
    if len(action) < mini:
        for i in range(len(options)):
            if i not in action:
                action.append(i)
                if len(action) >= mini:
                    break
    if len(action) > maxi:
        action = action[:maxi]
    return action
