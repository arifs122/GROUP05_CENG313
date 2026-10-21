"""Shared simple policy for the benchmark meta decks.

These are baseline pilots for deck lists mined from replays / online lists.
They are not top-tier policies; use them to compare a new AI policy against a
fixed, reproducible opponent.
"""

from __future__ import annotations

from pathlib import Path

from cg.api import OptionType, SelectContext, SelectType, all_attack, all_card_data

CARDS = {card.cardId: card for card in all_card_data()}
ATTACKS = {attack.attackId: attack for attack in all_attack()}


def _state(obs, player):
    return obs["current"]["players"][player]


def _active(obs, player):
    active = _state(obs, player).get("active") or []
    return active[0] if active else None


def _opp_active(obs, player):
    active = _state(obs, 1 - player).get("active") or []
    return active[0] if active else None


def _card_at(obs, player, area, index):
    try:
        area = int(area)
        index = int(index)
        state = _state(obs, player)
        if area == 2:
            return state["hand"][index]["id"]
        if area == 3:
            return state["discard"][index]["id"]
        if area == 4:
            return state["active"][index]["id"]
        if area == 5:
            return state["bench"][index]["id"]
    except (TypeError, KeyError, IndexError):
        return None
    return None


def _option_card(obs, player, option):
    kind = int(option.get("type", -1))
    if kind == int(OptionType.PLAY):
        return _card_at(obs, player, 2, option.get("index"))
    if kind in (int(OptionType.ATTACH), int(OptionType.EVOLVE), int(OptionType.ABILITY)):
        return _card_at(obs, player, option.get("area"), option.get("index"))
    if kind in (int(OptionType.CARD), int(OptionType.DISCARD)):
        return _card_at(obs, player, option.get("area"), option.get("index"))
    return None


def _pick(scores, lo, hi):
    count = max(lo, min(hi, len(scores)))
    if count == 0:
        return []
    return [i for _, i in sorted(scores, reverse=True)[:count]]


def _read_deck():
    candidates = []
    if "__file__" in globals():
        candidates.append(Path(__file__).with_name("deck.csv"))
    candidates += [Path("deck.csv"), Path("/kaggle_simulations/agent/deck.csv")]
    for path in candidates:
        if path.is_file():
            deck = [int(x) for x in path.read_text(encoding="utf-8").split() if x.strip()]
            if len(deck) == 60:
                return deck
    raise FileNotFoundError("deck.csv not found")


def make_agent(config: dict):
    setup_active = config.get("setup_active", {})
    setup_bench = config.get("setup_bench", {})
    evolve = config.get("evolve", {})
    play = config.get("play", {})
    search = config.get("search", {})
    abilities = config.get("abilities", {})
    energy_ids = set(config.get("energy", []))
    attach_targets = set(config.get("attach_targets", []))
    attack_override = config.get("attack_damage", {})
    attack_bonus = config.get("attack_bonus", {})

    def attack_damage(attack_id):
        name = ATTACKS[int(attack_id)].name if attack_id is not None and int(attack_id) in ATTACKS else ""
        if name in attack_override:
            return int(attack_override[name])
        attack = ATTACKS.get(int(attack_id)) if attack_id is not None else None
        return int(attack.damage) if attack and attack.damage else 0

    def attack_score(obs, player, attack_id):
        name = ATTACKS[int(attack_id)].name if attack_id is not None and int(attack_id) in ATTACKS else ""
        defender = _opp_active(obs, player)
        if defender is None:
            return 1
        damage = attack_damage(attack_id)
        if damage >= defender.get("hp", 0):
            return 12000
        return 1000 + damage * 10 + attack_bonus.get(name, 0)

    def main_score(obs, player, option):
        kind = int(option.get("type", -1))
        if kind == int(OptionType.ATTACK):
            return attack_score(obs, player, option.get("attackId"))
        if kind == int(OptionType.ABILITY):
            return abilities.get(_option_card(obs, player, option), 1500)
        if kind == int(OptionType.EVOLVE):
            return evolve.get(_option_card(obs, player, option), 1000)
        if kind == int(OptionType.PLAY):
            return play.get(_option_card(obs, player, option), 800)
        if kind == int(OptionType.ATTACH):
            cid = _option_card(obs, player, option)
            target = _card_at(obs, player, option.get("inPlayArea"), option.get("inPlayIndex"))
            base = 4500 if cid in energy_ids else 1200
            if target in attach_targets:
                base += 800
            active = _active(obs, player)
            if target is active and active and not (active.get("energies") or []):
                base += 1200
            return base
        if kind == int(OptionType.RETREAT):
            active = _active(obs, player)
            if active and active.get("hp", 0) <= active.get("maxHp", 0) // 2:
                return 1300
            return -1
        if kind == int(OptionType.END):
            return 0
        return 100

    def choose(obs, select, player):
        kind = int(select.get("type", -1))
        ctx = int(select.get("context", -1))
        options = select.get("option") or []
        lo = int(select.get("minCount", 0))
        hi = min(int(select.get("maxCount", 0)), len(options))
        if hi <= 0:
            return []
        if kind == int(SelectType.MAIN):
            return _pick([(main_score(obs, player, o), i) for i, o in enumerate(options)], lo, hi)
        if kind == int(SelectType.ATTACK):
            return _pick([(attack_score(obs, player, o.get("attackId")), i) for i, o in enumerate(options)], lo, hi)
        if kind == int(SelectType.YES_NO):
            for i, o in enumerate(options):
                if int(o.get("type", -1)) == int(OptionType.YES):
                    return [i]
            return [0]
        if kind == int(SelectType.COUNT):
            nums = [(int(o.get("number", 0)), i) for i, o in enumerate(options)]
            return [max(nums)[1]] if ctx in (38, 39, 40) else [min(nums)[1]]
        if ctx == int(SelectContext.SETUP_ACTIVE_POKEMON):
            return _pick([(setup_active.get(_option_card(obs, player, o), 20), i) for i, o in enumerate(options)], max(1, lo), hi)
        if ctx == int(SelectContext.SETUP_BENCH_POKEMON):
            return _pick([(setup_bench.get(_option_card(obs, player, o), 10), i) for i, o in enumerate(options)], lo, hi)
        if ctx in (int(SelectContext.TO_HAND), int(SelectContext.LOOK), int(SelectContext.TO_DECK)):
            return _pick([(search.get(_option_card(obs, player, o), 20), i) for i, o in enumerate(options)], lo, hi)
        if ctx == int(SelectContext.TO_BENCH):
            return _pick([(setup_bench.get(_option_card(obs, player, o), 10), i) for i, o in enumerate(options)], lo, hi)
        if ctx in (int(SelectContext.SWITCH), int(SelectContext.TO_ACTIVE), int(SelectContext.TO_FIELD)):
            return _pick([(search.get(_option_card(obs, player, o), 10), i) for i, o in enumerate(options)], max(1, lo), hi)
        if ctx in (int(SelectContext.EVOLVES_TO), int(SelectContext.EVOLVE)):
            return _pick([(evolve.get(_option_card(obs, player, o), 10), i) for i, o in enumerate(options)], max(1, lo), hi)
        if ctx in (int(SelectContext.DISCARD), int(SelectContext.DISCARD_CARD_OR_ATTACHED_CARD)):
            return _pick([(-search.get(_option_card(obs, player, o), 20), i) for i, o in enumerate(options)], lo, hi)
        return list(range(lo)) if lo else []

    def agent(obs_dict):
        select = obs_dict.get("select")
        if select is None:
            return _read_deck()
        player = int(obs_dict["current"]["yourIndex"])
        try:
            action = choose(obs_dict, select, player)
        except Exception:  # noqa: BLE001
            action = []
        options = select.get("option") or []
        lo = int(select.get("minCount", 0))
        hi = min(int(select.get("maxCount", 0)), len(options))
        action = [i for i in action if isinstance(i, int) and 0 <= i < len(options)]
        action = list(dict.fromkeys(action))
        while len(action) < lo:
            for i in range(len(options)):
                if i not in action:
                    action.append(i)
                    break
        return action[:hi]

    return agent
