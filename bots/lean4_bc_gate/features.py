"""Mazlum — replay parser and feature extraction (behavioral cloning).

Reads a Kaggle cabt episode replay and turns every decision into NumPy arrays:

* ``state``   — 1D vector: active HP, prize diff, hand count and both Active
  Pokémon ids.
* ``options`` — 2D matrix ``[MAX_OPTIONS, OPTION_FEATURE_SIZE]``, padded.
* ``mask``    — 1D vector ``[MAX_OPTIONS]``: 1 = real option, 0 = padding.

Card identity matters: without it two different cards in the same hand slot
produce identical features, which caps what any model can learn. Each option
therefore carries the acting card id and the target card id as **integer
indices** (``card_ids``), which ``model.MazlumNet`` feeds to an
``nn.Embedding`` lookup table. The float feature matrix stays compact (no
1500-way one-hot); only the attack's base damage, energy requirement and a few
numeric option fields are dense features.

Replay layout (verified on a real episode)::

    steps[t][player] = {action, observation, reward, status, visualize}

``steps[i]`` holds the observation, ``steps[i + 1][player]["action"]`` holds the
answer to it, so observations and actions must be paired across steps.

Usage::

    python src/mazlum/features.py
    python src/mazlum/features.py --replay replays/episode-125199547-replay.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPLAY = ROOT / "replays" / "episode-125199547-replay.json"


def default_replay() -> Path:
    """Preferred sample replay, else the first mined replay on disk."""
    if DEFAULT_REPLAY.is_file():
        return DEFAULT_REPLAY
    candidates = sorted((ROOT / "replays").glob("episode-*-replay.json"))
    return candidates[0] if candidates else DEFAULT_REPLAY

# Fixed option capacity: the widest decision seen in real replays had 33
# options, 40 leaves headroom without wasting memory.
MAX_OPTIONS = 40

# OptionType enum from cg.api (0..16); kept local so this module does not
# need the engine on sys.path.
OPTION_TYPE_COUNT = 17

# EnergyType enum from cg.api (COLORLESS .. TEAM_ROCKET).
ENERGY_TYPE_COUNT = 12

# Card ids are used directly as embedding indices (0 = "no card").
CARD_ID_LIMIT = 2048

# Deck fingerprint: counts of the 60 cards bucketed by ``card_id % N``.
# v13: 64 -> 16 buckets; the freed 48 dims carry dynamic strategy signals.
DECK_HISTOGRAM_BUCKETS = 16

# Card types / ids used by the v13 strategic features.
SUPPORTER = 3          # cg.api.CardType.SUPPORTER
BOSS_ORDERS_ID = 1182  # gust card in the Banette list (3 copies)
DARKNESS_ENERGY = 7

# Card-role priors for the Top2/Dragapult shell.  These are *features*, not
# hard policy rules: the network can still choose any legal option.
SETUP_CARD_IDS = {
    1086,  # Buddy-Buddy Poffin
    1152,  # Poké Pad
    1121,  # Ultra Ball
    1227,  # Lillie's Determination
    119,   # Dreepy
    120,   # Drakloak
    121,   # Dragapult ex
    1198,  # Crispin
    1079,  # Rare Candy (other shells)
}
TECH_CARD_IDS = {
    112,   # Munkidori
    1260,  # Risky Ruins
    1182,  # Boss's Orders
    1213,  # Judge
    235,   # Budew
    1071,  # Meowth ex
    140,   # Fezandipiti ex
    31,    # Chi-Yu
}
TARGET_CONTEXTS = {3, 4, 5, 13, 14, 16, 22, 25, 40}

STATE_FEATURE_NAMES = (
    "active_hp",
    "prize_diff",
    "hand_count",
    "my_active_id",
    "opp_active_id",
    "opp_active_hp",
    "opp_active_energy",
    "my_bench_ratio",
    "opp_bench_ratio",
    # v13 strategic signals (12): Poltergeist value, prize race, deck-out
    # pressure and public discard information.
    "opp_hand_count",
    "my_prizes",
    "opp_prizes",
    "my_prize_progress",
    "opp_prize_threat",
    "my_deck_ratio",
    "opp_deck_ratio",
    "my_discard_supporters",
    "opp_discard_supporters",
    "my_discard_boss",
    "opp_discard_boss",
    "opp_bench_energy",
    # TOKATÇI phase/setup features.
    "turn_norm",
    "bench_count",
    "opp_bench_count",
    "my_damaged_count",
    "my_munkidori_dark",
    "my_dreepy_count",
    "my_drakloak_count",
    "my_dragapult_count",
    "my_munkidori_count",
    "my_active_prize_value",
    "opp_active_prize_value",
    "stadium_active",
    "my_bench_energy",
    *(f"deck_bucket_{i}" for i in range(DECK_HISTOGRAM_BUCKETS)),
)

OPTION_FEATURE_NAMES = (
    *(f"type_{i}" for i in range(OPTION_TYPE_COUNT)),
    "has_area",
    "has_inplay_area",
    "target_active",
    "target_bench",
    "index",
    "area",
    "inplay_area",
    "number",
    "attack_id",
    "attack_damage",
    "attack_energy_cost",
    *(f"attack_energy_type_{i}" for i in range(ENERGY_TYPE_COUNT)),
    "attack_lethal",
    # TOKATÇI card-role / target features.
    "context_norm",
    "actor_is_setup",
    "actor_is_tech",
    "target_hp_ratio",
    "target_prize_value",
    "actor_prize_value",
    "is_target_context",
)
OPTION_FEATURE_SIZE = len(OPTION_FEATURE_NAMES)
STATE_FEATURE_SIZE = len(STATE_FEATURE_NAMES)

# Area ids used by the engine.
AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6
AREA_STADIUM = 7
AREA_LOOKING = 12

# Normalisation denominators for the numeric fields.
_INDEX_SCALE = 60.0
_AREA_SCALE = 12.0
_NUMBER_SCALE = 10.0
_ATTACK_ID_SCALE = 2000.0
_ID_SCALE = 2000.0
_DAMAGE_SCALE = 300.0
_ENERGY_COST_SCALE = 5.0

_ATTACKS: dict[int, object] | None = None
_CARDS: dict[int, object] | None = None


def _attack_table() -> dict[int, object]:
    """Attack metadata from the engine (empty dict when unavailable)."""
    global _ATTACKS
    if _ATTACKS is None:
        _ATTACKS = {}
        vendor = str(ROOT / "vendor")
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        try:
            from cg.api import all_attack

            _ATTACKS = {int(attack.attackId): attack for attack in all_attack()}
        except Exception:  # noqa: BLE001 - features must work without the engine
            _ATTACKS = {}
    return _ATTACKS


def _card_table() -> dict[int, object]:
    """Card metadata from the engine (empty dict when unavailable)."""
    global _CARDS
    if _CARDS is None:
        _CARDS = {}
        vendor = str(ROOT / "vendor")
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        try:
            from cg.api import all_card_data

            _CARDS = {int(card.cardId): card for card in all_card_data()}
        except Exception:  # noqa: BLE001 - features must work without the engine
            _CARDS = {}
    return _CARDS


@dataclass(frozen=True)
class Decision:
    """One observation/action pair taken from a replay."""

    step_index: int
    player: int
    observation: dict
    select: dict
    action: list[int]


def load_episode(path: Path) -> dict:
    """Load a replay JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_steps(episode: dict) -> Iterator[tuple[int, int, dict]]:
    """Walk ``steps`` and yield ``(step_index, player, step_entry)``."""
    for step_index, step in enumerate(episode.get("steps") or []):
        for player, agent in enumerate(step):
            yield step_index, player, agent


def iter_decisions(episode: dict) -> Iterator[Decision]:
    """Yield every decision with its action.

    The action for the observation at ``steps[i]`` is stored at
    ``steps[i + 1][player]["action"]``; the last step therefore has no
    answer and is skipped.
    """
    steps = episode.get("steps") or []
    for step_index, step in enumerate(steps[:-1]):
        for player, agent in enumerate(step):
            if agent.get("status") != "ACTIVE":
                continue
            observation = agent.get("observation") or {}
            select = observation.get("select")
            if not select:
                continue
            answer = steps[step_index + 1][player].get("action") or []
            if isinstance(answer, dict):
                # Some mined episodes store the choice as {"action": [...]}.
                answer = next(
                    (value for value in answer.values() if isinstance(value, list)), []
                )
            if not isinstance(answer, list):
                answer = []
            yield Decision(step_index, player, observation, select, list(answer))


def _get(obj, key: str, default=None):
    """Read a field from either a dict or a dataclass-like object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def has_board(observation: dict) -> bool:
    """True once the game is under way (turn started, an Active exists).

    Deck selection and the ``IS_FIRST`` / setup picks happen while ``current``
    is still empty, so their state vector would be all zeros.
    """
    current = observation.get("current") or {}
    if not current:
        return False
    if int(current.get("turn", 0) or 0) <= 0:
        return False
    return any((player.get("active") or [None])[0] for player in current.get("players") or [])


def _player_state(observation: dict, player: int) -> dict:
    current = observation.get("current") or {}
    players = current.get("players") or []
    return players[player] if player < len(players) else {}


def _active(observation: dict, player: int) -> dict | None:
    active = _player_state(observation, player).get("active") or []
    return active[0] if active else None


def _card_at(observation: dict, player: int, area, index) -> int | None:
    """Card id at ``(area, index)`` for ``player``, or None."""
    if area is None or index is None:
        return None
    area = int(area)
    index = int(index)
    state = _player_state(observation, player)
    current = observation.get("current") or {}
    select = observation.get("select") or {}

    try:
        if area == AREA_DECK:
            deck = select.get("deck") or []
            return int(deck[index]["id"]) if index < len(deck) else None
        if area == AREA_HAND:
            hand = state.get("hand") or []
            return int(hand[index]["id"]) if index < len(hand) else None
        if area == AREA_DISCARD:
            discard = state.get("discard") or []
            return int(discard[index]["id"]) if index < len(discard) else None
        if area == AREA_ACTIVE:
            active = state.get("active") or []
            pokemon = active[index] if index < len(active) else None
            return None if pokemon is None else int(pokemon["id"])
        if area == AREA_BENCH:
            bench = state.get("bench") or []
            return int(bench[index]["id"]) if index < len(bench) else None
        if area == AREA_PRIZE:
            prize = state.get("prize") or []
            card = prize[index] if index < len(prize) else None
            return None if card is None else int(card["id"])
        if area == AREA_STADIUM:
            stadium = current.get("stadium") or []
            return int(stadium[index]["id"]) if index < len(stadium) else None
        if area == AREA_LOOKING:
            looking = current.get("looking") or []
            card = looking[index] if index < len(looking) else None
            return None if card is None else int(card["id"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return None


def _option_cards(observation: dict, player: int, option: dict) -> tuple[int | None, int | None]:
    """Return ``(actor_card_id, target_card_id)`` for one option."""
    option_type = int(_get(option, "type", -1))

    actor: int | None = None
    target: int | None = None

    if option_type == 7:  # PLAY: card comes from hand
        actor = _card_at(observation, player, AREA_HAND, _get(option, "index"))
    elif option_type in (8, 9, 10, 11, 3, 4, 5, 6, 16):  # ATTACH/EVOLVE/...
        actor = _card_at(observation, player, _get(option, "area"), _get(option, "index"))
    elif option_type in (12, 13):  # RETREAT / ATTACK: the Active is the actor
        active = _active(observation, player)
        actor = None if active is None else int(active["id"])

    inplay_area = _get(option, "inPlayArea")
    if inplay_area is not None:
        target = _card_at(observation, player, inplay_area, _get(option, "inPlayIndex"))
    elif option_type == 13:  # ATTACK targets the opponent's Active
        defender = _active(observation, 1 - player)
        target = None if defender is None else int(defender["id"])

    return actor, target


def _attack_lethal(observation: dict, player: int, option: dict) -> float:
    """1.0 when this attack KOs the opponent's Active this turn.

    Rough calculator: printed damage plus the weakness multiplier. Effects,
    resistance and damage modifiers are ignored, so the flag is conservative.
    """
    attack_id = _get(option, "attackId")
    if attack_id is None:
        return 0.0
    attack = _attack_table().get(int(attack_id))
    defender = _active(observation, 1 - player)
    attacker = _active(observation, player)
    if attack is None or defender is None or attacker is None:
        return 0.0

    damage = int(_get(attack, "damage", 0) or 0)
    if damage <= 0:
        return 0.0

    defender_card = _card_table().get(int(defender["id"]))
    attacker_card = _card_table().get(int(attacker["id"]))
    if defender_card is not None and attacker_card is not None:
        weakness = _get(defender_card, "weakness")
        if weakness is not None and int(_get(attacker_card, "energyType", -1)) == int(weakness):
            damage *= 2

    return 1.0 if damage >= int(defender.get("hp", 0) or 0) else 0.0


def _active_hp(state: dict) -> float:
    active = state.get("active") or []
    pokemon = active[0] if active else None
    return float(_get(pokemon, "hp", 0) or 0)


def _active_id(observation: dict, player: int) -> float:
    active = _active(observation, player)
    return 0.0 if active is None else float(active["id"]) / _ID_SCALE


def _prize_diff(observation: dict, player: int) -> float:
    """Remaining prizes, opponent minus ours: positive means we are ahead."""
    mine = _player_state(observation, player)
    theirs = _player_state(observation, 1 - player)
    return float(len(theirs.get("prize") or []) - len(mine.get("prize") or []))


def _hand_count(state: dict) -> float:
    """Cards in hand. ``handCount`` is authoritative: the opponent's ``hand``
    list is always empty (hidden information), so len(hand) would read 0."""
    return float(state.get("handCount", 0) or 0)


def _prize_count(observation: dict, player: int) -> float:
    """Prizes still face-down for ``player`` (0-6)."""
    return float(len(_player_state(observation, player).get("prize") or []))


def _opp_hand_count(observation: dict, player: int) -> float:
    """Opponent's hand size — the damage scaling of Poltergeist."""
    return float(_player_state(observation, 1 - player).get("handCount", 0) or 0)


def _deck_ratio(state: dict) -> float:
    """Remaining deck in ``[0, 1]`` (deck-out / Iono value)."""
    return float(state.get("deckCount", 60) or 60) / 60.0


def _card_type(card) -> int:
    """CardType of a card dict from the observation (-1 when unknown)."""
    card_id = int((card or {}).get("id", 0) or 0)
    meta = _card_table().get(card_id)
    return int(getattr(meta, "cardType", -1)) if meta is not None else -1


def _discard(observation: dict, player: int) -> list:
    return _player_state(observation, player).get("discard") or []


def _discard_supporters(observation: dict, player: int) -> float:
    """Supporter cards in the public discard pile."""
    return float(sum(1 for card in _discard(observation, player) if _card_type(card) == SUPPORTER))


def _discard_has(observation: dict, player: int, card_id: int) -> float:
    """1.0 when ``card_id`` is already in the public discard (spent resource)."""
    return 1.0 if any(int((card or {}).get("id", 0) or 0) == card_id for card in _discard(observation, player)) else 0.0


def _bench_energy(state: dict) -> float:
    """Total energy attached to the bench (setup speed proxy)."""
    return float(sum(len(pokemon.get("energy") or []) for pokemon in (state.get("bench") or [])))


def _prize_value_for_id(card_id: int | None) -> int:
    """Prize value from card metadata: Mega ex=3, ex=2, else 1."""
    if not card_id:
        return 0
    meta = _card_table().get(int(card_id))
    if meta is None:
        return 1
    if bool(getattr(meta, "megaEx", False)):
        return 3
    if bool(getattr(meta, "ex", False)):
        return 2
    return 1


def _count_id_and_damaged(observation: dict, player: int, card_id: int | None = None) -> tuple[int, int]:
    """Return (count of card_id, damaged count) over Active+Bench."""
    state = _player_state(observation, player)
    cards = list(state.get("active") or []) + list(state.get("bench") or [])
    count = 0
    damaged = 0
    for pokemon in cards:
        if pokemon is None:
            continue
        if card_id is not None and int(pokemon.get("id", 0) or 0) == int(card_id):
            count += 1
        hp = float(pokemon.get("hp", 0) or 0)
        max_hp = float(pokemon.get("maxHp", 0) or 0)
        if max_hp > 0 and hp < max_hp:
            damaged += 1
    return count, damaged


def _munkidori_has_dark(observation: dict, player: int) -> float:
    state = _player_state(observation, player)
    for pokemon in (state.get("active") or []) + (state.get("bench") or []):
        if pokemon is None or int(pokemon.get("id", 0) or 0) != 112:
            continue
        energies = [int(x) for x in (pokemon.get("energies") or [])]
        cards = pokemon.get("energyCards") or []
        if DARKNESS_ENERGY in energies or any(int((c or {}).get("id", 0) or 0) == DARKNESS_ENERGY for c in cards):
            return 1.0
    return 0.0


def _target_pokemon(observation: dict, player: int, option: dict):
    """Return the target Pokémon dict for an option, if it points to one."""
    area = _get(option, "inPlayArea")
    index = _get(option, "inPlayIndex")
    if area is None or index is None:
        return None
    state = _player_state(observation, player)
    if int(area) == AREA_ACTIVE:
        arr = state.get("active") or []
    elif int(area) == AREA_BENCH:
        arr = state.get("bench") or []
    else:
        return None
    try:
        pokemon = arr[int(index)]
    except (IndexError, TypeError, ValueError):
        return None
    return pokemon if isinstance(pokemon, dict) else None


def deck_histogram(deck: list[int] | None) -> np.ndarray:
    """Counts of the deck's cards bucketed by ``card_id % DECK_HISTOGRAM_BUCKETS``.

    Normalised by deck size so the vector is a distribution; the model can then
    condition on which 60-card list it is piloting (Banette variants differ).
    """
    counts = np.zeros(DECK_HISTOGRAM_BUCKETS, dtype=np.float32)
    if not deck:
        return counts
    for card_id in deck:
        counts[int(card_id) % DECK_HISTOGRAM_BUCKETS] += 1.0
    return counts / max(1.0, float(len(deck)))


def _bench_ratio(observation: dict, player: int) -> float:
    """Bench occupancy in ``[0, 1]`` (benchMax defaults to 5)."""
    state = _player_state(observation, player)
    bench = state.get("bench") or []
    bench_max = int(state.get("benchMax", 5) or 5)
    return len(bench) / max(1, bench_max)


def _opp_active_hp(state: dict) -> float:
    active = state.get("active") or []
    pokemon = active[0] if active else None
    return float(_get(pokemon, "hp", 0) or 0)


def _opp_active_energy(state: dict) -> float:
    active = state.get("active") or []
    pokemon = active[0] if active else None
    return float(len(_get(pokemon, "energies", []) or []))


def state_vector(observation: dict, deck: list[int] | None = None) -> np.ndarray:
    """1D tactical state vector (v13, 37 dims).

    ``[active_hp, prize_diff, hand_count, my_active_id, opp_active_id,
    opp_active_hp, opp_active_energy, my_bench_ratio, opp_bench_ratio,
    opp_hand_count, my_prizes, opp_prizes, my_prize_progress, opp_prize_threat,
    my_deck_ratio, opp_deck_ratio, my_discard_supporters,
    opp_discard_supporters, my_discard_boss, opp_discard_boss,
    opp_bench_energy, deck_histogram(16)]``
    """
    player = int((observation.get("current") or {}).get("yourIndex", 0))
    state = _player_state(observation, player)
    opponent = _player_state(observation, 1 - player)
    my_prizes = _prize_count(observation, player)
    opp_prizes = _prize_count(observation, 1 - player)
    base = np.array(
        [
            _active_hp(state),
            _prize_diff(observation, player),
            _hand_count(state),
            _active_id(observation, player),
            _active_id(observation, 1 - player),
            _opp_active_hp(opponent),
            _opp_active_energy(opponent),
            _bench_ratio(observation, player),
            _bench_ratio(observation, 1 - player),
            _opp_hand_count(observation, player) / 10.0,
            my_prizes / 6.0,
            opp_prizes / 6.0,
            (6.0 - my_prizes) / 5.0,
            (6.0 - opp_prizes) / 5.0,
            _deck_ratio(state),
            _deck_ratio(opponent),
            _discard_supporters(observation, player) / 10.0,
            _discard_supporters(observation, 1 - player) / 10.0,
            _discard_has(observation, player, BOSS_ORDERS_ID),
            _discard_has(observation, 1 - player, BOSS_ORDERS_ID),
            _bench_energy(opponent) / 10.0,
            # TOKATÇI phase/setup signals.
            float((observation.get("current") or {}).get("turn", 0) or 0) / 20.0,
            len(state.get("bench") or []) / 5.0,
            len(opponent.get("bench") or []) / 5.0,
            _count_id_and_damaged(observation, player)[1] / 6.0,
            _munkidori_has_dark(observation, player),
            _count_id_and_damaged(observation, player, 119)[0] / 4.0,
            _count_id_and_damaged(observation, player, 120)[0] / 4.0,
            _count_id_and_damaged(observation, player, 121)[0] / 3.0,
            _count_id_and_damaged(observation, player, 112)[0] / 2.0,
            _prize_value_for_id((_active(observation, player) or {}).get("id")) / 3.0,
            _prize_value_for_id((_active(observation, 1 - player) or {}).get("id")) / 3.0,
            1.0 if (observation.get("current") or {}).get("stadium") else 0.0,
            _bench_energy(state) / 10.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, deck_histogram(deck)])


def _option_features(observation: dict, player: int, option: dict) -> np.ndarray:
    """Numeric description of a single legal option."""
    features = np.zeros(OPTION_FEATURE_SIZE, dtype=np.float32)

    option_type = int(_get(option, "type", -1))
    if 0 <= option_type < OPTION_TYPE_COUNT:
        features[option_type] = 1.0

    area = _get(option, "area")
    inplay_area = _get(option, "inPlayArea")
    number = _get(option, "number")
    attack_id = _get(option, "attackId")

    offset = OPTION_TYPE_COUNT
    features[offset + 0] = 1.0 if area is not None else 0.0
    features[offset + 1] = 1.0 if inplay_area is not None else 0.0
    features[offset + 2] = 1.0 if inplay_area == AREA_ACTIVE else 0.0
    features[offset + 3] = 1.0 if inplay_area == AREA_BENCH else 0.0
    features[offset + 4] = float(_get(option, "index", 0) or 0) / _INDEX_SCALE
    features[offset + 5] = float(area or 0) / _AREA_SCALE
    features[offset + 6] = float(inplay_area or 0) / _AREA_SCALE
    features[offset + 7] = float(number or 0) / _NUMBER_SCALE
    features[offset + 8] = float(attack_id or 0) / _ATTACK_ID_SCALE

    # Attack metadata: base damage and energy requirement.
    attack = _attack_table().get(int(attack_id)) if attack_id is not None else None
    if attack is not None:
        features[offset + 9] = float(attack.damage or 0) / _DAMAGE_SCALE
        energies = [int(energy) for energy in (attack.energies or [])]
        features[offset + 10] = len(energies) / _ENERGY_COST_SCALE
    else:
        energies = []

    energy_offset = offset + 11
    for energy in energies:
        if 0 <= energy < ENERGY_TYPE_COUNT:
            features[energy_offset + energy] += 1.0 / _ENERGY_COST_SCALE

    features[offset + 11 + ENERGY_TYPE_COUNT] = _attack_lethal(observation, player, option)

    # TOKATÇI role/context/target features.
    base_new = offset + 12 + ENERGY_TYPE_COUNT
    select = observation.get("select") or {}
    context = int(_get(select, "context", -1) or -1)
    actor_id, target_id = _option_cards(observation, player, option)
    features[base_new + 0] = float(context) / 48.0 if context >= 0 else 0.0
    features[base_new + 1] = 1.0 if actor_id in SETUP_CARD_IDS else 0.0
    features[base_new + 2] = 1.0 if actor_id in TECH_CARD_IDS else 0.0
    target_pokemon = _target_pokemon(observation, player, option)
    if target_pokemon is not None:
        hp = float(target_pokemon.get("hp", 0) or 0)
        max_hp = float(target_pokemon.get("maxHp", 0) or 0)
        features[base_new + 3] = hp / max_hp if max_hp > 0 else 0.0
    features[base_new + 4] = _prize_value_for_id(target_id) / 3.0
    features[base_new + 5] = _prize_value_for_id(actor_id) / 3.0
    features[base_new + 6] = 1.0 if context in TARGET_CONTEXTS else 0.0

    return features


def _card_index(card_id: int | None) -> int:
    """Embedding index for a card id (0 = no card / padding)."""
    if card_id is None:
        return 0
    return max(0, min(int(card_id), CARD_ID_LIMIT - 1))


def card_ids_matrix(observation: dict) -> np.ndarray:
    """``[MAX_OPTIONS, 2]`` integer indices: ``[actor_card_id, target_card_id]``.

    These are fed to ``nn.Embedding`` (training) and to the NumPy lookup table
    (inference); padding rows stay 0, which is the embedding ``padding_idx``.
    """
    select = observation.get("select") or {}
    raw_options = select.get("option") or []
    player = int((observation.get("current") or {}).get("yourIndex", 0))

    card_ids = np.zeros((MAX_OPTIONS, 2), dtype=np.int16)
    for row, option in enumerate(raw_options[:MAX_OPTIONS]):
        actor_id, target_id = _option_cards(observation, player, option)
        card_ids[row, 0] = _card_index(actor_id)
        card_ids[row, 1] = _card_index(target_id)

    return card_ids


def options_matrix(observation: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(options, mask)`` with ``MAX_OPTIONS`` rows.

    ``options`` is zero padded, ``mask`` is 1 for real options and 0 for the
    padding rows (also all zeros when the selection has no options). Options
    beyond ``MAX_OPTIONS`` are dropped; the widest decision in the sample
    replay had 33, so 40 rows leave headroom.
    """
    select = observation.get("select") or {}
    raw_options = select.get("option") or []
    player = int((observation.get("current") or {}).get("yourIndex", 0))

    options = np.zeros((MAX_OPTIONS, OPTION_FEATURE_SIZE), dtype=np.float32)
    mask = np.zeros(MAX_OPTIONS, dtype=np.float32)

    for row, option in enumerate(raw_options[:MAX_OPTIONS]):
        options[row] = _option_features(observation, player, option)
        mask[row] = 1.0

    return options, mask


def _episode_decks(episode: dict) -> list[list[int]] | None:
    """Both 60-card lists from ``steps[0][0].visualize[0].action``."""
    try:
        action = episode["steps"][0][0]["visualize"][0]["action"]
        if not isinstance(action, list) or len(action) != 2:
            return None
        return [list(map(int, deck)) for deck in action]
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def parse_episode(episode: dict) -> Iterator[
    tuple[Decision, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """Yield ``(decision, state, options, mask, card_ids)`` for every decision."""
    decks = _episode_decks(episode)
    for decision in iter_decisions(episode):
        deck = decks[decision.player] if decks and decision.player < len(decks) else None
        state = state_vector(decision.observation, deck)
        options, mask = options_matrix(decision.observation)
        card_ids = card_ids_matrix(decision.observation)
        yield decision, state, options, mask, card_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Mazlum replay parser / featurizer")
    parser.add_argument("--replay", type=Path, default=default_replay())
    args = parser.parse_args()

    episode = load_episode(args.replay)
    print(f"replay: {args.replay}")
    print(f"episode: {(episode.get('info') or {}).get('EpisodeId')}  steps: {len(episode.get('steps') or [])}")
    print(f"MAX_OPTIONS: {MAX_OPTIONS}  OPTION_FEATURE_SIZE: {OPTION_FEATURE_SIZE}")
    print(f"STATE_FEATURE_SIZE: {STATE_FEATURE_SIZE}  attacks loaded: {len(_attack_table())}")

    for decision, state, options, mask, card_ids in parse_episode(episode):
        # Pre-game picks (deck selection, IS_FIRST, setup) have no board yet.
        if not has_board(decision.observation):
            continue
        valid = int(mask.sum())
        if valid == 0:
            continue

        print(f"\nfirst valid decision: step={decision.step_index} player={decision.player}")
        print(f"  select: type={decision.select.get('type')} context={decision.select.get('context')}")
        print(f"  action: {decision.action}  valid_options={valid}")
        print(f"  state.shape    = {state.shape}   {state.tolist()}")
        print(f"  options.shape  = {options.shape}")
        print(f"  card_ids.shape = {card_ids.shape}  (actor, target) rows: {card_ids[:valid].tolist()}")
        print(f"  mask.shape     = {mask.shape}   sum={int(mask.sum())}")
        print(f"  mask preview   = {mask[:valid + 2].astype(int).tolist()}")
        break
    else:
        print("no valid decision found")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
