#!/usr/bin/env python3
"""Analyze Kaggle cabt episode replays.

Examples:
    python3 tools/replay_meta.py inspect replays/episode-*.json
    python3 tools/replay_meta.py summarize replays/*.json
    python3 tools/replay_meta.py summarize replays/*.json --out results/replay_meta.json
    python3 tools/replay_meta.py decks replays/*.json --archetype Alakazam
    python3 tools/replay_meta.py decisions replays/episode-125102132-replay.json

The replay JSON contains the full initial deck lists in
``steps[0][0]["visualize"][0]["action"]`` and every decision in ``steps``.
Replays are competition data: keep them in the git-ignored ``replays/`` folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

CARD_CSV = ROOT / "data" / "EN_Card_Data_R2_full.csv"

# Select / option enum values from cg.api (kept local so this tool works
# without importing the engine for simple analysis).
CONTEXT_NAMES = {
    0: "MAIN",
    1: "SETUP_ACTIVE",
    2: "SETUP_BENCH",
    3: "SWITCH",
    4: "TO_ACTIVE",
    5: "TO_BENCH",
    6: "TO_FIELD",
    7: "TO_HAND",
    8: "DISCARD",
    9: "TO_DECK",
    10: "TO_DECK_BOTTOM",
    11: "TO_PRIZE",
    12: "NOT_MOVE",
    13: "DAMAGE_COUNTER",
    14: "DAMAGE_COUNTER_ANY",
    15: "DAMAGE",
    16: "REMOVE_DAMAGE_COUNTER",
    17: "HEAL",
    18: "EVOLVES_FROM",
    19: "EVOLVES_TO",
    20: "DEVOLVE",
    21: "ATTACH_FROM",
    22: "ATTACH_TO",
    23: "DETACH_FROM",
    24: "LOOK",
    25: "EFFECT_TARGET",
    26: "DISCARD_ENERGY_CARD",
    27: "DISCARD_TOOL_CARD",
    28: "SWITCH_ENERGY_CARD",
    29: "DISCARD_CARD_OR_ATTACHED_CARD",
    30: "DISCARD_ENERGY",
    31: "TO_HAND_ENERGY",
    32: "TO_DECK_ENERGY",
    33: "SWITCH_ENERGY",
    34: "SKILL_ORDER",
    35: "ATTACK",
    36: "DISABLE_ATTACK",
    37: "EVOLVE",
    38: "DRAW_COUNT",
    39: "DAMAGE_COUNTER_COUNT",
    40: "REMOVE_DAMAGE_COUNTER_COUNT",
    41: "IS_FIRST",
    42: "MULLIGAN",
    43: "ACTIVATE",
    44: "FIRST_EFFECT",
    45: "MORE_DEVOLVE",
    46: "COIN_HEAD",
    47: "AFFECT_SPECIAL_CONDITION",
    48: "RECOVER_SPECIAL_CONDITION",
}

OPTION_NAMES = {
    0: "NUMBER",
    1: "YES",
    2: "NO",
    3: "CARD",
    4: "TOOL_CARD",
    5: "ENERGY_CARD",
    6: "ENERGY",
    7: "PLAY",
    8: "ATTACH",
    9: "EVOLVE",
    10: "ABILITY",
    11: "DISCARD",
    12: "RETREAT",
    13: "ATTACK",
    14: "END",
    15: "SKILL",
    16: "SPECIAL_CONDITION",
}

AREA_NAMES = {
    1: "DECK",
    2: "HAND",
    3: "DISCARD",
    4: "ACTIVE",
    5: "BENCH",
    6: "PRIZE",
    7: "STADIUM",
    8: "ENERGY",
    9: "TOOL",
    10: "PRE_EVOLUTION",
    11: "PLAYER",
    12: "LOOKING",
}


@dataclass
class CardInfo:
    id: int
    name: str
    kind: str
    expansion: str


def load_cards() -> dict[int, CardInfo]:
    if CARD_CSV.is_file():
        cards: dict[int, CardInfo] = {}
        with CARD_CSV.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                cid = int(row["Card ID"])
                cards.setdefault(
                    cid,
                    CardInfo(
                        id=cid,
                        name=row["Card Name"],
                        kind=row["Stage (Pokémon)/Type (Energy and Trainer)"],
                        expansion=row["Expansion"],
                    ),
                )
        return cards

    # Fallback: ask the local engine for card metadata.
    from cg.api import all_card_data

    type_names = {
        0: "Pokémon",
        1: "Item",
        2: "Pokémon Tool",
        3: "Supporter",
        4: "Stadium",
        5: "Basic Energy",
        6: "Special Energy",
    }
    return {
        card.cardId: CardInfo(card.cardId, card.name, type_names.get(int(card.cardType), "Other"), "")
        for card in all_card_data()
    }


CARDS = load_cards()

try:
    from cg.api import all_attack

    ATTACK_NAMES = {int(attack.attackId): attack.name for attack in all_attack()}
except Exception:  # noqa: BLE001
    ATTACK_NAMES = {}


def card_name(card_id: int | None) -> str:
    if card_id is None:
        return "?"
    card = CARDS.get(int(card_id))
    return card.name if card else f"#{card_id}"


def _is_pokemon(card: CardInfo) -> bool:
    return card.kind.startswith("Basic Pokémon") or card.kind.startswith("Stage 1") or card.kind.startswith("Stage 2")


ARCHETYPES: list[tuple[str, set[int], int]] = [
    ("Alakazam", {743}, 2),
    ("Mega Starmie", {1031}, 2),
    ("Mega Abomasnow", {723}, 2),
    ("Slowking-Kangaskhan", {163, 756}, 2),
    ("Slowking", {163}, 2),
    ("Mega Kangaskhan", {756}, 2),
    ("Dragapult", {121}, 2),
    ("Marnie's Grimmsnarl", {648}, 2),
    ("Cynthia's Garchomp", {381}, 2),
    ("Archaludon", {190}, 2),
    ("Mega Lucario", {678}, 2),
    ("Iono's Bellibolt", {269}, 2),
    ("Banette", {1375}, 2),
    ("Mega Lopunny", {849}, 2),
    ("Hydrapple", {150}, 2),
    ("Teal Mask Ogerpon", {96}, 2),
    ("Festival Lead", {1245, 90}, 2),
    ("Ceruledge", {320}, 2),
    ("Mega Excadrill", {1406}, 2),
    ("Munkidori-Crustle", {112, 345}, 2),
    ("Crustle-Centiskorch", {934, 345}, 2),
    ("Crustle", {345, 533}, 2),
    ("Mega Froslass", {861}, 2),
    ("N's Zorua", {292}, 2),
    ("Chandelure", {98, 495, 1379}, 2),
    ("Mega Charizard", {790, 928}, 2),
    ("Mega Gardevoir", {747}, 2),
    ("Mega Venusaur", {652}, 2),
    ("Mega Gengar", {772}, 2),
    ("Mega Absol", {687}, 2),
    ("Mega Blaziken", {326}, 2),
    ("Ethan's Typhlosion", {354}, 2),
    ("Hop's Trevenant", {879}, 2),
    ("Mega Starmie / Cinderace", {1031, 666}, 2),
]


def archetype(deck: list[int]) -> str:
    counts = Counter(deck)
    for name, key_ids, minimum in ARCHETYPES:
        if all(counts[cid] >= minimum for cid in key_ids):
            return name

    pokemon = [CARDS[cid] for cid in counts if cid in CARDS and _is_pokemon(CARDS[cid])]
    if pokemon:
        main = max(pokemon, key=lambda card: counts[card.id])
        return f"Other: {main.name}"
    return "Other"


def extract_decks(episode: dict) -> list[list[int]] | None:
    try:
        visualize = episode["steps"][0][0]["visualize"]
        action = visualize[0]["action"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(action, list) or len(action) != 2:
        return None
    return [list(map(int, deck)) for deck in action]


def team_names(episode: dict) -> list[str]:
    names = (episode.get("info") or {}).get("TeamNames") or []
    return [str(name) for name in names] + ["?", "?"]


def winner_index(episode: dict) -> int | None:
    rewards = episode.get("rewards") or []
    if len(rewards) != 2 or rewards[0] is None or rewards[1] is None:
        return None
    if rewards[0] == rewards[1]:
        return None
    return 0 if rewards[0] > rewards[1] else 1


def first_choice(episode: dict) -> dict[int, str]:
    """Return {player_index: 'first'|'second'} from IS_FIRST decisions.

    Replay steps store the observation at step i and the action that produced
    step i+1.  The action for an observation is therefore at step i+1.
    """
    choices: dict[int, str] = {}
    steps = episode.get("steps", [])
    for step_index, step in enumerate(steps):
        for player, agent in enumerate(step):
            observation = agent.get("observation") or {}
            select = observation.get("select")
            if not select or int(select.get("context", -1)) != 41:
                continue
            if step_index + 1 >= len(steps):
                continue
            action = steps[step_index + 1][player].get("action") or []
            if action and action[0] == 0:
                choices[player] = "first"
            elif action and action[0] == 1:
                choices[player] = "second"
    return choices


def first_player(episode: dict) -> int | None:
    for step in episode.get("steps", []):
        for agent in step:
            current = (agent.get("observation") or {}).get("current")
            if current and int(current.get("firstPlayer", -1)) >= 0:
                return int(current["firstPlayer"])
    for player, choice in first_choice(episode).items():
        return player if choice == "first" else 1 - player
    return None


def _card_from_area(observation: dict, player: int, area: int | None, index: int | None) -> int | None:
    if area is None or index is None:
        return None
    current = observation.get("current") or {}
    players = current.get("players") or []
    if player >= len(players):
        return None
    state = players[player]
    area = int(area)
    index = int(index)
    try:
        if area == 2:  # hand
            return state["hand"][index]["id"]
        if area == 3:  # discard
            return state["discard"][index]["id"]
        if area == 4:  # active
            pokemon = state["active"][index]
            return None if pokemon is None else pokemon["id"]
        if area == 5:  # bench
            return state["bench"][index]["id"]
        if area == 7:  # stadium
            return current["stadium"][index]["id"]
        if area == 12:  # looking
            return current["looking"][index]["id"]
    except (KeyError, IndexError, TypeError):
        return None
    return None


def option_label(observation: dict, player: int, option: dict) -> str:
    kind = OPTION_NAMES.get(int(option.get("type", -1)), str(option.get("type")))
    if kind == "PLAY":
        cid = _card_from_area(observation, player, 2, option.get("index"))
        return f"PLAY {card_name(cid)}"
    if kind in {"ATTACH", "EVOLVE"}:
        cid = _card_from_area(observation, player, option.get("area"), option.get("index"))
        target = _card_from_area(observation, player, option.get("inPlayArea"), option.get("inPlayIndex"))
        return f"{kind} {card_name(cid)} -> {card_name(target)}"
    if kind == "ABILITY":
        cid = _card_from_area(observation, player, option.get("area"), option.get("index"))
        return f"ABILITY {card_name(cid)}"
    if kind == "ATTACK":
        attack_id = option.get("attackId")
        return f"ATTACK {ATTACK_NAMES.get(int(attack_id), attack_id) if attack_id is not None else '?'}"
    if kind == "NUMBER":
        return f"NUMBER {option.get('number')}"
    return kind


def iter_decisions(episode: dict):
    """Yield (player, step_index, observation, select, action).

    The replay action at step i+1 is the action taken for the observation at
    step i, so pair them accordingly.
    """
    steps = episode.get("steps", [])
    for step_index, step in enumerate(steps):
        if step_index + 1 >= len(steps):
            continue
        for player, agent in enumerate(step):
            if agent.get("status") != "ACTIVE":
                continue
            observation = agent.get("observation") or {}
            select = observation.get("select")
            if select is None:
                continue
            action = steps[step_index + 1][player].get("action") or []
            yield player, step_index, observation, select, action


def inspect_episode(path: Path, show_actions: bool = False) -> None:
    episode = json.loads(path.read_text(encoding="utf-8"))
    names = team_names(episode)
    decks = extract_decks(episode)
    winner = winner_index(episode)
    first = first_player(episode)
    choices = first_choice(episode)

    print(f"\n{'=' * 70}\n{path.name}  episode={episode.get('info', {}).get('EpisodeId')}")
    print(f"teams: {names[0]} vs {names[1]}")
    print(f"winner: {names[winner] if winner is not None else 'draw'}")
    print(f"first player: {names[first] if first is not None else '?'}  choices={choices}")

    if not decks:
        print("could not extract decks")
        return

    for player, deck in enumerate(decks):
        counts = Counter(deck)
        print(f"\nP{player} {names[player]}  [{archetype(deck)}]  ({len(deck)} cards, {len(counts)} unique)")
        groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for cid, count in counts.items():
            card = CARDS.get(cid)
            groups[card.kind if card else "?"].append((count, cid))
        for kind in sorted(groups):
            print(f"  {kind}:")
            for count, cid in sorted(groups[kind], key=lambda item: card_name(item[1])):
                print(f"    {count}x {cid:>4} {card_name(cid)}")

    if not show_actions:
        return

    action_counts: dict[int, Counter] = {0: Counter(), 1: Counter()}
    main_counts: dict[int, Counter] = {0: Counter(), 1: Counter()}
    context_counts: dict[int, Counter] = {0: Counter(), 1: Counter()}
    for player, step_index, observation, select, action in iter_decisions(episode):
        context = CONTEXT_NAMES.get(int(select.get("context", -1)), str(select.get("context")))
        context_counts[player][context] += 1
        options = select.get("option") or []
        for index in action:
            if not isinstance(index, int) or index < 0 or index >= len(options):
                continue
            option = options[index]
            label = option_label(observation, player, option)
            action_counts[player][label.split(" ")[0]] += 1
            if context == "MAIN":
                main_counts[player][label] += 1
        if step_index < 0:
            break

    for player in (0, 1):
        print(f"\nP{player} {names[player]} action summary:")
        print(f"  contexts: {dict(context_counts[player].most_common())}")
        print(f"  main actions: {dict(main_counts[player].most_common())}")


def summarize(paths: list[Path]) -> dict:
    episodes = []
    deck_archetypes: Counter = Counter()
    wins: Counter = Counter()
    losses: Counter = Counter()
    matchups: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    choice_by_archetype: dict[str, Counter] = defaultdict(Counter)
    team_archetypes: dict[str, Counter] = defaultdict(Counter)

    for path in paths:
        episode = json.loads(path.read_text(encoding="utf-8"))
        decks = extract_decks(episode)
        if not decks:
            continue
        names = team_names(episode)
        types = [archetype(deck) for deck in decks]
        winner = winner_index(episode)
        choices = first_choice(episode)
        first = first_player(episode)

        for player, deck_type in enumerate(types):
            deck_archetypes[deck_type] += 1
            team_archetypes[names[player]][deck_type] += 1
            choice = choices.get(player)
            if choice:
                choice_by_archetype[deck_type][choice] += 1

        if winner is not None:
            win_type, lose_type = types[winner], types[1 - winner]
            wins[win_type] += 1
            losses[lose_type] += 1
            matchups[(win_type, lose_type)][0] += 1
            matchups[(lose_type, win_type)][1] += 1

        episodes.append(
            {
                "file": path.name,
                "episode_id": (episode.get("info") or {}).get("EpisodeId"),
                "teams": names[:2],
                "decks": decks,
                "archetypes": types,
                "winner": winner,
                "first_player": first,
                "choices": {str(k): v for k, v in choices.items()},
            }
        )

    rows = []
    for name, count in deck_archetypes.most_common():
        decided = wins[name] + losses[name]
        rows.append(
            {
                "archetype": name,
                "decks": count,
                "wins": wins[name],
                "losses": losses[name],
                "win_rate": wins[name] / decided if decided else None,
            }
        )

    return {
        "episodes": episodes,
        "archetypes": rows,
        "matchups": {
            f"{a} vs {b}": value for (a, b), value in sorted(matchups.items())
        },
        "choices": {name: dict(counter) for name, counter in choice_by_archetype.items()},
        "teams": {team: dict(counter) for team, counter in team_archetypes.items()},
    }


def print_summary(summary: dict) -> None:
    print(f"episodes: {len(summary['episodes'])}")
    print("\n## Archetypes")
    print("| Archetype | Decks | W-L | Win rate |")
    print("|---|---:|---:|---:|")
    for row in summary["archetypes"]:
        rate = "-" if row["win_rate"] is None else f"{row['win_rate']:.0%}"
        print(f"| {row['archetype']} | {row['decks']} | {row['wins']}-{row['losses']} | {rate} |")

    print("\n## First/second choices")
    for name, choices in summary["choices"].items():
        print(f"- {name}: {choices}")

    print("\n## Matchups (A vs B: A wins-B wins)")
    for key, (a_wins, b_wins) in summary["matchups"].items():
        print(f"- {key}: {a_wins}-{b_wins}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze cabt replay JSON files")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_parser = sub.add_parser("inspect", help="show decks and optionally action summary")
    inspect_parser.add_argument("replays", nargs="+", type=Path)
    inspect_parser.add_argument("--actions", action="store_true")
    inspect_parser.add_argument("--json", action="store_true", help="print raw summary as JSON")

    summary_parser = sub.add_parser("summarize", help="aggregate archetypes and matchups")
    summary_parser.add_argument("replays", nargs="+", type=Path)
    summary_parser.add_argument("--out", type=Path, default=None)
    summary_parser.add_argument("--json", action="store_true")

    decks_parser = sub.add_parser("decks", help="show the most common deck list(s)")
    decks_parser.add_argument("replays", nargs="+", type=Path)
    decks_parser.add_argument("--archetype", default=None)

    args = parser.parse_args()

    if args.command == "inspect":
        for path in args.replays:
            inspect_episode(path, show_actions=args.actions)
        return 0

    if args.command == "summarize":
        result = summarize(args.replays)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"written: {args.out}")
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_summary(result)
        return 0

    if args.command == "decks":
        result = summarize(args.replays)
        for row in result["archetypes"]:
            name = row["archetype"]
            if args.archetype and name != args.archetype:
                continue
            variants: Counter = Counter()
            for episode in result["episodes"]:
                for player, deck_type in enumerate(episode["archetypes"]):
                    if deck_type == name:
                        variants[tuple(sorted(Counter(episode["decks"][player]).items()))] += 1
            if not variants:
                continue
            signature, frequency = variants.most_common(1)[0]
            counts = {cid: count for cid, count in signature}
            print(f"\n## {name} (seen {frequency}x)")
            groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
            for cid, count in counts.items():
                card = CARDS.get(cid)
                groups[card.kind if card else "?"].append((count, cid))
            for kind in sorted(groups):
                print(f"  {kind}:")
                for count, cid in sorted(groups[kind], key=lambda item: card_name(item[1])):
                    print(f"    {count}x {cid:>4} {card_name(cid)}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
