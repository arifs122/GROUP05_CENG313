#!/usr/bin/env python3
"""Build a replay metadata table and an opponent/deck model.

Scans every episode replay under replays/ (deduplicated by EpisodeId), extracts
the actual decks, winner, first/second choice, simple action counts and final
board state, then aggregates:

  - archetype frequencies
  - exact deck variants
  - per-archetype key-card presence rates
  - per-archetype action tendencies
  - our own matchup records by policy version

Usage:
    python3 tools/opponent_model.py
    python3 tools/opponent_model.py --out results/opponent_model.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import replay_meta as rm  # noqa: E402

OUR_TEAM = "Attention Is All You Mew"
REPLAYS = ROOT / "replays"
OUT = ROOT / "results" / "opponent_model.json"

POLICY_LABELS = {
    "fixed_57592984": "fixed",
    "control_57593067": "control",
    "lean_57629056": "lean4",
    "lean5_57668293": "lean5",
    "oct24": "oct24",
    "first_57592508": "belief_first",
    "validate_57592508": "belief_first",
    "fixed_first": "fixed",
    "top_oct24": "top_team_oct24",
    "v3": "ladder_v3",
    "ladder": "ladder",
}


def source_label(path: Path) -> str:
    folder = path.parent.name
    if folder in POLICY_LABELS:
        return POLICY_LABELS[folder]
    if folder == "replays":
        return "top_team_original"
    return folder


def actions_for_player(episode: dict, player: int) -> tuple[Counter, Counter]:
    kinds: Counter = Counter()
    cards: Counter = Counter()
    for p, _step, observation, select, action in rm.iter_decisions(episode):
        if p != player:
            continue
        options = select.get("option") or []
        for index in action or []:
            if not isinstance(index, int) or index < 0 or index >= len(options):
                continue
            label = rm.option_label(observation, p, options[index])
            kind = label.split(" ", 1)[0]
            kinds[kind] += 1
            if kind == "PLAY":
                cards[label.split(" ", 1)[1]] += 1
    return kinds, cards


def final_state(episode: dict, player: int) -> dict:
    for step in reversed(episode.get("steps", [])):
        if player >= len(step):
            continue
        current = (step[player].get("observation") or {}).get("current")
        if not current:
            continue
        me = current["players"][player]
        return {
            "turn": current.get("turn"),
            "deck": me.get("deckCount"),
            "hand": len(me.get("hand") or []),
            "prizes": len(me.get("prize") or []),
            "bench": len(me.get("bench") or []),
        }
    return {}


def max_bench(episode: dict, player: int) -> int:
    best = 0
    for step in episode.get("steps", []):
        if player >= len(step):
            continue
        current = (step[player].get("observation") or {}).get("current")
        if not current:
            continue
        best = max(best, len(current["players"][player].get("bench") or []))
    return best


def load_episode(path: Path) -> dict | None:
    try:
        episode = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    decks = rm.extract_decks(episode)
    if not decks or len(decks) != 2:
        return None
    names = rm.team_names(episode)[:2]
    winner = rm.winner_index(episode)
    first = rm.first_player(episode)
    choices = rm.first_choice(episode)
    players = []
    for i in (0, 1):
        kinds, cards = actions_for_player(episode, i)
        players.append({
            "index": i,
            "team": names[i],
            "is_our_team": OUR_TEAM in names[i],
            "deck_archetype": rm.archetype(decks[i]),
            "final": final_state(episode, i),
            "max_bench": max_bench(episode, i),
            "choice": choices.get(i),
            "actions": dict(kinds),
            "cards_played": dict(cards),
        })
    return {
        "episode_id": (episode.get("info") or {}).get("EpisodeId"),
        "source": source_label(path),
        "teams": names,
        "winner": winner,
        "first": first,
        "choices": choices,
        "decks": decks,
        "players": players,
    }


def aggregate(episodes: list[dict]) -> dict:
    archetypes = Counter()
    variants: dict[str, Counter] = defaultdict(Counter)
    card_presence: dict[str, Counter] = defaultdict(Counter)
    action_totals: dict[str, defaultdict] = defaultdict(lambda: defaultdict(float))
    games: dict[str, int] = defaultdict(int)
    first_choice: dict[str, Counter] = defaultdict(Counter)
    our_matchups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    for e in episodes:
        for p in e["players"]:
            arch = p["deck_archetype"]
            archetypes[arch] += 1
            games[arch] += 1
            variants[arch][tuple(sorted(Counter(e["decks"][p["index"]]).items()))] += 1
            for cid in set(e["decks"][p["index"]]):
                card_presence[arch][cid] += 1
            first_choice[arch][p.get("choice") or "?"] += 1
            for kind, n in p["actions"].items():
                action_totals[arch][kind] += n
            action_totals[arch]["_max_bench"] += p["max_bench"]
            action_totals[arch]["_final_deck"] += (p.get("final") or {}).get("deck") or 0
            action_totals[arch]["_final_hand"] += (p.get("final") or {}).get("hand") or 0
        # our record by policy source
        ours = [p for p in e["players"] if p["is_our_team"]]
        if len(ours) == 1 and e["winner"] in (0, 1):
            me = ours[0]["index"]
            opp = e["players"][1 - me]
            result = "W" if e["winner"] == me else "L"
            bucket = our_matchups[e["source"]][opp["deck_archetype"]]
            bucket[0 if result == "W" else 1] += 1

    def card_list(arch: str) -> list[list]:
        n = max(1, games[arch])
        rows = [
            [rm.card_name(cid), count, round(count / n, 3)]
            for cid, count in card_presence[arch].most_common()
        ]
        return rows

    def behavior(arch: str) -> dict:
        n = max(1, games[arch])
        keys = ("PLAY", "ATTACH", "EVOLVE", "ABILITY", "ATTACK", "RETREAT", "CARD", "ENERGY")
        out = {k: round(action_totals[arch].get(k, 0) / n, 2) for k in keys}
        out["max_bench"] = round(action_totals[arch].get("_max_bench", 0) / n, 2)
        out["final_deck"] = round(action_totals[arch].get("_final_deck", 0) / n, 1)
        out["final_hand"] = round(action_totals[arch].get("_final_hand", 0) / n, 1)
        return out

    return {
        "episodes": len(episodes),
        "archetypes": dict(archetypes.most_common()),
        "first_choice": {arch: dict(counter) for arch, counter in first_choice.items()},
        "behavior": {arch: behavior(arch) for arch in archetypes},
        "cards": {arch: card_list(arch) for arch in archetypes},
        "variants": {
            arch: [
                {"deck": list(deck), "games": count}
                for deck, count in counter.most_common(5)
            ]
            for arch, counter in variants.items()
        },
        "our_matchups": {
            source: {arch: {"w": v[0], "l": v[1]} for arch, v in matchups.items()}
            for source, matchups in our_matchups.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build replay metadata + opponent model")
    parser.add_argument("--replays", type=Path, default=REPLAYS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    paths = sorted(args.replays.rglob("episode-*.json"))
    by_id: dict[int, dict] = {}
    priority = {
        "lean4": 10, "lean5": 10, "fixed": 9, "control": 9,
        "oct24": 8, "belief_first": 8, "top_team_oct24": 7,
        "top_team_original": 6, "ladder_v3": 6, "ladder": 6,
    }
    for path in paths:
        episode = load_episode(path)
        if not episode or episode["episode_id"] is None:
            continue
        eid = int(episode["episode_id"])
        old = by_id.get(eid)
        if old is None or priority.get(episode["source"], 0) > priority.get(old["source"], 0):
            by_id[eid] = episode

    episodes = list(by_id.values())
    model = aggregate(episodes)
    model["sources"] = dict(Counter(e["source"] for e in episodes))
    model["episode_meta"] = [
        {
            "episode_id": e["episode_id"],
            "source": e["source"],
            "teams": e["teams"],
            "winner": e["winner"],
            "first": e["first"],
            "choices": e["choices"],
            "archetypes": [p["deck_archetype"] for p in e["players"]],
            "turns": max((p.get("final") or {}).get("turn") or 0 for p in e["players"]),
            "our_team": [p["is_our_team"] for p in e["players"]],
        }
        for e in episodes
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"episodes: {len(episodes)}")
    print(f"sources: {model['sources']}")
    print("archetypes:")
    for arch, n in model["archetypes"].items():
        print(f"  {arch}: {n}")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
