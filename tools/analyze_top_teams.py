#!/usr/bin/env python3
"""Analyze winning replays for top leaderboard teams."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import replay_meta as rm  # noqa: E402

BASE = ROOT / "replays" / "top5_nov1"
TEAMS = {
    "17014244": "やる気元気ミワハルキ 2nd ver.",
    "17013640": "Petit Canard",
    "17132686": "Dipam Chakraborty",
    "17153775": "knshnb",
    "16981607": "RtoABC2",
}
OUR_DECK = [int(x) for x in (ROOT / "bots" / "dragapult_lean4" / "deck.csv").read_text().split()]


def final_state(e: dict, player: int) -> dict:
    for step in reversed(e.get("steps", [])):
        if player >= len(step):
            continue
        current = (step[player].get("observation") or {}).get("current")
        if current:
            me = current["players"][player]
            return {
                "turn": current.get("turn"),
                "deck": me.get("deckCount"),
                "hand": len(me.get("hand") or []),
                "prizes": len(me.get("prize") or []),
            }
    return {}


def analyze(team_id: str, name: str) -> dict:
    folder = BASE / team_id
    variants: Counter = Counter()
    variant_meta: dict = {}
    opponents = Counter()
    first_choice = Counter()
    turns = []
    finals = defaultdict(list)
    actions = Counter()
    cards_played = Counter()
    games = 0
    for path in sorted(folder.glob("*.json")):
        try:
            e = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        names = rm.team_names(e)[:2]
        ours = [i for i, n in enumerate(names) if name in str(n)]
        if len(ours) != 1:
            continue
        me = ours[0]
        winner = rm.winner_index(e)
        if winner != me:
            continue
        decks = rm.extract_decks(e)
        if not decks:
            continue
        deck = tuple(sorted(Counter(decks[me]).items()))
        variants[deck] += 1
        if deck not in variant_meta:
            variant_meta[deck] = {"archetype": rm.archetype(decks[me]), "cards": []}
        opponents[rm.archetype(decks[1 - me])] += 1
        first_choice[rm.first_choice(e).get(me, "?")] += 1
        state = final_state(e, me)
        if state:
            turns.append(state.get("turn") or 0)
            for key in ("deck", "hand", "prizes"):
                finals[key].append(state.get(key) or 0)
        for player, step, obs, sel, action in rm.iter_decisions(e):
            if player != me:
                continue
            options = sel.get("option") or []
            for idx in action or []:
                if not isinstance(idx, int) or idx < 0 or idx >= len(options):
                    continue
                label = rm.option_label(obs, player, options[idx])
                kind = label.split(" ", 1)[0]
                actions[kind] += 1
                if kind == "PLAY":
                    cards_played[label.split(" ", 1)[1]] += 1
        games += 1
    most = variants.most_common(8)
    out_variants = []
    for deck, count in most:
        ids = []
        for cid, c in deck:
            ids += [cid] * c
        out_variants.append({
            "games": count,
            "archetype": rm.archetype(ids),
            "counts": dict(deck),
            "names": [rm.card_name(cid) for cid in ids],
        })
    return {
        "team": name,
        "games": games,
        "opponents": dict(opponents.most_common()),
        "first_choice": dict(first_choice),
        "avg_turn": round(sum(turns) / max(1, len(turns)), 1),
        "final": {k: round(sum(v) / max(1, len(v)), 2) for k, v in finals.items()},
        "actions_per_game": {k: round(v / max(1, games), 2) for k, v in actions.most_common()},
        "cards_per_game": {k: round(v / max(1, games), 2) for k, v in cards_played.most_common(40)},
        "variants": out_variants,
    }


def main() -> int:
    result = {}
    for team_id, name in TEAMS.items():
        folder = BASE / team_id
        n = len(list(folder.glob("*.json"))) if folder.is_dir() else 0
        if not n:
            continue
        result[team_id] = analyze(team_id, name)
        r = result[team_id]
        print(f"\n## {name} ({team_id})  games={r['games']}")
        print("opponents:", r["opponents"])
        print("first:", r["first_choice"], "avg turn", r["avg_turn"], "final", r["final"])
        print("actions:", r["actions_per_game"])
        for v in r["variants"]:
            from collections import Counter as C
            counts = C(v["names"])
            print(f"  deck {v['games']}x [{v['archetype']}]: "
                  + ", ".join(f"{n}x{c}" for c, n in counts.most_common()))
    out = ROOT / "results" / "top5_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nwritten", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
