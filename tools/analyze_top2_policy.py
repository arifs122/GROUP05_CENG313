#!/usr/bin/env python3
"""Behavior analysis for Petit Canard's Dragapult submission."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import replay_meta as rm  # noqa: E402

TEAM = "Petit Canard"
FOLDER = ROOT / "replays" / "top2_policy"
OUT = ROOT / "results" / "top2_policy_analysis.json"


def final_state(e, player):
    for step in reversed(e.get("steps", [])):
        if player >= len(step):
            continue
        cur = (step[player].get("observation") or {}).get("current")
        if cur:
            me = cur["players"][player]
            return {
                "turn": cur.get("turn"),
                "deck": me.get("deckCount"),
                "hand": len(me.get("hand") or []),
                "prizes": len(me.get("prize") or []),
                "bench": len(me.get("bench") or []),
            }
    return {}


def main():
    games = []
    for path in sorted(FOLDER.glob("episode-*.json")):
        try:
            e = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        names = rm.team_names(e)[:2]
        ours = [i for i, n in enumerate(names) if TEAM in str(n)]
        if len(ours) != 1:
            continue
        me = ours[0]
        winner = rm.winner_index(e)
        decks = rm.extract_decks(e)
        if not decks:
            continue
        result = "W" if winner == me else "L"
        actions = Counter()
        cards = Counter()
        detail = Counter()
        first_actions = []
        n = 0
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
                detail[label] += 1
                if kind == "PLAY":
                    cards[label.split(" ", 1)[1]] += 1
                if n < 12:
                    first_actions.append(label)
                n += 1
        state = final_state(e, me)
        games.append({
            "episode": e["info"]["EpisodeId"],
            "result": result,
            "deck": tuple(sorted(Counter(decks[me]).items())),
            "archetype": rm.archetype(decks[me]),
            "opponent": rm.archetype(decks[1 - me]),
            "first": rm.first_choice(e).get(me, "?"),
            "state": state,
            "actions": actions,
            "cards": cards,
            "detail": detail,
            "first_actions": first_actions,
        })

    wins = [g for g in games if g["result"] == "W"]
    losses = [g for g in games if g["result"] == "L"]
    print(f"games={len(games)} wins={len(wins)} losses={len(losses)} "
          f"wr={len(wins)/max(1,len(games)):.1%}")
    print("opponents:", dict(Counter(g["opponent"] for g in games).most_common()))
    print("first:", dict(Counter(g["first"] for g in games)))
    print("decks:", Counter((g["archetype"], len(g["deck"])) for g in games))
    for label, subset in (("W", wins), ("L", losses)):
        print(f"\n{label} avg turn", round(sum((g['state'].get('turn') or 0) for g in subset)/max(1,len(subset)),1))
        print(" final", {k: round(sum((g['state'].get(k) or 0) for g in subset)/max(1,len(subset)),2)
                         for k in ("deck","hand","prizes","bench")})
        print(" actions", {k: round(v/max(1,len(subset)),2) for k,v in
                           sum((g["actions"] for g in subset), Counter()).most_common()})
        print(" cards", {k: round(v/max(1,len(subset)),2) for k,v in
                         sum((g["cards"] for g in subset), Counter()).most_common(25)})

    # Win rate when a card/action is used at least once.
    played = defaultdict(lambda: [0, 0])
    for g in games:
        for card in g["cards"]:
            played[card][0] += 1
            played[card][1] += int(g["result"] == "W")
    print("\ncard -> games, winrate")
    for card, (n, w) in sorted(played.items(), key=lambda kv: (-kv[1][0], kv[0])):
        print(f"  {card:28s} {n:3d}  {w/max(1,n):.1%}")

    # First-action patterns in wins vs losses.
    for label, subset in (("W", wins), ("L", losses)):
        seq = Counter(tuple(g["first_actions"]) for g in subset)
        print(f"\n{label} first sequences:")
        for s, n in seq.most_common(8):
            print(" ", n, "x", " -> ".join(s))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "games": len(games), "wins": len(wins), "losses": len(losses),
        "opponents": dict(Counter(g["opponent"] for g in games).most_common()),
        "first": dict(Counter(g["first"] for g in games)),
        "by_result": {
            label: {
                "avg_turn": round(sum((g["state"].get("turn") or 0) for g in subset) / max(1, len(subset)), 1),
                "final": {k: round(sum((g["state"].get(k) or 0) for g in subset) / max(1, len(subset)), 2)
                          for k in ("deck", "hand", "prizes", "bench")},
                "actions": dict(sum((g["actions"] for g in subset), Counter())),
                "cards": dict(sum((g["cards"] for g in subset), Counter())),
            }
            for label, subset in (("W", wins), ("L", losses))
        },
        "card_winrate": {card: {"games": n, "wins": w, "wr": w / max(1, n)}
                         for card, (n, w) in played.items()},
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("written", OUT)


if __name__ == "__main__":
    main()
