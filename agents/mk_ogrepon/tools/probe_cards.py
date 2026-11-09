#!/usr/bin/env python3
"""Verify the MK/Ogerpon card behaviours against CardImpl.h and Top1 replays."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BOT = ROOT / "agents" / "mk_ogrepon" / "bot"
sys.path.insert(0, str(BOT))
sys.path.insert(0, str(ROOT / "tools"))

import mk_data as D  # noqa: E402
import mk_state as S  # noqa: E402
import replay_meta as rm  # noqa: E402

TEAM_ID = "17014244"
TEAM_NAME = "やる気元気ミワハルキ 2nd ver."


def active0(obs: dict, player: int):
    arr = ((obs.get("current") or {}).get("players") or [{}, {}])[player].get("active", [])
    return arr[0] if arr else None


def find_serial(obs: dict, player: int, serial: int):
    pl = ((obs.get("current") or {}).get("players") or [{}, {}])[player]
    for p in list(pl.get("active", [])) + list(pl.get("bench", [])):
        if p and p.get("serial") == serial:
            return p
    return None


def pokemon_at(obs: dict, player: int, area, index):
    pl = ((obs.get("current") or {}).get("players") or [{}, {}])[player]
    if area == 4:
        arr = pl.get("active", [])
    elif area == 5:
        arr = pl.get("bench", [])
    else:
        return None
    return arr[index] if index is not None and 0 <= index < len(arr) else None


def main() -> int:
    deck = [int(x) for x in (BOT / "deck.csv").read_text(encoding="utf-8").split()]
    cards = D._cards()
    attacks = D._attacks()

    stats = {
        "deck": {
            "cards": len(deck),
            "unique": len(set(deck)),
            "basic_pokemon_lines": sum(1 for cid in set(deck) if D.is_basic(cid)),
            "basic_pokemon_copies": sum(1 for cid in deck if D.is_basic(cid)),
            "ace_spec": [cid for cid in set(deck) if bool(getattr(cards.get(cid), "aceSpec", False))],
            "tera": [cid for cid in set(deck) if D.is_tera(cid)],
            "mega_ex": [cid for cid in set(deck) if D.is_mega(cid)],
        },
        "teal_dance": Counter(),
        "kang_draw": Counter(),
        "fez_draw": Counter(),
        "lillie_draw": Counter(),
        "area_zero": Counter(),
        "prism_energy": Counter(),
        "attacks": Counter(),
        "prize_values": {str(cid): D.prize_value(cid) for cid in sorted(set(deck)) if D.is_pokemon(cid)},
    }

    base = ROOT / "replays" / "top5_nov1" / TEAM_ID
    episodes = 0
    for path in sorted(base.glob("*.json")):
        try:
            episode = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        names = rm.team_names(episode)[:2]
        ours = [i for i, n in enumerate(names) if TEAM_NAME in str(n)]
        if len(ours) != 1 or rm.winner_index(episode) != ours[0]:
            continue
        me = ours[0]
        episodes += 1
        steps = episode["steps"]
        for player, si, obs, sel, action in rm.iter_decisions(episode):
            cur = obs.get("current") or {}
            if not cur:
                continue
            if player == me:
                for idx in action:
                    opts = sel.get("option") or []
                    if not (0 <= idx < len(opts)):
                        continue
                    o = opts[idx]
                    label = rm.option_label(obs, player, o)
                    if label.startswith("ABILITY Teal Mask") and si + 2 < len(steps):
                        p = pokemon_at(obs, me, o.get("area"), o.get("index"))
                        after = steps[si + 2][me].get("observation") or {}
                        q = find_serial(after, me, p.get("serial")) if p and after.get("current") else None
                        if p and q:
                            stats["teal_dance"][(
                                len(q.get("energies", [])) - len(p.get("energies", [])),
                                after["current"]["players"][me]["deckCount"]
                                - cur["players"][me]["deckCount"],
                            )] += 1
                    elif label.startswith("ABILITY Mega Kangaskhan"):
                        after = steps[si + 1][me].get("observation") or {}
                        if after.get("current"):
                            stats["kang_draw"][(
                                cur["players"][me]["deckCount"] - after["current"]["players"][me]["deckCount"],
                                after["current"]["players"][me]["handCount"] - cur["players"][me]["handCount"],
                            )] += 1
                    elif label.startswith("ABILITY Fezandipiti"):
                        after = steps[si + 1][me].get("observation") or {}
                        if after.get("current"):
                            stats["fez_draw"][cur["players"][me]["deckCount"]
                                             - after["current"]["players"][me]["deckCount"]] += 1
                    elif label.startswith("PLAY Lillie's Determination"):
                        after = steps[si + 1][me].get("observation") or {}
                        if after.get("current"):
                            bc = cur["players"][me]["deckCount"]
                            bh = cur["players"][me]["handCount"]
                            ac = after["current"]["players"][me]["deckCount"]
                            stats["lillie_draw"][(
                                len(cur["players"][me]["prize"]), bc + bh - 1 - ac,
                            )] += 1
                    elif label == "PLAY Area Zero Underdepths":
                        after = steps[si + 1][me].get("observation") or {}
                        if after.get("current"):
                            stats["area_zero"][(
                                cur["players"][me].get("benchMax"),
                                after["current"]["players"][me].get("benchMax"),
                                cur["players"][1 - me].get("benchMax"),
                                after["current"]["players"][1 - me].get("benchMax"),
                            )] += 1
                    elif label.startswith("ATTACK "):
                        aid = S.option_int(o, "attackId", None)
                        if aid is not None:
                            stats["attacks"][getattr(D.attack(aid), "name", str(aid))] += 1

                for pi in range(2):
                    pl = ((obs.get("current") or {}).get("players") or [{}, {}])[pi]
                    for p in list(pl.get("active", [])) + list(pl.get("bench", [])):
                        if not p:
                            continue
                        if 16 in [c["id"] for c in p.get("energyCards", [])]:
                            stats["prism_energy"][tuple(p.get("energies", []))] += 1

    out = {
        "episodes": episodes,
        "deck": stats["deck"],
        "teal_dance": {"uses": sum(stats["teal_dance"].values()),
                        "deltas": {str(k): v for k, v in stats["teal_dance"].items()}},
        "kang_draw": {"uses": sum(stats["kang_draw"].values()),
                       "deltas": {str(k): v for k, v in stats["kang_draw"].items()}},
        "fez_draw": {"uses": sum(stats["fez_draw"].values()),
                      "deltas": {str(k): v for k, v in stats["fez_draw"].items()}},
        "lillie_draw": {"uses": sum(stats["lillie_draw"].values()),
                         "deltas": {str(k): v for k, v in stats["lillie_draw"].items()}},
        "area_zero": {"uses": sum(stats["area_zero"].values()),
                       "deltas": {str(k): v for k, v in stats["area_zero"].items()}},
        "prism_energy": {"observations": sum(stats["prism_energy"].values()),
                          "types": {str(k): v for k, v in stats["prism_energy"].most_common(8)}},
        "attacks": dict(stats["attacks"].most_common()),
        "prize_values": stats["prize_values"],
    }
    report = ROOT / "agents" / "mk_ogrepon" / "reports" / "card_behavior.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
