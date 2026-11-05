"""Audit human Slowking-vs-Dragapult replays: what do winners do differently?

Usage: python tools/mazlum/slowking_matchup_audit.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "mazlum"))
sys.path.insert(0, str(ROOT / "vendor"))

import features as F  # noqa: E402
from cg.api import all_card_data  # noqa: E402

TARGET = "Slowking-Kangaskhan"
OPP = "Dragapult"
CLEFAIRY = 272

names = {int(card.cardId): card.name for card in all_card_data()}
manifest = json.loads((ROOT / "replays" / "_collected.json").read_text(encoding="utf-8"))

stats = {
    "win": Counter(),
    "loss": Counter(),
}
episodes = 0
wins = 0

for entry in manifest["episodes"]:
    arch = entry.get("archetypes") or []
    if TARGET not in arch or OPP not in arch:
        continue
    side = arch.index(TARGET)
    winner = entry.get("winner_side")
    if winner is None:
        continue
    bucket = "win" if int(winner) == side else "loss"
    path = ROOT / "replays" / entry["file"]
    if not path.is_file():
        continue
    try:
        episode = F.load_episode(path)
        decisions = list(F.parse_episode(episode))
    except Exception:
        continue

    played = set()
    attacked = set()
    turns = 0
    for decision, _state, _opts, _mask, _ids in decisions:
        if int(decision.player) != side:
            continue
        obs = decision.observation
        turns = max(turns, int((obs.get("current") or {}).get("turn") or 0))
        select = obs.get("select") or {}
        options = select.get("option") or []
        if not decision.action:
            continue
        index = int(decision.action[0])
        if not 0 <= index < len(options):
            continue
        opt = options[index]
        otype = int(opt.get("type", -1))
        try:
            actor, _target = F._option_cards(obs, side, opt)
        except Exception:
            actor = None
        if otype == 7 and actor is not None:
            played.add(int(actor))
        elif otype == 13 and actor is not None:
            attacked.add(int(actor))

    episodes += 1
    if bucket == "win":
        wins += 1
    stats[bucket]["games"] += 1
    stats[bucket]["turns"] += turns
    stats[bucket]["clefairy_played"] += int(CLEFAIRY in played)
    stats[bucket]["clefairy_attacked"] += int(CLEFAIRY in attacked)
    stats[bucket]["kyurem_attacked"] += int(1005 in attacked or 144 in attacked)
    for cid in played:
        stats[bucket][f"card:{names.get(cid, cid)}"] += 1
    for cid in attacked:
        stats[bucket][f"attacker:{names.get(cid, cid)}"] += 1

print(f"episodes: {episodes} | Slowking wins: {wins} ({100 * wins / max(episodes, 1):.0f}%)")
for bucket in ("win", "loss"):
    s = stats[bucket]
    g = max(s["games"], 1)
    print(f"\n[{bucket}] games={s['games']} avg_turns={s['turns'] / g:.0f}")
    print(f"  Clefairy played:   {s['clefairy_played']}/{g} ({100 * s['clefairy_played'] / g:.0f}%)")
    print(f"  Clefairy attacked: {s['clefairy_attacked']}/{g} ({100 * s['clefairy_attacked'] / g:.0f}%)")
    print(f"  Kyurem attacked:   {s['kyurem_attacked']}/{g} ({100 * s['kyurem_attacked'] / g:.0f}%)")
    cards = [(k[5:], v) for k, v in s.items() if k.startswith("card:")]
    cards.sort(key=lambda kv: -kv[1])
    print("  top plays:", ", ".join(f"{n} {v}" for n, v in cards[:10]))
    atk = [(k[9:], v) for k, v in s.items() if k.startswith("attacker:")]
    atk.sort(key=lambda kv: -kv[1])
    print("  attackers:", ", ".join(f"{n} {v}" for n, v in atk[:8]))
