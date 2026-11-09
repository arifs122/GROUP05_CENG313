#!/usr/bin/env python3
"""Label the first decision that likely started a losing Lucario game."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
LUCARIO = ROOT / "agents" / "mk_ogrepon" / "lucario"
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(LUCARIO))
os.environ.setdefault("PTCG_BELIEF_LOG", "0")

import lucario_agent as L  # noqa: E402
import mk_state as S  # noqa: E402
import mk_data as D  # noqa: E402
import replay_meta as rm  # noqa: E402
from arena import load_bot, make_agent  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

ATTACKERS = {"Mega Lucario ex", "Hariyama", "Riolu", "Makuhita"}


def _active(obs, p):
    a = S.active(obs, p)
    return D.card_name(S.pokemon_id(a)) if a is not None else ""


def _bench(obs, p):
    return [D.card_name(S.pokemon_id(x)) for x in S.bench(obs, p) if S.pokemon_id(x)]


def _charged_bench(obs, me):
    out = []
    for p in S.bench(obs, me):
        cid = S.pokemon_id(p)
        if cid is not None and D.card_name(cid) in ATTACKERS and S.energy_count(p) >= 1:
            out.append(D.card_name(cid))
    return out


def _label_decision(obs, action, state, pending_mega):
    me = S.you(obs)
    labels = []
    opts = S.options(obs)
    for i in action:
        if not (0 <= i < len(opts)):
            continue
        o = opts[i]
        lab = rm.option_label(obs, me, o)
        otype = S.option_type(o)
        actor, target = S.option_cards(obs, o)
        actor_name = D.card_name(actor) if actor is not None else ""
        target_name = D.card_name(target) if target is not None else ""

        if otype == S.O_ATTACH:
            if "Hero" in actor_name and target_name not in ("Mega Lucario ex", "Hariyama"):
                labels.append(("cape_wrong", lab, target_name))
            elif actor_name == "Basic {F} Energy" and target_name not in ATTACKERS:
                labels.append(("wrong_energy", lab, target_name))

        if lab.startswith("PLAY Boss") or lab.startswith("PLAY Boss’s"):
            t, ko, _prize = L._best_boss_target(obs)
            if not t or (not ko and t not in L.SETUP_TARGETS):
                labels.append(("boss_waste", lab, t))

        if otype == S.O_RETREAT or lab == "PLAY Switch":
            if _active(obs, me) in ATTACKERS and not _charged_bench(obs, me):
                labels.append(("retreat_error", lab, _active(obs, me)))

        if otype == S.O_ATTACK:
            aid = S.option_int(o, "attackId", None)
            if aid == 983:  # Mega Brave
                pending_mega.append(True)

    if pending_mega and state.get("turn", 0) > state.get("_mega_turn", 0):
        pass
    return labels


def play_one(obs, opp_agent, max_steps=5000):
    me = 0
    history = []
    trace = []
    pending_mega = False
    mega_turn = -1
    late_checked = False
    search_count = 0
    mega_seen = False
    for _ in range(max_steps):
        cur = obs["current"]
        if cur["result"] >= 0:
            return int(cur["result"]), history, trace
        sel = obs["select"]
        if sel is None:
            return None, history, trace
        p = int(cur["yourIndex"])
        if p == 0:
            if int(cur["turn"]) >= 5 and not mega_seen and _active(obs, 0) in ("Riolu", "Makuhita"):
                hand_names = [D.card_name(S.card_id(c)) for c in (S.hand(obs, 0) or [])]
                if "Mega Lucario ex" in hand_names:
                    history.append({"turn": int(cur["turn"]), "kind": "delayed_mega",
                                    "detail": f"active={_active(obs,0)} bench={_bench(obs,0)}",
                                    "action": []})
                mega_seen = True
            opts0 = S.options(obs)
            a0 = _active(obs, 0)
            trace.append({
                "turn": int(cur["turn"]),
                "action": [rm.option_label(obs, 0, o) for o in opts0 if False],
                "active": a0,
                "energy": S.energy_count(S.active(obs,0)) if S.active(obs,0) else 0,
                "bench": _bench(obs, 0),
                "hand": S.hand_count(obs, 0),
            })
            # Check development once at the start of our second turn (turn 4).
            if not late_checked and int(cur["turn"]) >= 4:
                late_checked = True
                if len(_bench(obs, 0)) < 2:
                    history.append({
                        "turn": int(cur["turn"]), "kind": "late_bench",
                        "detail": f"bench={_bench(obs, 0)}", "action": [],
                    })
            if pending_mega and int(cur["turn"]) > mega_turn:
                # If Mega Brave was used and we have no attack option now.
                opts = S.options(obs)
                has_attack = any(S.option_type(o) == S.O_ATTACK for o in opts)
                if not has_attack and S.active(obs, 0) is not None:
                    history.append({
                        "turn": int(cur["turn"]), "kind": "mega_brave_lock",
                        "detail": f"active={_active(obs,0)} energy={S.energy_count(S.active(obs,0))}",
                        "action": [],
                    })
                pending_mega = False
        action = L.agent(obs) if p == 0 else opp_agent(obs)
        if p == 0:
            opts = S.options(obs)
            for i in action:
                if 0 <= i < len(opts):
                    lab = rm.option_label(obs, 0, opts[i])
                    o = opts[i]
                    if S.option_type(o) == S.O_ATTACK and S.option_int(o, "attackId", None) == 983:
                        pending_mega = True
                        mega_turn = int(cur["turn"])
            chosen_labels = []
            for i in action:
                if 0 <= i < len(opts):
                    chosen_labels.append(rm.option_label(obs, 0, opts[i]))
                    if "Mega Lucario ex" in chosen_labels[-1]:
                        mega_seen = True
                    if ("Poké Pad" in chosen_labels[-1] or "Fighting Gong" in chosen_labels[-1]
                            or "Ultra Ball" in chosen_labels[-1]):
                        search_count += 1
            if int(cur["turn"]) <= 5 and search_count >= 7:
                history.append({"turn": int(cur["turn"]), "kind": "search_spam",
                                "detail": f"search_plays={search_count}", "action": []})
            if trace:
                trace[-1]["chosen"] = chosen_labels
            history.extend({"turn": int(cur["turn"]), "kind": k[0], "detail": f"{k[1]} -> {k[2]}"}
                           for k in _label_decision(obs, action, {"turn": int(cur["turn"])}, []))
        obs = battle_select(action)
    return None, history, trace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--losses", type=int, default=40)
    parser.add_argument("--deck", type=Path, default=LUCARIO / "deck_974_pokepad.csv")
    parser.add_argument("--out", type=Path, default=LUCARIO / "reports" / "loss_analysis.json")
    args = parser.parse_args()

    deck = [int(x) for x in args.deck.read_text(encoding="utf-8").split() if x.strip()]
    L._DECK = deck
    pool = []
    for path in [
        "bots/_template", "bots/dragapult",
        "benchmarks/external_agents/alakazam", "bots/banette",
        "benchmarks/external_agents/lucario_mega_hariyama",
        "benchmarks/external_agents/bellibolt",
        "benchmarks/meta_decks/mega_abomasnow",
    ]:
        try:
            pool.append(load_bot(ROOT / path))
        except Exception:
            pass

    losses = []
    counts = Counter()
    examples = []
    games = 0
    for opp in pool:
        if len(losses) >= args.losses:
            break
        for g in range(max(20, args.games // len(pool))):
            if len(losses) >= args.losses:
                break
            obs, sd = battle_start(deck, opp.deck, g % 2 == 1)
            if obs is None:
                continue
            result, history, trace = play_one(obs, opp.agent)
            battle_finish()
            games += 1
            if result != 1:
                continue
            # Pick the first meaningful mistake; late bench is lowest priority
            # if a concrete decision error follows it.
            order = {"cape_wrong": 0, "wrong_energy": 1, "boss_waste": 2,
                     "retreat_error": 3, "delayed_mega": 4, "search_spam": 5,
                     "mega_brave_lock": 6, "late_bench": 7}
            history.sort(key=lambda h: (int(h["turn"]), order.get(h["kind"], 9)))
            first = history[0]["kind"] if history else "other"
            counts[first] += 1
            losses.append({"opponent": opp.name, "result": result, "first": first,
                           "history": history[:8], "trace": trace[:20]})
            if len(examples) < 8:
                examples.append(losses[-1])

    report = {
        "games": games,
        "losses": len(losses),
        "deck": str(args.deck),
        "first_mistake_counts": dict(counts.most_common()),
        "examples": examples,
        "top3": [k for k, _ in counts.most_common(3)],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("games", "losses", "first_mistake_counts", "top3")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
