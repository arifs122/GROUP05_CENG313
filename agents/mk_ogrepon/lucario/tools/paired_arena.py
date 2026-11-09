#!/usr/bin/env python3
"""Fork-paired local arena.

The competition engine's `BattleStart` uses `std::random_device` and does not
expose a seed.  We therefore start one deal in the parent and `fork()` the
process once per candidate: every child inherits the exact deck order, prizes,
hidden cards and C++ RNG state.  This gives true paired comparisons for policy
candidate weights on the same deal.

For deck-variant comparisons the deck is fixed at `BattleStart`, so those must
still be run as separate seat-swapped series; this tool is for policy/weights.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "agents" / "mk_ogrepon" / "lucario"))

import lucario_agent as L  # noqa: E402
from arena import is_valid, load_bot, load_deck  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from cg.sim import Battle  # noqa: E402

# ApiBattleStart sets config.recordLog=true, config.deviceRand=true; the latter
# makes every coin flip use std::random_device and breaks fork pairing.
# Absolute offset: sizeof(Game)=8 before config + offsetof(GameConfig, deviceRand)=555.
_DEVICE_RAND_OFFSET = 563


def _disable_device_rand():
    try:
        ctypes.c_ubyte.from_address(int(Battle.battle_ptr) + _DEVICE_RAND_OFFSET).value = 0
    except Exception:
        pass


def _child_play(obs, agent0, agent1, max_steps):
    try:
        for _ in range(max_steps):
            cur = obs["current"]
            if cur["result"] >= 0:
                return {"winner": int(cur["result"]), "error": None}
            sel = obs["select"]
            if sel is None:
                return {"winner": None, "error": "unexpected select None"}
            p = int(cur["yourIndex"])
            action = (agent0 if p == 0 else agent1)(obs)
            if not is_valid(action, sel):
                return {"winner": 1 - p, "error": "invalid action"}
            obs = battle_select(action)
        return {"winner": None, "error": "max steps"}
    except Exception as exc:  # noqa: BLE001
        return {"winner": None, "error": f"{type(exc).__name__}: {exc}"}


def paired_deal(deck0, deck1, reverse, candidates, agent1, max_steps=5000,
                candidate_player=0):
    """Start one deal and evaluate every candidate policy on a fork of it."""
    obs, start = battle_start(deck0, deck1, reverse)
    if obs is None:
        return None
    _disable_device_rand()
    results = []
    try:
        for cand in candidates:
            r, w = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(r)
                try:
                    # Candidate-specific global state must be set after fork.
                    L.W.update(cand.get("weights") or {})
                    if cand.get("deck"):
                        L._DECK = list(cand["deck"])
                    ca = cand.get("agent") or L.agent
                    a0 = ca if candidate_player == 0 else agent1
                    a1 = agent1 if candidate_player == 0 else ca
                    result = _child_play(obs, a0, a1, max_steps)
                except Exception as exc:  # noqa: BLE001
                    result = {"winner": None, "error": f"child {type(exc).__name__}: {exc}"}
                try:
                    os.write(w, json.dumps(result).encode("utf-8"))
                finally:
                    os.close(w)
                    os._exit(0)
            else:
                os.close(w)
                with os.fdopen(r, "rb") as handle:
                    raw = handle.read()
                try:
                    os.waitpid(pid, 0)
                except ChildProcessError:
                    pass
                try:
                    results.append(json.loads(raw.decode("utf-8")))
                except Exception:
                    results.append({"winner": None, "error": "child decode"})
    finally:
        battle_finish()
    return results


def paired_series(candidates, opponent_path, games=40, max_steps=5000,
                  deck_path: Path | None = None, seed_side=0):
    opponent = load_bot(ROOT / opponent_path)
    deck = list(L.load_deck()) if deck_path is None else load_deck(deck_path)
    L._DECK = list(deck)
    per = [[] for _ in candidates]
    for g in range(games):
        # Alternate the candidate's seat, and the engine's turn-order choice.
        candidate_as_p0 = (g % 2 == 0)
        reverse = (g // 2) % 2 == 1
        if candidate_as_p0:
            res = paired_deal(deck, opponent.deck, reverse, candidates,
                              opponent.agent, max_steps, candidate_player=0)
            if res is None:
                continue
            for i, r in enumerate(res):
                per[i].append(0 if r.get("winner") == 0 else 1 if r.get("winner") == 1 else 2)
        else:
            res = paired_deal(opponent.deck, deck, reverse, candidates,
                              opponent.agent, max_steps, candidate_player=1)
            if res is None:
                continue
            for i, r in enumerate(res):
                per[i].append(0 if r.get("winner") == 1 else 1 if r.get("winner") == 0 else 2)
    return per


def summarize(per):
    n = len(per[0]) if per else 0
    wins = sum(1 for x in per[0] if x == 0) if per else 0
    return {"n": n, "wins": wins, "win_rate": wins / max(1, n)}


def paired_report(per):
    """Paired difference between candidate 1 and candidate 0 per deal."""
    if len(per) < 2:
        return {}
    diffs = []
    a_win = b_win = both = 0
    for a, b in zip(per[0], per[1]):
        if a == 2 or b == 2:
            continue
        diffs.append((1 if b == 0 else 0) - (1 if a == 0 else 0))
        a_win += (a == 0)
        b_win += (b == 0)
        both += (a == 0 and b == 0)
    if not diffs:
        return {}
    n = len(diffs)
    m = mean(diffs)
    var = sum((d - m) ** 2 for d in diffs) / max(1, n - 1)
    se = (var / n) ** 0.5 if n > 1 else 0.0
    return {
        "n": n,
        "candidate0_wins": a_win,
        "candidate1_wins": b_win,
        "both_wins": both,
        "mean_diff": round(m, 4),
        "se": round(se, 4),
        "z": round(m / se, 3) if se > 0 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--deck", type=Path, default=ROOT / "agents" / "mk_ogrepon" / "lucario" / "deck.csv")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--candidate", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base_w = json.loads(args.baseline.read_text()) if args.baseline else {}
    cand_w = json.loads(args.candidate.read_text()) if args.candidate else dict(base_w)
    candidates = [{"label": "baseline", "weights": base_w},
                  {"label": "candidate", "weights": cand_w}]
    per = paired_series(candidates, args.opponent, args.games, deck_path=args.deck)
    payload = {
        "opponent": args.opponent,
        "deck": str(args.deck) if args.deck else "default",
        "candidates": candidates,
        "series": per,
        "summary": [summarize([p]) for p in per],
        "paired": paired_report(per),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(json.dumps(payload["paired"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
