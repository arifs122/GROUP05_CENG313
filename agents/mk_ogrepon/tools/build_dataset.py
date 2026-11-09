#!/usr/bin/env python3
"""Replay -> public-state value dataset for the MK/Ogerpon agent.

Only observations and the final match result are used.  Hidden opponent hand,
prizes, deck order, `visualize`, and `search_begin_input` never enter features.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BOT = ROOT / "agents" / "mk_ogrepon" / "bot"
sys.path.insert(0, str(BOT))
sys.path.insert(0, str(ROOT / "tools"))

import mk_features as F  # noqa: E402
import replay_meta as rm  # noqa: E402

TEAM_ID = "17014244"
TEAM_NAME = "やる気元気ミワハルキ 2nd ver."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "replays" / "top5_nov1" / TEAM_ID)
    parser.add_argument("--out", type=Path, default=ROOT / "agents" / "mk_ogrepon" / "data" / "value_dataset.npz")
    args = parser.parse_args()

    states: list[list[float]] = []
    labels: list[float] = []
    weights: list[float] = []
    episodes: list[int] = []
    steps: list[int] = []
    rounds: list[int] = []
    players: list[int] = []
    types: list[int] = []
    contexts: list[int] = []

    files = sorted(args.base.glob("*.json"))
    used = 0
    for path in files:
        try:
            episode = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        names = rm.team_names(episode)[:2]
        ours = [i for i, n in enumerate(names) if TEAM_NAME in str(n)]
        if len(ours) != 1:
            continue
        winner = rm.winner_index(episode)
        if winner is None:
            continue
        used += 1
        eid = int((episode.get("info") or {}).get("EpisodeId") or 0)
        for player, si, obs, sel, _action in rm.iter_decisions(episode):
            if sel is None:
                continue
            try:
                state = F.state_features(obs)
            except Exception:
                continue
            cur = obs.get("current") or {}
            states.append(state)
            # Match outcome from this player's perspective.
            labels.append(1.0 if player == winner else 0.0)
            weights.append(1.0)
            episodes.append(eid)
            steps.append(int(si))
            rounds.append(int(cur.get("round", 1) or 1))
            players.append(int(player))
            types.append(int(sel.get("type", -1)))
            contexts.append(int(sel.get("context", -1)))

    if len(states) < 100:
        raise SystemExit(f"too few samples: {len(states)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        state=np.asarray(states, dtype=np.float32),
        label=np.asarray(labels, dtype=np.float32),
        weight=np.asarray(weights, dtype=np.float32),
        episode=np.asarray(episodes, dtype=np.int64),
        step=np.asarray(steps, dtype=np.int64),
        round=np.asarray(rounds, dtype=np.int64),
        player=np.asarray(players, dtype=np.int8),
        type=np.asarray(types, dtype=np.int64),
        context=np.asarray(contexts, dtype=np.int64),
    )
    print(f"episodes={used} samples={len(states)} -> {args.out} "
          f"({args.out.stat().st_size/1024:.1f} KB)")
    print(f"label mean={np.mean(labels):.3f} players={np.bincount(players).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
