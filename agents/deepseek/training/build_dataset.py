#!/usr/bin/env python3
"""Build a leak-free Top2/Petit Canard dataset from public replays.

Rules:
* only the recorded Petit Canard observations are used;
* only fields that an agent actually receives are featurised;
* ``visualize``, ``search_begin_input``, opponent hand/prize identities and
  ``logs`` are never read as features;
* labels are the winner of the corresponding game (round), not the match;
* the validation split is episode-based.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CORE = Path(__file__).resolve().parents[1] / "top2_core"
for path in (ROOT / "vendor", CORE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import top2_core_features as F  # noqa: E402

TEAM = "Petit Canard"
DEFAULT_DIRS = [
    ROOT / "replays" / "top2_policy",
    ROOT / "replays" / "top2_more",
    ROOT / "replays" / "top5_nov1",
]
DEFAULT_DECK = ROOT / "deck_top2.csv"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "dataset_top2.npz"


def team_index(episode: dict, team: str = TEAM) -> int | None:
    names = (episode.get("info") or {}).get("TeamNames") or []
    for i, name in enumerate(names):
        if str(name) == team:
            return i
    return None


def match_winner(episode: dict) -> int | None:
    rewards = episode.get("rewards") or []
    if len(rewards) >= 2 and rewards[0] is not None and rewards[1] is not None:
        if rewards[0] > rewards[1]:
            return 0
        if rewards[1] > rewards[0]:
            return 1
    return None


def round_winners(episode: dict) -> dict[int, int | None]:
    """Map ``current.round`` to the winner of that game.

    The replay exposes ``players[p].win`` at the start of the next round; the
    difference between two round-start snapshots identifies the previous game's
    winner.  The final game uses the match reward.
    """
    starts: dict[int, list[int]] = {}
    for step in episode.get("steps") or []:
        for agent in step[:2]:
            cur = ((agent.get("observation") or {}).get("current"))
            if not cur:
                continue
            try:
                r = int(cur.get("round", 0) or 0)
            except (TypeError, ValueError):
                continue
            if r not in starts:
                starts[r] = [int((p or {}).get("win", 0) or 0) for p in (cur.get("players") or [])][:2]
                while len(starts[r]) < 2:
                    starts[r].append(0)

    rounds = sorted(starts)
    mw = match_winner(episode)
    out: dict[int, int | None] = {}
    for i, r in enumerate(rounds):
        if i + 1 < len(rounds):
            a = starts[r]
            b = starts[rounds[i + 1]]
            if b[0] > a[0]:
                out[r] = 0
            elif b[1] > a[1]:
                out[r] = 1
            else:
                out[r] = None
        else:
            out[r] = mw
    return out


def iter_decisions(episode: dict):
    """Yield (step_index, player, observation, select, action)."""
    steps = episode.get("steps") or []
    for step_index, step in enumerate(steps[:-1]):
        for player, agent in enumerate(step[:2]):
            if agent.get("status") != "ACTIVE":
                continue
            obs = agent.get("observation") or {}
            select = obs.get("select")
            if not select:
                continue
            action = steps[step_index + 1][player].get("action") or []
            if isinstance(action, dict):
                action = next((v for v in action.values() if isinstance(v, list)), [])
            if not isinstance(action, list):
                action = []
            yield step_index, player, obs, select, action


def load_episodes(dirs: list[Path]) -> dict[int, Path]:
    episodes: dict[int, Path] = {}
    for directory in dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("**/*replay.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            eid = int((data.get("info") or {}).get("EpisodeId") or -1)
            if eid >= 0:
                episodes.setdefault(eid, path)
    return episodes


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Top2 value/policy dataset")
    parser.add_argument("--dirs", type=Path, nargs="+", default=DEFAULT_DIRS)
    parser.add_argument("--deck", type=Path, default=DEFAULT_DECK)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    deck = [int(x) for x in args.deck.read_text(encoding="utf-8").split() if x.strip()]
    if len(deck) != 60:
        raise SystemExit(f"deck must contain 60 cards: {len(deck)}")
    episodes = load_episodes(args.dirs)
    if args.limit:
        episodes = dict(list(episodes.items())[: args.limit])
    print(f"unique episodes: {len(episodes)}")
    if not episodes:
        return 1

    rng = random.Random(args.seed)
    episode_ids = list(episodes)
    rng.shuffle(episode_ids)
    val_count = max(1, int(len(episode_ids) * args.val_fraction))
    val_ids = set(episode_ids[:val_count])
    print(f"val episodes: {len(val_ids)}")

    states: list[np.ndarray] = []
    options: list[np.ndarray] = []
    card_ids: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    game_wins: list[float] = []
    split: list[int] = []
    meta_episode: list[int] = []
    meta_step: list[int] = []
    meta_round: list[int] = []
    meta_turn: list[int] = []
    meta_player: list[int] = []
    meta_context: list[int] = []
    meta_selected_type: list[int] = []
    meta_max_count: list[int] = []
    meta_action_len: list[int] = []
    meta_source: list[int] = []

    source_by_dir = {str(p.resolve()): i for i, p in enumerate(args.dirs)}
    total_games = 0
    skipped = 0

    for eid, path in episodes.items():
        try:
            episode = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            continue
        me = team_index(episode)
        if me is None:
            skipped += 1
            continue
        winners = round_winners(episode)
        total_games += len(winners)
        source = source_by_dir.get(str(path.parent.resolve()), 0)
        for step_index, player, obs, select, action in iter_decisions(episode):
            if player != me:
                continue
            try:
                state = F.state_vector(obs, deck)
                option, mask = F.options_matrix(obs)
                cids = F.card_ids_matrix(obs)
            except Exception:
                skipped += 1
                continue
            if int(mask.sum()) <= 0:
                continue
            raw_options = list((select or {}).get("option") or [])
            max_count = int((select or {}).get("maxCount") or 0)
            label = -1
            if max_count == 1 and len(action) == 1:
                idx = int(action[0])
                if 0 <= idx < min(len(raw_options), F.MAX_OPTIONS):
                    label = idx
                else:
                    label = -1
            selected_type = -1
            if label >= 0:
                selected_type = int(raw_options[label].get("type", -1) or -1)
            cur = obs.get("current") or {}
            rnd = int(cur.get("round", 0) or 0)
            winner = winners.get(rnd)
            if winner is None:
                # Draw or unresolved round: keep it as a soft value target.
                win_value = 0.5
            else:
                win_value = 1.0 if int(winner) == me else 0.0

            states.append(state)
            options.append(option)
            card_ids.append(cids)
            masks.append(mask)
            labels.append(label)
            game_wins.append(win_value)
            split.append(1 if eid in val_ids else 0)
            meta_episode.append(eid)
            meta_step.append(step_index)
            meta_round.append(rnd)
            meta_turn.append(int(cur.get("turn", 0) or 0))
            meta_player.append(player)
            meta_context.append(int((select or {}).get("context", -1) or -1))
            meta_selected_type.append(selected_type)
            meta_max_count.append(max_count)
            meta_action_len.append(len(action))
            meta_source.append(source)

    n = len(states)
    if n == 0:
        print("no rows")
        return 1
    print(f"rows: {n}  skipped: {skipped}  games: {total_games}")
    state_arr = np.stack(states).astype(np.float16)
    option_arr = np.stack(options).astype(np.float16)
    card_arr = np.stack(card_ids).astype(np.int16)
    mask_arr = np.stack(masks).astype(np.float32)
    label_arr = np.asarray(labels, dtype=np.int64)
    win_arr = np.asarray(game_wins, dtype=np.float32)
    split_arr = np.asarray(split, dtype=np.int8)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        state=state_arr,
        options=option_arr,
        card_ids=card_arr,
        mask=mask_arr,
        label=label_arr,
        game_win=win_arr,
        split=split_arr,
        episode_id=np.asarray(meta_episode, dtype=np.int64),
        step_index=np.asarray(meta_step, dtype=np.int32),
        round=np.asarray(meta_round, dtype=np.int16),
        turn=np.asarray(meta_turn, dtype=np.int16),
        player=np.asarray(meta_player, dtype=np.int8),
        context=np.asarray(meta_context, dtype=np.int16),
        selected_type=np.asarray(meta_selected_type, dtype=np.int16),
        max_count=np.asarray(meta_max_count, dtype=np.int16),
        action_length=np.asarray(meta_action_len, dtype=np.int16),
        source=np.asarray(meta_source, dtype=np.int8),
    )
    manifest = {
        "episodes": len(episodes),
        "games": total_games,
        "rows": int(n),
        "train_rows": int((split_arr == 0).sum()),
        "val_rows": int((split_arr == 1).sum()),
        "val_episodes": len(val_ids),
        "state": int(F.STATE_FEATURE_SIZE),
        "option": int(F.OPTION_FEATURE_SIZE),
        "max_options": int(F.MAX_OPTIONS),
        "policy_rows": int(((label_arr >= 0) & (win_arr >= 0.5)).sum()),
        "deck": str(args.deck),
        "dirs": [str(p) for p in args.dirs],
    }
    (out.parent / "dataset_top2_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"wrote {out} ({out.stat().st_size / 1024 / 1024:.1f} MiB)")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
