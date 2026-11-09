#!/usr/bin/env python3
"""Build a leak-free dataset for the #2 ladder team's Grass/Ogerpon deck.

The selected list is the exact 60-card multiset used by
``やる気元気ミワハルキ 2nd ver.`` in the available top-team replays.
Only decisions made by that deck's pilot are kept.  Features are the shared
top2_core feature extractor with the ``grass`` profile.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CORE = Path(__file__).resolve().parents[1] / "top2_core"
TRAINING = Path(__file__).resolve().parent
for path in (ROOT / "vendor", CORE, TRAINING):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import top2_core_features as F  # noqa: E402
from build_dataset import iter_decisions, load_episodes, round_winners  # noqa: E402

DEFAULT_DECK = Path(__file__).resolve().parents[1] / "decks" / "grass_ogerpon_kangaskhan.csv"
DEFAULT_DIRS = [
    ROOT / "replays" / "top2_policy",
    ROOT / "replays" / "top2_more",
    ROOT / "replays" / "top5_nov1",
]
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "dataset_grass_ogerpon.npz"


def episode_decks(episode: dict) -> list[list[int]] | None:
    try:
        raw = episode["steps"][0][0]["visualize"][0]["action"]
        if not isinstance(raw, list) or len(raw) < 2:
            return None
        return [list(map(int, deck)) for deck in raw[:2]]
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def matching_players(episode: dict, deck: list[int]) -> list[int]:
    decks = episode_decks(episode)
    if not decks:
        return []
    target = Counter(deck)
    return [i for i, candidate in enumerate(decks) if Counter(candidate) == target]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Grass/Ogerpon BC dataset")
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
    splits: list[int] = []
    meta_episode: list[int] = []
    meta_step: list[int] = []
    meta_round: list[int] = []
    meta_turn: list[int] = []
    meta_player: list[int] = []
    meta_context: list[int] = []
    meta_selected_type: list[int] = []
    meta_max_count: list[int] = []
    meta_action_len: list[int] = []

    matched_episodes = 0
    matched_games = 0
    skipped = 0

    for eid, path in episodes.items():
        try:
            episode = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            continue
        players = matching_players(episode, deck)
        if not players:
            continue
        matched_episodes += 1
        winners = round_winners(episode)
        matched_games += len(winners)
        for step_index, player, obs, select, action in iter_decisions(episode):
            if player not in players:
                continue
            try:
                state = F.state_vector(obs, deck, profile="grass")
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
            selected_type = -1
            if label >= 0:
                selected_type = int(raw_options[label].get("type", -1) or -1)
            cur = obs.get("current") or {}
            rnd = int(cur.get("round", 0) or 0)
            winner = winners.get(rnd)
            win_value = 0.5 if winner is None else (1.0 if int(winner) == player else 0.0)

            states.append(state)
            options.append(option)
            card_ids.append(cids)
            masks.append(mask)
            labels.append(label)
            game_wins.append(win_value)
            splits.append(1 if eid in val_ids else 0)
            meta_episode.append(eid)
            meta_step.append(step_index)
            meta_round.append(rnd)
            meta_turn.append(int(cur.get("turn", 0) or 0))
            meta_player.append(player)
            meta_context.append(int((select or {}).get("context", -1) or -1))
            meta_selected_type.append(selected_type)
            meta_max_count.append(max_count)
            meta_action_len.append(len(action))

    n = len(states)
    if n == 0:
        print("no rows")
        return 1
    print(f"matched episodes: {matched_episodes}  games: {matched_games}  rows: {n}  skipped: {skipped}")

    state_arr = np.stack(states).astype(np.float16)
    option_arr = np.stack(options).astype(np.float16)
    card_arr = np.stack(card_ids).astype(np.int16)
    mask_arr = np.stack(masks).astype(np.float32)
    label_arr = np.asarray(labels, dtype=np.int64)
    win_arr = np.asarray(game_wins, dtype=np.float32)
    split_arr = np.asarray(splits, dtype=np.int8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
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
    )
    manifest = {
        "deck": str(args.deck),
        "episodes": matched_episodes,
        "games": matched_games,
        "rows": int(n),
        "train_rows": int((split_arr == 0).sum()),
        "val_rows": int((split_arr == 1).sum()),
        "val_episodes": len(val_ids),
        "state": int(F.STATE_FEATURE_SIZE),
        "option": int(F.OPTION_FEATURE_SIZE),
        "max_options": int(F.MAX_OPTIONS),
        "policy_rows": int(((label_arr >= 0) & (win_arr >= 0.5)).sum()),
        "profile": "grass",
        "dirs": [str(p) for p in args.dirs],
    }
    (args.out.parent / "dataset_grass_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024 / 1024:.1f} MiB)")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
