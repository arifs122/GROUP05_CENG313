"""Mazlum — build the Dragapult "brain transplant" dataset from replays.

For every mined episode that contains **Dragapult**:

* Dragapult won  -> keep Dragapult's own decisions (the winner's side)
* Dragapult lost -> keep the opponent's decisions (the anti-meta winner)

Rows are extracted directly from the replay files, so newly collected episodes
are picked up without rebuilding the full Banette dataset. The split is
episode-based, so no game leaks across train/val.

Usage::

    python tools/mazlum/build_dragapult_dataset.py
    python tools/mazlum/build_dragapult_dataset.py --target-episodes 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src" / "mazlum",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import features as F  # noqa: E402
from build_dataset import assign_split  # noqa: E402

MANIFEST = ROOT / "replays" / "_collected.json"
REPLAYS = ROOT / "replays"
OUT = ROOT / "data" / "mazlum" / "dataset_dragapult.npz"
TARGET = "Dragapult"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Dragapult dataset from replays")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--replays", type=Path, default=REPLAYS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--target", default=TARGET)
    parser.add_argument(
        "--opponent",
        default=None,
        help="keep only episodes whose other side is this archetype (non-mirror)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="max episodes (0 = all)")
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="only recompute the sample-weight source column on an existing npz",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    episodes = manifest.get("episodes") or []

    if args.weights_only:
        weight_map: dict[int, int] = {}
        for entry in episodes:
            archetypes = entry.get("archetypes") or []
            winner = entry.get("winner_side")
            if winner is None or len(archetypes) < 2 or args.target not in archetypes:
                continue
            weight_map[int(entry["episode_id"])] = 0 if archetypes[winner] == args.target else 2
        if not weight_map:
            print("error: no episodes with", args.target)
            return 1
        with np.load(args.out) as data:
            arrays = {key: data[key] for key in data.files}
        episode_ids = arrays["episode_id"]
        keys = np.array(sorted(weight_map), dtype=np.int64)
        values = np.array([weight_map[int(key)] for key in keys], dtype=np.int8)
        positions = np.clip(np.searchsorted(keys, episode_ids), 0, len(keys) - 1)
        matched = keys[positions] == episode_ids
        arrays["source"] = np.where(matched, values[positions], 0).astype(np.int8)
        np.savez_compressed(args.out, **arrays)
        rows = len(arrays["label"])
        print(
            f"weights-only: rows={rows} anti-meta={int((arrays['source'] == 2).sum())} "
            f"dragapult-wins={int((arrays['source'] == 0).sum())}"
        )
        print(f"wrote: {args.out}")
        return 0

    selected = []
    for entry in episodes:
        archetypes = entry.get("archetypes") or []
        winner = entry.get("winner_side")
        if winner is None or len(archetypes) < 2:
            continue
        if args.target in archetypes:
            if args.opponent:
                if archetypes.count(args.target) != 1:
                    continue
                if args.opponent not in archetypes:
                    continue
            selected.append(entry)
    if args.limit:
        selected = selected[: args.limit]

    print(f"episodes with {args.target}: {len(selected)}")
    if not selected:
        print("error: no matching episodes")
        return 1

    states, options, card_ids_list, masks, labels = [], [], [], [], []
    sources: list[int] = []
    episode_ids, step_indices, players = [], [], []
    wins = losses = skipped = 0

    for entry in selected:
        path = args.replays / entry["file"]
        if not path.is_file():
            skipped += 1
            continue
        try:
            episode = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skipped += 1
            continue

        winner = int(entry["winner_side"])
        # If Dragapult is the winner these are its own moves; otherwise the
        # winner is the anti-meta deck that beat Dragapult.
        anti_meta = (entry.get("archetypes") or [])[winner] != args.target
        if anti_meta:
            losses += 1
        else:
            wins += 1

        try:
            decisions = list(F.parse_episode(episode, players={winner}))
        except Exception:  # noqa: BLE001
            skipped += 1
            continue

        for decision, state, option_matrix, mask, card_ids in decisions:
            if not F.has_board(decision.observation):
                continue
            valid = int(mask.sum())
            if valid == 0 or not decision.action:
                continue
            label = int(decision.action[0])
            if not 0 <= label < valid:
                continue
            states.append(state.astype(np.float16))
            options.append(option_matrix.astype(np.float16))
            card_ids_list.append(card_ids.astype(np.int16))
            masks.append(mask)
            labels.append(label)
            episode_ids.append(int(entry["episode_id"]))
            step_indices.append(decision.step_index)
            players.append(decision.player)
            # Anti-meta rows: the opponent's moves that beat Dragapult get 2.0x
            # weight; Dragapult's own winning moves stay at 1.0x.
            sources.append(2 if anti_meta else 0)

    if not labels:
        print("error: no rows extracted")
        return 1

    print(f"{args.target} wins={wins} losses={losses} skipped={skipped}")
    print(f"rows: {len(labels)}")

    arrays = {
        "state": np.stack(states),
        "options": np.stack(options),
        "card_ids": np.stack(card_ids_list),
        "mask": np.stack(masks).astype(np.float32),
        "label": np.asarray(labels, dtype=np.int64),
        "episode_id": np.asarray(episode_ids, dtype=np.int64),
        "step_index": np.asarray(step_indices, dtype=np.int32),
        "player": np.asarray(players, dtype=np.int8),
    }
    split, val_ids = assign_split(arrays["episode_id"], args.val_fraction, args.seed)
    arrays["split"] = split
    arrays["source"] = np.asarray(sources, dtype=np.int8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)

    rows = len(labels)
    print(f"train: {int((split == 0).sum())}  val: {int((split == 1).sum())}  val episodes: {val_ids.size}")
    print(f"episodes: {len(np.unique(arrays['episode_id']))}")
    print(f"wrote: {args.out}  ({args.out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
