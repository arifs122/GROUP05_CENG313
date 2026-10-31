"""Build a wide multi-deck dataset with the v13 (37-dim) featurizer.

The existing ``build_dataset.py`` stacks Python lists at the end, which doubles
peak RAM (list of arrays + stacked copy). At ~900k rows that is ~6 GB and this
laptop has ~6.3 GB free, so this builder preallocates the output arrays and
fills them in place.

Episodes are visited in a seeded shuffle of the collection manifest so the
sample spans the whole meta (40 archetypes) instead of one mining run.

Usage::

    python tools/build_multi_dataset.py --capacity 900000 \
        --out data/mazlum/dataset_v14_multi.npz
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src" / "mazlum",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import features as F  # noqa: E402
from build_dataset import assign_split  # noqa: E402

DEFAULT_MANIFEST = Path(r"C:\Users\Fariz\OneDrive\Desktop\TransforMew\replays\_collected.json")
DEFAULT_REPLAYS = Path(r"C:\Users\Fariz\OneDrive\Desktop\TransforMew\replays")
DEFAULT_OUT = ROOT / "data" / "mazlum" / "dataset_v14_multi.npz"


def main() -> int:
    parser = argparse.ArgumentParser(description="Preallocated multi-deck dataset builder")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--replays", type=Path, default=DEFAULT_REPLAYS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--capacity", type=int, default=900_000)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = [e for e in manifest.get("episodes") or [] if e.get("file")]
    rng = random.Random(args.seed)
    rng.shuffle(entries)

    cap = args.capacity
    states = np.empty((cap, F.STATE_FEATURE_SIZE), dtype=np.float16)
    options = np.empty((cap, F.MAX_OPTIONS, F.OPTION_FEATURE_SIZE), dtype=np.float16)
    card_ids = np.empty((cap, F.MAX_OPTIONS, 2), dtype=np.int16)
    mask = np.empty((cap, F.MAX_OPTIONS), dtype=np.float16)
    label = np.empty(cap, dtype=np.int64)
    episode_id = np.empty(cap, dtype=np.int64)
    step_index = np.empty(cap, dtype=np.int32)
    player = np.empty(cap, dtype=np.int8)

    rows = 0
    skipped_no_board = skipped_bad = missing = 0
    episodes_used = 0

    for index, entry in enumerate(entries):
        if rows >= cap:
            break
        path = args.replays / entry["file"]
        if not path.is_file():
            missing += 1
            continue
        try:
            episode = F.load_episode(path)
        except (OSError, json.JSONDecodeError):
            missing += 1
            continue

        eid = int(entry["episode_id"])
        used = 0
        try:
            decisions = list(F.parse_episode(episode))
        except Exception:  # noqa: BLE001 - one broken replay must not stop the build
            missing += 1
            continue

        for decision, state, option_matrix, m, ids in decisions:
            if rows >= cap:
                break
            if not F.has_board(decision.observation):
                skipped_no_board += 1
                continue
            valid = int(m.sum())
            if valid == 0 or not decision.action:
                skipped_bad += 1
                continue
            lab = int(decision.action[0])
            if not 0 <= lab < valid:
                skipped_bad += 1
                continue
            states[rows] = state
            options[rows] = option_matrix
            card_ids[rows] = ids
            mask[rows] = m
            label[rows] = lab
            episode_id[rows] = eid
            step_index[rows] = decision.step_index
            player[rows] = decision.player
            rows += 1
            used += 1
        if used:
            episodes_used += 1

        if (index + 1) % 100 == 0:
            print(f"  files {index + 1}/{len(entries)}  rows {rows}/{cap}  episodes {episodes_used}", flush=True)

    if rows == 0:
        print("error: no rows collected")
        return 1

    arrays = {
        "state": states[:rows],
        "options": options[:rows],
        "card_ids": card_ids[:rows],
        "mask": mask[:rows].astype(np.float32),
        "label": label[:rows],
        "episode_id": episode_id[:rows],
        "step_index": step_index[:rows],
        "player": player[:rows],
    }
    split, val_ids = assign_split(arrays["episode_id"], args.val_fraction, args.seed)
    arrays["split"] = split

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)

    print(f"\nwrote: {args.out}  ({args.out.stat().st_size / 1e6:.2f} MB)")
    print(f"rows {rows}  episodes {episodes_used}  files missing/broken {missing}")
    print(f"skipped: no board {skipped_no_board}  bad action {skipped_bad}")
    print(f"train {int((split == 0).sum())} / val {int((split == 1).sum())} ({val_ids.size} val episodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
