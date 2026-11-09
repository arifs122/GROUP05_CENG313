"""Build a Banette-vs-Dragapult focused dataset with the v13 (37-dim) featurizer.

Only episodes where both archetypes appear are used; every decision from both
players is kept (the v13 recipe). Preallocated arrays keep peak RAM low.

Usage::

    python tools/build_targeted_dataset.py --capacity 400000 \
        --out data/mazlum/dataset_v15_dragapult_focus.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_SEARCH = [ROOT / "src" / "mazlum", Path(__file__).resolve().parents[1] / "src" / "mazlum"]
for path in _SEARCH:
    if (path / "features.py").is_file() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import features as F  # noqa: E402
from build_dataset import assign_split  # noqa: E402

MAIN = Path(r"C:\Users\Fariz\OneDrive\Desktop\TransforMew")
DEFAULT_MANIFEST = MAIN / "replays" / "_collected.json"
DEFAULT_REPLAYS = MAIN / "replays"
DEFAULT_OUT = ROOT / "data" / "mazlum" / "dataset_v15_dragapult_focus.npz"
PAIR = {"Banette", "Dragapult"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Banette-vs-Dragapult targeted dataset")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--replays", type=Path, default=DEFAULT_REPLAYS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--capacity", type=int, default=400_000)
    parser.add_argument("--episode-ids", type=Path, default=None,
                        help="JSON list of episode ids to include (overrides the pair filter)")
    parser.add_argument("--archetype", default=None, help="require this archetype in the episode")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--schema", type=int, default=1, help="feature schema (1 legacy, 2 extended)")
    parser.add_argument("--only-side", default=None,
                        help="keep only this archetype's decisions (halves memory)")
    args = parser.parse_args()

    F.configure(args.schema)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    wanted = None
    if args.episode_ids is not None:
        wanted = {int(x) for x in json.loads(args.episode_ids.read_text(encoding="utf-8"))}
        print(f"episode-id filter: {len(wanted)} ids", flush=True)

    episodes = []
    for e in manifest.get("episodes") or []:
        if not e.get("file"):
            continue
        arch = set(e.get("archetypes") or [])
        if wanted is not None:
            if int(e["episode_id"]) not in wanted:
                continue
        elif not (PAIR <= arch):
            continue
        if args.archetype and args.archetype not in arch:
            continue
        episodes.append(e)
    print(f"selected episodes: {len(episodes)}", flush=True)

    cap = args.capacity
    states = np.empty((cap, F.state_feature_size()), dtype=np.float16)
    options = np.empty((cap, F.MAX_OPTIONS, F.option_feature_size()), dtype=np.float16)
    card_ids = np.empty((cap, F.MAX_OPTIONS, 2), dtype=np.int16)
    mask = np.empty((cap, F.MAX_OPTIONS), dtype=np.float16)
    label = np.empty(cap, dtype=np.int64)
    episode_id = np.empty(cap, dtype=np.int64)
    step_index = np.empty(cap, dtype=np.int32)
    player = np.empty(cap, dtype=np.int8)

    rows = 0
    used = 0
    skipped_no_board = skipped_bad = missing = 0

    for index, entry in enumerate(episodes):
        if rows >= cap:
            break
        path = args.replays / entry["file"]
        if not path.is_file():
            missing += 1
            continue
        try:
            episode = F.load_episode(path)
            decisions = list(F.parse_episode(episode))
        except Exception:  # noqa: BLE001
            missing += 1
            continue

        eid = int(entry["episode_id"])
        before = rows
        side_index = None
        if args.only_side:
            arch = entry.get("archetypes") or []
            if args.only_side in arch:
                side_index = arch.index(args.only_side)
        for decision, state, option_matrix, m, ids in decisions:
            if rows >= cap:
                break
            if side_index is not None and int(decision.player) != side_index:
                continue
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
        if rows > before:
            used += 1

        if (index + 1) % 50 == 0:
            print(f"  episodes {index + 1}/{len(episodes)}  rows {rows}/{cap}", flush=True)

    if rows == 0:
        print("error: no rows")
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
    print(f"rows {rows}  episodes {used}  missing {missing}")
    print(f"skipped: no board {skipped_no_board}  bad action {skipped_bad}")
    print(f"train {int((split == 0).sum())} / val {int((split == 1).sum())} ({val_ids.size} val episodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
