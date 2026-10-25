"""Mazlum — build the behavioral-cloning dataset from replay JSON files.

Walks ``replays/*.json``, turns every valid decision into a training row with
``features.parse_episode()`` and stores everything in one compressed NumPy
archive:

    data/mazlum/dataset.npz
        state      float16 [N, STATE_FEATURE_SIZE]
        options    float16 [N, MAX_OPTIONS, OPTION_FEATURE_SIZE]
        card_ids   int16   [N, MAX_OPTIONS, 2]   # actor/target card ids (embedding)
        mask       float32 [N, MAX_OPTIONS]
        label      int64   [N]            # chosen option index (action[0])
        episode_id int64   [N]
        split      int8    [N]            # 0 = train, 1 = validation
        step_index int32   [N]
        player     int8    [N]

The split is **episode based**: every decision of an episode lands on the same
side, so validation measures generalisation to unseen games instead of leaking
rows from the same match.

Usage::

    python src/mazlum/build_dataset.py
    python src/mazlum/build_dataset.py --val-fraction 0.2 --seed 42
    python src/mazlum/build_dataset.py --replays replays --out data/mazlum/dataset.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import features as F  # noqa: E402

ROOT = HERE.parents[1]
DEFAULT_REPLAYS = ROOT / "replays"
DEFAULT_OUT = ROOT / "data" / "mazlum" / "dataset.npz"


def collect_rows(replays_dir: Path) -> dict[str, np.ndarray]:
    """Parse every replay in ``replays_dir`` and stack the decisions."""
    # ``_collected.json`` is the collector manifest, not a replay.
    paths = sorted(path for path in replays_dir.glob("*.json") if not path.name.startswith("_"))
    if not paths:
        raise FileNotFoundError(f"no replay JSON files in {replays_dir}")

    states: list[np.ndarray] = []
    options: list[np.ndarray] = []
    card_ids_list: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    episode_ids: list[int] = []
    step_indices: list[int] = []
    players: list[int] = []

    skipped_no_board = 0
    skipped_no_option = 0
    skipped_bad_action = 0
    per_episode: dict[int, int] = {}

    for path in paths:
        try:
            episode = F.load_episode(path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: skipping {path.name}: {exc}")
            continue

        episode_id = int((episode.get("info") or {}).get("EpisodeId") or -1)
        rows_before = len(labels)

        for decision, state, option_matrix, mask, card_ids in F.parse_episode(episode):
            # Deck selection, IS_FIRST and setup picks happen without a board.
            if not F.has_board(decision.observation):
                skipped_no_board += 1
                continue

            valid = int(mask.sum())
            if valid == 0:
                skipped_no_option += 1
                continue

            # Label comes from the first selected option index.
            if not decision.action:
                skipped_bad_action += 1
                continue
            label = int(decision.action[0])
            if not 0 <= label < valid:
                skipped_bad_action += 1
                continue

            states.append(state.astype(np.float16))
            options.append(option_matrix.astype(np.float16))
            card_ids_list.append(card_ids.astype(np.int16))
            masks.append(mask)
            labels.append(label)
            episode_ids.append(episode_id)
            step_indices.append(decision.step_index)
            players.append(decision.player)

        per_episode[episode_id] = len(labels) - rows_before

    if not labels:
        raise ValueError(f"no valid decisions found in {replays_dir}")

    print(f"files: {len(paths)}")
    print(f"skipped: no board={skipped_no_board} no options={skipped_no_option} bad action={skipped_bad_action}")
    episodes_total = len(per_episode)
    print(f"episodes: {episodes_total}  rows: {len(labels)}")
    if episodes_total and episodes_total <= 20:
        for episode_id, rows in per_episode.items():
            print(f"  episode {episode_id}: {rows} rows")

    # Stack one array at a time and free its Python list immediately: with
    # thousands of episodes the option list alone is several GB.
    arrays: dict[str, np.ndarray] = {}
    arrays["state"] = np.stack(states)
    states.clear()
    arrays["options"] = np.stack(options)
    options.clear()
    arrays["card_ids"] = np.stack(card_ids_list)
    card_ids_list.clear()
    arrays["mask"] = np.stack(masks).astype(np.float32)
    masks.clear()
    arrays["label"] = np.asarray(labels, dtype=np.int64)
    arrays["episode_id"] = np.asarray(episode_ids, dtype=np.int64)
    arrays["step_index"] = np.asarray(step_indices, dtype=np.int32)
    arrays["player"] = np.asarray(players, dtype=np.int8)
    return arrays


def assign_split(episode_ids: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Episode-level train/val split. Returns ``(split, val_episode_ids)``."""
    unique = np.unique(episode_ids)
    if val_fraction <= 0 or unique.size < 2:
        return np.zeros(episode_ids.shape[0], dtype=np.int8), np.array([], dtype=np.int64)

    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)

    n_val = int(round(unique.size * val_fraction))
    n_val = max(1, min(n_val, unique.size - 1))
    val_ids = np.sort(shuffled[:n_val])

    is_val = np.isin(episode_ids, val_ids)
    return is_val.astype(np.int8), val_ids.astype(np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Mazlum BC dataset from replays")
    parser.add_argument("--replays", type=Path, default=DEFAULT_REPLAYS, help="folder with replay JSON files")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output .npz path")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="share of episodes used for validation")
    parser.add_argument("--seed", type=int, default=42, help="split seed")
    args = parser.parse_args()

    if not args.replays.is_dir():
        print(f"error: replay folder not found: {args.replays}")
        return 1

    try:
        arrays = collect_rows(args.replays)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 1

    split, val_ids = assign_split(arrays["episode_id"], args.val_fraction, args.seed)
    arrays["split"] = split

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)

    rows = arrays["state"].shape[0]
    train_rows = int((split == 0).sum())
    val_rows = int((split == 1).sum())
    size_mb = args.out.stat().st_size / 1e6
    print(f"\nwrote: {args.out}  ({size_mb:.2f} MB)")
    print(f"rows: {rows}  train: {train_rows}  val: {val_rows}")
    print(f"episodes: {len(np.unique(arrays['episode_id']))}  val episodes: {val_ids.size}")
    if val_ids.size:
        print(f"  val episode ids: {', '.join(str(int(x)) for x in val_ids)}")
    for name, array in arrays.items():
        print(f"  {name:<10} {str(array.shape):<22} {array.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
