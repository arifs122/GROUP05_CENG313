"""Mazlum — pure Dragapult-win dataset (no anti-meta, no losses).

Filters ``dataset_dragapult.npz`` down to rows from episodes **Dragapult won**
(``source == 0``), recomputes the episode-level 80/20 split and writes
``dataset_dragapult_wins.npz``.

Usage::

    python tools/mazlum/make_wins_dataset.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src" / "mazlum",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_dataset import assign_split  # noqa: E402

DEFAULT_IN = ROOT / "data" / "mazlum" / "dataset_dragapult.npz"
DEFAULT_OUT = ROOT / "data" / "mazlum" / "dataset_dragapult_wins.npz"


def main() -> int:
    parser = argparse.ArgumentParser(description="Keep only Dragapult winning rows")
    parser.add_argument("--inp", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with np.load(args.inp) as data:
        arrays = {key: data[key] for key in data.files}

    source = arrays.get("source", np.zeros(len(arrays["label"]), dtype=np.int8))
    keep = source == 0
    print(f"rows before: {len(keep)}  Dragapult wins: {int(keep.sum())}")

    arrays = {key: value[keep] for key, value in arrays.items()}
    split, val_ids = assign_split(arrays["episode_id"], args.val_fraction, args.seed)
    arrays["split"] = split
    arrays["source"] = np.zeros(len(arrays["label"]), dtype=np.int8)

    np.savez_compressed(args.out, **arrays)

    print(
        f"rows: {len(arrays['label'])}  episodes: {len(np.unique(arrays['episode_id']))}  "
        f"train: {int((split == 0).sum())}  val: {int((split == 1).sum())} "
        f"({val_ids.size} episodes)"
    )
    print(f"wrote: {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
