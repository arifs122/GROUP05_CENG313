"""Mazlum — "clone your killer" (anti-meta) data pipeline.

1. Finds our official Mazlum submission on Kaggle and lists its episodes.
2. Downloads those replays and keeps only the **losses** (our side lost).
3. Parses *only the winner's decisions* — our bot's mistakes are never added.
4. Appends the new rows to ``data/mazlum/dataset.npz`` with ``source = 2`` so
   training can up-weight them (``--antimeta-weight``, default 2.0).

Our side is identified by the 60-card deck signature, because older episodes
carry no submission/team metadata.

Usage::

    python tools/mazlum/clone_killer.py
    python tools/mazlum/clone_killer.py --max-episodes 200 --delay 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "vendor", ROOT / "tools", ROOT / "tools" / "mazlum", ROOT / "src" / "mazlum"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import features as F  # noqa: E402
import replay_meta as rm  # noqa: E402
import collect_replays as cr  # noqa: E402

DEFAULT_DATASET = ROOT / "data" / "mazlum" / "dataset.npz"
ANTIMETA_OUT = ROOT / "data" / "mazlum" / "antimeta.npz"
OUR_DECK = ROOT / "bots" / "mazlum" / "deck.csv"
TMP_DIR = ROOT / "replays" / "_antimeta_tmp"
ANTIMETA_EPISODE_BASE = 8_000_000
SOURCE_ANTIMETA = 2


def deck_signature(deck: list[int]) -> tuple[int, ...]:
    return tuple(sorted(int(card) for card in deck))


def find_mazlum_submissions() -> list[int]:
    """Our submissions whose file name mentions mazlum (newest first)."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    subs = api.competition_submissions(cr.COMPETITION) or []
    ids = []
    for sub in subs:
        name = str(getattr(sub, "file_name", "") or "")
        if "mazlum" in name.lower() and "COMPLETE" in str(getattr(sub, "status", "")):
            ids.append(int(getattr(sub, "ref", 0)))
    return ids


def collect_rows(episode_paths: list[Path], our_deck: list[int]) -> tuple[list[tuple], dict]:
    """Extract the winner's decisions from every lost episode."""
    rows: list[tuple] = []
    stats = {"episodes": 0, "losses": 0, "wins": 0, "unknown_side": 0, "mirror": 0, "rows": 0}
    our_signature = deck_signature(our_deck)
    episode_index = 0

    for path in episode_paths:
        try:
            episode = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stats["episodes"] += 1

        decks = rm.extract_decks(episode)
        if not decks:
            continue
        signatures = [deck_signature(deck) for deck in decks]
        if len(signatures) == 2 and signatures[0] == signatures[1]:
            # Mirror match: both decks are our list, so "our side" is ambiguous.
            stats["mirror"] += 1
            continue
        our_side = next(
            (side for side, signature in enumerate(signatures) if signature == our_signature),
            None,
        )
        if our_side is None:
            stats["unknown_side"] += 1
            continue

        winner = rm.winner_index(episode)
        if winner is None:
            continue
        if winner == our_side:
            stats["wins"] += 1
            continue
        stats["losses"] += 1

        episode_id = ANTIMETA_EPISODE_BASE + episode_index
        episode_index += 1

        try:
            decisions = list(F.parse_episode(episode, players={winner}))
        except Exception:  # noqa: BLE001 - one bad replay must not kill the run
            continue

        for decision, state, options, mask, card_ids in decisions:
            if not F.has_board(decision.observation):
                continue
            valid = int(mask.sum())
            if valid == 0 or not decision.action:
                continue
            label = int(decision.action[0])
            if not 0 <= label < valid:
                continue
            rows.append((state, options, card_ids, mask, label, episode_id, decision.step_index))

    stats["rows"] = len(rows)
    return rows, stats


def to_arrays(rows: list[tuple]) -> dict[str, np.ndarray]:
    return {
        "state": np.stack([row[0] for row in rows]).astype(np.float16),
        "options": np.stack([row[1] for row in rows]).astype(np.float16),
        "card_ids": np.stack([row[2] for row in rows]).astype(np.int16),
        "mask": np.stack([row[3] for row in rows]).astype(np.float32),
        "label": np.asarray([row[4] for row in rows], dtype=np.int64),
        "episode_id": np.asarray([row[5] for row in rows], dtype=np.int64),
        "step_index": np.asarray([row[6] for row in rows], dtype=np.int32),
        "player": np.zeros(len(rows), dtype=np.int8),
        "split": np.zeros(len(rows), dtype=np.int8),
        "source": np.full(len(rows), SOURCE_ANTIMETA, dtype=np.int8),
    }


def merge_dataset(dataset_path: Path, new: dict[str, np.ndarray]) -> dict[str, int]:
    with np.load(dataset_path) as data:
        before = int(data["state"].shape[0])
        source = data["source"] if "source" in data.files else np.zeros(before, dtype=np.int8)
        if set(np.unique(new["episode_id"]).tolist()) & set(np.unique(data["episode_id"]).tolist()):
            raise ValueError("anti-meta episode ids collide with the dataset; refusing to merge")
        merged = {
            key: np.concatenate([data[key], new[key]]) if key in new else data[key]
            for key in data.files
        }
        merged["source"] = np.concatenate([source, new["source"]])

    np.savez_compressed(dataset_path, **merged)
    return {"before": before, "after": int(merged["state"].shape[0])}


def main() -> int:
    parser = argparse.ArgumentParser(description="Clone the opponents that beat Mazlum")
    parser.add_argument("--submission", type=int, default=None, help="Mazlum submission id (auto-detected)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=ANTIMETA_OUT)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--max-episodes", type=int, default=500)
    parser.add_argument("--no-merge", action="store_true")
    args = parser.parse_args()

    if not OUR_DECK.is_file():
        print(f"error: deck not found: {OUR_DECK}")
        return 1
    our_deck = [int(line.strip()) for line in OUR_DECK.read_text(encoding="utf-8").splitlines() if line.strip()]

    submissions = [args.submission] if args.submission else find_mazlum_submissions()
    if not submissions:
        print("error: no completed Mazlum submission found on Kaggle")
        return 1
    print(f"mazlum submissions: {submissions}")

    cr._DELAY = max(0.0, args.delay)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for submission_id in submissions:
        try:
            episodes = cr.list_episodes(submission_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  sub {submission_id}: episodes failed ({exc})")
            continue
        print(f"  sub {submission_id}: {len(episodes)} episodes")
        for row in episodes[: args.max_episodes]:
            try:
                paths.append(cr.download_replay(row["id"], TMP_DIR / f"ep-{row['id']}"))
            except Exception as exc:  # noqa: BLE001
                print(f"    episode {row['id']}: download failed ({exc})")

    print(f"downloaded replays: {len(paths)}")
    rows, stats = collect_rows(paths, our_deck)
    print(
        f"episodes={stats['episodes']} losses={stats['losses']} wins={stats['wins']} "
        f"unknown_side={stats['unknown_side']} mirror={stats['mirror']} rows={stats['rows']}"
    )
    if not rows:
        print("error: no anti-meta rows were collected")
        return 1

    arrays = to_arrays(rows)
    np.savez_compressed(args.out, **arrays)
    print(f"wrote: {args.out}")

    if not args.no_merge:
        counts = merge_dataset(args.dataset, arrays)
        print(f"merged into: {args.dataset}  rows {counts['before']} -> {counts['after']}")

    for path in TMP_DIR.glob("ep-*"):
        for file in path.glob("*.json"):
            file.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
