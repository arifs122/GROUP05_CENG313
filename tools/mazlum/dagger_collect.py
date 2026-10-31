"""Mazlum — DAgger data collection.

Plays N games of **Mazlum vs the Banette expert** and, on every position Mazlum
has to act in, asks the expert the classic DAgger question: *"in this exact
position, which legal move would you play?"* The pair
``(state, options, card_ids, mask, expert_label)`` is recorded, then appended to
the existing ``data/mazlum/dataset.npz`` as extra **training** rows.

Why DAgger: BC only sees expert states, so the learner drifts into states the
expert never visited. Here Mazlum drives the rollout (its own state
distribution) and the expert labels those states.

Usage::

    python tools/mazlum/dagger_collect.py --games 100
    python tools/mazlum/dagger_collect.py --games 10 --no-merge
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# The engine must be importable before ``arena`` (it imports cg at module load).
for path in (ROOT / "vendor", ROOT / "tools", ROOT / "src" / "mazlum"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import arena  # noqa: E402
import features as F  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

MAZLUM_BOT = ROOT / "bots" / "mazlum"
EXPERT_BOT = ROOT / "bots" / "banette"
DEFAULT_DATASET = ROOT / "data" / "mazlum" / "dataset.npz"
DAGGER_OUT = ROOT / "data" / "mazlum" / "dagger.npz"

# Synthetic episode ids keep DAgger rows separate from replay episodes.
DAGGER_EPISODE_BASE = 9_500_000


def collect_game(mazlum_agent, expert_agent, deck_mazlum, deck_banette, reverse, max_steps, episode_id):
    """Play one game, record Mazlum's states with expert labels."""
    obs, start_data = battle_start(deck_mazlum, deck_banette, reverse)
    rows: list[tuple] = []
    if obs is None:
        return rows, None, 0

    steps = 0
    result = None
    try:
        while steps < max_steps:
            current = obs["current"]
            if int(current["result"]) >= 0:
                result = int(current["result"])
                break

            select = obs["select"]
            if select is None:
                break

            player = int(current["yourIndex"])
            if player == 0:
                # Mazlum's state: expert labels the same observation.
                expert_action = expert_agent(obs)
                if arena.is_valid(expert_action, select) and expert_action:
                    options, mask = F.options_matrix(obs)
                    label = int(expert_action[0])
                    # Labels must index the padded 40-row matrix, not the raw option list.
                    if 0 <= label < int(mask.sum()):
                        rows.append(
                            (
                                F.state_vector(obs, deck_mazlum),
                                options,
                                F.card_ids_matrix(obs),
                                mask,
                                label,
                                episode_id,
                                steps,
                            )
                        )
                action = mazlum_agent(obs)
            else:
                action = expert_agent(obs)

            if not arena.is_valid(action, select):
                break
            obs = battle_select(action)
            steps += 1
    finally:
        battle_finish()

    return rows, result, steps


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
        # DAgger rows always train; the original validation split is untouched.
        "split": np.zeros(len(rows), dtype=np.int8),
        # source: 1 = DAgger, 0 = replay (used for sample weights at training).
        "source": np.ones(len(rows), dtype=np.int8),
    }


def merge_dataset(dataset_path: Path, new: dict[str, np.ndarray]) -> dict[str, int]:
    """Append ``new`` arrays to ``dataset_path`` (training split)."""
    with np.load(dataset_path) as data:
        existing_episodes = set(np.unique(data["episode_id"]).tolist())
        new_episodes = set(np.unique(new["episode_id"]).tolist())
        if new_episodes & existing_episodes:
            raise ValueError("DAgger episode ids already exist in the dataset; refusing to duplicate")

        before = int(data["state"].shape[0])
        source = data["source"] if "source" in data.files else np.zeros(before, dtype=np.int8)
        merged = {
            key: np.concatenate([data[key], new[key]]) if key in new else data[key]
            for key in data.files
        }
        merged["source"] = np.concatenate([source, new["source"]])

    # The handle must be closed before overwriting the archive on Windows.
    np.savez_compressed(dataset_path, **merged)
    return {"before": before, "after": int(merged["state"].shape[0])}


def main() -> int:
    parser = argparse.ArgumentParser(description="DAgger: Mazlum states + Banette expert labels")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DAGGER_OUT, help="where to store the new rows")
    parser.add_argument("--no-merge", action="store_true", help="only write the dagger file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.dataset.is_file():
        print(f"error: dataset not found: {args.dataset}")
        return 1

    mazlum = arena.load_bot(MAZLUM_BOT)
    expert = arena.load_bot(EXPERT_BOT)
    # Expert synchronisation: both bots pilot the SAME 60 cards (the learner's
    # consensus Banette list) so expert labels live in the learner's action space.
    deck = mazlum.deck
    print(f"learner: {mazlum.name}  expert: {expert.name}")
    print(f"shared deck: {len(deck)} cards (bots/mazlum/deck.csv)")
    print(f"games: {args.games}")

    all_rows: list[tuple] = []
    wins = 0
    errors = 0
    started = time.perf_counter()

    for game_index in range(args.games):
        episode_id = DAGGER_EPISODE_BASE + game_index
        rows, result, steps = collect_game(
            mazlum.agent,
            expert.agent,
            deck,
            deck,
            reverse=(game_index % 2 == 1),
            max_steps=args.max_steps,
            episode_id=episode_id,
        )
        all_rows.extend(rows)
        if result == 0:
            wins += 1
        if not rows:
            errors += 1
        if (game_index + 1) % 10 == 0 or game_index == 0:
            print(
                f"  game {game_index + 1:3d}/{args.games}  rows={len(all_rows)}  "
                f"mazlum wins={wins}  steps={steps}"
            )

    if not all_rows:
        print("error: no DAgger rows were collected")
        return 1

    arrays = to_arrays(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)

    seconds = time.perf_counter() - started
    print(
        f"\ncollected {len(all_rows)} rows from {args.games} games "
        f"({seconds:.1f}s, mazlum wins={wins}, empty games={errors})"
    )
    print(f"wrote: {args.out}")
    if arrays["label"].size:
        majority = float((arrays["label"] == 0).mean())
        print(f"expert label = 0 share: {majority:.1%}")

    if not args.no_merge:
        counts = merge_dataset(args.dataset, arrays)
        print(f"merged into: {args.dataset}  rows {counts['before']} -> {counts['after']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
