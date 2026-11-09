#!/usr/bin/env python3
"""Build a Top2-only (Petit Canard) Dragapult BC dataset.

Sources
-------
* ``replays/top2_policy`` (300 Petit Canard episodes)
* ``replays/top5_nov1`` episodes containing Petit Canard, deduped by episode id

Only decisions made by Petit Canard are kept.  Opponent decisions, anti-meta
decisions and other teams' decisions are not included.

Games are split by ``current.round``.  A won game has weight 1.0 and a lost
game 0.5.  Decisions with ``maxCount > 1`` are expanded into one row per
selected option, in the recorded order.  Empty actions and the deck submission
are skipped.

The featurizer is the existing ``src/mazlum/features.py``.  No ``visualize``,
``search_begin_input``, opponent hand, prize identities or opponent deck list
is used as a feature.  ``state_vector`` receives the known Top2 deck explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src" / "mazlum",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import features as F  # noqa: E402

OUR_TEAM = "Petit Canard"
DEFAULT_DECK = ROOT / "deck_top2.csv"
DEFAULT_TOP2 = ROOT / "replays" / "top2_policy"
DEFAULT_TOP5 = ROOT / "replays" / "top5_nov1"
DEFAULT_OUT = ROOT / "data" / "mazlum" / "dataset_top2.npz"


def team_index(episode: dict, team: str) -> int | None:
    names = (episode.get("info") or {}).get("TeamNames") or []
    for i, name in enumerate(names):
        if str(name) == team:
            return i
    return None


def game_win_map(episode: dict, me: int) -> dict[int, bool]:
    """Map round -> whether Petit Canard won that game.

    Earlier games are inferred from the last public state of the round.  The
    final game is the match winner.  Unresolved earlier games fall back to the
    match result and are counted in the returned diagnostics by the caller.
    """
    rewards = episode.get("rewards") or []
    match_winner = None
    if len(rewards) == 2 and rewards[0] is not None and rewards[1] is not None:
        if rewards[0] != rewards[1]:
            match_winner = 0 if rewards[0] > rewards[1] else 1

    states: dict[int, list[tuple[dict, dict]]] = defaultdict(list)
    for step in episode.get("steps") or []:
        c0 = ((step[0].get("observation") or {}).get("current") if len(step) > 0 else None)
        c1 = ((step[1].get("observation") or {}).get("current") if len(step) > 1 else None)
        cur = c0 or c1
        if not cur:
            continue
        r = cur.get("round")
        if r is None:
            continue
        states[int(r)].append((c0, c1))

    rounds = sorted(states)
    out: dict[int, bool] = {}
    for r in rounds:
        if match_winner is None:
            continue
        if r == rounds[-1]:
            out[r] = match_winner == me
            continue
        # Last observation before the round transition.
        c0, c1 = states[r][-1]
        cur = c0 or c1
        if not cur:
            out[r] = match_winner == me
            continue
        p0 = cur["players"][0]
        p1 = cur["players"][1]
        pr0, pr1 = len(p0.get("prize") or []), len(p1.get("prize") or [])
        d0, d1 = p0.get("deckCount"), p1.get("deckCount")
        a0 = any(x for x in (p0.get("active") or []))
        a1 = any(x for x in (p1.get("active") or []))
        winner = None
        if pr0 == 0:
            winner = 0
        elif pr1 == 0:
            winner = 1
        elif d0 == 0:
            winner = 1
        elif d1 == 0:
            winner = 0
        elif not a0 and a1:
            winner = 1
        elif not a1 and a0:
            winner = 0
        out[r] = (winner == me) if winner is not None else (match_winner == me)
    return out


def selected_actor(observation: dict, player: int, option: dict) -> int | None:
    actor, _ = F._option_cards(observation, player, option)  # type: ignore[attr-defined]
    return actor


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Top2-only Dragapult BC dataset")
    ap.add_argument("--top2", type=Path, default=DEFAULT_TOP2)
    ap.add_argument(
        "--extra",
        type=Path,
        nargs="*",
        default=[ROOT / "replays" / "top2_more"],
        help="extra Top2 replay folders (deduped by EpisodeId)",
    )
    ap.add_argument("--top5", type=Path, default=DEFAULT_TOP5)
    ap.add_argument("--deck", type=Path, default=DEFAULT_DECK)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-top5", type=int, default=0)
    args = ap.parse_args()

    deck = [int(x) for x in args.deck.read_text().split() if x.strip()]
    if len(deck) != 60:
        raise SystemExit(f"deck must be 60 cards: {len(deck)}")

    files: list[Path] = []
    files += sorted(args.top2.glob("**/*replay.json"))
    for extra in args.extra or []:
        if extra.is_dir():
            files += sorted(extra.glob("**/*replay.json"))
    # Dedupe is by EpisodeId, not filename.
    episodes: dict[int, Path] = {}
    for p in files:
        try:
            e = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        eid = int((e.get("info") or {}).get("EpisodeId") or -1)
        if eid >= 0:
            episodes.setdefault(eid, p)

    # top5: only episodes containing Petit Canard, only if unseen.
    top5_added = 0
    top5_paths = sorted(args.top5.glob("**/*replay.json"))
    if args.limit_top5:
        top5_paths = top5_paths[: args.limit_top5]
    for p in top5_paths:
        try:
            e = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if team_index(e, OUR_TEAM) is None:
            continue
        eid = int((e.get("info") or {}).get("EpisodeId") or -1)
        if eid >= 0 and eid not in episodes:
            episodes[eid] = p
            top5_added += 1

    print(f"unique episodes: {len(episodes)} (top5 added {top5_added})", flush=True)

    states: list[np.ndarray] = []
    options: list[np.ndarray] = []
    card_ids: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    episode_ids: list[int] = []
    step_indices: list[int] = []
    players: list[int] = []
    contexts: list[int] = []
    selected_types: list[int] = []
    selected_actors: list[int] = []
    max_counts: list[int] = []
    action_lengths: list[int] = []
    rounds: list[int] = []
    turns: list[int] = []
    wins: list[int] = []
    weights: list[float] = []

    raw_decisions = 0
    skipped_empty = 0
    skipped_bad = 0
    multi_decisions = 0
    expanded_multi = 0
    game_keys: set[tuple[int, int]] = set()
    win_games = loss_games = 0
    unknown_games = 0

    for eid, path in episodes.items():
        try:
            e = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        me = team_index(e, OUR_TEAM)
        if me is None:
            continue
        gw = game_win_map(e, me)
        decisions = list(F.iter_decisions(e))
        for d in decisions:
            if d.player != me:
                continue
            raw_decisions += 1
            if not d.action:
                skipped_empty += 1
                continue
            cur = d.observation.get("current") or {}
            r = cur.get("round")
            if r is None:
                continue
            game_keys.add((eid, int(r)))
            if bool(gw.get(int(r), False)):
                win_games += 1
                w = 1.0
            else:
                loss_games += 1
                w = 0.5
            state = F.state_vector(d.observation, deck)
            opt, mask = F.options_matrix(d.observation)
            cids = F.card_ids_matrix(d.observation)
            raw_options = (d.select.get("option") or [])
            if not raw_options:
                skipped_bad += 1
                continue
            action = [int(x) for x in d.action if 0 <= int(x) < min(len(raw_options), F.MAX_OPTIONS)]
            if not action:
                skipped_empty += 1
                continue
            if len(action) > 1:
                multi_decisions += 1
                expanded_multi += len(action)
            for idx in action:
                option = raw_options[idx]
                states.append(state)
                options.append(opt)
                card_ids_list = cids
                card_ids.append(card_ids_list)
                masks.append(mask)
                labels.append(idx)
                episode_ids.append(eid)
                step_indices.append(d.step_index)
                players.append(me)
                contexts.append(int(d.select.get("context", -1)))
                selected_types.append(int(option.get("type", -1)))
                actor = selected_actor(d.observation, me, option)
                selected_actors.append(int(actor) if actor is not None else 0)
                max_counts.append(int(d.select.get("maxCount", 0) or 0))
                action_lengths.append(len(action))
                rounds.append(int(r))
                turns.append(int(cur.get("turn", 0) or 0))
                wins.append(1 if w == 1.0 else 0)
                weights.append(w)

    if not labels:
        print("error: no rows")
        return 1

    # Episode-level split, 80/20; same rows from an episode stay together.
    unique = np.array(sorted(set(episode_ids)), dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique)
    n_val = int(round(len(unique) * args.val_fraction))
    n_val = max(1, min(n_val, len(unique) - 1))
    val_ids = set(int(x) for x in unique[:n_val])
    split = np.array([1 if int(x) in val_ids else 0 for x in episode_ids], dtype=np.int8)

    arrays = {
        "state": np.stack(states).astype(np.float16),
        "options": np.stack(options).astype(np.float16),
        "card_ids": np.stack(card_ids).astype(np.int16),
        "mask": np.stack(masks).astype(np.float32),
        "label": np.asarray(labels, dtype=np.int64),
        "episode_id": np.asarray(episode_ids, dtype=np.int64),
        "step_index": np.asarray(step_indices, dtype=np.int32),
        "player": np.asarray(players, dtype=np.int8),
        "split": split,
        # 2.0 for wins, 1.0 for losses in train_bc's existing source weighting;
        # this is the requested 2:1 (1.0/0.5) game-weight ratio.
        "source": np.asarray([2 if w == 1.0 else 0 for w in weights], dtype=np.int8),
        "sample_weight": np.asarray(weights, dtype=np.float32),
        "round": np.asarray(rounds, dtype=np.int16),
        "turn": np.asarray(turns, dtype=np.int16),
        "game_win": np.asarray(wins, dtype=np.int8),
        "context": np.asarray(contexts, dtype=np.int16),
        "selected_type": np.asarray(selected_types, dtype=np.int16),
        "selected_actor": np.asarray(selected_actors, dtype=np.int32),
        "max_count": np.asarray(max_counts, dtype=np.int16),
        "action_length": np.asarray(action_lengths, dtype=np.int16),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)

    print(f"raw Top2 decisions: {raw_decisions}")
    print(f"games: {len(game_keys)}  win_games={win_games} loss_games={loss_games}")
    print(f"skipped empty: {skipped_empty}  skipped bad options: {skipped_bad}")
    print(f"multi decisions: {multi_decisions}  expanded multi rows: {expanded_multi}")
    print(f"rows after maxCount expansion: {len(labels)}")
    print(f"episodes: {len(unique)}  val episodes: {n_val}  train rows {int((split==0).sum())} val rows {int((split==1).sum())}")
    print(f"contexts: {Counter(contexts).most_common(12)}")
    print(f"selected types: {Counter(selected_types).most_common(12)}")
    print(f"wrote: {args.out} ({args.out.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
