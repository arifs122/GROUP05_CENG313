"""Mazlum — scale the Dragapult pool (wins AND losses).

The Banette collector only keeps episodes where a target archetype *won*. For
the Dragapult brain we also need the games Dragapult **lost** (the opponent's
anti-meta moves), so this collector:

1. probes each submission's deck once (cached in the manifest),
2. if the submission plays Dragapult, downloads **all** its episodes,
3. keeps every episode that actually contains Dragapult (winner recorded),
4. stops when the Dragapult pool reaches ``--target-episodes``.

Backoff comes from ``collect_replays.download_replay`` (429-safe); a single
worker + delay keeps the API happy.

Usage::

    python tools/mazlum/collect_dragapult.py --target-episodes 1800 --teams 400
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "tools", ROOT / "tools" / "mazlum"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import collect_replays as cr  # noqa: E402

TARGET = "Dragapult"
MANIFEST = ROOT / "replays" / "_collected.json"
OUT = ROOT / "replays"
TMP = ROOT / "replays" / "_dragon_tmp"


def dragapult_count(manifest: dict) -> int:
    return sum(1 for entry in manifest["episodes"] if TARGET in entry.get("archetypes", []))


def _api_call(fn, *args, attempts: int = 5):
    """Call a Kaggle list endpoint, backing off exponentially on 429."""
    delay = 4.0
    for attempt in range(attempts):
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001
            rate_limited = "429" in str(exc) or "Too Many Requests" in str(exc)
            if not rate_limited or attempt == attempts - 1:
                raise
            pause = delay + random.uniform(0, 2)
            print(f"    429 during {fn.__name__}; sleeping {pause:.1f}s (attempt {attempt + 1}/{attempts})")
            time.sleep(pause)
            delay = min(delay * 2, 120.0)
    raise RuntimeError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Dragapult wins and losses")
    parser.add_argument("--target-episodes", type=int, default=1800)
    parser.add_argument("--teams", type=int, default=400)
    parser.add_argument("--subs-per-team", type=int, default=2)
    parser.add_argument("--max-per-submission", type=int, default=250)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--list-delay", type=float, default=0.5, help="sleep between list API calls")
    parser.add_argument("--start-team", type=int, default=0, help="skip the first N leaderboard teams")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = cr.load_manifest(args.out / cr.MANIFEST_NAME)
    manifest.setdefault("submissions", {})
    manifest.setdefault("episodes", [])
    kept = {int(entry["episode_id"]) for entry in manifest["episodes"]}
    have = dragapult_count(manifest)

    print(f"dragapult episodes: {have}  target: {args.target_episodes}  kept total: {len(kept)}")
    if have >= args.target_episodes:
        print("already at target")
        return 0

    cr._DELAY = max(0.0, args.delay)
    TMP.mkdir(parents=True, exist_ok=True)
    teams = cr.list_teams(args.teams)
    if args.start_team:
        teams = teams[args.start_team:]
    print(f"teams scanned: {len(teams)}")

    scanned_submissions = 0
    probes_failed = 0
    started = time.perf_counter()

    for team_id, team_name in teams:
        if have >= args.target_episodes:
            break
        try:
            time.sleep(args.list_delay)
            submissions = _api_call(cr.list_submissions, team_id, args.subs_per_team)
        except Exception as exc:  # noqa: BLE001
            print(f"  {team_name}: submissions failed ({exc})")
            continue

        for submission_id in submissions:
            if have >= args.target_episodes:
                break
            try:
                time.sleep(args.list_delay)
                episodes = _api_call(cr.list_episodes, submission_id)
            except Exception as exc:  # noqa: BLE001
                print(f"  sub {submission_id}: episodes failed ({exc})")
                continue
            if not episodes:
                continue

            scanned_submissions += 1
            cached = manifest["submissions"].get(str(submission_id))
            if cached is not None:
                archetypes = list(cached)
            else:
                probe_dir = TMP / f"probe-{submission_id}"
                try:
                    probe_path = cr.download_replay(episodes[0]["id"], probe_dir)
                    archetypes, _, _ = cr.deck_archetypes(probe_path)
                except Exception as exc:  # noqa: BLE001
                    probes_failed += 1
                    print(f"  sub {submission_id}: probe failed ({exc})")
                    continue
                finally:
                    shutil.rmtree(probe_dir, ignore_errors=True)
                manifest["submissions"][str(submission_id)] = archetypes
                cr.save_manifest(args.out / cr.MANIFEST_NAME, manifest)

            if TARGET not in archetypes:
                continue

            todo = [
                int(row["id"])
                for row in episodes
                if int(row["id"]) not in kept
            ][: args.max_per_submission]
            print(
                f"  {team_name} sub {submission_id}: {archetypes} -> {len(todo)} episodes to try"
            )

            for episode_id in todo:
                if have >= args.target_episodes:
                    break
                if episode_id in kept:
                    continue
                episode_dir = TMP / f"ep-{episode_id}"
                try:
                    path = cr.download_replay(episode_id, episode_dir)
                except Exception as exc:  # noqa: BLE001
                    print(f"    episode {episode_id}: download failed ({exc})")
                    shutil.rmtree(episode_dir, ignore_errors=True)
                    continue

                try:
                    episode_archetypes, winner, names = cr.deck_archetypes(path)
                except Exception as exc:  # noqa: BLE001
                    print(f"    episode {episode_id}: parse failed ({exc})")
                    shutil.rmtree(episode_dir, ignore_errors=True)
                    continue

                if TARGET not in episode_archetypes or winner is None:
                    shutil.rmtree(episode_dir, ignore_errors=True)
                    continue

                final = args.out / path.name
                shutil.move(str(path), final)
                shutil.rmtree(episode_dir, ignore_errors=True)

                kept.add(episode_id)
                manifest["episodes"].append(
                    {
                        "episode_id": episode_id,
                        "submission_id": submission_id,
                        "team": team_name,
                        "winner_side": winner,
                        "archetypes": episode_archetypes,
                        "teams": names,
                        "file": final.name,
                    }
                )
                have += 1
                cr.save_manifest(args.out / cr.MANIFEST_NAME, manifest)
                print(
                    f"    kept {have}/{args.target_episodes}: {episode_id} "
                    f"({episode_archetypes[0]} vs {episode_archetypes[1]})"
                )

    shutil.rmtree(TMP, ignore_errors=True)
    minutes = (time.perf_counter() - started) / 60
    print(
        f"\nsubmissions scanned: {scanned_submissions}  probe failures: {probes_failed}  "
        f"elapsed: {minutes:.1f} min"
    )
    print(f"dragapult episodes: {have} -> {args.out}")
    if have < args.target_episodes:
        print(f"warning: only {have} of {args.target_episodes} requested episodes found")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
