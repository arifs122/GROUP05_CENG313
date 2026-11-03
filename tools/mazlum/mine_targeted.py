"""Targeted mining: keep only Banette-vs-Dragapult episodes.

Reuses the collectors in ``collect_replays``. Probes submissions (cached in the
shared manifest), downloads episodes only for subs that play Banette or
Dragapult, and keeps an episode solely when **both** archetypes appear.

Usage::

    python tools/mazlum/mine_targeted.py --target-new 150 --minutes 45
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "tools", ROOT / "tools" / "mazlum"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import collect_replays as cr  # noqa: E402
from collect_dragapult import _api_call  # noqa: E402

PAIR = {"Banette", "Dragapult"}
MANIFEST = ROOT / "replays" / "_collected.json"
OUT = ROOT / "replays"
TMP = ROOT / "replays" / "_target_tmp"


def pair_count(manifest: dict) -> int:
    return sum(1 for e in manifest["episodes"] if PAIR <= set(e.get("archetypes") or []))


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine Banette-vs-Dragapult episodes only")
    parser.add_argument("--target-new", type=int, default=150)
    parser.add_argument("--minutes", type=float, default=45.0)
    parser.add_argument("--teams", type=int, default=563)
    parser.add_argument("--subs-per-team", type=int, default=8)
    parser.add_argument("--max-per-submission", type=int, default=60)
    parser.add_argument("--only-sub-archetype", default=None,
                        help="only scan submissions whose deck contains this archetype")
    parser.add_argument("--delay", type=float, default=0.8)
    parser.add_argument("--list-delay", type=float, default=0.3)
    args = parser.parse_args()

    manifest = cr.load_manifest(MANIFEST)
    manifest.setdefault("submissions", {})
    manifest.setdefault("episodes", [])
    kept = {int(e["episode_id"]) for e in manifest["episodes"]}
    start_count = pair_count(manifest)
    have = start_count

    print(f"pair episodes: {have}  target: +{args.target_new}", flush=True)
    cr._DELAY = max(0.0, args.delay)
    TMP.mkdir(parents=True, exist_ok=True)
    teams = cr.list_teams(args.teams)
    print(f"teams: {len(teams)}", flush=True)

    started = time.perf_counter()
    probes = new_rows = 0
    for team_id, team_name in teams:
        if have - start_count >= args.target_new:
            break
        if time.perf_counter() - started > args.minutes * 60:
            print("time budget reached", flush=True)
            break
        try:
            time.sleep(args.list_delay)
            subs = _api_call(cr.list_submissions, team_id, args.subs_per_team)
        except Exception as exc:  # noqa: BLE001
            print(f"  {team_name}: submissions failed ({exc})", flush=True)
            continue

        for submission_id in subs:
            if have - start_count >= args.target_new:
                break
            try:
                time.sleep(args.list_delay)
                episodes = _api_call(cr.list_episodes, submission_id)
            except Exception as exc:  # noqa: BLE001
                continue
            if not episodes:
                continue

            archetypes = manifest["submissions"].get(str(submission_id))
            if archetypes is None:
                probe_dir = TMP / f"probe-{submission_id}"
                try:
                    probe = cr.download_replay(episodes[0]["id"], probe_dir)
                    archetypes, _, _ = cr.deck_archetypes(probe)
                except Exception:  # noqa: BLE001
                    continue
                finally:
                    shutil.rmtree(probe_dir, ignore_errors=True)
                manifest["submissions"][str(submission_id)] = archetypes
                cr.save_manifest(MANIFEST, manifest)
                probes += 1

            if not (set(archetypes) & PAIR):
                continue
            if args.only_sub_archetype and args.only_sub_archetype not in archetypes:
                continue

            todo = [int(row["id"]) for row in episodes if int(row["id"]) not in kept][: args.max_per_submission]
            if not todo:
                continue
            print(f"  {team_name} sub {submission_id} {archetypes} -> {len(todo)} episodes", flush=True)

            for episode_id in todo:
                if have - start_count >= args.target_new:
                    break
                if episode_id in kept:
                    continue
                episode_dir = TMP / f"ep-{episode_id}"
                try:
                    path = cr.download_replay(episode_id, episode_dir)
                    ep_arch, winner, names = cr.deck_archetypes(path)
                except Exception:  # noqa: BLE001
                    shutil.rmtree(episode_dir, ignore_errors=True)
                    continue

                if not (PAIR <= set(ep_arch)) or winner is None:
                    shutil.rmtree(episode_dir, ignore_errors=True)
                    continue

                final = args.out if hasattr(args, "out") else OUT
                final_file = final / path.name
                shutil.move(str(path), final_file)
                shutil.rmtree(episode_dir, ignore_errors=True)
                kept.add(episode_id)
                manifest["episodes"].append(
                    {
                        "episode_id": episode_id,
                        "submission_id": submission_id,
                        "team": team_name,
                        "winner_side": winner,
                        "archetypes": ep_arch,
                        "teams": names,
                        "file": final_file.name,
                    }
                )
                have += 1
                new_rows += 1
                cr.save_manifest(MANIFEST, manifest)
                if new_rows % 5 == 0:
                    print(f"    kept +{new_rows}: {episode_id} ({ep_arch[0]} vs {ep_arch[1]})", flush=True)

    shutil.rmtree(TMP, ignore_errors=True)
    minutes = (time.perf_counter() - started) / 60
    print(
        f"\nprobed {probes} submissions, +{new_rows} pair episodes "
        f"({start_count} -> {have}) in {minutes:.1f} min",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
