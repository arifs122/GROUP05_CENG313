"""Mine our own Kaggle matches (v16 meta) into the replay archive.

Lists episodes of our competition submissions, downloads the ones not yet in
the manifest, and keeps those where our Banette deck appears. New episode ids
are written to a JSON list so the dataset builder can isolate them.

Usage::

    python tools/mazlum/mine_own.py --submissions 57835594 57643520 --delay 0.6
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "tools", ROOT / "tools" / "mazlum"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import collect_replays as cr  # noqa: E402

MANIFEST = ROOT / "replays" / "_collected.json"
OUT = ROOT / "replays"
TMP = ROOT / "replays" / "_own_tmp"
TARGET = "Banette"


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine our own Kaggle matches")
    parser.add_argument("--submissions", type=int, nargs="+", default=[57835594, 57643520])
    parser.add_argument("--delay", type=float, default=0.6)
    parser.add_argument("--max-per-sub", type=int, default=400)
    parser.add_argument("--any-deck", action="store_true", help="keep every deck, not just Banette")
    parser.add_argument("--ids-out", type=Path, default=ROOT / "data" / "mazlum" / "v16_episode_ids.json")
    args = parser.parse_args()

    manifest = cr.load_manifest(MANIFEST)
    manifest.setdefault("episodes", [])
    manifest.setdefault("submissions", {})
    kept = {int(e["episode_id"]) for e in manifest["episodes"]}
    cr._DELAY = max(0.0, args.delay)
    TMP.mkdir(parents=True, exist_ok=True)

    added: list[int] = []
    for submission_id in args.submissions:
        try:
            episodes = cr.list_episodes(submission_id)
        except Exception as exc:  # noqa: BLE001
            print(f"sub {submission_id}: list failed ({exc})", flush=True)
            continue
        todo = [int(row["id"]) for row in episodes if int(row["id"]) not in kept][: args.max_per_sub]
        print(f"sub {submission_id}: {len(episodes)} episodes, {len(todo)} new to fetch", flush=True)

        for index, episode_id in enumerate(todo, 1):
            episode_dir = TMP / f"ep-{episode_id}"
            try:
                path = cr.download_replay(episode_id, episode_dir)
                archetypes, winner, names = cr.deck_archetypes(path)
            except Exception as exc:  # noqa: BLE001
                shutil.rmtree(episode_dir, ignore_errors=True)
                print(f"  {episode_id}: failed ({exc})", flush=True)
                continue

            if winner is None or not archetypes:
                shutil.rmtree(episode_dir, ignore_errors=True)
                continue
            if not args.any_deck and TARGET not in archetypes:
                shutil.rmtree(episode_dir, ignore_errors=True)
                continue

            final = OUT / path.name
            shutil.move(str(path), final)
            shutil.rmtree(episode_dir, ignore_errors=True)
            kept.add(episode_id)
            manifest["episodes"].append(
                {
                    "episode_id": episode_id,
                    "submission_id": submission_id,
                    "team": "mazlum",
                    "winner_side": winner,
                    "archetypes": archetypes,
                    "teams": names,
                    "file": final.name,
                }
            )
            added.append(episode_id)
            cr.save_manifest(MANIFEST, manifest)
            if index % 10 == 0:
                print(f"  {index}/{len(todo)} fetched, +{len(added)} kept", flush=True)

    shutil.rmtree(TMP, ignore_errors=True)
    args.ids_out.parent.mkdir(parents=True, exist_ok=True)
    args.ids_out.write_text(json.dumps(sorted(added)), encoding="utf-8")
    print(f"\nadded {len(added)} episodes -> {args.ids_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
