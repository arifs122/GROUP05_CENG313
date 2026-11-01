#!/usr/bin/env python3
"""Download up to N winning replays for each top leaderboard team."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / ".venv" / "bin" / "kaggle"
OUT = ROOT / "replays" / "top5_nov1"
TMP = Path("/tmp/opencode/top5_download")
TARGET = 50
MAX_ATTEMPTS = 220

TEAMS = {
    "17014244": ("やる気元気ミワハルキ 2nd ver.", "57697887"),
    "17013640": ("Petit Canard", "57699225"),
    "17132686": ("Dipam Chakraborty", "57666077"),
    "17153775": ("knshnb", "57867588"),
    "16981607": ("RtoABC2", "57751477"),
}


def episodes(submission: str) -> list[str]:
    cmd = [str(KAGGLE), "competitions", "episodes", submission, "--format", "csv"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print("episodes failed", submission, p.stderr[-300:])
        return []
    lines = p.stdout.replace("\r", "").splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("id,")), None)
    if start is None:
        return []
    rows = list(csv.DictReader(lines[start:]))
    return [r["id"] for r in rows if r.get("type") == "EpisodeType.EPISODE_TYPE_PUBLIC"]


def download(eid: str, dest_dir: Path) -> Path | None:
    target = dest_dir / f"episode-{eid}-replay.json"
    if target.is_file():
        return target
    for attempt in range(3):
        p = subprocess.run(
            [str(KAGGLE), "competitions", "replay", eid, "-p", str(dest_dir), "-q"],
            capture_output=True, text=True,
        )
        if target.is_file():
            return target
        if "429" in p.stderr or "Too Many Requests" in p.stderr:
            time.sleep(8)
            continue
        break
    return target if target.is_file() else None


def winner_index(e: dict) -> int | None:
    rewards = e.get("rewards") or []
    for i, r in enumerate(rewards):
        if r == 1:
            return i
    return None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for team_id, (team_name, submission) in TEAMS.items():
        team_dir = OUT / team_id
        team_dir.mkdir(parents=True, exist_ok=True)
        eps = episodes(submission)
        print(f"\n=== {team_id} {team_name} sub={submission} episodes={len(eps)} ===", flush=True)
        wins = len(list(team_dir.glob("episode-*.json")))
        manifest[team_id] = {"team": team_name, "submission": submission,
                             "wins": [int(p.name.split("-")[1]) for p in team_dir.glob("episode-*.json")]}
        print(f"{team_name}: existing {wins} wins", flush=True)
        cursor = 0
        while wins < TARGET and cursor < min(len(eps), MAX_ATTEMPTS):
            batch = eps[cursor:cursor + 40]
            cursor += len(batch)
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(download, eid, TMP): eid for eid in batch}
                for future in as_completed(futures):
                    eid = futures[future]
                    dest = future.result()
                    if dest is None:
                        continue
                    try:
                        e = json.loads(dest.read_text(encoding="utf-8"))
                    except Exception:
                        dest.unlink(missing_ok=True)
                        continue
                    names = (e.get("info") or {}).get("TeamNames") or []
                    idx = winner_index(e)
                    if idx is not None and idx < len(names) and team_name in str(names[idx]):
                        final = team_dir / dest.name
                        if final.exists():
                            dest.unlink(missing_ok=True)
                            continue
                        shutil.move(str(dest), str(final))
                        wins += 1
                        manifest[team_id]["wins"].append(eid)
                        print(f"{team_name}: win {wins}/{TARGET} (episode {eid})", flush=True)
                        if wins >= TARGET:
                            break
                    else:
                        dest.unlink(missing_ok=True)
        print(f"DONE {team_id}: {wins} wins", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("written", OUT / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
