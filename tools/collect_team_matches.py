#!/usr/bin/env python3
"""Download a bounded set of public matches (wins and losses) for one team."""
from __future__ import annotations
import csv, json, shutil, subprocess, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / ".venv" / "bin" / "kaggle"
TEAM = "Petit Canard"
SUB = "57699225"
OUT = ROOT / "replays" / "top2_policy"
TMP = Path("/tmp/opencode/top2_policy_tmp")
LIMIT = 180
WORKERS = 4


def episodes():
    p = subprocess.run([str(KAGGLE), "competitions", "episodes", SUB, "--format", "csv"],
                       capture_output=True, text=True)
    lines = p.stdout.replace("\r", "").splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("id,")), None)
    if start is None:
        return []
    rows = list(csv.DictReader(lines[start:]))
    return [r["id"] for r in rows if r.get("type") == "EpisodeType.EPISODE_TYPE_PUBLIC"]


def download(eid):
    target = TMP / f"episode-{eid}-replay.json"
    if target.is_file():
        return target
    for _ in range(3):
        p = subprocess.run([str(KAGGLE), "competitions", "replay", eid, "-p", str(TMP), "-q"],
                           capture_output=True, text=True)
        if target.is_file():
            return target
        if "429" in p.stderr or "Too Many Requests" in p.stderr:
            time.sleep(6)
        else:
            break
    return target if target.is_file() else None


def winner_index(e):
    for i, r in enumerate(e.get("rewards") or []):
        if r == 1:
            return i
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    eps = episodes()
    print("episodes", len(eps), flush=True)
    seen = {p.name for p in OUT.glob("episode-*.json")}
    ids = [e for e in eps if f"episode-{e}-replay.json" not in seen][:LIMIT]
    print("to download", len(ids), flush=True)
    for start in range(0, len(ids), 40):
        batch = ids[start:start + 40]
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(download, e): e for e in batch}
            for f in as_completed(futures):
                dest = f.result()
                if dest is None:
                    continue
                try:
                    e = json.loads(dest.read_text(encoding="utf-8"))
                except Exception:
                    dest.unlink(missing_ok=True)
                    continue
                names = (e.get("info") or {}).get("TeamNames") or []
                if TEAM not in " ".join(str(x) for x in names):
                    dest.unlink(missing_ok=True)
                    continue
                final = OUT / dest.name
                if final.exists():
                    dest.unlink(missing_ok=True)
                else:
                    shutil.move(str(dest), str(final))
        print("downloaded", len(list(OUT.glob("episode-*.json"))), flush=True)
    print("done", len(list(OUT.glob("episode-*.json"))))


if __name__ == "__main__":
    main()
