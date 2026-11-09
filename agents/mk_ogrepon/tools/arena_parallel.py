#!/usr/bin/env python3
"""Run a BO3 arena benchmark in parallel shards and merge the JSON results."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot0", type=Path, default=ROOT / "agents" / "mk_ogrepon" / "bot")
    parser.add_argument("--bot1", type=Path, required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bo3", action="store_true", default=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    workers = max(1, min(args.workers, args.games))
    per = [args.games // workers + (1 if i < args.games % workers else 0)
           for i in range(workers)]
    tmp = Path("/tmp/opencode/mk_parallel")
    tmp.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, n in enumerate(per):
        if n <= 0:
            continue
        out = tmp / f"shard_{i}.json"
        cmd = [
            sys.executable, str(ROOT / "tools" / "arena.py"),
            "--bot0", str(args.bot0), "--bot1", str(args.bot1),
            "--games", str(n), "--seed", str(args.seed + i * 1000),
            "--out", str(out), "--quiet",
        ]
        if args.bo3:
            cmd.append("--bo3")
        jobs.append((i, n, out, cmd))

    env = dict(os.environ)
    env.setdefault("MK_SEARCH", "1")
    env.setdefault("MK_BUDGET", "0.18")
    env.setdefault("MK_PRIOR_WEIGHT", "60")
    env.setdefault("MK_VALUE_WEIGHT", "0")

    def run(job):
        i, n, out, cmd = job
        p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
        return i, n, out, p.returncode, p.stdout, p.stderr

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for fut in as_completed(futures):
            i, n, out, rc, stdout, stderr = fut.result()
            print(f"shard {i}: games={n} rc={rc} out={out}")
            if rc != 0:
                print(stdout[-1000:])
                print(stderr[-1000:])
            results.append((i, n, out))

    merged = {
        "format": "bo3", "p0": {"label": "mk_ogrepon"},
        "p1": {"label": str(args.bot1)}, "games": 0, "seed": args.seed,
        "wins": [0, 0, 0], "errors": 0, "results": [],
    }
    for i, n, out in sorted(results):
        if not out.is_file():
            continue
        d = json.loads(out.read_text())
        merged["games"] += int(d.get("games", 0))
        for k in range(3):
            merged["wins"][k] += int(d.get("wins", [0, 0, 0])[k])
        merged["errors"] += int(d.get("errors", 0))
        merged["results"].extend(d.get("results", []))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    rounds = [r for m in merged["results"] for r in m.get("rounds", [])]
    played = max(1, merged["games"])
    print()
    print(f"games={merged['games']} workers={workers} seed={args.seed}")
    print(f"P0 wins: {merged['wins'][0]} ({merged['wins'][0]/played:.1%})")
    print(f"P1 wins: {merged['wins'][1]} ({merged['wins'][1]/played:.1%})")
    print(f"draws: {merged['wins'][2]}  errors: {merged['errors']}")
    if rounds:
        print(f"rounds={len(rounds)} "
              f"round_wins={{0:{sum(1 for r in rounds if r['winner']==0)}, "
              f"1:{sum(1 for r in rounds if r['winner']==1)}}} "
              f"avg_round_sec={sum(r['seconds'] for r in rounds)/len(rounds):.3f}")
    print(f"merged -> {args.out}")
    return 0 if merged["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
