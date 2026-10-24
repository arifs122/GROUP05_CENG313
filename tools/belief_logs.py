#!/usr/bin/env python3
"""Summarize observation-only BELIEF lines from Kaggle agent logs.

Usage:
    python3 tools/belief_logs.py logs/episode-*-agent-*-logs.json
    python3 tools/belief_logs.py logs/*.json --json results/belief_summary.json

Kaggle stores one JSON object per agent call, with stdout/stderr strings.  The
bot writes at most one `BELIEF {...}` line per turn; search rollouts are excluded.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def extract_beliefs(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            stdout = node.get("stdout")
            if isinstance(stdout, str):
                for line in stdout.splitlines():
                    if line.startswith("BELIEF "):
                        try:
                            out.append(json.loads(line[len("BELIEF "):]))
                        except json.JSONDecodeError:
                            pass
            for value in node.values():
                if isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    return out


def summarize(records: list[dict]) -> dict:
    arches = Counter()
    phases = Counter()
    boss = []
    judge = []
    lillie = []
    setup = defaultdict(list)
    energy = defaultdict(list)
    my_boss = []

    for row in records:
        op = row.get("op") or {}
        me = row.get("me") or {}
        phase = row.get("ph", "?")
        phases[phase] += 1
        best = op.get("best")
        if best:
            arches[best] += 1
        for key, target in (("boss", boss), ("judge", judge), ("lillie", lillie)):
            value = op.get(key)
            if isinstance(value, (int, float)):
                target.append(float(value))
        if isinstance(me.get("setup"), (int, float)):
            setup[phase].append(float(me["setup"]))
        if isinstance(me.get("energy"), (int, float)):
            energy[phase].append(float(me["energy"]))
        if isinstance(me.get("boss"), (int, float)):
            my_boss.append(float(me["boss"]))

    def avg(values):
        return round(sum(values) / len(values), 3) if values else None

    def entropy(row):
        arch = row.get("op", {}).get("arch") or []
        return -sum(p * math.log(max(p, 1e-9)) for _, p in arch if p > 0)

    return {
        "decisions": len(records),
        "phases": dict(phases),
        "top_archetypes": arches.most_common(15),
        "opponent_prizes": {
            "boss": avg(boss),
            "judge": avg(judge),
            "lillie": avg(lillie),
        },
        "own_prizes": {
            "boss": avg(my_boss),
            "setup_by_phase": {phase: avg(values) for phase, values in setup.items()},
            "energy_by_phase": {phase: avg(values) for phase, values in energy.items()},
        },
        "mean_archetype_entropy_top3": round(
            sum(entropy(row) for row in records) / max(1, len(records)), 3
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize BELIEF lines")
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    files: list[Path] = []
    for path in args.logs:
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        else:
            files.append(path)

    records = []
    for path in files:
        try:
            records.extend(extract_beliefs(path))
        except (OSError, json.JSONDecodeError):
            continue

    summary = summarize(records)
    summary["files"] = [str(path) for path in files]
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"written: {args.json}")

    print(f"files: {len(files)}  decisions: {summary['decisions']}")
    print("phases:", summary["phases"])
    print("top archetypes:")
    for name, count in summary["top_archetypes"]:
        print(f"  {name}: {count}")
    print("opponent key-card averages:", summary["opponent_prizes"])
    print("own averages:", summary["own_prizes"])
    print("mean top-3 archetype entropy:", summary["mean_archetype_entropy_top3"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
