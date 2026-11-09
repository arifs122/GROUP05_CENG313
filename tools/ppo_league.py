#!/usr/bin/env python3
"""PPO-vs-PPO deck league.

Runs the same PPO bundle against itself with different 60-card lists, so the
result measures the deck/policy combination rather than a weak local bot.
Every match goes through ``tools/arena.py``; the PPO bundle's ``main.py`` is
used unchanged, and ``PYTHONPATH`` is pointed at the bundle so its lazy
``ptcg`` imports keep working after the arena's module cleanup.

Examples
--------
    # all CSVs in a folder, both deck orders, 15 games per cell
    .venv/bin/python tools/ppo_league.py \
        --decks decks/ppo_candidates --games 15 --workers 6 \
        --out results/ppo_league.json

    # a hand-picked set
    .venv/bin/python tools/ppo_league.py \
        --decks bots/ppo_public/deck.csv decks/meta/mega_excadrill_top_b.csv \
        --names alakazam1074 mega_knshnb --games 30 --workers 4

The output JSON contains the per-cell win matrix, game counts, overall win
rates and a Bradley-Terry strength estimate.  The console prints the ranking
and a compact matrix.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT / "tools" / "arena.py"
DEFAULT_BASE = ROOT / "bots" / "ppo_public"


def _slug(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text or "deck"


def _read_deck(path: Path) -> list[int]:
    cards = [int(x) for x in path.read_text(encoding="utf-8").split() if x.strip()]
    if len(cards) != 60:
        raise ValueError(f"{path}: expected 60 cards, found {len(cards)}")
    return cards


def _make_bot(base: Path, deck: Path, dest: Path) -> Path:
    """Create a self-contained bot dir that reuses the base PPO bundle."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("model.pt", "ptcg", "cg", "data"):
        src = base / name
        if not src.exists():
            raise FileNotFoundError(f"{base}: missing {name}")
        os.symlink(src, dest / name)
    (dest / "main.py").write_bytes((base / "main.py").read_bytes())
    (dest / "deck.csv").write_bytes(deck.read_bytes())
    return dest


def _run_pair(
    arena: Path,
    base: Path,
    bot_a: Path,
    bot_b: Path,
    games: int,
    seed: int,
    bo3: bool,
    out_json: Path,
    timeout: float,
) -> dict:
    cmd = [
        sys.executable,
        str(arena),
        "--bot0",
        str(bot_a),
        "--bot1",
        str(bot_b),
        "--games",
        str(games),
        "--seed",
        str(seed),
        "--quiet",
        "--out",
        str(out_json),
    ]
    if bo3:
        cmd.append("--bo3")
    env = {**os.environ, "PYTHONPATH": str(base)}
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "seconds": time.time() - t0}
    if proc.returncode != 0 or not out_json.exists():
        tail = (proc.stdout + proc.stderr)[-1200:]
        return {
            "ok": False,
            "error": f"arena rc={proc.returncode}",
            "tail": tail,
            "seconds": time.time() - t0,
        }
    try:
        data = json.loads(out_json.read_text())
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"bad json: {exc}", "seconds": time.time() - t0}
    wins = data.get("wins") or [0, 0, 0]
    return {
        "ok": True,
        "wins": [int(x) for x in wins],
        "games": int(data.get("games", games)),
        "errors": int(data.get("errors", 0)),
        "seconds": time.time() - t0,
    }


def _bradley_terry(names: list[str], cells: dict[tuple[int, int], dict]) -> dict[str, float]:
    """Fit a simple Bradley-Terry model to the pair results."""
    n = len(names)
    if n < 2:
        return {names[0]: 0.0} if names else {}
    try:
        import numpy as np
        from scipy.optimize import minimize
    except Exception:  # pragma: no cover - scipy is present in the project venv
        return {name: 0.0 for name in names}

    rows: list[tuple[int, int, int, int]] = []
    for (i, j), cell in cells.items():
        if not cell.get("ok"):
            continue
        w, l = cell["wins"][0], cell["wins"][1]
        if w + l:
            rows.append((i, j, w, l))
    if not rows:
        return {name: 0.0 for name in names}

    def nll(theta):
        # theta has n-1 free values; the first rating is pinned to 0.
        s = np.concatenate(([0.0], theta))
        total = 0.0
        for i, j, w, l in rows:
            p = 1.0 / (1.0 + math.exp(-(s[i] - s[j])))
            p = min(max(p, 1e-9), 1.0 - 1e-9)
            total -= w * math.log(p) + l * math.log(1.0 - p)
        return total + 1e-3 * float(np.dot(theta, theta))

    res = minimize(
        nll,
        np.zeros(n - 1),
        method="L-BFGS-B",
        options={"maxiter": 1000},
    )
    s = np.concatenate(([0.0], res.x))
    # center for readability
    s = s - s.mean()
    return {name: float(s[i]) for i, name in enumerate(names)}


def _print_matrix(names: list[str], cells: dict[tuple[int, int], dict], top: int = 20) -> None:
    show = min(top, len(names))
    width = max(10, max(len(n) for n in names[:show]) + 1)
    header = " " * width + "".join(f"{n[:8]:>9s}" for n in names[:show])
    print(header)
    for i in range(show):
        row = f"{names[i][: width - 1]:<{width}s}"
        for j in range(show):
            cell = cells.get((i, j))
            if not cell or not cell.get("ok"):
                row += f"{'--':>9s}"
            else:
                w, l = cell["wins"][0], cell["wins"][1]
                n = w + l
                row += f"{(w / n if n else 0.0):>8.1%} "
        print(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decks", nargs="+", type=Path, required=True,
                        help="deck CSV files, or directories containing them")
    parser.add_argument("--names", nargs="*", default=None,
                        help="display names (same order/count as --decks)")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE,
                        help="PPO bundle with model.pt, main.py, ptcg/, cg/, data/")
    parser.add_argument("--games", type=int, default=20, help="games per ordered pair")
    parser.add_argument("--repeats", type=int, default=1,
                        help="independent arena processes per cell (fresh engine RNG)")
    parser.add_argument("--workers", type=int, default=4, help="parallel arena processes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bo3", action="store_true", help="best-of-three matches")
    parser.add_argument("--one-way", action="store_true",
                        help="run each unordered pair once (A as P0) instead of both orders")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds per pair")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "ppo_league.json")
    parser.add_argument("--keep", action="store_true", help="keep temporary bot dirs")
    parser.add_argument("--matrix-top", type=int, default=20)
    args = parser.parse_args()

    deck_paths: list[Path] = []
    for entry in args.decks:
        if entry.is_dir():
            deck_paths.extend(sorted(entry.glob("*.csv")))
        else:
            deck_paths.append(entry)
    if not deck_paths:
        raise SystemExit("no deck CSVs found")

    # stable de-duplication by exact 60-card multiset
    unique: list[Path] = []
    seen: set[tuple[int, ...]] = set()
    for path in deck_paths:
        cards = tuple(sorted(_read_deck(path)))
        if cards in seen:
            continue
        seen.add(cards)
        unique.append(path)
    deck_paths = unique

    if args.names:
        if len(args.names) != len(deck_paths):
            raise SystemExit(f"--names has {len(args.names)} entries, {len(deck_paths)} decks")
        names = args.names
    else:
        names = [p.stem for p in deck_paths]

    base = args.base.resolve()
    if not (base / "model.pt").exists():
        raise SystemExit(f"{base}: model.pt not found")

    tmp_root = Path(tempfile.mkdtemp(prefix="ppo_league_", dir="/tmp/opencode"))
    bots: list[Path] = []
    try:
        for name, deck in zip(names, deck_paths):
            bots.append(_make_bot(base, deck.resolve(), tmp_root / _slug(name)))

        pairs: list[tuple[int, int]] = []
        for i in range(len(bots)):
            for j in range(len(bots)):
                if i == j:
                    continue
                if args.one_way and j < i:
                    continue
                pairs.append((i, j))

        repeats = max(1, args.repeats)
        tasks = [(i, j, r) for (i, j) in pairs for r in range(repeats)]
        print(f"PPO league: {len(bots)} decks, {len(pairs)} ordered pairs, "
              f"{args.games} games x {repeats} runs/cell, {args.workers} workers")
        print(f"temporary bots: {tmp_root}")

        cells: dict[tuple[int, int], dict] = {
            (i, j): {"ok": True, "wins": [0, 0, 0], "games": 0, "errors": 0,
                     "seconds": 0.0, "runs": 0, "failures": []}
            for (i, j) in pairs
        }
        pending = {pair: 0 for pair in pairs}
        pairs_done = 0
        started = time.time()
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {}
            for (i, j, r) in tasks:
                out_json = tmp_root / f"pair_{i}_{j}_{r}.json"
                # The engine seeds itself per process; the CLI seed still
                # varies turn-order/shuffle choices where the arena uses it.
                seed = args.seed + i * 1000 + j + r * 100000
                fut = pool.submit(
                    _run_pair, ARENA, base, bots[i], bots[j],
                    args.games, seed, args.bo3, out_json, args.timeout,
                )
                futures[fut] = (i, j, r)
            for fut in as_completed(futures):
                i, j, _r = futures[fut]
                cell = fut.result()
                dest = cells[(i, j)]
                if cell.get("ok"):
                    for k in range(3):
                        dest["wins"][k] += cell["wins"][k]
                    dest["games"] += cell["games"]
                    dest["errors"] += cell["errors"]
                    dest["seconds"] += cell["seconds"]
                    dest["runs"] += 1
                else:
                    dest["failures"].append(cell.get("error"))
                pending[(i, j)] += 1
                if pending[(i, j)] == repeats:
                    pairs_done += 1
                    if dest["runs"] == 0:
                        dest["ok"] = False
                        dest["error"] = "all repeats failed"
                        print(f"[{pairs_done:4d}/{len(pairs)}] {names[i][:30]:30s} vs "
                              f"{names[j][:30]:30s} ERROR", flush=True)
                    else:
                        w, l = dest["wins"][0], dest["wins"][1]
                        print(f"[{pairs_done:4d}/{len(pairs)}] {names[i][:30]:30s} vs "
                              f"{names[j][:30]:30s} {w:4d}-{l:<4d} "
                              f"({w / max(1, w + l):5.1%}, runs={dest['runs']})",
                              flush=True)

        elapsed = time.time() - started
        # aggregate
        total_w = [0] * len(names)
        total_g = [0] * len(names)
        for (i, j), cell in cells.items():
            if not cell.get("ok"):
                continue
            w, l = cell["wins"][0], cell["wins"][1]
            total_w[i] += w
            total_w[j] += l
            total_g[i] += w + l
            total_g[j] += w + l
        overall = {
            names[i]: (total_w[i] / total_g[i] if total_g[i] else 0.0)
            for i in range(len(names))
        }
        bt = _bradley_terry(names, cells)
        ranking = sorted(names, key=lambda n: (-bt[n], -overall[n]))

        print(f"\ncompleted in {elapsed / 60:.1f} min")
        print("\nRANKING (Bradley-Terry strength, overall win rate)")
        for rank, name in enumerate(ranking, 1):
            print(f"{rank:2d}. {name:32s} BT={bt[name]:+6.3f}  overall={overall[name]:5.1%}")

        print("\nMATRIX (row win rate vs column; blank = not played)")
        _print_matrix(names, cells, top=args.matrix_top)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "base": str(base),
                "games": args.games,
                "repeats": repeats,
                "bo3": args.bo3,
                "one_way": args.one_way,
                "seed": args.seed,
                "workers": args.workers,
                "decks": [str(p) for p in deck_paths],
                "names": names,
            },
            "elapsed_seconds": elapsed,
            "cells": [
                {
                    "i": i,
                    "j": j,
                    "a": names[i],
                    "b": names[j],
                    **cell,
                }
                for (i, j), cell in sorted(cells.items())
            ],
            "overall": overall,
            "bradley_terry": bt,
            "ranking": ranking,
        }
        args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")
        if args.keep:
            print(f"kept bots in {tmp_root}")
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
