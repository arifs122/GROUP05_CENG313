#!/usr/bin/env python3
"""Team-clustered stats for high-level replays.

Reads results/master_replays.csv and restricts to source=top20.  Each game
contributes one observation per side.  Confidence intervals resample TEAMS,
not games, because decisions from the same pilot are correlated.
"""
from __future__ import annotations
import argparse, csv, random, statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_rows(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("source") != "top20":
                continue
            if r.get("winner", "") == "":
                continue
            try:
                wi = int(r["winner"])
            except ValueError:
                continue
            rows.append(r)
    return rows


def records_for(rows, arch=None, opp=None):
    """Return list of (team, win) for deck ``arch`` (optionally vs opp)."""
    out = []
    for r in rows:
        a0, a1 = r["arch0"], r["arch1"]
        for side, other in ((0, 1), (1, 0)):
            a = a0 if side == 0 else a1
            b = a1 if side == 0 else a0
            if arch is not None and a != arch:
                continue
            if opp is not None and b != opp:
                continue
            team = r["team0"] if side == 0 else r["team1"]
            out.append((team, int(int(r["winner"]) == side)))
    return out


def summarize(records, reps=3000, seed=13):
    if not records:
        return {}
    by = defaultdict(list)
    for team, win in records:
        by[team].append(win)
    teams = list(by)
    games = len(records); wins = sum(w for _, w in records)
    team_rates = [sum(v) / len(v) for v in by.values()]
    # Cluster bootstrap: resample teams with replacement.
    rng = random.Random(seed)
    boot = []
    for _ in range(reps):
        sample = [rng.choice(teams) for _ in teams]
        ww = sum(sum(by[t]) for t in sample)
        nn = sum(len(by[t]) for t in sample)
        boot.append(ww / nn if nn else 0.5)
    boot.sort()
    lo = boot[int(0.025 * len(boot))]
    hi = boot[int(0.975 * len(boot)) - 1]
    top_team, top_list = max(by.items(), key=lambda kv: len(kv[1]))
    top_n = len(top_list)
    return {
        "games": games,
        "wins": wins,
        "win_rate": wins / games if games else 0.5,
        "teams": len(teams),
        "mean_team_rate": statistics.mean(team_rates),
        "median_team_rate": statistics.median(team_rates),
        "ci_low": lo,
        "ci_high": hi,
        "top_team": top_team,
        "top_team_games": top_n,
        "top_team_share": top_n / games if games else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, default=ROOT / "results/master_replays.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--reps", type=int, default=3000)
    args = ap.parse_args()
    rows = load_rows(args.master)
    print("top20 rows", len(rows), flush=True)
    archs = Counter()
    for r in rows:
        archs[r["arch0"]] += 1; archs[r["arch1"]] += 1
    arch_order = [a for a, n in archs.most_common() if n >= 5]
    # Overall deck stats.
    with (args.out_dir / "team_arch_stats.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["archetype", "games", "wins", "win_rate", "teams", "mean_team_rate",
                    "median_team_rate", "ci95_low", "ci95_high", "top_team",
                    "top_team_games", "top_team_share"])
        for a in arch_order:
            s = summarize(records_for(rows, arch=a), reps=args.reps)
            w.writerow([a, s["games"], s["wins"], f"{s['win_rate']:.4f}", s["teams"],
                        f"{s['mean_team_rate']:.4f}", f"{s['median_team_rate']:.4f}",
                        f"{s['ci_low']:.4f}", f"{s['ci_high']:.4f}", s["top_team"],
                        s["top_team_games"], f"{s['top_team_share']:.4f}"])
            print(f"{a:24} n={s['games']:4} wr={s['win_rate']:.3f} teams={s['teams']:2} "
                  f"CI=[{s['ci_low']:.3f},{s['ci_high']:.3f}] equal={s['mean_team_rate']:.3f} "
                  f"top={s['top_team']} {s['top_team_share']:.0%}", flush=True)
    # Dragapult matchup stats.
    with (args.out_dir / "team_dragapult_matchups.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["opponent", "games", "wins", "win_rate", "teams", "mean_team_rate",
                    "ci95_low", "ci95_high", "top_team", "top_team_games"])
        for b in arch_order:
            if b == "Dragapult":
                continue
            s = summarize(records_for(rows, arch="Dragapult", opp=b), reps=args.reps)
            if not s or s["games"] < 5:
                continue
            w.writerow([b, s["games"], s["wins"], f"{s['win_rate']:.4f}", s["teams"],
                        f"{s['mean_team_rate']:.4f}", f"{s['ci_low']:.4f}",
                        f"{s['ci_high']:.4f}", s["top_team"], s["top_team_games"]])
            print(f"Dragapult vs {b:22} n={s['games']:4} wr={s['win_rate']:.3f} "
                  f"teams={s['teams']:2} CI=[{s['ci_low']:.3f},{s['ci_high']:.3f}] "
                  f"equal={s['mean_team_rate']:.3f}", flush=True)
    # All pairs with >=10 games, for matrix audit.
    with (args.out_dir / "team_matchup_ci.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["deck_a", "deck_b", "games", "win_rate", "teams", "mean_team_rate", "ci95_low", "ci95_high"])
        seen = set()
        for a in arch_order:
            for b in arch_order:
                if a == b: continue
                s = summarize(records_for(rows, arch=a, opp=b), reps=args.reps)
                if not s or s["games"] < 10: continue
                w.writerow([a, b, s["games"], f"{s['win_rate']:.4f}", s["teams"],
                            f"{s['mean_team_rate']:.4f}", f"{s['ci_low']:.4f}", f"{s['ci_high']:.4f}"])


if __name__ == "__main__":
    main()
