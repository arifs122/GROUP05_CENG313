#!/usr/bin/env python3
"""Build a deduplicated master table from local cabt replays.

This is a lightweight metadata pass: only the first 2 MiB of each replay is
read, which is where EpisodeId, TeamNames, rewards, and the two initial deck
lists live.  Full decision extraction is a separate step.

Outputs (default: results/):
  master_replays.csv
  archetype_shares.csv
  archetype_matchups.csv
  source_counts.csv
"""
from __future__ import annotations

import argparse, json, re, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import replay_meta as R  # noqa: E402

EP_RE = re.compile(r'"EpisodeId"\s*:\s*(\d+)')
TEAM_RE = re.compile(r'"TeamNames"\s*:\s*(\[[^\]]*\])')
REW_RE = re.compile(r'"rewards"\s*:\s*\[([^\]]*)\]')
ACT_RE = re.compile(
    r'"action"\s*:\s*\[\s*\[\s*([0-9,\s]+?)\s*\]\s*,\s*\[\s*([0-9,\s]+?)\s*\]\s*\]'
)


def parse_one(path: Path) -> dict | None:
    try:
        with path.open("rb") as fh:
            text = fh.read(2_000_000).decode("utf-8", "ignore")
        m = EP_RE.search(text)
        a = ACT_RE.search(text)
        if not m or not a:
            return None
        d0 = [int(x) for x in re.findall(r"\d+", a.group(1))]
        d1 = [int(x) for x in re.findall(r"\d+", a.group(2))]
        if len(d0) != 60 or len(d1) != 60:
            return None
        teams: list[str] = []
        tm = TEAM_RE.search(text)
        if tm:
            try:
                teams = [str(x) for x in json.loads(tm.group(1))]
            except Exception:
                pass
        teams += ["?", "?"]
        rewards: list[float | None] = [None, None]
        rm = REW_RE.search(text)
        if rm:
            try:
                vals = json.loads("[" + rm.group(1) + "]")
                if len(vals) == 2:
                    rewards = vals
            except Exception:
                pass
        return {
            "episode_id": int(m.group(1)),
            "path": str(path),
            "teams": teams[:2],
            "decks": [d0, d1],
            "rewards": rewards,
        }
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", type=Path, default=ROOT / "replays")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.replays.rglob("*replay.json"))
    # high_elo flat files only; skip the top20/ and ours/ hardlink views.
    paths = [p for p in paths if p.parent.name not in {"top20", "ours"}]
    print(f"candidate files: {len(paths)}", flush=True)
    parsed: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(parse_one, p) for p in paths]
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            if row:
                parsed.append(row)
            if i % 1000 == 0:
                print(f"parsed {i}/{len(paths)}", flush=True)
    parsed.sort(key=lambda r: r["episode_id"])

    # Source metadata: high_elo queue kind/team; ours log; original parent folder.
    queue = {}
    # Use the complete top20 plan, not only the last download queue.
    plan_path = args.out_dir / "high_elo_plan.json"
    if not plan_path.is_file():
        plan_path = Path("/tmp/opencode/high_elo_plan/plan.json")
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text())
        for team in plan.get("top20", []):
            for sub in team.get("submissions", []):
                for eid in sub.get("eids", []):
                    queue[int(eid)] = {"kind": "top20", "team": team.get("team", "")}
        for sub in plan.get("ours", []):
            for eid in sub.get("eids", []):
                queue[int(eid)] = {"kind": "ours", "team": "ours"}
    qp = args.replays / "high_elo" / "download_queue.json"
    if qp.is_file():
        for q in json.loads(qp.read_text()):
            queue.setdefault(int(q["eid"]), q)
    ours_ids: set[int] = set()
    op = args.replays / "high_elo" / "ours_download_log.jsonl"
    if op.is_file():
        for line in op.read_text().splitlines():
            try:
                ours_ids.add(int(json.loads(line)["eid"]))
            except Exception:
                pass

    rows: list[dict] = []
    seen: set[int] = set()
    for r in parsed:
        eid = int(r["episode_id"])
        if eid in seen:
            continue
        seen.add(eid)
        p = Path(r["path"])
        if "high_elo" in p.parts:
            q = queue.get(eid)
            if q is not None:
                source = q.get("kind", "high_elo")
                source_team = q.get("team", "")
            elif eid in ours_ids:
                source, source_team = "ours", "ours"
            else:
                source, source_team = "high_elo_other", ""
        else:
            try:
                source = p.relative_to(args.replays).parts[0]
            except ValueError:
                source = "unknown"
            source_team = ""
        arch = [R.archetype(d) for d in r["decks"]]
        rew = r["rewards"]
        winner = None
        if len(rew) == 2 and rew[0] is not None and rew[1] is not None and rew[0] != rew[1]:
            winner = 0 if rew[0] > rew[1] else 1
        rows.append({
            "episode_id": eid,
            "source": source,
            "source_team": source_team,
            "team0": r["teams"][0],
            "team1": r["teams"][1],
            "deck0": " ".join(map(str, r["decks"][0])),
            "deck1": " ".join(map(str, r["decks"][1])),
            "arch0": arch[0],
            "arch1": arch[1],
            "reward0": "" if rew[0] is None else rew[0],
            "reward1": "" if rew[1] is None else rew[1],
            "winner": "" if winner is None else winner,
            "path": r["path"],
        })

    import csv
    fields = list(rows[0]) if rows else []
    with (args.out_dir / "master_replays.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    source_counts = Counter(r["source"] for r in rows)
    with (args.out_dir / "source_counts.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["source", "episodes"])
        for k, v in source_counts.most_common():
            w.writerow([k, v])

    # Deck appearances and win/loss.
    deck = Counter()
    wins = Counter()
    match = defaultdict(lambda: [0, 0])  # (A,B) -> games, A wins
    for r in rows:
        a, b = r["arch0"], r["arch1"]
        deck[a] += 1; deck[b] += 1
        if r["winner"] == "":
            continue
        wi = int(r["winner"])
        if wi == 0:
            match[(a, b)][0] += 1; match[(a, b)][1] += 1
            wins[a] += 1; match[(b, a)][0] += 1
        else:
            match[(b, a)][0] += 1; match[(b, a)][1] += 1
            wins[b] += 1; match[(a, b)][0] += 1
    total_decks = sum(deck.values())
    with (args.out_dir / "archetype_shares.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["archetype", "appearances", "share", "wins", "win_rate"])
        for k, v in deck.most_common():
            w.writerow([k, v, f"{v/total_decks:.4f}" if total_decks else "", wins[k], f"{wins[k]/v:.4f}" if v else ""])

    # Ordered matchup matrix, only pairs with games.
    with (args.out_dir / "archetype_matchups.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["deck_a", "deck_b", "games", "a_wins", "a_win_rate"])
        for (a, b), (g, aw) in sorted(match.items(), key=lambda kv: (-kv[1][0], kv[0])):
            if g:
                w.writerow([a, b, g, aw, f"{aw/g:.4f}"])


    # High-level-only report: top20 replays, excluding our own games.
    hi = [r for r in rows if r["source"] == "top20"]
    hdeck = Counter(); hwins = Counter(); hmatch = defaultdict(lambda: [0,0])
    for r in hi:
        a,b=r["arch0"],r["arch1"]; hdeck[a]+=1; hdeck[b]+=1
        if r["winner"]=="": continue
        wi=int(r["winner"])
        if wi==0:
            hmatch[(a,b)][0]+=1; hmatch[(a,b)][1]+=1; hwins[a]+=1; hmatch[(b,a)][0]+=1
        else:
            hmatch[(b,a)][0]+=1; hmatch[(b,a)][1]+=1; hwins[b]+=1; hmatch[(a,b)][0]+=1
    htotal=sum(hdeck.values())
    with (args.out_dir/"high_elo_archetype_shares.csv").open("w",newline="",encoding="utf-8") as fh:
        w=csv.writer(fh); w.writerow(["archetype","appearances","share","wins","win_rate"])
        for k,v in hdeck.most_common():
            w.writerow([k,v,f"{v/htotal:.4f}" if htotal else "",hwins[k],f"{hwins[k]/v:.4f}" if v else ""])
    with (args.out_dir/"high_elo_archetype_matchups.csv").open("w",newline="",encoding="utf-8") as fh:
        w=csv.writer(fh); w.writerow(["deck_a","deck_b","games","a_wins","a_win_rate"])
        for (a,b),(g,aw) in sorted(hmatch.items(),key=lambda kv:(-kv[1][0],kv[0])):
            if g: w.writerow([a,b,g,aw,f"{aw/g:.4f}"])
    print(f"high_elo top20 rows: {len(hi)}; deck appearances: {htotal}")
    print(f"unique episodes: {len(rows)}")
    print("sources:")
    for k, v in source_counts.most_common():
        print(f"  {k:20} {v}")
    print("\ndeck shares:")
    for k, v in deck.most_common(20):
        print(f"  {k:32} {v:5} {v/total_decks*100:5.1f}%  win {wins[k]/v*100 if v else 0:5.1f}%")
    print("\nDragapult matchups (top 15 by games):")
    for (a, b), (g, aw) in sorted(match.items(), key=lambda kv: -kv[1][0]):
        if a == "Dragapult" and g >= 5:
            print(f"  Dragapult vs {b:30} {aw:4}/{g:4} {aw/g*100:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
