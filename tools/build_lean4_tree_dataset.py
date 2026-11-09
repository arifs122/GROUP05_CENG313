#!/usr/bin/env python3
"""Build option-level decision rows for the lean4 deck from replays.

Each row: state(50) + option(48) + context features -> selected (0/1).
Only decisions made by a player using the exact lean4 deck are included.
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "mazlum"))
sys.path.insert(0, str(ROOT / "tools"))
import features as F  # noqa: E402
import replay_meta as R  # noqa: E402

LEAN = sorted(int(x) for x in (ROOT / "bots/dragapult_lean4/deck.csv").read_text().split() if x.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, default=ROOT / "results/master_replays.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data/lean4_tree_dataset.npz")
    ap.add_argument("--max-games", type=int, default=0)
    args = ap.parse_args()
    rows = list(csv.DictReader(args.master.open(encoding="utf-8")))
    games = []
    for r in rows:
        d0 = sorted(map(int, r["deck0"].split())); d1 = sorted(map(int, r["deck1"].split()))
        if d0 == LEAN or d1 == LEAN:
            games.append(r)
    # Prefer lean submission games first, then ours.
    games.sort(key=lambda r: (r["source"] != "lean_57629056", r["episode_id"]))
    if args.max_games:
        games = games[: args.max_games]
    print("games", len(games), Counter(r["source"] for r in games), flush=True)

    states=[]; options=[]; cids=[]; labels=[]; contexts=[]; sel_types=[]; mins=[]; maxs=[]; eps=[]; steps=[]; results=[]; decks=[]
    for gi,r in enumerate(games):
        try:
            ep = json.loads(Path(r["path"]).read_text())
        except Exception:
            continue
        decks0 = R.extract_decks(ep)
        if not decks0:
            continue
        d0 = sorted(decks0[0]); d1 = sorted(decks0[1])
        myidx = 0 if d0 == LEAN else (1 if d1 == LEAN else None)
        if myidx is None:
            continue
        winner = r["winner"]; won = 1 if winner != "" and int(winner)==myidx else 0
        n = 0
        for d in F.iter_decisions(ep):
            if d.player != myidx or not d.action:
                continue
            state = F.state_vector(d.observation, LEAN)
            opt, mask = F.options_matrix(d.observation)
            ci = F.card_ids_matrix(d.observation)
            sel = d.select or {}
            try:
                ctx = int(sel.get("context", -1)); st = int(sel.get("type", -1))
                mn = int(sel.get("minCount", 0)); mx = int(sel.get("maxCount", 0))
            except Exception:
                continue
            selected = set(int(x) for x in d.action if 0 <= int(x) < F.MAX_OPTIONS)
            if not selected:
                continue
            for i in range(int(mask.sum())):
                states.append(state); options.append(opt[i]); cids.append(ci[i])
                labels.append(1 if i in selected else 0)
                contexts.append(ctx); sel_types.append(st); mins.append(mn); maxs.append(mx)
                eps.append(int(r["episode_id"])); steps.append(int(d.step_index)); results.append(won)
                decks.append(myidx)
                n += 1
        if gi % 20 == 0:
            print("game", gi, r["episode_id"], "rows", n, flush=True)
    if not states:
        raise SystemExit("no rows")
    state = np.stack(states).astype(np.float16)
    option = np.stack(options).astype(np.float16)
    card_ids = np.stack(cids).astype(np.int16)
    label = np.asarray(labels, dtype=np.int8)
    context = np.asarray(contexts, dtype=np.int16)
    sel_type = np.asarray(sel_types, dtype=np.int16)
    min_count = np.asarray(mins, dtype=np.int16)
    max_count = np.asarray(maxs, dtype=np.int16)
    episode = np.asarray(eps, dtype=np.int64)
    step = np.asarray(steps, dtype=np.int32)
    target = np.asarray(results, dtype=np.int8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, state=state, option=option, card_ids=card_ids,
                        label=label, context=context, select_type=sel_type,
                        min_count=min_count, max_count=max_count,
                        episode_id=episode, step_index=step, won=target)
    print("rows", len(label), "selected", int(label.sum()), "episodes", len(set(eps)))
    print("contexts", Counter(contexts).most_common(15))
    print("wrote", args.out, f"{args.out.stat().st_size/1e6:.1f} MB")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
