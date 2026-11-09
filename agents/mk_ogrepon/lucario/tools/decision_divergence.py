#!/usr/bin/env python3
"""Compare our Lucario MAIN choices with strong Mega Lucario replay choices."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
LUCARIO = HERE.parent
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(LUCARIO))
os.environ.setdefault("PTCG_BELIEF_LOG", "0")

import lucario_agent as L  # noqa: E402
import mk_state as S  # noqa: E402
import mk_data as D  # noqa: E402
import replay_meta as rm  # noqa: E402

EXTERNAL = ROOT / "benchmarks" / "external_agents" / "lucario_mega_hariyama"
SEARCH_NAMES = {"Ultra Ball", "Poké Pad", "Dusk Ball", "Fighting Gong",
                "Buddy-Buddy Poffin", "Night Stretcher", "Switch"}


def external_agent():
    name = "external_lucario_divergence"
    spec = importlib.util.spec_from_file_location(name, EXTERNAL / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.USE_SEARCH = False
    module.SEARCH_BUDGET = 0.0
    return module.agent


def action_category(obs, idx: int) -> str:
    opts = S.options(obs)
    if not (0 <= idx < len(opts)):
        return "other"
    o = opts[idx]
    t = S.option_type(o)
    if t == S.O_ATTACH:
        return "energy"
    if t == S.O_ABILITY:
        return "ability"
    if t == S.O_ATTACK:
        return "attack"
    if t == S.O_RETREAT:
        return "retreat"
    if t == S.O_EVOLVE:
        return "evolve"
    if t == S.O_END:
        return "end"
    if t == S.O_PLAY:
        i = S.option_int(o, "index", None)
        cid = S.card_at(obs, S.you(obs), S.HAND, i) if i is not None else None
        name = D.card_name(cid)
        if "Boss" in name:
            return "boss"
        if name in SEARCH_NAMES:
            return "search"
        if D.is_pokemon(cid):
            return "basic"
        return "other"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays", type=Path, default=ROOT / "replays")
    parser.add_argument("--out", type=Path,
                        default=LUCARIO / "reports" / "decision_divergence.json")
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()

    ext = external_agent()
    total = 0
    agree = 0
    ext_total = 0
    ext_agree = 0
    cats = defaultdict(lambda: [0, 0])
    ext_cats = defaultdict(lambda: [0, 0])
    pairs = Counter()
    examples = []
    episodes = 0
    seen = set()

    for path in sorted(args.replays.rglob("*.json")):
        try:
            e = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        eid = (e.get("info") or {}).get("EpisodeId") or e.get("id")
        if eid in seen:
            continue
        seen.add(eid)
        decks = rm.extract_decks(e)
        if not decks:
            continue
        players = [i for i, d in enumerate(decks) if 678 in d]
        if len(players) != 1:
            continue
        me = players[0]
        L._DECK = list(decks[me])
        L._POFFIN_CACHE = None
        episodes += 1
        for player, si, obs, sel, action in rm.iter_decisions(e):
            if player != me or not sel or int(sel.get("type", -1)) != 0:
                continue
            if not action or len(action) != 1:
                continue
            opts = sel.get("option") or []
            idx = int(action[0])
            if not (0 <= idx < len(opts)):
                continue
            try:
                pred = L.agent(obs)
            except Exception:
                continue
            if not pred:
                continue
            pred_idx = int(pred[0])
            try:
                ext_pred = ext(obs)
                ext_idx = int(ext_pred[0]) if ext_pred else -1
            except Exception:
                ext_idx = -1
            cat = action_category(obs, idx)
            total += 1
            ok = pred_idx == idx
            agree += ok
            cats[cat][0] += ok
            cats[cat][1] += 1
            if ext_idx >= 0:
                ext_total += 1
                ext_ok = ext_idx == idx
                ext_agree += ext_ok
                ext_cats[cat][0] += ext_ok
                ext_cats[cat][1] += 1
            if not ok:
                try:
                    expert = rm.option_label(obs, me, opts[idx])
                    ours = rm.option_label(obs, me, opts[pred_idx]) if 0 <= pred_idx < len(opts) else "?"
                except Exception:
                    expert, ours = "?", "?"
                pairs[(cat, expert, ours)] += 1
                if len(examples) < args.examples:
                    examples.append({"episode": eid, "turn": obs.get("current", {}).get("turn"),
                                     "category": cat, "expert": expert, "ours": ours,
                                     "bench": [D.card_name(S.pokemon_id(p)) for p in S.bench(obs, me)],
                                     "active": D.card_name(S.pokemon_id(S.active(obs, me))) if S.active(obs, me) else "",
                                     "hand": S.hand_count(obs, me)})

    cat_report = {}
    for c, (a, n) in cats.items():
        cat_report[c] = {"agree": a, "total": n, "rate": round(a / max(1, n), 3)}
    ext_cat_report = {}
    for c, (a, n) in ext_cats.items():
        ext_cat_report[c] = {"agree": a, "total": n, "rate": round(a / max(1, n), 3)}
    report = {
        "episodes": episodes,
        "main_decisions": total,
        "our_top1_agreement": round(agree / max(1, total), 4),
        "external_top1_agreement": round(ext_agree / max(1, ext_total), 4),
        "by_category": cat_report,
        "external_by_category": ext_cat_report,
        "top_mismatches": [
            {"count": c, "category": k[0], "expert": k[1], "ours": k[2]}
            for k, c in pairs.most_common(30)
        ],
        "examples": examples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("episodes", "main_decisions", "our_top1_agreement",
                       "external_top1_agreement", "by_category")}, indent=2))
    print("top mismatches:")
    for x in report["top_mismatches"][:10]:
        print(f"  {x['count']:4d} {x['category']:8s} expert={x['expert']!r} ours={x['ours']!r}")
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
