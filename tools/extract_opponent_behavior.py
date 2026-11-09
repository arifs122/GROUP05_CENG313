#!/usr/bin/env python3
"""Extract opponent/opponent-facing behavior from high-level replays.

Only source=top20 rows are used.  Every selected option becomes one event;
actions from all decks are kept (opponent model), but no action is treated as
our policy label here.
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "mazlum")); sys.path.insert(0, str(ROOT / "tools"))
import features as F  # noqa: E402
import replay_meta as R  # noqa: E402


def parse_one(r: dict):
    try:
        ep = json.loads(Path(r["path"]).read_text())
    except Exception:
        return None, None, None
    events=[]; profile={}; card_counts=Counter()
    n=0
    # first event tracking per player
    first=defaultdict(dict)
    for d in F.iter_decisions(ep):
        if not d.action: continue
        side=d.player; opp=1-side
        arch=r[f"arch{side}"]; opp_arch=r[f"arch{opp}"]
        team=r[f"team{side}"]; won = 1 if r["winner"]!="" and int(r["winner"])==side else 0
        cur=d.observation.get("current") or {}; turn=int(cur.get("turn",0) or 0)
        sel=d.select or {}; ctx=int(sel.get("context",-1)); st=int(sel.get("type",-1))
        opts=sel.get("option") or []
        for ix in d.action:
            try: o=opts[int(ix)]
            except Exception: continue
            typ=int(o.get("type",-1)); actor,target=F._option_cards(d.observation,side,o)
            attack=int(o.get("attackId") or -1)
            event=(arch,opp_arch,team,won,turn,ctx,st,typ,actor or 0,target or 0,attack)
            events.append(event); n+=1
            if actor: card_counts[int(actor)]+=1
            # first occurrence per game for key event types
            key=None
            if typ==13: key='first_attack'
            elif typ==12: key='first_retreat'
            elif typ==10: key='first_ability'
            elif typ==9: key='first_evolve'
            elif typ==7: key='first_play'
            elif typ==14: key='first_end'
            if key and key not in first[side]:
                first[side][key]=turn
            if actor==1182 and 'first_boss' not in first[side]: first[side]['first_boss']=turn
        n+=0
    for side in [0,1]:
        arch=r[f"arch{side}"]
        p=first[side]
        row={'episode_id':r['episode_id'],'source':'top20','arch':arch,'opp_arch':r[f"arch{1-side}"],
             'team':r[f'team{side}'],'won':1 if r['winner']!="" and int(r['winner'])==side else 0,
             'first_attack':p.get('first_attack',''),'first_boss':p.get('first_boss',''),
             'first_ability':p.get('first_ability',''),'first_evolve':p.get('first_evolve','')}
        profile.setdefault(side,row)
    return events, profile, card_counts


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--master',type=Path,default=ROOT/'results/master_replays.csv'); ap.add_argument('--out-dir',type=Path,default=ROOT/'results'); ap.add_argument('--workers',type=int,default=4); args=ap.parse_args()
    rows=[r for r in csv.DictReader(args.master.open(encoding='utf-8')) if r['source']=='top20']
    print('games',len(rows),flush=True)
    all_events=[]; profiles=[]; cards=Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs=[ex.submit(parse_one,r) for r in rows]
        for i,f in enumerate(as_completed(futs),1):
            ev,pr,cc=f.result()
            if ev: all_events+=ev
            if pr: profiles+=list(pr.values())
            if cc: cards.update(cc)
            if i%200==0: print('parsed',i,flush=True)
    with (args.out_dir/'opponent_events.csv').open('w',newline='',encoding='utf-8') as fh:
        w=csv.writer(fh); w.writerow(['arch','opp_arch','team','won','turn','context','select_type','option_type','actor','target','attack'])
        w.writerows(all_events)
    with (args.out_dir/'opponent_profiles.csv').open('w',newline='',encoding='utf-8') as fh:
        w=csv.writer(fh); w.writerow(['episode_id','source','arch','opp_arch','team','won','first_attack','first_boss','first_ability','first_evolve'])
        w.writerows([[p[k] for k in ['episode_id','source','arch','opp_arch','team','won','first_attack','first_boss','first_ability','first_evolve']] for p in profiles])
    with (args.out_dir/'opponent_cards.csv').open('w',newline='',encoding='utf-8') as fh:
        w=csv.writer(fh); w.writerow(['card_id','card','count'])
        for cid,n in cards.most_common(): w.writerow([cid,R.card_name(cid),n])
    print('events',len(all_events),'profiles',len(profiles),'cards',len(cards),flush=True)
    # quick aggregate by arch
    agg=defaultdict(lambda:{'games':0,'first_attack':0,'first_attack_n':0,'first_boss':0,'first_boss_n':0})
    for p in profiles:
        a=p['arch']; agg[a]['games']+=1
        for k in ['first_attack','first_boss']:
            if str(p[k]).strip()!='':
                agg[a][k]+=int(p[k]); agg[a][k+'_n']+=1
    for a,v in sorted(agg.items(),key=lambda kv:-kv[1]['games'])[:20]:
        print(f"{a:26} games={v['games']:4} first_atk={v['first_attack']/max(1,v['first_attack_n']):.1f} first_boss={v['first_boss']/max(1,v['first_boss_n']):.1f}")

if __name__=='__main__': main()
