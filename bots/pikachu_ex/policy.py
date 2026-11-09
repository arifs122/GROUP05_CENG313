"""Small rule pilot for the Pikachu ex / Blissey energy-recycling deck.

No search.  It prioritises:
  1. get Pikachu ex into play and attach Prism / basic energy,
  2. evolve Chansey into Blissey ex and use Happy Switch,
  3. use Glass Trumpet / Powerglass to recycle energy,
  4. attack with the strongest legal attack.
"""
from __future__ import annotations
from collections import defaultdict
from cg.api import CardType, SelectContext, SelectType, OptionType, all_attack, all_card_data

# card ids
PIKACHU = 210
CHANSEY = 124
BLISSEY = 125
MEOWTH = 1071
FEZ = 140
PRISM = 16
LIGHTNING = 4
GRASS = 1
METAL = 8
CRISPIN = 1198
LILLIE = 1227
JUDGE = 1213
BOSS = 1182
ULTRA = 1121
POKE_PAD = 1152
GLASS = 1098
POWERGLASS = 1163
CRYSTAL = 1165
NIGHT = 1097
ENERGY_RET = 1118
SWITCH = 1123

CARDS = {c.cardId: c for c in all_card_data()}
ATTACK_DAMAGE = {a.attackId: int(a.damage or 0) for a in all_attack()}
ATTACK_NAME = {a.attackId: a.name for a in all_attack()}

def _g(o, k, d=None):
    if isinstance(o, dict):
        return o.get(k, d)
    return getattr(o, k, d)

def _id(c):
    return int(_g(c, "id", 0) or 0)

def _at(obs, area, index, pi):
    try:
        cur = obs.get("current") or {}
        ps = cur.get("players") or [{}, {}]
        p = ps[pi] or {}
        if area == 1: return (obs.get("select") or {}).get("deck", [])[index]
        if area == 2: return (p.get("hand") or [])[index]
        if area == 3: return (p.get("discard") or [])[index]
        if area == 4: return (p.get("active") or [])[index]
        if area == 5: return (p.get("bench") or [])[index]
        if area == 6: return (p.get("prize") or [])[index]
        if area == 7: return (cur.get("stadium") or [])[index]
        if area == 12: return (cur.get("looking") or [])[index]
    except Exception:
        pass
    return None

def _name(cid):
    c = CARDS.get(cid)
    return c.name if c else str(cid)

def _is_energy(cid):
    c = CARDS.get(cid)
    return c is not None and c.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)

def _is_pokemon(cid):
    c = CARDS.get(cid)
    return c is not None and c.cardType == CardType.POKEMON

def _basic_energy_count(players):
    n = 0
    for p in players:
        for c in (p.get("hand") or []):
            if _id(c) in (PRISM, LIGHTNING, GRASS, METAL): n += 1
    return n

def _bench(obs, pi):
    try: return (obs["current"]["players"][pi].get("bench") or [])
    except Exception: return []

def _active(obs, pi):
    try:
        a = obs["current"]["players"][pi].get("active") or []
        return a[0] if a else None
    except Exception: return None

def _has(obs, pi, cid):
    a = _active(obs, pi)
    return (a is not None and _id(a) == cid) or any(_id(x) == cid for x in _bench(obs, pi))

def _ability_value(cid, obs, me):
    if cid == BLISSEY: return 95
    if cid == MEOWTH: return 85
    if cid == FEZ: return 65
    return 40

def _play_value(cid, obs, me, hand):
    active = _active(obs, me)
    bench = _bench(obs, me)
    has_pika = _has(obs, me, PIKACHU)
    bench_blissey = any(_id(x) == BLISSEY for x in bench)
    discard = obs["current"]["players"][me].get("discard") or []
    if cid == PIKACHU: return 82 if has_pika else 100
    if cid == CHANSEY: return 88
    if cid == BLISSEY: return 70
    if cid == MEOWTH: return 65
    if cid == FEZ: return 42
    if cid == GLASS: return 92 if (has_pika and bench_blissey and any(_is_energy(_id(c)) for c in discard)) else 10
    if cid == CRISPIN: return 90 if has_pika else 30
    if cid == LILLIE: return 78 if len(hand) <= 7 else 30
    if cid == ULTRA: return 86
    if cid == POKE_PAD: return 84
    if cid == JUDGE: return 45 if (obs["current"]["players"][1-me].get("handCount") or 0) >= 5 else 20
    if cid == BOSS: return 60 if len(bench) else 30
    if cid == NIGHT: return 45 if any(_is_energy(_id(c)) or _id(c) == PIKACHU for c in discard) else 10
    if cid == ENERGY_RET: return 55 if any(_is_energy(_id(c)) for c in discard) else 10
    if cid == SWITCH: return 45 if (active is not None and _id(active) != PIKACHU) else 15
    if cid in (CRYSTAL, POWERGLASS): return 100
    return 5

def _attach_value(cid, target, obs, me, hand):
    if target is None: return 0
    tid = _id(target)
    c = CARDS.get(cid)
    if c is not None and c.cardType == CardType.TOOL:
        return 120 if tid == PIKACHU else 70
    if not _is_energy(cid): return 0
    # Energy goes to Pikachu first, then Blissey/Chansey.
    n = len(_g(target, "energies", []) or [])
    if tid == PIKACHU:
        return 120 if n < 3 else 20
    if tid == BLISSEY: return 75 if n < 2 else 15
    if tid == CHANSEY: return 55 if n < 1 else 10
    return 8

def _card_value(cid, ctx, obs, me, effect_id=0):
    active = _active(obs, me)
    bench = _bench(obs, me)
    if ctx in (SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.TO_ACTIVE, SelectContext.SWITCH):
        if cid == PIKACHU: return 100
        if cid == BLISSEY: return 90
        if cid == CHANSEY: return 70
        if cid == MEOWTH: return 35
        if cid == FEZ: return 35
        return 5
    if ctx == SelectContext.SETUP_BENCH_POKEMON:
        if cid == CHANSEY: return 100
        if cid == PIKACHU: return 85
        if cid == MEOWTH: return 60
        if cid == FEZ: return 45
        return 20
    if ctx in (SelectContext.TO_BENCH, SelectContext.TO_HAND):
        if cid == PIKACHU: return 100
        if cid == BLISSEY: return 92
        if cid == CHANSEY: return 88
        if cid == CRISPIN: return 80
        if cid == LILLIE: return 72
        if cid == ULTRA: return 70
        if cid == POKE_PAD: return 68
        if cid == GLASS: return 62
        if _is_energy(cid): return 55
        if cid == MEOWTH: return 50
        if cid == FEZ: return 45
        if cid == BOSS: return 40
        return 15
    if ctx in (SelectContext.DISCARD, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
               SelectContext.DISCARD_ENERGY_CARD, SelectContext.DISCARD_TOOL_CARD):
        # high score = expendable
        if cid in (PIKACHU, CHANSEY, BLISSEY, CRISPIN, LILLIE, ULTRA, POKE_PAD, GLASS, CRYSTAL): return 0
        if cid in (JUDGE, BOSS, SWITCH): return 60
        if _is_energy(cid): return 45
        return 35
    if ctx == SelectContext.ATTACH_FROM:
        if cid == PRISM: return 110
        if _is_energy(cid): return 80
        return 0
    if ctx == SelectContext.ATTACH_TO:
        if cid == PIKACHU: return 110
        if cid == BLISSEY: return 90
        if cid == CHANSEY: return 60
        return 5
    if ctx == SelectContext.SWITCH_ENERGY:
        if cid == PIKACHU: return 100
        if cid == BLISSEY: return 50
        if cid == CHANSEY: return 30
        return 5
    if ctx == SelectContext.DAMAGE_COUNTER or ctx == SelectContext.DAMAGE_COUNTER_ANY:
        # hurt the opponent; active is usually the prize target
        try:
            area = _g(obs.get("select"), "effect", None)
        except Exception:
            pass
        return 50
    if ctx == SelectContext.REMOVE_DAMAGE_COUNTER:
        return 50
    if ctx == SelectContext.TO_DECK or ctx == SelectContext.TO_DECK_BOTTOM:
        return -_card_value(cid, SelectContext.TO_HAND, obs, me, effect_id)
    if _is_energy(cid): return 30
    return 10

def _main_scores(obs, select, me):
    cur = obs.get("current") or {}
    ps = cur.get("players") or [{}, {}]
    hand = (ps[me] or {}).get("hand") or []
    opts = select.get("option") or []
    scores = []
    for o in opts:
        typ = int(_g(o, "type", -1))
        score = -10
        if typ == int(OptionType.PLAY):
            c = _at(obs, int(_g(o, "area", 2)), int(_g(o, "index", 0)), me)
            cid = _id(c)
            score = _play_value(cid, obs, me, hand)
        elif typ == int(OptionType.EVOLVE):
            tgt = _at(obs, int(_g(o, "inPlayArea", 0)), int(_g(o, "inPlayIndex", 0)), me)
            score = 100 if _id(tgt) == CHANSEY else 50
        elif typ == int(OptionType.ATTACH):
            card = _at(obs, int(_g(o, "area", 2)), int(_g(o, "index", 0)), me)
            tgt = _at(obs, int(_g(o, "inPlayArea", 0)), int(_g(o, "inPlayIndex", 0)), me)
            score = _attach_value(_id(card), tgt, obs, me, hand)
        elif typ == int(OptionType.ABILITY):
            card = _at(obs, int(_g(o, "area", 2)), int(_g(o, "index", 0)), me)
            score = _ability_value(_id(card), obs, me)
        elif typ == int(OptionType.RETREAT):
            active = _active(obs, me)
            bench_pika = any(_id(x) == PIKACHU for x in _bench(obs, me))
            score = 40 if (active is not None and _id(active) != PIKACHU and bench_pika) else -5
        elif typ == int(OptionType.ATTACK):
            score = ATTACK_DAMAGE.get(int(_g(o, "attackId", 0) or 0), 0)
        elif typ == int(OptionType.END):
            score = 0
        scores.append(score)
    return scores

def act(obs: dict) -> list[int]:
    select = obs.get("select")
    if select is None:
        return _load_deck()
    opts = select.get("option") or []
    if not opts:
        return []
    st = int(_g(select, "type", -1))
    ctx = int(_g(select, "context", -1))
    mn = max(0, int(_g(select, "minCount", 0) or 0))
    mx = min(int(_g(select, "maxCount", 0) or 0), len(opts))
    if mx <= 0: return []
    cur = obs.get("current") or {}
    me = int(_g(cur, "yourIndex", 0) or 0)

    if st == int(SelectType.MAIN):
        scores = _main_scores(obs, select, me)
        order = sorted(range(len(opts)), key=lambda i: scores[i], reverse=True)
        return order[:max(1, mx)]

    if st == int(SelectType.ATTACK):
        best = max(range(len(opts)), key=lambda i: (ATTACK_DAMAGE.get(int(_g(opts[i], "attackId", 0) or 0), 0), -i))
        return [best]

    if st == int(SelectType.YES_NO):
        for i, o in enumerate(opts):
            if int(_g(o, "type", -1)) == int(OptionType.YES):
                return [i]
        return [0]

    if st == int(SelectType.COUNT):
        nums = [int(_g(o, "number", 0) or 0) for o in opts]
        idx = (max if ctx in (SelectContext.DRAW_COUNT, SelectContext.DAMAGE_COUNTER_COUNT,
                              SelectContext.REMOVE_DAMAGE_COUNTER_COUNT) else min)
        return [idx(range(len(nums)), key=lambda i: (nums[i], -i))]

    effect = select.get("effect") or {}
    effect_id = int(_g(effect, "id", 0) or 0) if effect else 0
    scores = []
    for o in opts:
        cid = 0
        try:
            cid = _id(_at(obs, int(_g(o, "area", 0)), int(_g(o, "index", 0)), int(_g(o, "playerIndex", me))))
        except Exception:
            pass
        scores.append(_card_value(cid, ctx, obs, me, effect_id))
    order = sorted(range(len(opts)), key=lambda i: scores[i], reverse=True)
    if ctx in (SelectContext.DISCARD, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
               SelectContext.DISCARD_ENERGY_CARD, SelectContext.DISCARD_TOOL_CARD,
               SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM):
        n = max(mn, min(mx, mn if mn > 0 else 0))
    else:
        n = max(mn, mx)
    n = max(mn, min(n, mx))
    return order[:n]

_DECK = None
def _load_deck():
    global _DECK
    if _DECK is None:
        from pathlib import Path
        p = Path(__file__).with_name("deck.csv")
        _DECK = [int(x) for x in p.read_text().split() if x.strip()]
    return list(_DECK)
