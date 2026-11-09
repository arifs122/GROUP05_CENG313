"""
Mega Lucario ex heuristic ajanı (motor-bağımsız).

KULLANIM:
  1) Sadece to_state(obs) ve to_acts(obs) fonksiyonlarını cabt SDK'ya göre doldurun.
  2) agent(obs, config) fonksiyonunu Kaggle'ın istediği imzaya uyarlayın.
  3) W sözlüğündeki sayılar self-play ile ayarlanacak (weights.json).

Kart etkileri hafızadan yazıldı; motordaki metinle doğrulayın.
"""
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import mk_data as D
import mk_state as S

# ---------- Ortak veri yapıları ----------

@dataclass
class Act:
    raw: Any            # motorun kendi action nesnesi (aynen geri döndürülür)
    kind: str           # basic|evolve|attach|item|supporter|ability|attack|retreat|pick|end
    card: str = ""      # oynanan kart / saldıran Pokémon / seçilen kart
    target: str = ""    # hedef Pokémon adı
    zone: str = ""      # active|bench|opp_active|opp_bench
    ko: bool = False    # bu saldırı rakibi KO ediyor mu (biliniyorsa)
    prize: int = 1      # KO ederse alınacak prize
    attack_id: int = 0  # saldırı kimliği (varsa)
    damage: int = 0     # hesaplanan hasar


@dataclass
class State:
    turn: int = 1
    my_prize: int = 6
    opp_prize: int = 6
    hand: int = 5
    active: str = ""
    bench: List[str] = field(default_factory=list)
    opp_active: str = ""
    opp_bench: List[str] = field(default_factory=list)
    energy: dict = field(default_factory=dict)  # Pokémon adı -> takılı enerji sayısı
    hand_cards: List[str] = field(default_factory=list)


# ---------- Ayarlanabilir ağırlıklar (CMA-ES ile ayarlanacak) ----------

W = {
    "ko": 700, "attack": 180,
    "basic": 650, "evolve": 900, "mega_evolve": 980,
    "attach": 620, "poffin": 720, "ultra": 720, "gong": 730,
    "lillie": 760, "lillie_low_hand": 50, "boss_setup": 720, "boss_ko": 950,
    "lunatone": 980, "retreat": 450, "ppp": 780, "cape": 350,
    "night_stretcher": 300, "switch": 300, "bench_target": 4,
}
_WEIGHT_CANDIDATES = []
if "__file__" in globals():
    _WEIGHT_CANDIDATES.append(Path(__file__).with_name("weights.json"))
_WEIGHT_CANDIDATES.extend([Path.cwd() / "weights.json", Path("/kaggle_simulations/agent/weights.json")])
for _wp in _WEIGHT_CANDIDATES:
    if _wp.is_file():
        try:
            W.update(json.loads(_wp.read_text(encoding="utf-8")))
        except Exception:
            pass
        break

ATTACH_PRI = {"Mega Lucario ex": 3, "Riolu": 3, "Hariyama": 2, "Makuhita": 2}
ATTACKERS = set(ATTACH_PRI)
SETUP_TARGETS = {"Abra", "Kadabra", "Dreepy", "Drakloak", "Duskull", "Dusclops",
                 "Munkidori", "Munkidori ex", "Joltik", "Gastly"}
PICK_ORDER = ["Riolu", "Makuhita", "Mega Lucario ex", "Hariyama",
              "Basic {F} Energy", "Lunatone", "Solrock", "Budew"]

CARD_VALUE = {
    "Riolu": 100, "Makuhita": 100, "Mega Lucario ex": 120, "Hariyama": 120,
    "Lunatone": 80, "Solrock": 80, "Budew": 50, "Basic {F} Energy": 55,
    "Buddy-Buddy Poffin": 70, "Ultra Ball": 75, "Night Stretcher": 65,
    "Switch": 50, "Boss’s Orders": 85, "Lillie's Determination": 85,
    "Premium Power Pro": 70, "Fighting Gong": 75, "Hero’s Cape": 90,
}
DISCARD_CONTEXTS = {
    S.C_DISCARD, S.C_DISCARD_ENERGY_CARD, S.C_DISCARD_TOOL_CARD,
    S.C_DISCARD_CARD_OR_ATTACHED, S.C_DISCARD_ENERGY, S.C_TO_DECK,
    S.C_TO_DECK_BOTTOM, S.C_TO_PRIZE,
}


# ---------- Skor fonksiyonu ----------

_POFFIN_CACHE = None

def _poffin_targets() -> dict[str, int]:
    global _POFFIN_CACHE
    if _POFFIN_CACHE is not None:
        return _POFFIN_CACHE
    targets: dict[str, int] = {}
    try:
        deck = list(_DECK) if _DECK is not None else load_deck()
        for cid in deck:
            card = D.card(cid)
            if card is not None and D.is_basic(cid) and int(getattr(card, "hp", 99) or 99) <= 70:
                name = D.card_name(cid)
                targets[name] = targets.get(name, 0) + 1
    except Exception:
        pass
    _POFFIN_CACHE = targets
    return targets


def score(a: Act, s: State) -> float:
    k = a.kind
    c = a.card.replace("’", "'")
    in_play = [s.active] + s.bench
    n_bench = len(s.bench)

    if k == "attack":
        if a.attack_id == 982 and s.turn <= 5:
            return W["attack"] + 0.5 * a.damage + 120
        if a.attack_id == 983 and s.turn <= 5 and not a.ko:
            return W["attack"] + 0.5 * a.damage - 80
        if a.ko:
            if a.prize >= s.opp_prize:
                return 5000.0
            return W["ko"] + 100 * a.prize
        return W["attack"] + 0.5 * a.damage

    if k == "basic":
        if c in ("Riolu", "Makuhita"):
            return W["basic"] if n_bench < 4 else 200
        if c in ("Lunatone", "Solrock"):
            return 680 if n_bench < 5 and c not in in_play else 150
        return 300 if n_bench < 2 else 80  # Budew vb.

    if k == "evolve":
        base = W["mega_evolve"] if c == "Mega Lucario ex" else W["evolve"]
        if c == "Mega Lucario ex" and "Mega Lucario ex" in s.hand_cards:
            base += 50
        return base

    if k == "attach":
        if a.target in ("Solrock", "Lunatone"):
            base = 470
        else:
            base = W["attach"] if a.target in ATTACH_PRI else 200
        return base + 50 * ATTACH_PRI.get(a.target, 0) - 5 * s.energy.get(a.target, 0) \
            + (30 if a.target == s.active else 0)

    if k == "item":
        if c in ("Buddy-Buddy Poffin",):
            if n_bench >= 4:
                return 80
            if c == "Buddy-Buddy Poffin":
                targets = _poffin_targets()
                if not any(count > in_play.count(name) for name, count in targets.items()):
                    return -10
            return W["poffin"]
        if c in ("Ultra Ball", "Poké Pad", "Dusk Ball"):
            return W["ultra"] if s.hand >= 3 else 120
        if c == "Fighting Gong":
            return W["gong"]
        if c == "Premium Power Pro":
            ready = s.energy.get(s.active, 0) >= 1 and s.active in ATTACKERS
            return W["ppp"] if ready else -10
        if c == "Hero's Cape":
            return W["cape"] if a.target in ("Mega Lucario ex", "Hariyama") else -10
        if c == "Carmine":
            return 760
        if c == "Night Stretcher":
            return W["night_stretcher"]
        if c == "Switch":
            if s.active not in ATTACKERS:
                return W["switch"]
            if ("Mega Lucario ex" in s.bench and s.energy.get("Mega Lucario ex", 0) >= 1
                    and s.active in ("Riolu", "Makuhita", "Hariyama", "Budew",
                                     "Lunatone", "Solrock")):
                return W["switch"]
            return -100
        return 100

    if k == "supporter":
        if c == "Lillie's Determination":
            return W["lillie"] + W["lillie_low_hand"] * max(0, 4 - s.hand)
        if c == "Boss's Orders":
            if a.ko:
                return W["boss_ko"]
            return W["boss_setup"] if a.target in SETUP_TARGETS else -200
        return 400

    if k == "ability":
        if c == "Lunatone" and "Solrock" in in_play:
            return W["lunatone"]
        if c == "Hariyama":
            return W["boss_setup"]
        return 350

    if k == "retreat":
        bench_ready = any(p in ATTACKERS and s.energy.get(p, 0) >= 1 for p in s.bench)
        mega_ready = ("Mega Lucario ex" in s.bench
                      and s.energy.get("Mega Lucario ex", 0) >= 1
                      and s.active != "Mega Lucario ex")
        return W["retreat"] if ((s.active not in ATTACKERS and bench_ready) or mega_ready) else -500

    if k == "pick":
        # arama/seçim: en gerekli kartı seç
        if c in PICK_ORDER:
            idx = PICK_ORDER.index(c)
            bonus = 0
            if c == "Riolu" and in_play.count("Riolu") < 2:
                bonus = 100
            if c == "Basic {F} Energy" and any(s.energy.get(p, 0) == 0 for p in in_play if p in ATTACKERS):
                bonus = 60
            return 500 - 10 * idx + bonus
        return 100

    if k == "discard":
        return -CARD_VALUE.get(c, 30)
    if k == "boss_target":
        if c in SETUP_TARGETS:
            return 700
        return 250
    if k == "number":
        return 100
    if k == "yes":
        return 100
    if k == "no":
        return 0
    if k == "end":
        return 0
    return 50


# ---------- Motor adaptörü ----------

_DECK = None

def load_deck() -> list[int]:
    global _DECK
    if _DECK is not None:
        return list(_DECK)
    candidates = []
    env_path = os.environ.get("MK_DECK")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([Path.cwd() / "deck.csv", Path("/kaggle_simulations/agent/deck.csv")])
    if "__file__" in globals():
        candidates.insert(0, Path(__file__).with_name("deck.csv"))
    for path in candidates:
        if path.is_file():
            cards = [int(x) for x in path.read_text(encoding="utf-8").split() if x.strip()]
            if len(cards) == 60:
                _DECK = cards
                return list(_DECK)
    raise FileNotFoundError("lucario deck.csv not found")


def _name(obs, cid) -> str:
    return D.card_name(cid) if cid is not None else ""


def _target_name(obs, option) -> str:
    _actor, target = S.option_cards(obs, option)
    return _name(obs, target)


def _zone_of(obs, option) -> str:
    area = S.option_int(option, "inPlayArea", None)
    if area == S.ACTIVE:
        return "active"
    if area == S.BENCH:
        return "bench"
    pi = S.option_int(option, "playerIndex", None)
    if pi is not None and pi != S.you(obs):
        return "opp_bench"
    return ""


def _attack_damage(obs, attack_id: int, attacker, defender) -> int:
    if attacker is None or defender is None:
        return 0
    atk = D.attack(attack_id)
    if atk is None:
        return 0
    cid = S.pokemon_id(attacker)
    did = S.pokemon_id(defender)
    if cid is None or did is None:
        return 0
    dmg = int(getattr(atk, "damage", 0) or 0)
    # Solrock's Cosmic Beam fails without a benched Lunatone.
    if attack_id == 980:
        if not any(S.pokemon_id(p) == 675 for p in S.bench(obs, S.you(obs))):
            return 0
        return dmg
    # Aura Jab's 130 is the on-hit damage; the bench attachment is separate.
    weak = D.weakness(did)
    if weak is not None and D.energy_type(cid) == weak:
        dmg *= 2
    res = D.resistance(did)
    if res is not None and D.energy_type(cid) == res:
        dmg = max(0, dmg - 30)
    return max(0, dmg)


def _best_attack(obs, attacker, defender) -> tuple[int, int]:
    best = (0, 0)
    for aid in D.attacks_of(S.pokemon_id(attacker)):
        dmg = _attack_damage(obs, aid, attacker, defender)
        if dmg > best[1]:
            best = (aid, dmg)
    return best


def _best_boss_target(obs):
    me = S.you(obs)
    active = S.active(obs, me)
    if active is None:
        return "", False, 1
    opponents = S.bench(obs, 1 - me)
    best = None
    for p in opponents:
        cid = S.pokemon_id(p)
        if cid is None:
            continue
        aid, dmg = _best_attack(obs, active, p)
        ko = dmg >= S.hp(p) and dmg > 0
        key = (ko, D.prize_value(cid), -S.hp(p), D.card_name(cid) in SETUP_TARGETS)
        if best is None or key > best[0]:
            best = (key, D.card_name(cid), ko, D.prize_value(cid))
    if best is None:
        return "", False, 1
    return best[1], best[2], best[3]


def _best_cape_target(obs) -> str:
    me = S.you(obs)
    for p in [S.active(obs, me)] + S.bench(obs, me):
        if p is not None and S.pokemon_id(p) in (678, 674):
            return D.card_name(S.pokemon_id(p))
    return ""


def to_state(obs) -> State:
    me = S.you(obs)
    st = State()
    st.turn = S.turn(obs)
    st.my_prize = S.prize_count(obs, me)
    st.opp_prize = S.prize_count(obs, 1 - me)
    st.hand = S.hand_count(obs, me)
    st.hand_cards = [D.card_name(S.card_id(c)) for c in (S.hand(obs, me) or [])
                     if S.card_id(c) is not None]
    a = S.active(obs, me)
    st.active = D.card_name(S.pokemon_id(a)) if a is not None else ""
    st.bench = [D.card_name(S.pokemon_id(p)) for p in S.bench(obs, me)
                if S.pokemon_id(p) is not None]
    oa = S.active(obs, 1 - me)
    st.opp_active = D.card_name(S.pokemon_id(oa)) if oa is not None else ""
    st.opp_bench = [D.card_name(S.pokemon_id(p)) for p in S.bench(obs, 1 - me)
                    if S.pokemon_id(p) is not None]
    for p in S.field(obs, me):
        cid = S.pokemon_id(p)
        if cid is not None:
            nm = D.card_name(cid)
            st.energy[nm] = st.energy.get(nm, 0) + S.energy_count(p)
    return st


def to_acts(obs) -> List[Act]:
    me = S.you(obs)
    stype = S.select_type(obs)
    ctx = S.select_context(obs)
    acts: List[Act] = []
    for i, o in enumerate(S.options(obs)):
        otype = S.option_type(o)
        actor, target = S.option_cards(obs, o)
        kind = "pick"
        card = _name(obs, actor)
        tname = _name(obs, target)
        zone = _zone_of(obs, o)
        ko = False
        prize = 1
        attack_id = 0
        damage = 0

        if stype == S.S_MAIN:
            if otype == S.O_PLAY:
                idx = S.option_int(o, "index", None)
                cid = S.card_at(obs, me, S.HAND, idx) if idx is not None else None
                ctype = D.card_type(cid)
                card = _name(obs, cid)
                if ctype == D.POKEMON:
                    kind = "basic"
                elif ctype == D.SUPPORTER:
                    kind = "supporter"
                else:
                    kind = "item"
                if card.replace("’", "'") == "Boss's Orders":
                    tname, ko, prize = _best_boss_target(obs)
                    kind = "supporter"
                elif card.replace("’", "'") == "Hero's Cape":
                    tname = _best_cape_target(obs)
                    kind = "item"
            elif otype == S.O_ATTACH:
                ctype = D.card_type(actor)
                if ctype == D.TOOL:
                    kind = "item"
                else:
                    kind = "attach"
            elif otype == S.O_EVOLVE:
                kind = "evolve"
            elif otype == S.O_ABILITY:
                kind = "ability"
            elif otype == S.O_ATTACK:
                kind = "attack"
                attacker = S.active(obs, me)
                defender = S.active(obs, 1 - me)
                aid = S.option_int(o, "attackId", None)
                dmg = _attack_damage(obs, aid, attacker, defender) if aid is not None else 0
                tname = _name(obs, S.pokemon_id(defender))
                ko = dmg >= S.hp(defender) and dmg > 0 if defender is not None else False
                prize = D.prize_value(S.pokemon_id(defender)) if defender is not None else 1
                attack_id = int(aid or 0)
                damage = int(dmg)
            elif otype == S.O_RETREAT:
                kind = "retreat"
            elif otype == S.O_END:
                kind = "end"
        else:
            if ctx in DISCARD_CONTEXTS and otype in (S.O_CARD, S.O_ENERGY_CARD,
                                                      S.O_ENERGY):
                kind = "discard"
            elif ctx == S.C_SWITCH:
                kind = "boss_target"
                card = tname
            elif otype == S.O_YES:
                kind = "yes"
            elif otype == S.O_NO:
                kind = "no"
            elif otype == S.O_NUMBER:
                kind = "number"
                card = f"NUMBER {S.option_int(o, 'number', 0)}"
            elif otype == S.O_ATTACK:
                kind = "attack"
                attacker = S.active(obs, me)
                defender = S.active(obs, 1 - me)
                aid = S.option_int(o, "attackId", None)
                dmg = _attack_damage(obs, aid, attacker, defender) if aid is not None else 0
                tname = _name(obs, S.pokemon_id(defender))
                ko = dmg >= S.hp(defender) and dmg > 0 if defender is not None else False
                prize = D.prize_value(S.pokemon_id(defender)) if defender is not None else 1
                attack_id = int(aid or 0)
                damage = int(dmg)
        acts.append(Act(raw=[i], kind=kind, card=card, target=tname, zone=zone,
                         ko=bool(ko), prize=int(prize), attack_id=attack_id,
                         damage=damage))
    return acts


def _pick_multi(obs, acts: List[Act], mn: int, mx: int) -> list[int]:
    ctx = S.select_context(obs)
    n = min(mx, len(acts))
    if n <= 0:
        return []
    if ctx in DISCARD_CONTEXTS:
        count = max(mn, min(mx, 1))
        ranked = sorted(range(len(acts)), key=lambda i: (CARD_VALUE.get(acts[i].card, 30), i))
        return ranked[:count]
    state = to_state(obs)
    ranked = sorted(range(len(acts)), key=lambda i: (score(acts[i], state), -i), reverse=True)
    count = max(mn, min(mx, len(ranked)))
    return ranked[:count]


_PRIOR = None
_PRIOR_TRIED = False


def _prior():
    global _PRIOR, _PRIOR_TRIED
    if _PRIOR_TRIED:
        return _PRIOR
    _PRIOR_TRIED = True
    try:
        from lucario_model import LucarioPrior
        _PRIOR = LucarioPrior()
    except Exception:
        _PRIOR = None
    return _PRIOR


def _choose_main(obs, s: State, acts: List[Act]):
    scores = [score(a, s) for a in acts]
    best_i = max(range(len(acts)), key=lambda i: (scores[i], -i))
    # Guaranteed game-winning lethal remains a hard rule.
    if scores[best_i] >= 5000.0:
        return acts[best_i].raw
    prior = _prior()
    if prior is None:
        return acts[best_i].raw
    try:
        logits = prior.logits(obs)
    except Exception:
        return acts[best_i].raw
    if len(logits) != len(acts):
        return acts[best_i].raw
    w = float(os.environ.get("LUCARIO_PRIOR_WEIGHT", "260"))
    mode = os.environ.get("LUCARIO_PRIOR_MODE", "blend")
    if mode == "argmax":
        i = int(max(range(len(logits)), key=lambda j: (float(logits[j]), -j)))
        return acts[i].raw
    mean = sum(float(x) for x in logits) / max(1, len(logits))
    ranked = []
    for i, a in enumerate(acts):
        ranked.append((scores[i] + w * (float(logits[i]) - mean), i))
    i = max(ranked, key=lambda x: (x[0], -x[1]))[1]
    return acts[i].raw


def choose(s: State, acts: List[Act]):
    # Deterministic tie-break keeps local measurements reproducible and makes
    # fork-paired candidate tests exact.
    best, best_sc = None, -1e9
    for a in acts:
        sc = score(a, s)
        if sc > best_sc + 1e-9:
            best, best_sc = a, sc
    return best.raw if best is not None else acts[0].raw


def _fallback(obs) -> list[int]:
    sel = S.select(obs)
    if sel is None:
        return load_deck()
    opts = S.options(obs)
    if not opts:
        return []
    mn = min(S.min_count(obs), len(opts))
    mx = min(S.max_count(obs), len(opts))
    if mx <= 0:
        return []
    return list(range(max(mn, min(mx, 1))))


def agent(obs, config=None):
    sel = S.select(obs)
    if sel is None:
        return load_deck()
    try:
        stype = S.select_type(obs)
        ctx = S.select_context(obs)
        mn, mx = S.min_count(obs), min(S.max_count(obs), len(S.options(obs)))
        if mx <= 0:
            return []
        # Yes/no and count prompts are better handled explicitly.
        if stype == S.S_YES_NO:
            opts = S.options(obs)
            yes = next((i for i, o in enumerate(opts) if S.option_type(o) == S.O_YES), 0)
            no = next((i for i, o in enumerate(opts) if S.option_type(o) == S.O_NO), 0)
            if ctx == S.C_IS_FIRST:
                return [yes]
            if ctx == S.C_MULLIGAN:
                has_basic = any(D.is_basic(c) for c in S.hand_ids(obs, S.you(obs)))
                return [no if has_basic else yes]
            if ctx in (S.C_COIN_HEAD,):
                return [yes]
            return [yes]
        if stype == S.S_COUNT:
            nums = [(S.option_int(o, "number", 0) or 0, i) for i, o in enumerate(S.options(obs))]
            if ctx in (S.C_DRAW_COUNT, S.C_DAMAGE_COUNTER_COUNT,
                       S.C_REMOVE_DAMAGE_COUNTER_COUNT):
                deck = S.deck_count(obs, S.you(obs))
                prizes = S.prize_count(obs, S.you(obs))
                if ctx == S.C_DRAW_COUNT and deck <= max(2, prizes * 2) + 4:
                    return [min(nums, key=lambda x: (x[0], x[1]))[1]]
                return [max(nums, key=lambda x: (x[0], -x[1]))[1]]
            return [min(nums, key=lambda x: (x[0], x[1]))[1]]
        acts = to_acts(obs)
        if not acts:
            return _fallback(obs)
        if mn <= 1 and mx <= 1:
            if stype == S.S_MAIN:
                return _choose_main(obs, to_state(obs), acts)
            # Discard effects should choose the least valuable card.
            if ctx in DISCARD_CONTEXTS:
                best = min(range(len(acts)), key=lambda i: (CARD_VALUE.get(acts[i].card, 30), i))
                return acts[best].raw
            if ctx == S.C_SWITCH:
                # Own switch vs Boss's Orders (opponent bench).
                opts = S.options(obs)
                pi = S.option_int(opts[0], "playerIndex", None)
                if pi is not None and pi == S.you(obs):
                    state = to_state(obs)
                    best = max(range(len(acts)), key=lambda i: (acts[i].card in ATTACKERS,
                                                                state.energy.get(acts[i].card, 0), -i))
                    return acts[best].raw
            return choose(to_state(obs), acts)
        return _pick_multi(obs, acts, mn, mx)
    except Exception:
        return _fallback(obs)
