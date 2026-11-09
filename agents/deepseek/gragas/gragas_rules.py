"""Rule fallback and multi-select heuristics for the Grass/Ogerpon deck."""

from __future__ import annotations

import gragas_data as D
import gragas_state as S

MAIN = 0
CARD = 1
ATTACHED_CARD = 2
CARD_OR_ATTACHED_CARD = 3
ENERGY = 4
SKILL = 5
ATTACK = 6
EVOLVE = 7
COUNT = 8
YES_NO = 9
SPECIAL_CONDITION = 10

SETUP_ACTIVE = 1
SETUP_BENCH = 2
SWITCH = 3
TO_ACTIVE = 4
TO_BENCH = 5
TO_FIELD = 6
TO_HAND = 7
DISCARD = 8
TO_DECK = 9
TO_DECK_BOTTOM = 10
TO_PRIZE = 11
NOT_MOVE = 12
DAMAGE_COUNTER = 13
DAMAGE_COUNTER_ANY = 14
DAMAGE = 15
REMOVE_DAMAGE_COUNTER = 16
HEAL = 17
EVOLVES_FROM = 18
EVOLVES_TO = 19
DEVOLVE = 20
ATTACH_FROM = 21
ATTACH_TO = 22
DETACH_FROM = 23
LOOK = 24
EFFECT_TARGET = 25
DISCARD_ENERGY_CARD = 26
DISCARD_TOOL_CARD = 27
SWITCH_ENERGY_CARD = 28
DISCARD_CARD_OR_ATTACHED_CARD = 29
DISCARD_ENERGY = 30
TO_HAND_ENERGY = 31
TO_DECK_ENERGY = 32
SWITCH_ENERGY = 33
SKILL_ORDER = 34
ATTACK_CONTEXT = 35
DISABLE_ATTACK = 36
EVOLVE_CONTEXT = 37
DRAW_COUNT = 38
DAMAGE_COUNTER_COUNT = 39
REMOVE_DAMAGE_COUNTER_COUNT = 40
IS_FIRST = 41
MULLIGAN = 42
ACTIVATE = 43
FIRST_EFFECT = 44
MORE_DEVOLVE = 45
COIN_HEAD = 46

NUMBER = 0
YES = 1
NO = 2
PLAY = 7
ATTACH = 8
EVOLVE_OPT = 9
ABILITY = 10
DISCARD_OPT = 11
RETREAT = 12
ATTACK_OPT = 13
END = 14

OGERPON_TEAL = 96
OGERPON_WELL = 108
KANGASKHAN = 756
LATIAS = 184
CLEFAIRY = 272
MUNKIDORI = 112
FEZANDIPITI = 140
PECHARUNT = 230
MEOWTH = 1071
CHI_YU = 31
GRASS = 1
PRISM = 16
UNFAIR = 1080
BUG_CATCHING = 1094
NIGHT_STRETCHER = 1097
ENERGY_SWITCH = 1116
ULTRA_BALL = 1121
TERA_ORB = 1127
BOSS = 1182
CIPHER = 1188
N_PLAN = 1221
LILLIE = 1227
AREA_ZERO = 1250

DISCARD_CONTEXTS = {
    DISCARD, TO_DECK, TO_DECK_BOTTOM, TO_PRIZE,
    DISCARD_ENERGY_CARD, DISCARD_TOOL_CARD, DISCARD_CARD_OR_ATTACHED_CARD,
    DISCARD_ENERGY, TO_DECK_ENERGY,
}


class GrassRules:
    def __init__(self, deck: list[int]) -> None:
        self.deck = list(deck)

    def choose(self, obs: dict) -> list[int]:
        sel = S.select(obs)
        if sel is None:
            return list(self.deck)
        options = list(S.get(sel, "option", []) or [])
        if not options:
            return []
        mn, mx = S.min_count(obs), S.max_count(obs)
        mx = min(mx, len(options))
        if mx <= 0:
            return []
        if S.select_type(obs) == MAIN:
            choice = self.choose_main(obs, options)
        else:
            choice = self.choose_other(obs, options, mn, mx)
        return self._legalize(choice, len(options), mn, mx)

    # ------------------------------------------------------------------
    def choose_main(self, obs: dict, options: list) -> list[int]:
        scores = []
        for i, option in enumerate(options):
            try:
                scores.append((self.main_score(obs, option), i))
            except Exception:
                scores.append((0.0, i))
        lethal = [item for item in scores if item[0] >= 20000.0]
        if lethal:
            return [max(lethal, key=lambda x: (x[0], -x[1]))[1]]
        best = max(scores, key=lambda x: (x[0], -x[1]))
        return [int(best[1])]

    def main_score(self, obs: dict, option) -> float:
        otype = S.option_type(option)
        if otype == ATTACK_OPT:
            return self.attack_score(obs, option)
        if otype == ABILITY:
            return self.ability_score(obs, option)
        if otype == PLAY:
            return self.play_score(obs, option)
        if otype == ATTACH:
            return self.attach_score(obs, option)
        if otype == RETREAT:
            return self.retreat_score(obs, option)
        if otype == END:
            return 0.0
        if otype == DISCARD_OPT:
            return 500.0
        return 500.0

    # ------------------------------------------------------------------
    def _damage(self, obs: dict, attack_id: int, actor_id: int | None = None,
                defender=None) -> int:
        atk = D.attack(attack_id)
        if atk is None:
            return 0
        me = S.current_player(obs)
        actor = S.active(obs, me)
        if defender is None:
            defender = S.active(obs, 1 - me)
        if actor_id is None:
            actor_id = S.card_id(actor)
        damage = int(getattr(atk, "damage", 0) or 0)
        # Attack-specific scaling.
        if int(attack_id) == 120:  # Myriad Leaf Shower
            damage = 30 + 30 * (len(S.energies(actor)) + len(S.energies(defender)))
        elif int(attack_id) == 1092:  # Rapid-Fire Combo
            damage = 200
        elif int(attack_id) == 243:  # Eon Blade
            damage = 200
        elif int(attack_id) == 136:  # Torrential Pump
            damage = 100
            if len(S.energies(actor)) >= 3:
                damage = 220
        elif int(attack_id) == 371:  # Full Moon Rondo
            damage = 20 + 20 * (len(S.bench(obs, me)) + len(S.bench(obs, 1 - me)))
        elif int(attack_id) == 20:  # Ground Melter
            damage = 120 if S.stadium(obs) else 60
        if damage > 0 and actor_id is not None and defender is not None:
            ac = D.card(actor_id)
            dc = D.card(S.card_id(defender))
            if ac is not None and dc is not None:
                if getattr(dc, "weakness", None) == getattr(ac, "energyType", None):
                    damage *= 2
                if getattr(dc, "resistance", None) == getattr(ac, "energyType", None):
                    damage = max(0, damage - 30)
        return damage

    def attack_score(self, obs: dict, option) -> float:
        me = S.current_player(obs)
        aid = S.option_int(option, "attackId", None)
        if aid is None:
            return 0.0
        defender = S.active(obs, 1 - me)
        damage = self._damage(obs, int(aid), S.active_id(obs, me), defender)
        hp = S.hp(defender)
        prize = D.prize_value(S.card_id(defender))
        score = 5000.0 + damage * 7.0
        if defender is not None and damage >= hp > 0:
            score = 20000.0 + prize * 1000.0
        if int(aid) == 136:  # Torrential Pump bench line
            score += 2500.0
        if int(aid) == 183:  # Cruel Arrow: 100 to any target
            target = self._best_opponent_target(obs)
            if target is not None:
                thp, tid = target
                score += 1500.0
                if thp <= 100:
                    score = max(score, 12000.0 + D.prize_value(tid) * 1000.0)
        return score

    def _best_opponent_target(self, obs: dict):
        me = S.current_player(obs)
        choices = [S.active(obs, 1 - me)] + S.bench(obs, 1 - me)
        best = None
        for p in choices:
            if p is None:
                continue
            cid = S.card_id(p)
            hp = S.hp(p)
            if cid is None or hp <= 0:
                continue
            key = (-hp, D.prize_value(cid))
            if best is None or key > best[0]:
                best = (key, hp, cid)
        return None if best is None else (best[1], best[2])

    def ability_score(self, obs: dict, option) -> float:
        actor, _ = S.option_cards(obs, option)
        me = S.current_player(obs)
        if actor == OGERPON_TEAL:
            return 9000.0 if any(S.card_id(c) == GRASS for c in S.hand(obs, me)) else 3000.0
        if actor == KANGASKHAN:
            return 8200.0 if S.card_id(S.active(obs, me)) == KANGASKHAN else 5000.0
        if actor == MUNKIDORI:
            damaged = sum(1 for p in S.all_field(obs, me) if S.is_damaged(p))
            return 6000.0 if damaged else 1500.0
        if actor == FEZANDIPITI:
            return 7200.0
        if actor == CLEFAIRY or actor == LATIAS or actor == PECHARUNT:
            return 3500.0
        if actor == MEOWTH:
            return 6800.0
        return 3000.0

    def play_score(self, obs: dict, option) -> float:
        me = S.current_player(obs)
        hand_index = S.option_int(option, "index", None)
        cid = S.card_at(obs, me, S.AREA_HAND, hand_index)
        if cid is None:
            cid = S.option_int(option, "cardId", None)
        if cid is None:
            return 2000.0
        bench = S.bench(obs, me)
        bench_max = int(S.get(S.player_state(obs, me), "benchMax", 5) or 5)
        space = max(0, bench_max - len(bench))

        if cid in (OGERPON_TEAL, OGERPON_WELL, KANGASKHAN, LATIAS, CLEFAIRY,
                   MUNKIDORI, MEOWTH, FEZANDIPITI, PECHARUNT, CHI_YU):
            if space <= 0:
                return 500.0
            return {
                KANGASKHAN: 9200.0,
                OGERPON_TEAL: 9000.0,
                OGERPON_WELL: 8600.0,
                LATIAS: 8000.0,
                CLEFAIRY: 7800.0,
                MUNKIDORI: 6200.0,
                MEOWTH: 6200.0,
                FEZANDIPITI: 6200.0,
                PECHARUNT: 4600.0,
                CHI_YU: 3800.0,
            }.get(cid, 4000.0)
        if cid == BUG_CATCHING:
            return 9200.0
        if cid == LILLIE:
            h = S.hand_count(obs, me)
            return 9000.0 if S.prize_count(obs, me) == 6 else (8200.0 if h <= 4 else 5000.0)
        if cid == ULTRA_BALL:
            return 7200.0 if S.hand_count(obs, me) >= 4 else 4000.0
        if cid == TERA_ORB:
            field = [S.card_id(p) for p in S.all_field(obs, me)]
            return 7800.0 if not any(x in (OGERPON_TEAL, OGERPON_WELL) for x in field) else 4200.0
        if cid == ENERGY_SWITCH:
            return 6800.0 if self._needs_energy(obs) else 3500.0
        if cid == N_PLAN:
            return 6800.0 if self._needs_energy(obs) else 3500.0
        if cid == BOSS:
            return self.gust_score(obs)
        if cid == UNFAIR:
            return 7200.0
        if cid == AREA_ZERO:
            return 6600.0 if space > 0 else 3000.0
        if cid == CIPHER:
            return 6200.0
        if cid == NIGHT_STRETCHER:
            disc = set(S.discard_ids(obs, me))
            return 6800.0 if disc & {OGERPON_TEAL, OGERPON_WELL, KANGASKHAN, GRASS} else 2500.0
        return 3000.0

    def _needs_energy(self, obs: dict) -> bool:
        me = S.current_player(obs)
        active = S.active(obs, me)
        return active is not None and len(S.energies(active)) < 3

    def attach_score(self, obs: dict, option) -> float:
        me = S.current_player(obs)
        actor, _ = S.option_cards(obs, option)
        area = S.option_int(option, "inPlayArea", None)
        index = S.option_int(option, "inPlayIndex", None)
        target = None
        if area == S.AREA_ACTIVE:
            target = S.active(obs, me)
        elif area == S.AREA_BENCH:
            bench = S.bench(obs, me)
            if index is not None and 0 <= int(index) < len(bench):
                target = bench[int(index)]
        tid = S.card_id(target)
        etype = D.energy_type(actor)
        base = 5000.0
        if tid == KANGASKHAN:
            base = 9500.0
        elif tid in (OGERPON_TEAL, OGERPON_WELL):
            base = 9000.0
        elif tid in (LATIAS, CLEFAIRY):
            base = 6200.0
        elif tid in (MUNKIDORI, MEOWTH, FEZANDIPITI):
            base = 5200.0
        if target is S.active(obs, me):
            base += 800.0
        if etype in (GRASS, PRISM):
            base += 1000.0
        return base

    def retreat_score(self, obs: dict, option) -> float:
        me = S.current_player(obs)
        active = S.active(obs, me)
        aid = S.card_id(active)
        if aid in (OGERPON_TEAL, OGERPON_WELL, KANGASKHAN):
            return -500.0
        bench = S.bench(obs, me)
        if any(S.card_id(p) in (KANGASKHAN, OGERPON_TEAL, OGERPON_WELL) for p in bench):
            return 2500.0
        return 0.0

    def gust_score(self, obs: dict) -> float:
        me = S.current_player(obs)
        best = 0.0
        damage = 200
        for p in S.bench(obs, 1 - me):
            cid = S.card_id(p)
            hp = S.hp(p)
            if cid is None:
                continue
            prize = D.prize_value(cid)
            if damage >= hp:
                score = 9000.0 + prize * 700.0
            elif hp <= 100:
                score = 6000.0 + prize * 400.0
            else:
                score = 3000.0
            best = max(best, score)
        return best

    # ------------------------------------------------------------------
    def choose_other(self, obs: dict, options: list, mn: int, mx: int) -> list[int]:
        stype = S.select_type(obs)
        ctx = S.select_context(obs)
        if stype == COUNT:
            return [self._count_choice(obs, options, ctx)]
        if stype == YES_NO:
            if ctx == IS_FIRST:
                idx = self._yes_index(options)
                return [idx if idx >= 0 else 0]
            idx = self._yes_index(options)
            return [idx if idx >= 0 else 0]
        if stype == ATTACK:
            return [self._attack_choice(obs, options)]
        if stype == ENERGY:
            scored = []
            for i, o in enumerate(options):
                actor, _ = S.option_cards(obs, o)
                score = 7000.0 if D.energy_type(actor) in (GRASS, PRISM) else 3000.0
                if ctx in (DISCARD_ENERGY, TO_DECK_ENERGY):
                    score = -score
                scored.append((score, i))
            ordered = sorted(scored, key=lambda x: (x[0], x[1])) if ctx in (DISCARD_ENERGY, TO_DECK_ENERGY) else sorted(scored, key=lambda x: (-x[0], x[1]))
            count = mn if ctx in (DISCARD_ENERGY, TO_DECK_ENERGY) else mx
            count = max(mn, min(count, mx))
            return [i for _, i in ordered[:count]]
        if stype in (ATTACHED_CARD, CARD_OR_ATTACHED_CARD):
            scored = []
            for i, o in enumerate(options):
                actor, _ = S.option_cards(obs, o)
                score = self.card_value(actor)
                if ctx in (DISCARD_ENERGY_CARD, DISCARD_TOOL_CARD, DISCARD_CARD_OR_ATTACHED_CARD):
                    score = -score
                scored.append((score, i))
            ordered = sorted(scored, key=lambda x: (-x[0], x[1]))
            count = mn if ctx in (DISCARD_ENERGY_CARD, DISCARD_TOOL_CARD, DISCARD_CARD_OR_ATTACHED_CARD) else mx
            count = max(mn, min(count, mx))
            return [i for _, i in ordered[:count]]
        if stype in (EVOLVE, EVOLVE_CONTEXT):
            return list(range(max(mn, min(mx, mx))))

        scored = [(self.card_option_score(obs, o, ctx), i) for i, o in enumerate(options)]
        if ctx in DISCARD_CONTEXTS:
            ordered = sorted(scored, key=lambda x: (x[0], x[1]))
            count = mn
        else:
            ordered = sorted(scored, key=lambda x: (-x[0], x[1]))
            count = mx
        count = max(mn, min(count, mx))
        return [i for _, i in ordered[:count]]

    def card_option_score(self, obs: dict, option, ctx: int) -> float:
        actor, target = S.option_cards(obs, option)
        cid = target if target is not None else actor
        if ctx == SETUP_ACTIVE:
            return {
                KANGASKHAN: 9500.0, OGERPON_TEAL: 9000.0, OGERPON_WELL: 8800.0,
                LATIAS: 8000.0, CLEFAIRY: 7600.0, MUNKIDORI: 6000.0,
                MEOWTH: 6000.0, FEZANDIPITI: 5800.0,
            }.get(cid, 4000.0)
        if ctx in (SETUP_BENCH, TO_BENCH, TO_FIELD):
            return {
                OGERPON_TEAL: 9500.0, OGERPON_WELL: 9000.0, KANGASKHAN: 9000.0,
                LATIAS: 8000.0, CLEFAIRY: 7600.0, MUNKIDORI: 6400.0,
                MEOWTH: 6200.0, FEZANDIPITI: 5600.0, PECHARUNT: 4600.0,
            }.get(cid, 4000.0)
        if ctx in (SWITCH, TO_ACTIVE):
            return {
                KANGASKHAN: 9200.0, OGERPON_TEAL: 8800.0, OGERPON_WELL: 8500.0,
                LATIAS: 8000.0, CLEFAIRY: 7200.0,
            }.get(cid, 4000.0)
        if ctx in (DAMAGE_COUNTER, DAMAGE_COUNTER_ANY, DAMAGE, EFFECT_TARGET):
            owner = S.option_target_player(obs, option)
            me = S.current_player(obs)
            if owner is not None and owner == me:
                return 1000.0
            p = None
            area = S.option_int(option, "area", None)
            index = S.option_int(option, "index", None)
            if area == S.AREA_ACTIVE:
                p = S.active(obs, 1 - me)
            elif area == S.AREA_BENCH:
                bench = S.bench(obs, 1 - me)
                if index is not None and 0 <= int(index) < len(bench):
                    p = bench[int(index)]
            hp = S.hp(p) if p is not None else (D.card(cid).hp if D.card(cid) else 100)
            prize = D.prize_value(cid)
            if hp <= 10:
                return 12000.0 + prize * 1000.0
            if hp <= 30:
                return 9000.0 + prize * 700.0
            if hp <= 60:
                return 6500.0 + prize * 500.0
            return 3000.0 - hp / 10.0
        if ctx in (ATTACH_TO, ATTACH_FROM):
            meta = D.card(cid)
            if meta is not None and int(getattr(meta, "cardType", -1)) == 0:
                me = S.current_player(obs)
                p = next((x for x in S.all_field(obs, me) if S.card_id(x) == cid), None)
                if p is None:
                    return 1000.0
                if cid == KANGASKHAN:
                    return 9500.0
                if cid in (OGERPON_TEAL, OGERPON_WELL):
                    return 9000.0
                return 6000.0
            return 7000.0 if D.energy_type(cid) in (GRASS, PRISM) else 3000.0
        if ctx in DISCARD_CONTEXTS:
            return self.discard_value(obs, cid)
        return self.card_value(cid)

    def card_value(self, cid: int | None) -> float:
        if cid is None:
            return 0.0
        return {
            KANGASKHAN: 10000.0, OGERPON_TEAL: 9500.0, OGERPON_WELL: 9000.0,
            LATIAS: 8200.0, CLEFAIRY: 8000.0, MUNKIDORI: 6500.0,
            MEOWTH: 6500.0, FEZANDIPITI: 6400.0, PECHARUNT: 4800.0,
            CHI_YU: 4000.0, GRASS: 6000.0, PRISM: 5500.0,
            BUG_CATCHING: 7200.0, ULTRA_BALL: 6500.0, LILLIE: 6500.0,
            BOSS: 6500.0, TERA_ORB: 6800.0, ENERGY_SWITCH: 6000.0,
            N_PLAN: 6000.0, UNFAIR: 6000.0, AREA_ZERO: 5000.0,
            CIPHER: 5000.0, NIGHT_STRETCHER: 5000.0,
        }.get(int(cid), 3000.0)

    def discard_value(self, obs: dict, cid: int | None) -> float:
        value = self.card_value(cid)
        if cid in (KANGASKHAN, OGERPON_TEAL, OGERPON_WELL, LATIAS, CLEFAIRY):
            value += 3000.0
        if cid in (GRASS, PRISM):
            value += 1000.0
        return value

    def _count_choice(self, obs: dict, options: list, ctx: int) -> int:
        nums = [S.option_int(o, "number", 0) or 0 for o in options]
        if ctx in (DRAW_COUNT, DAMAGE_COUNTER_COUNT, REMOVE_DAMAGE_COUNTER_COUNT):
            return int(max(range(len(nums)), key=lambda i: (nums[i], -i))) if nums else 0
        return int(min(range(len(nums)), key=lambda i: (nums[i], i))) if nums else 0

    def _attack_choice(self, obs: dict, options: list) -> int:
        best = 0
        best_score = -1
        for i, o in enumerate(options):
            aid = S.option_int(o, "attackId", None)
            score = self._damage(obs, int(aid)) if aid is not None else 0
            if score > best_score:
                best_score = score
                best = i
        return int(best)

    def _yes_index(self, options: list) -> int:
        for i, o in enumerate(options):
            if S.option_type(o) == YES:
                return i
        return -1

    def _legalize(self, choice: list[int], n: int, mn: int, mx: int) -> list[int]:
        out: list[int] = []
        for value in choice:
            try:
                idx = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < n and idx not in out:
                out.append(idx)
            if len(out) >= mx:
                break
        if len(out) < mn:
            for idx in range(n):
                if idx not in out:
                    out.append(idx)
                if len(out) >= mn:
                    break
        return out[:mx]
