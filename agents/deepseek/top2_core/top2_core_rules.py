"""Layer 0/1: legal, deterministic Top2 Dragapult rules and damage maths.

The rules never rely on hidden information.  They are the safety net for the
search layer: every branch ends with a legal selection, and a lethal attack is
always taken before anything speculative.
"""

from __future__ import annotations

import os
from typing import Any

import top2_core_data as D
import top2_core_state as S

# SelectType / SelectContext / OptionType values from cg.api.  They are kept
# as integers so the module also works with raw replay dictionaries.
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
AFFECT_SPECIAL_CONDITION = 47
RECOVER_SPECIAL_CONDITION = 48

# OptionType values.
NUMBER = 0
YES = 1
NO = 2
CARD_OPT = 3
TOOL_CARD_OPT = 4
ENERGY_CARD_OPT = 5
ENERGY_OPT = 6
PLAY = 7
ATTACH = 8
EVOLVE_OPT = 9
ABILITY = 10
DISCARD_OPT = 11
RETREAT = 12
ATTACK_OPT = 13
END = 14
SKILL_OPT = 15
SPECIAL_CONDITION_OPT = 16

DISCARD_CONTEXTS = {
    DISCARD,
    TO_DECK,
    TO_DECK_BOTTOM,
    TO_PRIZE,
    DISCARD_ENERGY_CARD,
    DISCARD_TOOL_CARD,
    DISCARD_CARD_OR_ATTACHED_CARD,
    DISCARD_ENERGY,
    TO_DECK_ENERGY,
}
TARGET_CONTEXTS = {
    SWITCH,
    TO_ACTIVE,
    TO_BENCH,
    TO_FIELD,
    DAMAGE_COUNTER,
    DAMAGE_COUNTER_ANY,
    DAMAGE,
    REMOVE_DAMAGE_COUNTER,
    HEAL,
    EVOLVES_FROM,
    EVOLVES_TO,
    DEVOLVE,
    ATTACH_FROM,
    ATTACH_TO,
    DETACH_FROM,
    EFFECT_TARGET,
}

LETHAL_SCORE = 20000.0


class RuleEngine:
    """A legal, deck-aware Top2 policy with a hand-written damage model."""

    def __init__(self, deck: list[int], searcher=None, prior=None) -> None:
        self.deck = list(deck)
        self.searcher = searcher
        self.prior = prior
        self.prior_weight = float(os.environ.get("TOP2_POLICY_WEIGHT", "220"))
        self.lethal_overrides = 0
        self.search_overrides = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
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
    # MAIN
    # ------------------------------------------------------------------
    def choose_main(self, obs: dict, options: list) -> list[int]:
        scores = []
        for i, option in enumerate(options):
            try:
                score = self.main_score(obs, option)
            except Exception:
                score = 0.0
            scores.append((float(score), i))

        # The learned policy is only a prior over legal MAIN options.  It never
        # overrides the hard lethal rule and is disabled when no model exists.
        if self.prior is not None and self.prior_weight != 0.0 and len(options) > 1:
            try:
                prior = self.prior.logits(obs, self.deck)
                if prior is not None and len(prior) >= len(options):
                    scores = [
                        (score + self.prior_weight * float(prior[i]), i)
                        for score, i in scores
                    ]
            except Exception:
                pass

        # Lethal first: this is a rule guarantee, not a search suggestion.
        lethal = [item for item in scores if item[0] >= LETHAL_SCORE]
        if lethal:
            self.lethal_overrides += 1
            return [max(lethal, key=lambda item: (item[0], -item[1]))[1]]

        best = max(scores, key=lambda item: (item[0], -item[1]))
        if self.searcher is not None:
            try:
                alternative = self.searcher.choose(obs, scores)
            except Exception:
                alternative = None
            if alternative is not None and alternative != best[1]:
                self.search_overrides += 1
                return [int(alternative)]
        return [int(best[1])]

    def main_score(self, obs: dict, option: Any) -> float:
        otype = S.option_type(option)
        if otype == ATTACK_OPT:
            return self.attack_score(obs, option)
        if otype == EVOLVE_OPT:
            return self.evolve_score(obs, option)
        if otype == ABILITY:
            return self.ability_score(obs, option)
        if otype == ATTACH:
            return self.attach_score(obs, option)
        if otype == PLAY:
            return self.play_score(obs, option)
        if otype == RETREAT:
            return self.retreat_score(obs, option)
        if otype == DISCARD_OPT:
            return self.main_discard_score(obs, option)
        if otype == END:
            return 0.0
        if otype in (SKILL_OPT, SPECIAL_CONDITION_OPT):
            return 2500.0
        return 500.0

    # ------------------------------------------------------------------
    # Attack / damage counter model
    # ------------------------------------------------------------------
    def _damage(self, obs: dict, attack_id: int, actor_id: int | None = None,
                defender: Any = None) -> int:
        atk = D.attack(attack_id)
        if atk is None:
            return 0
        damage = int(getattr(atk, "damage", 0) or 0)
        me = S.current_player(obs)
        if actor_id is None:
            actor_id = S.active_id(obs, me)
        if defender is None:
            defender = S.active(obs, 1 - me)
        defender_id = S.card_id(defender)
        if damage > 0 and actor_id is not None and defender_id is not None:
            ac = D.card(actor_id)
            dc = D.card(defender_id)
            if ac is not None and dc is not None:
                weakness = getattr(dc, "weakness", None)
                resistance = getattr(dc, "resistance", None)
                energy = getattr(ac, "energyType", None)
                if weakness is not None and energy is not None and int(energy) == int(weakness):
                    damage *= 2
                if resistance is not None and energy is not None and int(energy) == int(resistance):
                    damage = max(0, damage - 30)
        return damage

    def _bench_counter_plan(self, obs: dict, counters: int = 6) -> tuple[int, int]:
        """Greedy Phantom Dive plan.  Returns (KOs, useful damage)."""
        me = S.current_player(obs)
        targets = []
        for p in S.bench(obs, 1 - me):
            cid = S.card_id(p)
            hp = S.hp(p)
            if cid is None or hp <= 0:
                continue
            targets.append((hp, -D.prize_value(cid), cid))
        targets.sort()
        remaining = counters
        kos = 0
        useful = 0
        for hp, _negprize, _cid in targets:
            if remaining <= 0:
                break
            need = (hp + 9) // 10
            if need <= remaining:
                remaining -= need
                kos += 1
                useful += hp
            elif remaining > 0:
                useful += remaining * 10
                remaining = 0
        return kos, useful

    def attack_score(self, obs: dict, option: Any) -> float:
        me = S.current_player(obs)
        attack_id = S.option_int(option, "attackId", None)
        if attack_id is None:
            return 0.0
        actor_id = S.active_id(obs, me)
        defender = S.active(obs, 1 - me)
        defender_id = S.card_id(defender)
        damage = self._damage(obs, int(attack_id), actor_id, defender)
        defender_hp = S.hp(defender)

        score = 5000.0 + damage * 7.0
        ko = defender is not None and damage >= defender_hp and damage > 0
        if ko:
            prize = D.prize_value(defender_id)
            score = LETHAL_SCORE + prize * 1200.0
            if defender_id in (D.DRAGAPULT, D.DRAKLOAK, D.DREEPY):
                score += 1500.0

        if int(attack_id) == D.PHANTOM_DIVE:
            kos, useful = self._bench_counter_plan(obs, 6)
            score += kos * 7000.0 + useful * 1.5
            if kos >= 2:
                score += 4000.0
            if defender is not None and damage >= defender_hp:
                score += 1000.0
        elif int(attack_id) == D.CRUEL_ARROW:
            # 100 damage to any Pokémon, no weakness/resistance on the bench.
            best = self._best_opponent_target(obs, 100)
            if best is not None:
                hp, cid = best
                score += 1200.0
                if hp <= 100:
                    score = max(score, 12000.0 + D.prize_value(cid) * 1000.0)
        elif int(attack_id) == D.ITCHY_POLLEN:
            # Budew's item lock is worth more in the first turns.
            score = 3600.0 + (2500.0 if int(S.get(S.current(obs), "turn", 0) or 0) <= 3 else 0.0)
            if ko:
                score = LETHAL_SCORE + 500.0
        elif int(attack_id) == D.GROUND_MELTER:
            if S.stadium(obs):
                score += 3000.0
        elif int(attack_id) == D.MIND_BEND:
            score += 1200.0

        # Prefer attacks that do not require a discard/extra setup.
        return score

    def _best_opponent_target(self, obs: dict, damage: int = 0) -> tuple[int, int] | None:
        """Return (hp, card_id) of the best opposing Pokémon to hit."""
        me = S.current_player(obs)
        choices = []
        a = S.active(obs, 1 - me)
        if a is not None:
            choices.append(a)
        choices.extend(S.bench(obs, 1 - me))
        best = None
        for p in choices:
            cid = S.card_id(p)
            hp = S.hp(p)
            if cid is None or hp <= 0:
                continue
            ko = 1 if damage >= hp else 0
            key = (ko, -hp, D.prize_value(cid))
            if best is None or key > best[0]:
                best = (key, hp, cid)
        return None if best is None else (best[1], best[2])

    def max_damage_available(self, obs: dict) -> int:
        """Rough maximum damage our current board can deal next attack."""
        me = S.current_player(obs)
        best = 0
        for p in S.all_field(obs, me):
            cid = S.card_id(p)
            if cid is None:
                continue
            c = D.card(cid)
            if c is None:
                continue
            for aid in getattr(c, "attacks", []) or []:
                atk = D.attack(aid)
                if atk is None:
                    continue
                if len(getattr(atk, "energies", []) or []) <= len(S.energies(p)):
                    best = max(best, int(getattr(atk, "damage", 0) or 0))
        return best

    # ------------------------------------------------------------------
    # MAIN option scores
    # ------------------------------------------------------------------
    def evolve_score(self, obs: dict, option: Any) -> float:
        actor, target = S.option_cards(obs, option)
        if actor is None:
            return 0.0
        if actor == D.DRAGAPULT:
            score = 9500.0
        elif actor == D.DRAKLOAK:
            score = 8500.0
        else:
            score = 5000.0
        target_id = target
        if target_id is not None:
            if target_id == D.DREEPY:
                score += 800.0
            if target_id == D.DRAKLOAK:
                score += 500.0
            me = S.current_player(obs)
            if target_id == S.active_id(obs, me):
                score += 700.0
            if target_id in (S.active_id(obs, me),):
                pass
        return score

    def ability_score(self, obs: dict, option: Any) -> float:
        actor, _target = S.option_cards(obs, option)
        if actor == D.DRAKLOAK:
            return 7600.0
        if actor == D.MUNKIDORI:
            me = S.current_player(obs)
            damaged = [p for p in S.all_field(obs, me) if S.is_damaged(p)]
            if not damaged:
                return 1200.0
            counters = sum(max(0, (S.max_hp(p) - S.hp(p)) // 10) for p in damaged)
            if counters <= 0:
                return 1200.0
            kos, useful = self._bench_counter_plan(obs, min(3, counters))
            if kos:
                return 9500.0
            # Moving damage off the active or setting up a Phantom Dive KO.
            return 5000.0 + useful * 2.0
        if actor == D.FEZANDIPITI:
            return 8200.0
        if actor == D.MEOWTH:
            return 7000.0
        return 3500.0

    def _target_energy_needs(self, target_id: int | None) -> tuple[int, int, int]:
        """Return (attached fire, attached psychic, attached dark)."""
        return (0, 0, 0)

    def _attached_types(self, pokemon: Any) -> tuple[int, int, int]:
        vals = S.energies(pokemon)
        cards = S.energy_cards(pokemon)
        types = list(vals)
        for cid in cards:
            et = D.energy_type(cid)
            if et >= 0:
                types.append(et)
        return (
            sum(1 for e in types if e == D.FIRE),
            sum(1 for e in types if e == D.PSYCHIC),
            sum(1 for e in types if e == D.DARK),
        )

    def attach_score(self, obs: dict, option: Any) -> float:
        me = S.current_player(obs)
        actor, _target = S.option_cards(obs, option)
        energy_card = actor
        target_area = S.option_int(option, "inPlayArea", None)
        target_index = S.option_int(option, "inPlayIndex", None)
        target = None
        if target_area is not None and target_index is not None:
            if int(target_area) == S.AREA_ACTIVE:
                target = S.active(obs, me)
            elif int(target_area) == S.AREA_BENCH:
                bench = S.bench(obs, me)
                if 0 <= int(target_index) < len(bench):
                    target = bench[int(target_index)]
        target_id = S.card_id(target)
        etype = D.energy_type(energy_card)

        if target_id is None:
            return 2500.0
        active_id = S.active_id(obs, me)
        is_active = target is S.active(obs, me) or target_id == active_id
        base = 5000.0
        if target_id == D.DRAGAPULT:
            base = 10000.0 if is_active else 9500.0
        elif target_id == D.DRAKLOAK:
            base = 8200.0 if is_active else 7800.0
        elif target_id == D.DREEPY:
            base = 7200.0 if is_active else 6800.0
        elif target_id == D.MUNKIDORI:
            base = 8500.0 if etype == D.DARK else 2500.0
        elif target_id == D.BUDEW:
            base = 2600.0
        elif target_id in (D.CHI_YU, D.FEZANDIPITI, D.MEOWTH):
            base = 5000.0
        else:
            base = 3500.0

        # If a weak active is blocking a Dragapult, power the bench attacker
        # rather than giving the energy to the blocker.
        if target_id == D.DRAGAPULT and active_id in (D.BUDEW, D.DREEPY, D.MUNKIDORI):
            base += 1500.0

        fire, psychic, _dark = self._attached_types(target)
        if target_id in (D.DREEPY, D.DRAKLOAK, D.DRAGAPULT):
            if etype not in (D.FIRE, D.PSYCHIC):
                base -= 3000.0
            elif etype == D.FIRE and fire == 0:
                base += 1500.0
            elif etype == D.PSYCHIC and psychic == 0:
                base += 1500.0
            elif fire > 0 and psychic > 0:
                base -= 1500.0
            elif (etype == D.FIRE and fire > 0) or (etype == D.PSYCHIC and psychic > 0):
                base -= 1200.0
        if target_id == D.MUNKIDORI and etype != D.DARK:
            base -= 3000.0
        return base

    def play_score(self, obs: dict, option: Any) -> float:
        me = S.current_player(obs)
        cid = S.option_int(option, "index", None)
        card_id = S.card_at(obs, me, S.AREA_HAND, cid)
        if card_id is None:
            # Some engines expose cardId directly for card effects.
            card_id = S.option_int(option, "cardId", None)
        if card_id is None:
            return 2000.0
        bench = S.bench(obs, me)
        bench_space = max(0, int(S.get(S.player_state(obs, me), "benchMax", 5) or 5) - len(bench))
        turn = int(S.get(S.current(obs), "turn", 0) or 0)
        opp = 1 - me

        if card_id in (D.DREEPY, D.BUDEW, D.MUNKIDORI, D.CHI_YU, D.FEZANDIPITI, D.MEOWTH):
            if bench_space <= 0:
                return 500.0
            if card_id == D.DREEPY:
                return 7200.0
            if card_id == D.BUDEW:
                return 6800.0
            if card_id == D.MUNKIDORI:
                return 6500.0
            return 5600.0

        if card_id == D.POFFIN:
            if bench_space <= 0:
                return 1000.0
            return 9000.0 if len(bench) < 3 else 7200.0
        if card_id == D.POKE_PAD:
            # Search the Drakloak/Dragapult line.
            field_ids = [S.card_id(p) for p in S.all_field(obs, me)]
            missing = sum(1 for x in (D.DRAKLOAK, D.DRAGAPULT) if x not in field_ids)
            return 8500.0 if missing else 5600.0
        if card_id == D.ULTRA_BALL:
            hand_n = S.hand_count(obs, me)
            field_ids = [S.card_id(p) for p in S.all_field(obs, me)]
            need_line = any(x not in field_ids for x in (D.DRAGAPULT, D.DRAKLOAK))
            # Ultra Ball costs two other cards.  Never fire it when the hand
            # only contains the Pokémon we are trying to keep.
            if hand_n >= 6 and need_line:
                return 8300.0
            if hand_n >= 5 and need_line:
                return 6800.0
            if hand_n >= 4 and need_line and self._has_disposable(obs, 2):
                return 5200.0
            return 400.0
        if card_id == D.LILLIE:
            h = S.hand_count(obs, me)
            if S.prize_count(obs, me) == 6:
                return 9000.0
            return 8200.0 if h <= 4 else (6200.0 if h <= 6 else 3500.0)
        if card_id == D.JUDGE:
            oh = S.hand_count(obs, opp)
            return 7200.0 if oh >= 6 else (4800.0 if oh >= 4 else 2800.0)
        if card_id == D.CRISPIN:
            return 8000.0 if self._needs_energy(obs) else 5200.0
        if card_id == D.BOSS:
            return self.gust_score(obs)
        if card_id == D.NIGHT_STRETCHER:
            disc = set(S.discard_ids(obs, me))
            if disc & {D.DRAGAPULT, D.DRAKLOAK, D.DREEPY, D.FIRE, D.PSYCHIC}:
                return 6800.0
            return 2200.0
        if card_id == D.CRUSHING_HAMMER:
            foe = S.active(obs, opp)
            return 6200.0 if S.energies(foe) else 3000.0
        if card_id == D.UNFAIR_STAMP:
            return 7200.0 if S.prize_count(obs, me) >= S.prize_count(obs, opp) else 5200.0
        if card_id == D.WATCHTOWER:
            return 4200.0 if self._opponent_ability_heavy(obs) else 2400.0
        if card_id == D.RISKY_RUINS:
            # It also damages our own Dreepy, so keep it situational.
            own_dreepy = sum(1 for p in bench if S.card_id(p) == D.DREEPY)
            return 3400.0 if own_dreepy == 0 else 1800.0
        if card_id in (D.FIRE, D.PSYCHIC, D.DARK):
            return 2500.0  # energy is normally attached, not played
        return 3000.0

    def _needs_energy(self, obs: dict) -> bool:
        me = S.current_player(obs)
        for p in S.all_field(obs, me):
            cid = S.card_id(p)
            if cid in (D.DREEPY, D.DRAKLOAK, D.DRAGAPULT):
                fire, psychic, _ = self._attached_types(p)
                if fire == 0 or psychic == 0:
                    return True
        return False

    def _opponent_ability_heavy(self, obs: dict) -> bool:
        me = S.current_player(obs)
        for p in S.all_field(obs, 1 - me):
            cid = S.card_id(p)
            c = D.card(cid)
            if c is not None and getattr(c, "skills", None):
                return True
        return False

    def active_max_damage(self, obs: dict) -> int:
        me = S.current_player(obs)
        p = S.active(obs, me)
        cid = S.card_id(p)
        c = D.card(cid)
        if c is None:
            return 0
        best = 0
        for aid in getattr(c, "attacks", []) or []:
            atk = D.attack(aid)
            if atk is None:
                continue
            if len(getattr(atk, "energies", []) or []) <= len(S.energies(p)):
                best = max(best, self._damage(obs, int(aid), cid, S.active(obs, 1 - me)))
        return best

    def gust_score(self, obs: dict) -> float:
        me = S.current_player(obs)
        active_dmg = self.active_max_damage(obs)
        # If the active is a weak support Pokémon, a powered bench Dragapult
        # can still attack after a retreat.  That is a legitimate gust line.
        bench_dmg = 0
        active_id = S.active_id(obs, me)
        if active_id in (D.BUDEW, D.DREEPY, D.MUNKIDORI):
            for p in S.bench(obs, me):
                if S.card_id(p) == D.DRAGAPULT and len(S.energies(p)) >= 2:
                    bench_dmg = max(bench_dmg, 200)
                elif S.card_id(p) == D.DRAKLOAK and len(S.energies(p)) >= 2:
                    bench_dmg = max(bench_dmg, 70)
        reach = max(active_dmg, bench_dmg)
        best = 0.0
        for p in S.bench(obs, 1 - me):
            cid = S.card_id(p)
            hp = S.hp(p)
            if cid is None:
                continue
            prize = D.prize_value(cid)
            if reach >= hp:
                score = 9000.0 + prize * 700.0
            elif hp <= 60:
                score = 5200.0 + prize * 300.0
            else:
                score = 2600.0 + prize * 150.0
            if cid in (D.DREEPY, D.DRAKLOAK):
                score += 700.0
            best = max(best, score)
        return best

    def retreat_score(self, obs: dict, option: Any) -> float:
        me = S.current_player(obs)
        a = S.active(obs, me)
        aid = S.card_id(a)
        # Never retreat the Dragapult/Drakloak line just to retreat.  The line
        # needs to stay active so it can evolve and attack next turn.
        if aid in (D.DRAGAPULT, D.DRAKLOAK):
            return -1000.0
        powered = False
        best_bench = None
        for p in S.bench(obs, me):
            cid = S.card_id(p)
            if cid == D.DRAGAPULT and len(S.energies(p)) >= 1:
                powered = True
                best_bench = cid
            elif cid == D.DRAKLOAK and len(S.energies(p)) >= 1 and best_bench is None:
                powered = True
                best_bench = cid
        if aid == D.DREEPY:
            return 5600.0 if powered and best_bench == D.DRAGAPULT else (3600.0 if powered else -500.0)
        if aid == D.BUDEW:
            # Budew's 10-damage attack is worse than promoting a powered
            # Dragapult/Drakloak, even after giving up the item lock.
            return 6900.0 if powered and best_bench == D.DRAGAPULT else (4200.0 if powered else -500.0)
        if aid == D.MUNKIDORI:
            return 3800.0 if powered else -500.0
        return -500.0

    def main_discard_score(self, obs: dict, option: Any) -> float:
        actor, _ = S.option_cards(obs, option)
        if actor == D.RISKY_RUINS:
            return 2800.0
        return 600.0

    # ------------------------------------------------------------------
    # Non-MAIN selections
    # ------------------------------------------------------------------
    def choose_other(self, obs: dict, options: list, mn: int, mx: int) -> list[int]:
        stype = S.select_type(obs)
        context = S.select_context(obs)
        if stype == COUNT:
            return [self._choose_count(obs, options, context)]
        if stype == YES_NO:
            return [self._choose_yes_no(obs, options, context, mx)]
        if stype == ATTACK:
            return [self._choose_attack(obs, options)]
        if stype == SPECIAL_CONDITION:
            return [self._choose_condition(obs, options, context)]
        if stype == ENERGY:
            return self._choose_energy(obs, options, context, mn, mx)
        if stype in (ATTACHED_CARD, CARD_OR_ATTACHED_CARD):
            return self._choose_attached(obs, options, context, mn, mx)
        if stype == EVOLVE:
            return self._choose_evolve(obs, options, mn, mx)
        if stype == SKILL:
            return self._choose_skill(obs, options, mn, mx)

        # CARD selections (the common case).
        scored = [(self.card_option_score(obs, option, context), i) for i, option in enumerate(options)]
        if context in DISCARD_CONTEXTS:
            count = mn
            ordered = sorted(scored, key=lambda item: (item[0], item[1]))
        else:
            count = mx
            ordered = sorted(scored, key=lambda item: (-item[0], item[1]))
        count = max(mn, min(count, mx))
        return [i for _score, i in ordered[:count]]

    def card_option_score(self, obs: dict, option: Any, context: int) -> float:
        actor, target = S.option_cards(obs, option)
        owner = S.option_target_player(obs, option)
        me = S.current_player(obs)
        card_id = target if target is not None else actor
        if card_id is None:
            card_id = S.option_int(option, "cardId", None)

        if context in (SETUP_ACTIVE,):
            return self.setup_active_score(obs, card_id)
        if context in (SETUP_BENCH, TO_BENCH, TO_FIELD):
            return self.bench_pick_score(obs, card_id)
        if context in (SWITCH, TO_ACTIVE):
            return self.active_pick_score(obs, card_id)
        if context in (DAMAGE_COUNTER, DAMAGE_COUNTER_ANY, DAMAGE, EFFECT_TARGET):
            return self.attack_target_score(obs, option, owner, card_id)
        if context in (REMOVE_DAMAGE_COUNTER, HEAL):
            return self.heal_target_score(obs, card_id)
        if context in (EVOLVES_FROM, EVOLVES_TO):
            return self.card_value(card_id) + 3000.0
        if context == DEVOLVE:
            return self.card_value(card_id) + 2000.0
        if context in (ATTACH_TO, ATTACH_FROM):
            meta = D.card(card_id)
            if meta is not None and int(getattr(meta, "cardType", -1)) == 0:
                return self.attach_target_score(obs, card_id)
            return self.energy_pick_score(obs, card_id)
        if context == DETACH_FROM:
            return 3000.0 - self.card_value(card_id) / 10.0
        if context == LOOK:
            return self.card_value(card_id)
        if context in DISCARD_CONTEXTS:
            return self.discard_value(obs, card_id)
        # Search/draw to hand and generic card picks.
        return self.card_pick_score(obs, card_id, context)

    def card_pick_score(self, obs: dict, card_id: int | None, context: int) -> float:
        if card_id is None:
            return 0.0
        base = self.card_value(card_id)
        me = S.current_player(obs)
        field = [S.card_id(p) for p in S.all_field(obs, me)]
        hand_ids = [S.card_id(c) for c in S.hand(obs, me)]
        field_set = set(field)
        has_dreepy = D.DREEPY in field_set or D.DREEPY in hand_ids
        has_drakloak = D.DRAKLOAK in field_set or D.DRAKLOAK in hand_ids
        has_dragapult = D.DRAGAPULT in field_set or D.DRAGAPULT in hand_ids
        bench_space = max(0, int(S.get(S.player_state(obs, me), "benchMax", 5) or 5) - len(S.bench(obs, me)))

        if card_id == D.DREEPY:
            if not has_dreepy and bench_space > 0:
                return base + 6000.0
            return base + (1500.0 if bench_space > 0 else -2000.0)
        if card_id == D.DRAKLOAK:
            if not has_dreepy and not has_drakloak:
                return base - 2500.0
            if not has_drakloak:
                return base + 3500.0
            return base - 1000.0
        if card_id == D.DRAGAPULT:
            if not has_drakloak:
                return base - 4000.0
            if not has_dragapult:
                return base + 4500.0
            return base - 1500.0
        if card_id in (D.FIRE, D.PSYCHIC):
            return base + (1500.0 if self._needs_energy(obs) else 0.0)
        if card_id == D.DARK:
            return base + (800.0 if D.MUNKIDORI in field else 0.0)
        if card_id in (D.POFFIN, D.POKE_PAD, D.ULTRA_BALL, D.CRISPIN):
            return base + 1000.0
        return base

    def _has_disposable(self, obs: dict, count: int) -> bool:
        me = S.current_player(obs)
        key = {D.DRAGAPULT, D.DRAKLOAK, D.DREEPY, D.BUDEW, D.MUNKIDORI,
               D.FEZANDIPITI, D.MEOWTH, D.CHI_YU}
        disposable = 0
        for c in S.hand(obs, me):
            cid = S.card_id(c)
            if cid not in key:
                disposable += 1
        return disposable >= count

    def card_value(self, card_id: int | None) -> float:
        if card_id is None:
            return 0.0
        values = {
            D.DRAGAPULT: 10000.0,
            D.DRAKLOAK: 9000.0,
            D.DREEPY: 8000.0,
            D.MUNKIDORI: 7000.0,
            D.BUDEW: 6500.0,
            D.FEZANDIPITI: 6200.0,
            D.MEOWTH: 6200.0,
            D.CHI_YU: 5200.0,
            D.POFFIN: 7000.0,
            D.POKE_PAD: 7200.0,
            D.ULTRA_BALL: 6500.0,
            D.LILLIE: 6200.0,
            D.JUDGE: 5600.0,
            D.CRISPIN: 7000.0,
            D.BOSS: 6500.0,
            D.NIGHT_STRETCHER: 5000.0,
            D.CRUSHING_HAMMER: 4800.0,
            D.UNFAIR_STAMP: 5600.0,
            D.WATCHTOWER: 3000.0,
            D.RISKY_RUINS: 3000.0,
            D.FIRE: 4600.0,
            D.PSYCHIC: 4600.0,
            D.DARK: 3800.0,
        }
        return values.get(int(card_id), 3000.0)

    def setup_active_score(self, obs: dict, card_id: int | None) -> float:
        # Follow the Top2 opening pattern: a Dreepy or Budew is the usual
        # active; Munkidori is the fallback when the line is not available.
        if card_id == D.DREEPY:
            return 9000.0
        if card_id == D.BUDEW:
            return 8700.0
        if card_id == D.MUNKIDORI:
            return 8000.0
        if card_id == D.CHI_YU:
            return 6500.0
        if card_id in (D.MEOWTH, D.FEZANDIPITI):
            return 6000.0
        return 3000.0

    def bench_pick_score(self, obs: dict, card_id: int | None) -> float:
        if card_id == D.DREEPY:
            return 9500.0
        if card_id == D.BUDEW:
            return 7000.0
        if card_id == D.MUNKIDORI:
            return 6500.0
        if card_id in (D.FEZANDIPITI, D.MEOWTH):
            return 5200.0
        if card_id == D.CHI_YU:
            return 4800.0
        return 3000.0

    def active_pick_score(self, obs: dict, card_id: int | None) -> float:
        me = S.current_player(obs)
        # Find the actual Pokémon dict to account for attached energy.
        chosen = None
        for p in S.all_field(obs, me):
            if S.card_id(p) == card_id:
                chosen = p
                break
        energy = len(S.energies(chosen)) if chosen is not None else 0
        if card_id == D.DRAGAPULT:
            return 10000.0 + energy * 300.0
        if card_id == D.DRAKLOAK:
            return 8500.0 + energy * 250.0
        if card_id == D.DREEPY:
            return 7000.0 + energy * 200.0
        if card_id == D.BUDEW:
            return 6000.0
        if card_id == D.MUNKIDORI:
            return 5500.0
        return 4000.0

    def attack_target_score(self, obs: dict, option: Any, owner: int | None,
                            card_id: int | None) -> float:
        me = S.current_player(obs)
        if owner is not None and int(owner) == me:
            # Our own Pokémon; used by effects that move damage to ourselves.
            return 1000.0 - self.card_value(card_id) / 100.0
        p = self._find_pokemon(obs, owner if owner is not None else 1 - me,
                               S.option_int(option, "area", None),
                               S.option_int(option, "index", None))
        hp = S.hp(p) if p is not None else (D.card(card_id).hp if D.card(card_id) else 100)
        prize = D.prize_value(card_id)
        # A damage counter option: prefer a KO, then the lowest HP.
        if hp <= 10:
            return 12000.0 + prize * 1000.0
        if hp <= 30:
            return 9000.0 + prize * 700.0
        if hp <= 60:
            return 6500.0 + prize * 500.0
        return 3500.0 - hp / 10.0 + prize * 200.0

    def _find_pokemon(self, obs: dict, player: int, area: int | None, index: int | None):
        if area is None or index is None:
            return None
        if int(area) == S.AREA_ACTIVE:
            return S.active(obs, player)
        if int(area) == S.AREA_BENCH:
            bench = S.bench(obs, player)
            if 0 <= int(index) < len(bench):
                return bench[int(index)]
        return None

    def heal_target_score(self, obs: dict, card_id: int | None) -> float:
        me = S.current_player(obs)
        p = None
        for candidate in S.all_field(obs, me):
            if S.card_id(candidate) == card_id:
                p = candidate
                break
        if p is None:
            return 1000.0
        damage = S.max_hp(p) - S.hp(p)
        return 3000.0 + damage * 4.0 + self.card_value(card_id) / 20.0

    def attach_target_score(self, obs: dict, card_id: int | None) -> float:
        me = S.current_player(obs)
        p = None
        for candidate in S.all_field(obs, me):
            if S.card_id(candidate) == card_id:
                p = candidate
                break
        if p is None:
            return 1000.0
        active_id = S.active_id(obs, me)
        is_active = S.card_id(p) == active_id
        if card_id == D.DRAGAPULT:
            return 10000.0 if is_active else 9500.0
        if card_id == D.DRAKLOAK:
            return 8200.0 if is_active else 7800.0
        if card_id == D.DREEPY:
            return 7200.0 if is_active else 6800.0
        if card_id == D.MUNKIDORI:
            return 6500.0
        if card_id == D.BUDEW:
            return 2200.0
        return 4800.0

    def energy_pick_score(self, obs: dict, energy_card: int | None) -> float:
        if energy_card is None:
            return 0.0
        etype = D.energy_type(energy_card)
        if etype in (D.FIRE, D.PSYCHIC):
            return 7000.0
        if etype == D.DARK:
            return 4500.0
        return 3000.0

    def discard_value(self, obs: dict, card_id: int | None) -> float:
        if card_id is None:
            return 10000.0
        value = self.card_value(card_id)
        # Duplicates and extra stadiums are cheaper to throw away.
        me = S.current_player(obs)
        hand_ids = [S.card_id(c) for c in S.hand(obs, me)]
        count = sum(1 for x in hand_ids if x == card_id)
        if count >= 2:
            value *= 0.7
        if card_id in (D.RISKY_RUINS, D.CRUSHING_HAMMER):
            value *= 0.6
        if card_id in (D.DRAGAPULT, D.DRAKLOAK, D.DREEPY):
            value += 2000.0
        return value

    def _choose_count(self, obs: dict, options: list, context: int) -> int:
        numbers = [S.option_int(o, "number", 0) or 0 for o in options]
        if context in (DRAW_COUNT, DAMAGE_COUNTER_COUNT, REMOVE_DAMAGE_COUNTER_COUNT):
            best = max(range(len(numbers)), key=lambda i: (numbers[i], -i))
        else:
            best = min(range(len(numbers)), key=lambda i: (numbers[i], i))
        return int(best)

    def _choose_yes_no(self, obs: dict, options: list, context: int, mx: int) -> int:
        if context == MULLIGAN:
            # The engine only asks when the opening hand has no Basic.
            return self._index_of_type(options, YES)
        if context in (IS_FIRST, ACTIVATE, FIRST_EFFECT, COIN_HEAD, MORE_DEVOLVE):
            idx = self._index_of_type(options, YES)
            return idx if idx >= 0 else 0
        idx = self._index_of_type(options, YES)
        return idx if idx >= 0 else 0

    def _index_of_type(self, options: list, option_type: int) -> int:
        for i, o in enumerate(options):
            if S.option_type(o) == option_type:
                return i
        return -1

    def _choose_attack(self, obs: dict, options: list) -> int:
        best = 0
        best_score = -1e9
        for i, o in enumerate(options):
            aid = S.option_int(o, "attackId", None)
            if aid is None:
                score = 0
            else:
                score = self._damage(obs, int(aid))
            if score > best_score:
                best_score = score
                best = i
        return int(best)

    def _choose_condition(self, obs: dict, options: list, context: int) -> int:
        if context == RECOVER_SPECIAL_CONDITION:
            return 0
        priority = {3: 5, 2: 4, 4: 3, 1: 2, 0: 1}  # paralyze/sleep/confuse/burn/poison
        best = 0
        best_score = -1
        for i, o in enumerate(options):
            cond = S.option_int(o, "specialConditionType", 0) or 0
            score = priority.get(int(cond), 0)
            if score > best_score:
                best_score = score
                best = i
        return int(best)

    def _choose_energy(self, obs: dict, options: list, context: int, mn: int, mx: int) -> list[int]:
        scored = []
        for i, o in enumerate(options):
            actor, _ = S.option_cards(obs, o)
            if context in (DISCARD_ENERGY, TO_DECK_ENERGY):
                score = -self.energy_pick_score(obs, actor)
            else:
                score = self.energy_pick_score(obs, actor)
            scored.append((score, i))
        if context in (DISCARD_ENERGY, TO_DECK_ENERGY):
            ordered = sorted(scored, key=lambda item: (item[0], item[1]))
            count = mn
        else:
            ordered = sorted(scored, key=lambda item: (-item[0], item[1]))
            count = mx
        count = max(mn, min(count, mx))
        return [i for _s, i in ordered[:count]]

    def _choose_attached(self, obs: dict, options: list, context: int, mn: int, mx: int) -> list[int]:
        scored = []
        for i, o in enumerate(options):
            actor, _ = S.option_cards(obs, o)
            if context in (DISCARD_ENERGY_CARD, DISCARD_TOOL_CARD,
                           DISCARD_CARD_OR_ATTACHED_CARD):
                score = self.discard_value(obs, actor)
                scored.append((score, i))
            elif context == SWITCH_ENERGY_CARD:
                scored.append((-self.energy_pick_score(obs, actor), i))
            else:
                scored.append((self.discard_value(obs, actor), i))
        if context in (DISCARD_ENERGY_CARD, DISCARD_TOOL_CARD,
                       DISCARD_CARD_OR_ATTACHED_CARD):
            ordered = sorted(scored, key=lambda item: (item[0], item[1]))
            count = mn
        else:
            ordered = sorted(scored, key=lambda item: (-item[0], item[1]))
            count = mx
        count = max(mn, min(count, mx))
        return [i for _s, i in ordered[:count]]

    def _choose_evolve(self, obs: dict, options: list, mn: int, mx: int) -> list[int]:
        scored = [(self.evolve_score(obs, o), i) for i, o in enumerate(options)]
        ordered = sorted(scored, key=lambda item: (-item[0], item[1]))
        count = max(mn, min(mx, mx))
        return [i for _s, i in ordered[:count]]

    def _choose_skill(self, obs: dict, options: list, mn: int, mx: int) -> list[int]:
        # Skill order is rarely important; use the first legal order.
        count = max(mn, min(mx, mx))
        return list(range(count))

    # ------------------------------------------------------------------
    # Legalisation
    # ------------------------------------------------------------------
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
