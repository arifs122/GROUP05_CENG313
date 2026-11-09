"""Layer 0/1: legal, deck-aware Mega Kangaskhan / Ogerpon rule engine.

The engine is deliberately defensive: every path returns a legal selection and
the public entry point falls back to a legal first option if a scorer raises.
No hidden information (opponent hand/prize/deck, replay fields) is read.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import mk_data as D
import mk_state as S

LETHAL = 20000.0


class RuleEngine:
    def __init__(self, deck: list[int], searcher=None, prior=None,
                 prior_weight: float = 0.0) -> None:
        self.deck = list(deck)
        self.searcher = searcher
        self.prior = prior
        self.prior_weight = float(prior_weight)
        self.search_overrides = 0
        self.lethal_overrides = 0

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def choose(self, obs: dict) -> list[int]:
        sel = S.select(obs)
        if sel is None:
            return list(self.deck)
        opts = S.options(obs)
        if not opts:
            return []
        mn = S.min_count(obs)
        mx = min(S.max_count(obs), len(opts))
        if mx <= 0:
            return []
        if S.select_type(obs) == S.S_MAIN:
            choice = self.choose_main(obs, opts)
        else:
            choice = self.choose_other(obs, opts, mn, mx)
        return self.legalize(choice, len(opts), mn, mx)

    def legalize(self, choice: Any, n: int, mn: int, mx: int) -> list[int]:
        if not isinstance(choice, (list, tuple)):
            choice = [choice] if isinstance(choice, int) else []
        out: list[int] = []
        for item in choice:
            try:
                idx = int(item)
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

    # ------------------------------------------------------------------
    # MAIN
    # ------------------------------------------------------------------
    def choose_main(self, obs: dict, options: list) -> list[int]:
        scores: list[tuple[float, int]] = []
        lethal: list[tuple[float, int]] = []
        for i, option in enumerate(options):
            try:
                score = self.main_score(obs, option)
            except Exception:
                score = 0.0
            scores.append((float(score), i))
            if score >= LETHAL:
                lethal.append((float(score), i))

        # Rule guarantee: take a guaranteed lethal attack before anything.
        if lethal:
            self.lethal_overrides += 1
            return [max(lethal, key=lambda item: (item[0], -item[1]))[1]]

        if self.prior is not None and self.prior_weight > 0:
            try:
                logits = self.prior.logits(obs)
            except Exception:
                logits = None
            if logits is not None and len(logits) == len(scores):
                mean = sum(float(x) for x in logits) / max(1, len(logits))
                adjusted = []
                for s, i in scores:
                    # The prior is weakest on raw energy placement and setup
                    # plays; keep the rule score dominant there.
                    w = self.prior_weight
                    otype = S.option_type(options[i])
                    if otype == S.O_ATTACH:
                        w = 0.0
                    elif otype == S.O_PLAY:
                        cid = S.option_int(options[i], "index", None)
                        cid = S.card_at(obs, S.you(obs), S.HAND, cid) if cid is not None else None
                        if cid in (D.OGERPON_TEAL, D.OGERPON_WELL, D.KANGASKHAN,
                                   D.LATIAS, D.CLEFAIRY):
                            w *= 0.25
                    adjusted.append((s + w * (float(logits[i]) - mean), i))
                scores = adjusted

        best = max(scores, key=lambda item: (item[0], -item[1]))
        if best[0] <= 0.0:
            end = self.index_of_type(options, S.O_END)
            if end is not None:
                return [end]
        if self.searcher is not None:
            try:
                alt = self.searcher.choose(obs, scores)
            except Exception:
                alt = None
            if alt is not None and alt != best[1]:
                self.search_overrides += 1
                return [int(alt)]
        return [best[1]]

    def main_score(self, obs: dict, option: Any) -> float:
        otype = S.option_type(option)
        if otype == S.O_ATTACK:
            return self.attack_score(obs, option)
        if otype == S.O_ABILITY:
            return self.ability_score(obs, option)
        if otype == S.O_ATTACH:
            return self.attach_score(obs, option)
        if otype == S.O_PLAY:
            return self.play_score(obs, option)
        if otype == S.O_RETREAT:
            return self.retreat_score(obs, option)
        if otype == S.O_EVOLVE:
            return 5000.0
        if otype == S.O_DISCARD:
            return 1000.0
        if otype == S.O_END:
            return 0.0
        if otype in (S.O_SKILL, S.O_SPECIAL_CONDITION):
            return 2500.0
        return 500.0

    # ------------------------------------------------------------------
    # Board / plan helpers
    # ------------------------------------------------------------------
    def me(self, obs: dict) -> int:
        return S.you(obs)

    def my_active(self, obs: dict) -> Any:
        return S.active(obs, self.me(obs))

    def opp_active(self, obs: dict) -> Any:
        return S.active(obs, 1 - self.me(obs))

    def my_bench(self, obs: dict) -> list:
        return S.bench(obs, self.me(obs))

    def opp_bench(self, obs: dict) -> list:
        return S.bench(obs, 1 - self.me(obs))

    def my_field(self, obs: dict) -> list:
        return S.field(obs, self.me(obs))

    def opp_field(self, obs: dict) -> list:
        return S.field(obs, 1 - self.me(obs))

    def my_hand_ids(self, obs: dict) -> list[int]:
        return S.hand_ids(obs, self.me(obs))

    def bench_space(self, obs: dict) -> int:
        st = S.my_state(obs)
        cap = S.as_int(S.get(st, "benchMax", 5), 5) or 5
        return max(0, cap - len(self.my_bench(obs)))

    def in_play(self, obs: dict, cid: int) -> bool:
        for p in self.my_field(obs):
            if S.pokemon_id(p) == cid or cid in S.pre_evolution(p):
                return True
        return False

    def count_in_play(self, obs: dict, cid: int) -> int:
        n = 0
        for p in self.my_field(obs):
            if S.pokemon_id(p) == cid or cid in S.pre_evolution(p):
                n += 1
        return n

    def has_grass_in_hand(self, obs: dict) -> bool:
        return D.GRASS in self.my_hand_ids(obs)

    def has_prism_in_hand(self, obs: dict) -> bool:
        return D.PRISM in self.my_hand_ids(obs)

    def energy_on_field(self, obs: dict) -> int:
        return sum(S.energy_count(p) for p in self.my_field(obs))

    def basic_energy_on_bench(self, obs: dict) -> int:
        n = 0
        for p in self.my_bench(obs):
            for cid in S.energy_cards(p):
                if D.card_type(cid) == D.BASIC_ENERGY:
                    n += 1
        return n

    def has_tera_in_play(self, obs: dict) -> bool:
        return any(S.pokemon_id(p) in D.TERA_IDS for p in self.my_field(obs))

    def has_latias(self, obs: dict) -> bool:
        return self.in_play(obs, D.LATIAS)

    def has_clefairy(self, obs: dict) -> bool:
        return self.in_play(obs, D.CLEFAIRY)

    def has_kang(self, obs: dict) -> bool:
        return self.in_play(obs, D.KANGASKHAN)

    def opponent_archetype(self, obs: dict) -> str:
        opp = 1 - self.me(obs)
        counts: Counter[int] = Counter()
        for p in S.field(obs, opp):
            cid = S.pokemon_id(p)
            if cid is not None:
                counts[cid] += 1
            for pre in S.pre_evolution(p):
                counts[pre] += 1
            for e in S.energy_cards(p):
                counts[e] += 1
        for cid in S.discard_ids(obs, opp):
            counts[cid] += 1
        for card in S.stadium(obs):
            if S.as_int(S.get(card, "playerIndex", -1), -1) == opp:
                cid = S.card_id(card)
                if cid is not None:
                    counts[cid] += 1
        best_name = "Unknown"
        best_score = 0
        for name, keys in D.ARCHETYPE_KEYS:
            if not keys:
                continue
            score = sum(min(2, counts[k]) for k in keys)
            if score > best_score:
                best_score = score
                best_name = name
        return best_name

    def opponent_has_dragon(self, obs: dict) -> bool:
        opp = 1 - self.me(obs)
        for p in S.field(obs, opp):
            cid = S.pokemon_id(p)
            if cid is not None and D.energy_type(cid) == D.DRAGON_T:
                return True
            if any(D.energy_type(x) == D.DRAGON_T for x in S.pre_evolution(p)):
                return True
        return False

    def my_prizes_left(self, obs: dict) -> int:
        return S.prize_count(obs, self.me(obs))

    def opp_prizes_left(self, obs: dict) -> int:
        return S.prize_count(obs, 1 - self.me(obs))

    def estimated_turns_left(self, obs: dict) -> int:
        # Each remaining prize usually needs one to two of our turns.
        return max(2, self.my_prizes_left(obs) * 2)

    def draw_safe(self, obs: dict, extra: int = 0) -> bool:
        deck = S.deck_count(obs, self.me(obs))
        return deck > self.estimated_turns_left(obs) + 4 + int(extra)

    def deck_low(self, obs: dict) -> bool:
        deck = S.deck_count(obs, self.me(obs))
        return deck <= self.estimated_turns_left(obs) + 2

    # ------------------------------------------------------------------
    # Damage model
    # ------------------------------------------------------------------
    def _modified_damage(self, obs: dict, attacker_id: int | None, defender_id: int | None,
                         damage: int, benched: bool = False) -> int:
        if damage <= 0 or defender_id is None:
            return max(0, damage)
        if benched:
            return max(0, damage)
        atk_type = D.energy_type(attacker_id)
        weak = D.weakness(defender_id)
        if weak is not None and atk_type == weak:
            damage *= 2
        # Lillie's Clefairy ex: opponent Dragons become Psychic-weak.
        if (self.has_clefairy(obs) and D.energy_type(defender_id) == D.DRAGON_T
                and atk_type == D.PSYCHIC_T):
            damage *= 2
        res = D.resistance(defender_id)
        if res is not None and atk_type == res:
            damage = max(0, damage - 30)
        return max(0, damage)

    def _active_energy_counts(self, obs: dict) -> tuple[int, int]:
        a = self.my_active(obs)
        b = self.opp_active(obs)
        return S.energy_count(a) if a is not None else 0, S.energy_count(b) if b is not None else 0

    def attack_damage(self, obs: dict, attack_id: int, attacker: Any,
                      defender: Any, benched: bool = False) -> int:
        if attacker is None or defender is None:
            return 0
        attacker_id = S.pokemon_id(attacker)
        defender_id = S.pokemon_id(defender)
        if defender_id is None:
            return 0
        if benched and D.is_tera(defender_id):
            return 0
        base = D.base_damage(attack_id)
        dmg = base
        if attack_id == D.MYRIAD_LEAF_SHOWER:
            mine, theirs = self._active_energy_counts(obs)
            dmg = 30 + 30 * (mine + theirs)
        elif attack_id == D.FULL_MOON_RONDO:
            dmg = 20 + 20 * (len(self.my_bench(obs)) + len(self.opp_bench(obs)))
        elif attack_id == D.RAPID_FIRE_COMBO:
            # Coin until tails: expected heads is 1, guaranteed base is 200.
            dmg = 250
        elif attack_id == D.GROUND_MELTER:
            if S.stadium(obs):
                dmg = 120
        elif attack_id == D.CRUEL_ARROW:
            dmg = 100
        elif attack_id in (D.SOB, D.TORRENTIAL_PUMP, D.EON_BLADE,
                           D.POISON_CHAIN, D.MIND_BEND, D.TUCK_TAIL, D.ALLURE):
            dmg = base
        return self._modified_damage(obs, attacker_id, defender_id, dmg, benched)

    def attack_min_damage(self, obs: dict, attack_id: int, attacker: Any,
                          defender: Any, benched: bool = False) -> int:
        if attacker is None or defender is None:
            return 0
        if attack_id == D.RAPID_FIRE_COMBO:
            attacker_id = S.pokemon_id(attacker)
            defender_id = S.pokemon_id(defender)
            raw = 200
            return self._modified_damage(obs, attacker_id, defender_id, raw, benched)
        return self.attack_damage(obs, attack_id, attacker, defender, benched)

    def attack_potential(self, obs: dict, pokemon: Any) -> int:
        """Best damage this Pokémon can deal right now, ignoring type details."""
        if pokemon is None:
            return 0
        cid = S.pokemon_id(pokemon)
        if cid is None:
            return 0
        have = S.energy_count(pokemon)
        defender = self.opp_active(obs)
        best = 0
        for aid in D.attacks_of(cid):
            a = D.attack(aid)
            if a is None:
                continue
            if len(getattr(a, "energies", []) or []) > have:
                continue
            if aid == D.ALLURE:
                dmg = 0
            elif aid == D.CRUEL_ARROW:
                dmg = 100
            else:
                dmg = self.attack_damage(obs, aid, pokemon, defender)
            best = max(best, dmg)
        return best

    def best_opponent_target(self, obs: dict, attack_id: int,
                             include_bench: bool = False) -> tuple[Any, int, int, bool] | None:
        """Return (pokemon, damage, min_damage, benched) for the best target."""
        me = self.me(obs)
        attacker = self.my_active(obs)
        if attacker is None:
            return None
        candidates: list[tuple[Any, bool]] = []
        a = self.opp_active(obs)
        if a is not None:
            candidates.append((a, False))
        if include_bench:
            for p in self.opp_bench(obs):
                candidates.append((p, True))
        best = None
        for target, benched in candidates:
            dmg = self.attack_damage(obs, attack_id, attacker, target, benched)
            mind = self.attack_min_damage(obs, attack_id, attacker, target, benched)
            hp = S.hp(target)
            cid = S.pokemon_id(target)
            ko = 1 if mind >= hp and mind > 0 else 0
            key = (ko, D.prize_value(cid), -hp, 1 if not benched else 0, dmg)
            if best is None or key > best[0]:
                best = (key, target, dmg, mind, benched)
        if best is None:
            return None
        return best[1], best[2], best[3], best[4]

    # ------------------------------------------------------------------
    # Attacks
    # ------------------------------------------------------------------
    def attack_score(self, obs: dict, option: Any) -> float:
        attack_id = S.option_int(option, "attackId", None)
        if attack_id is None:
            return 0.0
        attack_id = int(attack_id)
        me = self.me(obs)
        attacker = self.my_active(obs)
        defender = self.opp_active(obs)
        if attacker is None:
            return 0.0

        # Cruel Arrow can hit any Pokémon; Torrential Pump can also hit bench.
        include_bench = attack_id in (D.CRUEL_ARROW, D.TORRENTIAL_PUMP)
        target_info = self.best_opponent_target(obs, attack_id, include_bench)
        if target_info is None:
            return 0.0
        target, dmg, mind, benched = target_info
        target_id = S.pokemon_id(target)
        hp = S.hp(target)
        prize = D.prize_value(target_id)

        score = 5000.0 + dmg * 8.0
        if mind >= hp and mind > 0:
            score = LETHAL + prize * 1200.0
            if target_id in (121, 120, 119, 743, 742, 741, 756, 747, 163):
                score += 1400.0
            if benched:
                score += 1200.0
        elif dmg >= hp and dmg > 0:
            # Expected lethal behind a coin flip: strong but not a guarantee.
            score = 15000.0 + prize * 900.0
        elif benched:
            score += 2200.0 + prize * 300.0

        # Tactical bonuses.
        if attack_id == D.MYRIAD_LEAF_SHOWER:
            score += 300.0 + 80.0 * S.energy_count(attacker)
        elif attack_id == D.RAPID_FIRE_COMBO:
            score += 500.0
        elif attack_id == D.FULL_MOON_RONDO:
            if self.opponent_has_dragon(obs):
                score += 1800.0
            score += 60.0 * (len(self.my_bench(obs)) + len(self.opp_bench(obs)))
        elif attack_id == D.SOB:
            # Cheap lock attack: good when we cannot reach a bigger attack yet.
            score = max(score, 5600.0)
            if S.energy_count(attacker) <= 1:
                score += 500.0
            score += 400.0
        elif attack_id == D.TORRENTIAL_PUMP:
            best_bench = None
            for p in self.opp_bench(obs):
                cid = S.pokemon_id(p)
                if cid is None or D.is_tera(cid):
                    continue
                bh = S.hp(p)
                key = (1 if bh <= 120 else 0, D.prize_value(cid), -bh)
                if best_bench is None or key > best_bench[0]:
                    best_bench = (key, p)
            if best_bench is not None:
                score += 1200.0 + best_bench[0][1] * 400.0
            if S.energy_count(attacker) < 5:
                score -= 800.0
        elif attack_id == D.CRUEL_ARROW:
            score += 700.0
        elif attack_id == D.POISON_CHAIN:
            active_id = S.pokemon_id(attacker)
            if active_id == D.PECHARUNT:
                score += 2400.0
            score += 500.0
        elif attack_id == D.GROUND_MELTER:
            if S.stadium(obs) and D.AREA_ZERO in S.stadium_ids(obs) and self.has_tera_in_play(obs):
                score -= 2600.0
        elif attack_id == D.ALLURE:
            if self.draw_safe(obs) and S.hand_count(obs, me) <= 5:
                score = max(score, 7200.0)
            else:
                score = 0.0
        elif attack_id == D.EON_BLADE:
            score -= 1800.0
        elif attack_id == D.MIND_BEND:
            score += 500.0
        elif attack_id == D.TUCK_TAIL:
            score -= 400.0

        # Avoid a 0-damage attack when something else is available.
        if dmg <= 0 and attack_id not in (D.ALLURE,):
            score -= 1000.0
        return score

    # ------------------------------------------------------------------
    # Abilities
    # ------------------------------------------------------------------
    def ability_score(self, obs: dict, option: Any) -> float:
        actor, _target = S.option_cards(obs, option)
        deck_safe = self.draw_safe(obs)
        if actor == D.OGERPON_TEAL:
            if not self.has_grass_in_hand(obs):
                return 0.0
            return 11200.0 if deck_safe else 0.0
        if actor == D.KANGASKHAN:
            active = self.my_active(obs)
            if S.pokemon_id(active) == D.KANGASKHAN and deck_safe:
                return 9600.0
            return 0.0
        if actor == D.MUNKIDORI:
            if not self.munkidori_has_dark(obs):
                return 0.0
            counters = self.own_damage_counters(obs)
            if counters >= 1:
                if self.munkidori_can_ko(obs, counters):
                    return 9400.0
                return 7600.0
            return 4000.0
        if actor == D.FEZANDIPITI:
            return 9000.0 if deck_safe else 0.0
        if actor == D.MEOWTH:
            return 7800.0
        return 5200.0

    def own_damage_counters(self, obs: dict) -> int:
        total = 0
        for p in self.my_field(obs):
            total += max(0, (S.max_hp(p) - S.hp(p)) // 10)
        return total

    def munkidori_can_ko(self, obs: dict, counters: int) -> bool:
        if counters <= 0:
            return False
        for p in self.opp_field(obs):
            if S.hp(p) <= min(3, counters) * 10:
                return True
        return False

    def munkidori_has_dark(self, obs: dict) -> bool:
        for p in self.my_field(obs):
            if S.pokemon_id(p) != D.MUNKIDORI:
                continue
            for cid in S.energy_cards(p):
                if cid == D.PRISM or D.energy_type(cid) == D.DARKNESS_T:
                    return True
            if D.DARKNESS_T in S.energies(p):
                return True
        return False

    # ------------------------------------------------------------------
    # Attach
    # ------------------------------------------------------------------
    def attach_score(self, obs: dict, option: Any) -> float:
        actor, target = S.option_cards(obs, option)
        me = self.me(obs)
        target_area = S.option_int(option, "inPlayArea", None)
        target_index = S.option_int(option, "inPlayIndex", None)
        target_p = S.pokemon_at(obs, me, target_area, target_index)
        target_id = S.pokemon_id(target_p)
        if target_id is None:
            return 2500.0
        energy_id = actor
        etype = D.energy_type(energy_id)
        have = S.energy_count(target_p)
        is_active = target_p is self.my_active(obs)

        if target_id == D.OGERPON_TEAL:
            need = 3
            base = 9500.0 if is_active else 8200.0
            if have >= need:
                base = 2600.0
            elif etype == D.GRASS_T:
                base += 700.0
        elif target_id == D.OGERPON_WELL:
            base = 9000.0 if is_active else 7800.0
            if have >= 3:
                base = 2400.0
            elif etype == D.WATER_T or etype == D.PRISM:
                base += 350.0
        elif target_id == D.KANGASKHAN:
            base = 9500.0 if is_active else 6400.0
            if have >= 3:
                base = 2200.0
            elif etype == D.PRISM:
                base += 450.0
        elif target_id == D.CLEFAIRY:
            base = 9000.0 if is_active else 7200.0
            if have >= 2:
                base = 2200.0
            if self.opponent_has_dragon(obs):
                base += 1700.0
                if not is_active:
                    base += 700.0
        elif target_id == D.LATIAS:
            base = 7000.0 if is_active else 4000.0
            if have >= 3:
                base = 1800.0
        elif target_id == D.MUNKIDORI:
            base = 7200.0 if etype in (D.DARKNESS_T, D.PRISM) else 3000.0
            if have >= 1:
                base -= 900.0
        elif target_id == D.FEZANDIPITI:
            base = 6200.0 if is_active else 4200.0
        elif target_id == D.CHI_YU:
            base = 5800.0 if is_active else 3800.0
        elif target_id == D.PECHARUNT:
            base = 5200.0 if is_active else 3500.0
        elif target_id == D.MEOWTH:
            base = 3600.0
        else:
            base = 3400.0

        if is_active and have == 0 and target_id in D.ATTACKERS:
            base += 900.0
        # Type fit.  Prism Energy is the only way to pay coloured costs on
        # non-Grass attackers (it provides every type while on a Basic).
        types = S.energies(target_p)
        has_rainbow = 10 in types
        if target_id == D.OGERPON_TEAL:
            if etype == D.GRASS_T:
                base += 650.0
            elif etype == D.PRISM:
                base += 150.0
        elif target_id == D.CLEFAIRY:
            if etype == D.PRISM and not has_rainbow and 5 not in types:
                base += 750.0
            elif etype != D.PRISM and not has_rainbow and 5 not in types:
                base -= 700.0
        elif target_id == D.LATIAS:
            psychic = sum(1 for t in types if t in (5, 10))
            if etype == D.PRISM and psychic < 2:
                base += 700.0
            elif etype != D.PRISM and psychic < 2:
                base -= 650.0
        elif target_id == D.MUNKIDORI:
            if etype == D.PRISM:
                base += 700.0
            elif not has_rainbow and 7 not in types:
                base -= 600.0
        elif target_id == D.PECHARUNT:
            if etype == D.PRISM:
                base += 600.0
            elif not has_rainbow and 7 not in types:
                base -= 550.0
        elif target_id == D.CHI_YU:
            if etype == D.PRISM:
                base += 600.0
            elif not has_rainbow and 2 not in types:
                base -= 500.0
        elif target_id == D.OGERPON_WELL:
            if etype == D.PRISM:
                base += 450.0
            elif not has_rainbow and 3 not in types:
                base -= 200.0
        elif target_id == D.KANGASKHAN and etype == D.PRISM:
            base += 250.0
        return base

    # ------------------------------------------------------------------
    # Play
    # ------------------------------------------------------------------
    def play_score(self, obs: dict, option: Any) -> float:
        cid = S.option_int(option, "index", None)
        cid = S.card_at(obs, self.me(obs), S.HAND, cid) if cid is not None else None
        if cid is None:
            return 0.0
        me = self.me(obs)
        hand_n = S.hand_count(obs, me)
        bench_space = self.bench_space(obs)
        deck_safe = self.draw_safe(obs)
        early = S.turn(obs) <= 4

        if cid == D.OGERPON_TEAL:
            score = 9300.0 if bench_space > 0 else 1500.0
            if self.count_in_play(obs, cid) >= 2:
                score -= 900.0
            if self.opponent_has_dragon(obs) and not self.has_clefairy(obs) and self.count_in_play(obs, cid) >= 2:
                score -= 2200.0
            if self.has_grass_in_hand(obs):
                score += 400.0
            return score
        if cid == D.OGERPON_WELL:
            score = 9000.0 if bench_space > 0 else 1500.0
            if self.count_in_play(obs, cid) >= 1:
                score -= 600.0
            return score
        if cid == D.KANGASKHAN:
            score = 9200.0 if bench_space > 0 else 1200.0
            if self.has_kang(obs):
                score -= 1000.0
            if self.my_prizes_left(obs) <= 3:
                score -= 900.0
            return score
        if cid == D.LATIAS:
            score = 8800.0 if bench_space > 0 else 1200.0
            if self.has_latias(obs):
                score -= 1800.0
            return score
        if cid == D.CLEFAIRY:
            score = 8600.0 if bench_space > 0 else 1200.0
            if self.has_clefairy(obs):
                score -= 1600.0
            if self.opponent_has_dragon(obs):
                score = (11600.0 if not self.has_clefairy(obs) else 8200.0) if bench_space > 0 else 1200.0
            return score
        if cid == D.MEOWTH:
            score = 8500.0 if bench_space > 0 else 1000.0
            if self.in_play(obs, cid):
                score -= 1800.0
            if S.supporter_played(obs):
                score -= 500.0
            return score
        if cid == D.FEZANDIPITI:
            score = 8200.0 if bench_space > 0 else 1000.0
            if self.in_play(obs, cid):
                score -= 1500.0
            return score
        if cid == D.MUNKIDORI:
            score = 6900.0 if bench_space > 0 else 900.0
            if self.in_play(obs, cid):
                score -= 1500.0
            if not (self.has_prism_in_hand(obs) or self.own_damage_counters(obs) > 0):
                score -= 1400.0
            return score
        if cid == D.CHI_YU:
            score = 6500.0 if bench_space > 0 else 900.0
            if self.in_play(obs, cid):
                score -= 1800.0
            if S.stadium(obs) and D.AREA_ZERO in S.stadium_ids(obs):
                score -= 500.0
            if not early:
                score -= 700.0
            return score
        if cid == D.PECHARUNT:
            score = 6000.0 if bench_space > 0 else 900.0
            if self.in_play(obs, cid):
                score -= 1900.0
            if not early:
                score -= 600.0
            return score
        if cid == D.AREA_ZERO:
            score = 4000.0
            if self.has_tera_in_play(obs):
                score = 9000.0
            if D.AREA_ZERO in S.stadium_ids(obs):
                score = 800.0
            if not self.has_tera_in_play(obs) and not early:
                score = 2600.0
            return score
        if cid == D.LILLIE:
            if not deck_safe:
                return 0.0
            score = 9000.0 if hand_n <= 6 or self.my_prizes_left(obs) == 6 else 8200.0
            if self.my_prizes_left(obs) == 6:
                score += 500.0
            return score
        if cid == D.BUG_CATCHING:
            if S.deck_count(obs, me) <= 0:
                return 0.0
            need = (self.count_in_play(obs, D.OGERPON_TEAL) < 3
                    or not self.has_grass_in_hand(obs))
            return 8600.0 if need else 7000.0
        if cid == D.ULTRA_BALL:
            if hand_n < 4:
                return 0.0
            need = (bench_space > 0 or not self.in_play(obs, D.KANGASKHAN)
                    or self.count_in_play(obs, D.OGERPON_TEAL) < 2)
            return 9000.0 if need else 6000.0
        if cid == D.TERA_ORB:
            if S.deck_count(obs, me) <= 0:
                return 0.0
            return 8400.0 if self.count_in_play(obs, D.OGERPON_TEAL) < 3 else 5500.0
        if cid == D.ENERGY_SWITCH:
            active = self.my_active(obs)
            active_id = S.pokemon_id(active)
            # The only Basic Energy in this deck is Grass.  Moving it onto a
            # Psychic/Fire/Darkness attacker is usually wasted.
            compatible = active_id in (D.OGERPON_TEAL, D.KANGASKHAN,
                                       D.FEZANDIPITI, D.MEOWTH, D.OGERPON_WELL)
            need = active is not None and S.energy_count(active) < 3
            ready = self.basic_energy_on_bench(obs) > 0
            if not (need and ready and compatible):
                return 0.0
            score = 9800.0
            if self.has_latias(obs):
                score += 200.0
            return score
        if cid == D.N_PLAN:
            active = self.my_active(obs)
            need = active is not None and S.energy_count(active) < 3
            ready = sum(S.energy_count(p) for p in self.my_bench(obs)) > 0
            return 9700.0 if need and ready else 0.0
        if cid == D.BOSS:
            if not self.opp_bench(obs):
                return 0.0
            active = self.my_active(obs)
            can_attack = active is not None and S.energy_count(active) >= 2
            if can_attack and self.gust_best_score(obs) > 0:
                return 8600.0
            return 0.0
        if cid == D.NIGHT_STRETCHER:
            useful = any(D.is_pokemon(c) or D.card_type(c) == D.BASIC_ENERGY
                         for c in S.discard_ids(obs, me))
            return 7000.0 if useful else 2500.0
        if cid == D.CIPHERMANIAC:
            return 6600.0 if deck_safe else 3000.0
        if cid == D.UNFAIR_STAMP:
            if not deck_safe:
                return 0.0
            return 8600.0 if hand_n <= 4 else 7000.0
        # Unknown card: play it if there is nothing better, but keep it low.
        return 4200.0

    # ------------------------------------------------------------------
    # Retreat
    # ------------------------------------------------------------------
    def retreat_score(self, obs: dict, option: Any) -> float:
        active = self.my_active(obs)
        if active is None or not self.my_bench(obs):
            return -1000.0
        active_id = S.pokemon_id(active)
        active_pot = self.attack_potential(obs, active)
        best = self.best_benched_attacker(obs)
        if best is None:
            return -1000.0
        best_p, best_id, best_pot, best_energy = best
        if best_pot <= 0:
            return -1000.0

        active_support = active_id in D.SUPPORT_POKEMON
        active_weak = S.hp(active) <= 70 and self.opp_can_ko_active(obs)
        active_prize_risk = (D.prize_value(active_id) >= 3 and self.opp_prizes_left(obs) <= 3)

        # A 3-prize Mega ex must not sit in front when the opponent is close to
        # winning; promote a lower-prize attacker that can actually act.
        if (self.opponent_has_dragon(obs) and best_id == D.CLEFAIRY
                and best_energy >= 2):
            return 8200.0
        if active_prize_risk and D.prize_value(best_id) < D.prize_value(active_id):
            return 7800.0
        if active_pot <= 0 and best_pot > 0:
            return 7200.0
        if active_support and best_pot >= 80:
            return 6900.0
        if best_pot > active_pot + 80 and best_energy >= 2:
            return 6400.0
        if active_weak and best_pot > 0 and D.prize_value(best_id) <= D.prize_value(active_id):
            return 6200.0
        return -1000.0

    def best_benched_attacker(self, obs: dict) -> tuple[Any, int, int, int] | None:
        best = None
        for p in self.my_bench(obs):
            cid = S.pokemon_id(p)
            if cid is None:
                continue
            pot = self.attack_potential(obs, p)
            energy = S.energy_count(p)
            key = (pot, energy, D.ACTIVE_ORDER.get(cid, 0))
            if best is None or key > best[0]:
                best = (key, p, cid, pot, energy)
        if best is None:
            return None
        return best[1], best[2], best[3], best[4]

    def opp_can_ko_active(self, obs: dict) -> bool:
        # Conservative public estimate: highest printed attack among visible
        # opposing Pokémon, with weakness taken into account.
        active = self.my_active(obs)
        if active is None:
            return False
        active_id = S.pokemon_id(active)
        hp = S.hp(active)
        for p in self.opp_field(obs):
            cid = S.pokemon_id(p)
            if cid is None:
                continue
            for aid in D.attacks_of(cid):
                a = D.attack(aid)
                if a is None:
                    continue
                dmg = int(getattr(a, "damage", 0) or 0)
                if dmg <= 0:
                    continue
                at = D.energy_type(cid)
                w = D.weakness(active_id)
                if w is not None and at == w:
                    dmg *= 2
                if dmg >= hp:
                    return True
        return False

    # ------------------------------------------------------------------
    # Non-MAIN selections
    # ------------------------------------------------------------------
    def choose_other(self, obs: dict, options: list, mn: int, mx: int) -> list[int]:
        stype = S.select_type(obs)
        ctx = S.select_context(obs)
        if stype == S.S_YES_NO:
            return [self.choose_yes_no(obs, options, ctx, mx)]
        if stype == S.S_COUNT:
            return [self.choose_count(obs, options, ctx)]
        if stype == S.S_ATTACK:
            return [self.choose_attack(obs, options)]
        if stype == S.S_SPECIAL_CONDITION:
            return [self.choose_condition(obs, options, ctx)]
        if stype == S.S_ENERGY:
            return self.choose_energy(obs, options, ctx, mn, mx)
        if stype == S.S_ATTACHED_CARD:
            return self.choose_attached(obs, options, ctx, mn, mx)
        if stype == S.S_SKILL:
            return [self.choose_skill(obs, options)]
        if stype == S.S_EVOLVE:
            return self.choose_evolve(obs, options, mn, mx)
        # CARD / CARD_OR_ATTACHED / generic
        return self.choose_cards(obs, options, ctx, mn, mx)

    def choose_yes_no(self, obs: dict, options: list, ctx: int, mx: int) -> int:
        yes = self.index_of_type(options, S.O_YES)
        no = self.index_of_type(options, S.O_NO)
        if ctx == S.C_IS_FIRST:
            return yes if yes is not None else max(0, mx - 1)
        if ctx == S.C_MULLIGAN:
            # Redraw only when the opening hand has no Basic Pokémon.
            has_basic = any(D.is_basic(c) for c in self.my_hand_ids(obs))
            if has_basic:
                return no if no is not None else 0
            return yes if yes is not None else 0
        if ctx == S.C_COIN_HEAD:
            return yes if yes is not None else 0
        if ctx == S.C_MORE_DEVOLVE:
            return no if no is not None else 0
        if ctx in (S.C_ACTIVATE, S.C_FIRST_EFFECT):
            # Optional effects in this deck (Meowth, Teal Dance prompts,
            # Torrential Pump) are beneficial unless they would deck us out.
            effect_id = None
            eff = S.get(S.select(obs), "effect", None)
            if eff is not None:
                effect_id = S.card_id(eff)
            if effect_id in D.TERA_IDS and not self.draw_safe(obs):
                return no if no is not None else 0
            return yes if yes is not None else 0
        return yes if yes is not None else 0

    def choose_count(self, obs: dict, options: list, ctx: int) -> int:
        numbers = [S.option_int(o, "number", 0) or 0 for o in options]
        if not numbers:
            return 0
        if ctx in (S.C_DRAW_COUNT, S.C_DAMAGE_COUNTER_COUNT, S.C_REMOVE_DAMAGE_COUNTER_COUNT):
            if ctx == S.C_DRAW_COUNT and not self.draw_safe(obs):
                return min(range(len(numbers)), key=lambda i: (numbers[i], i))
            return max(range(len(numbers)), key=lambda i: (numbers[i], -i))
        return min(range(len(numbers)), key=lambda i: (numbers[i], i))

    def choose_attack(self, obs: dict, options: list) -> int:
        scores = [(self.attack_score(obs, o), i) for i, o in enumerate(options)]
        return max(scores, key=lambda x: (x[0], -x[1]))[1]

    def choose_condition(self, obs: dict, options: list, ctx: int) -> int:
        # Poison is the most valuable condition in this deck (Pecharunt).
        priority = {0: 5, 3: 4, 2: 3, 4: 2, 1: 1}
        if ctx == S.C_RECOVER_SPECIAL_CONDITION:
            priority = {1: 5, 0: 4, 3: 3, 2: 2, 4: 1}
        best = 0
        best_key = -1
        for i, o in enumerate(options):
            v = S.option_int(o, "specialConditionType", None)
            key = priority.get(int(v) if v is not None else -1, 0)
            if key > best_key:
                best_key = key
                best = i
        return best

    def choose_energy(self, obs: dict, options: list, ctx: int, mn: int, mx: int) -> list[int]:
        if ctx in (S.C_DISCARD_ENERGY, S.C_TO_DECK_ENERGY):
            scored = [(self.energy_discard_score(obs, o), i) for i, o in enumerate(options)]
            scored.sort()
            return [i for _s, i in scored[:max(mn, min(mx, 1))]]
        if ctx in (S.C_SWITCH_ENERGY, S.C_TO_HAND_ENERGY):
            scored = [(self.energy_source_score(obs, o), i) for i, o in enumerate(options)]
            scored.sort(reverse=True)
            count = max(1, min(mx, mn if mn > 0 else 1))
            return [i for _s, i in scored[:count]]
        scored = [(self.energy_discard_score(obs, o), i) for i, o in enumerate(options)]
        scored.sort()
        return [i for _s, i in scored[:max(mn, min(mx, 1))]]

    def choose_attached(self, obs: dict, options: list, ctx: int, mn: int, mx: int) -> list[int]:
        if ctx in (S.C_DISCARD_ENERGY_CARD, S.C_DISCARD_TOOL_CARD,
                   S.C_DISCARD_CARD_OR_ATTACHED):
            scored = [(self.energy_source_score(obs, o), i) for i, o in enumerate(options)]
            scored.sort()
            count = max(mn, min(mx, 1))
            return [i for _s, i in scored[:count]]
        # Source selection for Energy Switch / N's Plan: best bench energy first.
        scored = [(self.energy_source_score(obs, o), i) for i, o in enumerate(options)]
        scored.sort(reverse=True)
        count = max(mn, min(mx, 1))
        return [i for _s, i in scored[:count]]

    def choose_skill(self, obs: dict, options: list) -> int:
        # Pick the card with the highest static role value first.
        best = max(range(len(options)), key=lambda i: (
            D.role_value(S.option_cards(obs, options[i])[0]), -i))
        return best

    def choose_evolve(self, obs: dict, options: list, mn: int, mx: int) -> list[int]:
        scored = []
        for i, o in enumerate(options):
            actor, target = S.option_cards(obs, o)
            scored.append((D.role_value(actor) + (10 if target else 0), i))
        scored.sort(reverse=True)
        count = max(mn, min(mx, len(scored)))
        return [i for _s, i in scored[:count]]

    def choose_cards(self, obs: dict, options: list, ctx: int, mn: int, mx: int) -> list[int]:
        if ctx == S.C_SETUP_ACTIVE:
            scored = [(self.setup_active_score(obs, o), i) for i, o in enumerate(options)]
            scored.sort(reverse=True)
            return [i for _s, i in scored[:max(mn, min(mx, 1))]]
        if ctx == S.C_SETUP_BENCH:
            scored = [(self.setup_bench_score(obs, o), i) for i, o in enumerate(options)]
            scored.sort(reverse=True)
            count = max(mn, min(mx, len(scored)))
            return [i for _s, i in scored[:count]]
        if ctx == S.C_SWITCH:
            scored = [(self.switch_option_score(obs, o), i) for i, o in enumerate(options)]
            scored.sort(reverse=True)
            return [i for _s, i in scored[:max(mn, min(mx, 1))]]
        if ctx == S.C_ATTACH_FROM:
            scored = [(self.attach_target_option_score(obs, o), i) for i, o in enumerate(options)]
            scored.sort(reverse=True)
            return [i for _s, i in scored[:max(mn, min(mx, 1))]]
        if ctx == S.C_ATTACH_TO:
            scored = [(self.energy_pick_score(obs, o), i) for i, o in enumerate(options)]
            scored.sort(reverse=True)
            return [i for _s, i in scored[:max(mn, min(mx, 1))]]
        if ctx in (S.C_DAMAGE_COUNTER, S.C_DAMAGE_COUNTER_ANY, S.C_DAMAGE,
                   S.C_EFFECT_TARGET):
            scored = [(self.damage_target_score(obs, o, ctx), i) for i, o in enumerate(options)]
            scored.sort(reverse=True)
            return [i for _s, i in scored[:max(mn, min(mx, 1))]]
        if ctx in (S.C_REMOVE_DAMAGE_COUNTER, S.C_HEAL):
            scored = [(self.heal_target_score(obs, o), i) for i, o in enumerate(options)]
            scored.sort(reverse=True)
            return [i for _s, i in scored[:max(mn, min(mx, 1))]]
        if ctx in (S.C_TO_ACTIVE, S.C_TO_BENCH, S.C_TO_FIELD):
            scored = [(self.board_pick_score(obs, o, ctx), i) for i, o in enumerate(options)]
            scored.sort(reverse=True)
            count = max(mn, min(mx, 1))
            return [i for _s, i in scored[:count]]
        if ctx in (S.C_TO_HAND, S.C_LOOK):
            scored = [(self.search_pick_score(obs, o, ctx), i) for i, o in enumerate(options)]
            scored.sort(reverse=True)
            count = max(mn, min(mx, len(scored)))
            if mn == 0:
                # Optional search: return empty if nothing is worth taking.
                if not scored or scored[0][0] < 35:
                    return []
                count = 1
            return [i for _s, i in scored[:count]]
        if ctx in (S.C_TO_DECK, S.C_TO_DECK_BOTTOM):
            areas = {S.option_int(o, "area", None) for o in options}
            if areas & {S.DECK, S.LOOKING}:
                # Ciphermaniac-style effect: choose the best cards to put on
                # top of the deck, not the cheapest ones.
                scored = [(self.search_pick_score(obs, o, ctx), i) for i, o in enumerate(options)]
                scored.sort(reverse=True)
                count = max(mn, min(mx, len(scored)))
                return [i for _s, i in scored[:count]]
            scored = [(self.discard_option_score(obs, o), i) for i, o in enumerate(options)]
            scored.sort()
            count = max(mn, min(mx, len(scored)))
            return [i for _s, i in scored[:count]]
        if ctx in (S.C_DISCARD, S.C_TO_PRIZE, S.C_NOT_MOVE):
            scored = [(self.discard_option_score(obs, o), i) for i, o in enumerate(options)]
            scored.sort()
            count = max(mn, min(mx, len(scored)))
            return [i for _s, i in scored[:count]]
        # Generic: take the highest-value cards.
        scored = [(self.search_pick_score(obs, o, ctx), i) for i, o in enumerate(options)]
        scored.sort(reverse=True)
        count = max(mn, min(mx, 1))
        return [i for _s, i in scored[:count]]

    # ------------------------------------------------------------------
    # Selection scores
    # ------------------------------------------------------------------
    def setup_active_score(self, obs: dict, option: Any) -> float:
        cid = S.option_cards(obs, option)[0]
        score = float(D.ACTIVE_ORDER.get(cid, 30))
        if cid == D.KANGASKHAN:
            score += 12 if self.opp_prizes_left(obs) > 3 else -8
        elif cid == D.OGERPON_TEAL:
            score += 10 if self.has_grass_in_hand(obs) else 3
        elif cid == D.CLEFAIRY:
            if self.opponent_has_dragon(obs):
                score += 18
        elif cid == D.CHI_YU:
            if S.turn(obs) <= 2:
                score += 8
        elif cid == D.LATIAS:
            score += 6 if self.my_bench(obs) else 0
        return score

    def setup_bench_score(self, obs: dict, option: Any) -> float:
        cid = S.option_cards(obs, option)[0]
        score = float(D.BENCH_ORDER.get(cid, 25))
        if cid == D.OGERPON_TEAL:
            score += 8 if self.count_in_play(obs, cid) < 2 else 0
        elif cid == D.LATIAS:
            score += 10 if not self.has_latias(obs) else -5
        elif cid == D.CLEFAIRY:
            score += 45 if self.opponent_has_dragon(obs) and not self.has_clefairy(obs) else 0
        elif cid == D.MEOWTH:
            score += 8 if not self.in_play(obs, cid) else -6
        elif cid == D.KANGASKHAN:
            score += 6 if not self.has_kang(obs) else -4
        elif cid == D.MUNKIDORI:
            score += 6 if self.has_prism_in_hand(obs) else 0
        return score

    def switch_option_score(self, obs: dict, option: Any) -> float:
        # Context SWITCH is used both by Boss's Orders (opponent bench) and by
        # our own retreat (our bench).  The playerIndex field disambiguates.
        owner = S.option_owner(obs, option)
        if owner == self.me(obs):
            return self.board_pick_score(obs, option, S.C_TO_ACTIVE)
        return self.gust_option_score(obs, option)

    def gust_option_score(self, obs: dict, option: Any) -> float:
        owner = 1 - self.me(obs)
        cid = S.card_at(obs, owner, S.option_int(option, "area", None),
                        S.option_int(option, "index", None))
        if cid is None:
            return 0.0
        hp = S.hp(S.pokemon_at(obs, owner, S.option_int(option, "area", None),
                               S.option_int(option, "index", None)))
        prize = D.prize_value(cid)
        score = prize * 120.0 + max(0, 250 - hp) * 0.8
        if cid in (121, 120, 119, 743, 742, 741, 756, 747, 163, 1406, 150):
            score += 180.0
        # Can our active attack KO it?
        active = self.my_active(obs)
        if active is not None:
            for aid in D.attacks_of(S.pokemon_id(active)):
                target = S.pokemon_at(obs, owner, S.option_int(option, "area", None),
                                      S.option_int(option, "index", None))
                dmg = self.attack_damage(obs, aid, active, target)
                if dmg >= hp and dmg > 0:
                    score += 1000.0
                    break
        return score

    def gust_best_score(self, obs: dict) -> float:
        best = 0.0
        owner = 1 - self.me(obs)
        for i, p in enumerate(self.opp_bench(obs)):
            cid = S.pokemon_id(p)
            if cid is None:
                continue
            hp = S.hp(p)
            score = D.prize_value(cid) * 120.0 + max(0, 250 - hp) * 0.8
            if cid in (121, 120, 119, 743, 742, 741, 756, 747, 163, 1406, 150):
                score += 180.0
            best = max(best, score)
        return best

    def attach_target_option_score(self, obs: dict, option: Any) -> float:
        owner = self.me(obs)
        area = S.option_int(option, "area", None)
        index = S.option_int(option, "index", None)
        p = S.pokemon_at(obs, owner, area, index)
        cid = S.pokemon_id(p)
        if cid is None:
            return 0.0
        is_active = p is self.my_active(obs)
        score = float(D.ACTIVE_ORDER.get(cid, 20))
        if is_active:
            score += 25
        if cid in (D.OGERPON_TEAL, D.KANGASKHAN, D.CLEFAIRY, D.OGERPON_WELL):
            score += 10
        # Energy Switch can only move the Basic Grass cards in this deck.
        if cid in (D.CLEFAIRY, D.LATIAS, D.MUNKIDORI, D.PECHARUNT, D.CHI_YU):
            score -= 35
        score += S.energy_count(p) * 2
        return score

    def energy_pick_score(self, obs: dict, option: Any) -> float:
        cid = S.option_cards(obs, option)[0]
        if cid is None:
            # The option may be an energy unit without a card id.
            return 10.0
        target_id = self.last_attach_target(obs)
        score = 20.0
        if cid == D.GRASS:
            score += 40.0
            if target_id in D.TERA_IDS:
                score += 30.0
        elif cid == D.PRISM:
            score += 35.0
            if target_id in (D.MUNKIDORI, D.KANGASKHAN, D.LATIAS, D.CLEFAIRY):
                score += 15.0
        else:
            score += D.role_value(cid) * 0.1
        return score

    def last_attach_target(self, obs: dict) -> int | None:
        # The engine resolves Teal Dance by asking for the grass card after the
        # ability; the target is the Ogerpon that offered the ability.  We can
        # recover the most recent ability target from the observation logs.
        for entry in reversed(list(S.get(obs, "logs", []) or [])):
            typ = S.as_int(S.get(entry, "type", -1), -1)
            if typ == 12:  # LogType.EVOLVE not relevant; kept for safety
                continue
        # Fallback: choose the Ogerpon with the fewest energy.
        best = None
        for p in self.my_field(obs):
            cid = S.pokemon_id(p)
            if cid in D.TERA_IDS:
                key = (S.energy_count(p), cid == D.OGERPON_TEAL)
                if best is None or key < best[0]:
                    best = (key, cid)
        return best[1] if best else None

    def damage_target_score(self, obs: dict, option: Any, ctx: int) -> float:
        owner = S.option_owner(obs, option)
        area = S.option_int(option, "area", None)
        index = S.option_int(option, "index", None)
        p = S.pokemon_at(obs, owner, area, index)
        cid = S.pokemon_id(p)
        if cid is None:
            return 0.0
        if owner == self.me(obs):
            return self.heal_target_score(obs, option)
        hp = S.hp(p)
        benched = area == S.BENCH
        if benched and D.is_tera(cid):
            return -100.0
        # Use remaining counters if the engine exposes them.
        counters = S.as_int(S.get(S.select(obs), "remainDamageCounter", 0), 0) or 0
        damage = max(10, counters * 10) if ctx in (S.C_DAMAGE_COUNTER, S.C_DAMAGE_COUNTER_ANY) else 100
        score = D.prize_value(cid) * 140.0 + max(0, 300 - hp) * 0.5
        if hp <= damage:
            score += 1200.0
        if cid in (121, 120, 119, 743, 742, 741, 756, 747, 163):
            score += 220.0
        if benched:
            score += 60.0
        return score

    def heal_target_score(self, obs: dict, option: Any) -> float:
        owner = S.option_owner(obs, option)
        area = S.option_int(option, "area", None)
        index = S.option_int(option, "index", None)
        p = S.pokemon_at(obs, owner, area, index)
        cid = S.pokemon_id(p)
        if cid is None:
            return 0.0
        counters = max(0, (S.max_hp(p) - S.hp(p)) // 10)
        score = counters * 10.0 + D.prize_value(cid) * 40.0
        if p is self.my_active(obs):
            score += 60.0
        return score

    def board_pick_score(self, obs: dict, option: Any, ctx: int) -> float:
        cid = S.option_cards(obs, option)[0]
        if cid is None:
            return 0.0
        if ctx == S.C_TO_ACTIVE:
            score = float(D.ACTIVE_ORDER.get(cid, 20)) + (20 if cid in D.ATTACKERS else 0)
            p = S.pokemon_at(obs, self.me(obs), S.option_int(option, "area", None),
                             S.option_int(option, "index", None))
            if p is not None:
                score += self.attack_potential(obs, p) * 0.6
                score += S.energy_count(p) * 18.0
                score += S.hp(p) * 0.08
            return score
        if ctx == S.C_TO_BENCH:
            return float(D.BENCH_ORDER.get(cid, 20))
        return float(D.BENCH_ORDER.get(cid, 20))

    def search_pick_score(self, obs: dict, option: Any, ctx: int) -> float:
        owner = S.option_owner(obs, option)
        area = S.option_int(option, "area", None)
        index = S.option_int(option, "index", None)
        cid = S.card_at(obs, owner, area, index)
        if cid is None:
            return 0.0
        return self.card_search_value(obs, cid, from_discard=area == S.DISCARD)

    def card_search_value(self, obs: dict, cid: int, from_discard: bool = False) -> float:
        score = float(D.role_value(cid))
        in_play = self.count_in_play(obs, cid)
        hand = Counter(self.my_hand_ids(obs))
        hand_n = hand.get(cid, 0)
        bench_space = self.bench_space(obs)

        if D.is_pokemon(cid):
            if in_play == 0:
                score += 35.0
            elif in_play >= 2:
                score -= 25.0
            if cid == D.KANGASKHAN:
                score += 30.0 if not self.has_kang(obs) else -15.0
            elif cid == D.OGERPON_TEAL:
                score += 28.0 if in_play < 2 else -8.0
            elif cid == D.LATIAS:
                score += 25.0 if not self.has_latias(obs) else -20.0
            elif cid == D.CLEFAIRY:
                score += 80.0 if (self.opponent_has_dragon(obs) and not self.has_clefairy(obs)) else 8.0
            elif cid == D.MEOWTH:
                score += 20.0 if not self.in_play(obs, cid) else -10.0
            elif cid == D.MUNKIDORI:
                score += 15.0 if self.has_prism_in_hand(obs) or self.own_damage_counters(obs) else -5.0
            elif cid == D.PECHARUNT:
                score -= 12.0
            if bench_space <= 0 and not self.has_latias(obs):
                score -= 40.0
        else:
            if cid == D.GRASS:
                score += 25.0 if self.count_in_play(obs, D.OGERPON_TEAL) else -5.0
                if hand_n >= 2:
                    score -= 10.0
            elif cid == D.PRISM:
                score += 18.0 if self.has_prism_in_hand(obs) is False else 0.0
            elif cid == D.LILLIE:
                score += 20.0 if S.hand_count(obs, self.me(obs)) <= 4 else 5.0
            elif cid == D.BOSS:
                score += 15.0 if self.opp_bench(obs) else 0.0
            elif cid == D.AREA_ZERO:
                score += 20.0 if self.has_tera_in_play(obs) and D.AREA_ZERO not in S.stadium_ids(obs) else 0.0
            elif cid == D.ULTRA_BALL:
                score += 15.0 if bench_space > 0 else 0.0
            elif cid in (D.ENERGY_SWITCH, D.N_PLAN):
                active = self.my_active(obs)
                score += 10.0 if active is not None and S.energy_count(active) < 3 else 0.0
        if from_discard:
            score -= 5.0
        if hand_n >= 1:
            score -= 8.0 * min(2, hand_n)
        return score

    def discard_option_score(self, obs: dict, option: Any) -> float:
        cid = S.option_cards(obs, option)[0]
        if cid is None:
            return 0.0
        return self.card_keep_value(obs, cid)

    def card_keep_value(self, obs: dict, cid: int) -> float:
        value = float(D.role_value(cid))
        in_play = self.count_in_play(obs, cid)
        hand = Counter(self.my_hand_ids(obs))
        hand_n = hand.get(cid, 0)
        if D.is_pokemon(cid):
            if in_play == 0:
                value += 30.0
            elif in_play >= 2:
                value -= 15.0
            if cid == D.KANGASKHAN and not self.has_kang(obs):
                value += 25.0
            if cid == D.OGERPON_TEAL and in_play < 2:
                value += 20.0
            if cid == D.LATIAS and not self.has_latias(obs):
                value += 18.0
            if cid == D.CLEFAIRY and self.opponent_has_dragon(obs) and not self.has_clefairy(obs):
                value += 25.0
        else:
            if cid in (D.GRASS, D.PRISM):
                active = self.my_active(obs)
                if active is not None and S.energy_count(active) == 0 and hand_n <= 1:
                    value += 30.0
            if cid == D.GRASS:
                value += 5.0 if self.count_in_play(obs, D.OGERPON_TEAL) else -5.0
                if hand_n >= 2:
                    value -= 10.0
            elif cid == D.PRISM:
                if hand_n >= 2:
                    value -= 10.0
            elif cid == D.UNFAIR_STAMP:
                value += 15.0
        if hand_n >= 2:
            # Duplicates are easier to give up, but key cards remain valuable.
            value -= 7.0 * (hand_n - 1)
        return value

    def energy_discard_score(self, obs: dict, option: Any) -> float:
        cid, target = S.option_cards(obs, option)
        if cid is None:
            cid = S.option_int(option, "cardId", None)
        if cid == D.PRISM:
            return 40.0
        if cid == D.GRASS:
            return 25.0
        # Attached energy: moving it away from a support Pokémon is cheapest.
        return 30.0

    def energy_source_score(self, obs: dict, option: Any) -> float:
        cid, target = S.option_cards(obs, option)
        area = S.option_int(option, "area", None)
        index = S.option_int(option, "index", None)
        p = S.pokemon_at(obs, self.me(obs), area, index)
        target_id = S.pokemon_id(p)
        score = 20.0
        if area == S.BENCH:
            score += 25.0
        if target_id in D.TERA_IDS:
            score += 10.0 if S.energy_count(p) >= 3 else -5.0
        if target_id in (D.KANGASKHAN, D.CLEFAIRY, D.OGERPON_WELL):
            score -= 8.0
        if cid == D.PRISM:
            score += 8.0
        return score

    def index_of_type(self, options: list, otype: int) -> int | None:
        for i, o in enumerate(options):
            if S.option_type(o) == otype:
                return i
        return None
