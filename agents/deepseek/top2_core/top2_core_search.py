"""Layer 2: determinised 1-ply search for the Top2 Dragapult agent.

The engine exposes a search API that simulates one branch from the *agent's*
observation.  Hidden information (both decks, prizes, opponent hand) must be
sampled by us.  We use only public information plus the known Top2 deck and a
small library of public archetype lists.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any

import top2_core_data as D
import top2_core_state as S

try:
    from cg.api import search_begin, search_end, search_step, to_observation_class
except Exception:  # pragma: no cover - engine unavailable in some unit tests
    search_begin = search_end = search_step = to_observation_class = None


WIN_SCORE = 100000.0
LOSS_SCORE = -100000.0


class SearchEngine:
    def __init__(self, deck: list[int], rules=None, value_fn=None) -> None:
        self.deck = list(deck)
        self.rules = rules
        self.value_fn = value_fn
        self.value_weight = float(os.environ.get("TOP2_VALUE_WEIGHT", "45"))
        self.rng = random.Random()
        self.per_move = float(os.environ.get("TOP2_SEARCH_SECONDS", "0.20"))
        self.total_budget = float(os.environ.get("TOP2_SEARCH_TOTAL", "240"))
        self.trials = max(1, int(os.environ.get("TOP2_SEARCH_TRIALS", "3")))
        self.candidates = max(2, int(os.environ.get("TOP2_SEARCH_CANDIDATES", "4")))
        self.advantage = float(os.environ.get("TOP2_SEARCH_ADVANTAGE", "6.0"))
        self.spent = 0.0
        self.enabled = search_begin is not None and self.total_budget > 0
        self.stats = {"calls": 0, "overrides": 0, "errors": 0, "trials": 0}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def choose(self, obs: dict, scored: list[tuple[float, int]]) -> int | None:
        if not self.enabled or self.spent >= self.total_budget:
            return None
        sel = S.select(obs)
        if sel is None or S.select_type(obs) != 0:
            return None
        if S.get(sel, "context", 0) not in (0, None):
            return None
        if int(S.get(sel, "maxCount", 1) or 1) != 1:
            return None
        options = list(S.get(sel, "option", []) or [])
        if len(options) < 2:
            return None
        if not scored:
            return None

        rule_best_score, rule_best = max(scored, key=lambda item: (item[0], -item[1]))
        if rule_best_score >= 20000.0:  # lethal is a hard rule
            return None

        # Keep the rule-best candidate in the race, then add the strongest
        # alternatives.  END/RETREAT are only searched when the rules already
        # consider them viable.
        ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
        picks: list[int] = []
        for score, idx in ranked:
            if idx == rule_best or score > 1000.0:
                if idx not in picks:
                    picks.append(int(idx))
            if len(picks) >= self.candidates:
                break
        if len(picks) < 2:
            return None

        start = time.perf_counter()
        me = S.current_player(obs)
        totals: dict[int, list[float]] = {idx: [] for idx in picks}
        for trial in range(self.trials):
            if time.perf_counter() - start > self.per_move:
                break
            hidden = self._sample_hidden(obs, me)
            if hidden is None:
                continue
            for idx in picks:
                if time.perf_counter() - start > self.per_move:
                    break
                value = self._rollout(obs, idx, hidden, me)
                if value is not None:
                    totals[idx].append(value)
                    self.stats["trials"] += 1
        elapsed = time.perf_counter() - start
        self.spent += elapsed
        self.stats["calls"] += 1

        averages = {idx: sum(vals) / len(vals) for idx, vals in totals.items() if vals}
        if not averages:
            return None
        search_best = max(averages, key=lambda idx: (averages[idx], -idx))
        if search_best == rule_best:
            return None
        if averages.get(rule_best, LOSS_SCORE * 2) == LOSS_SCORE * 2:
            # The rule choice failed in simulation; do not let a broken
            # determinisation cause an override.
            return None
        if averages[search_best] - averages.get(rule_best, LOSS_SCORE) < self.advantage:
            return None
        self.stats["overrides"] += 1
        if os.environ.get("TOP2_SEARCH_DEBUG"):
            print(
                f"SEARCH override rule={rule_best}({rule_best_score:.1f}) "
                f"-> {search_best} avgs={ {k: round(v, 1) for k, v in averages.items()} }",
                flush=True,
            )
        return int(search_best)

    # ------------------------------------------------------------------
    # Determinisation
    # ------------------------------------------------------------------
    def _sample_hidden(self, obs: dict, me: int):
        try:
            my_deck, my_prize = self._sample_own(obs, me)
            opp = self._sample_opponent(obs, me)
            if opp is None:
                return None
            return (my_deck, my_prize, *opp)
        except Exception:
            self.stats["errors"] += 1
            return None

    def _sample_own(self, obs: dict, me: int) -> tuple[list[int], list[int]]:
        known = S.known_cards(obs, me)
        pool = self._remove_cards(self.deck, known)
        deck_count = S.deck_count(obs, me)
        prize_count = S.prize_count(obs, me)
        need = deck_count + prize_count
        if len(pool) < need:
            pool.extend(self.deck)
        self.rng.shuffle(pool)
        prizes = list(pool[:prize_count])
        deck = list(pool[prize_count:prize_count + deck_count])
        return deck, prizes

    def _sample_opponent(self, obs: dict, me: int):
        opp = 1 - me
        seen = self._seen_opponent_cards(obs, opp)
        deck_list = self._choose_meta_deck(seen)
        pool = self._remove_cards(deck_list, seen)
        deck_count = S.deck_count(obs, opp)
        prize_count = S.prize_count(obs, opp)
        hand_count = S.hand_count(obs, opp)
        need = deck_count + prize_count + hand_count
        if len(pool) < need:
            pool.extend(deck_list)
        self.rng.shuffle(pool)
        prizes = list(pool[:prize_count])
        hand = list(pool[prize_count:prize_count + hand_count])
        deck = list(pool[prize_count + hand_count:prize_count + hand_count + deck_count])
        active = []
        a = S.active(obs, opp)
        if a is None:
            active = [self._likely_basic(deck_list)]
        return deck, prizes, hand, active

    def _seen_opponent_cards(self, obs: dict, opp: int) -> list[int]:
        out: list[int] = []
        for p in S.all_field(obs, opp):
            cid = S.card_id(p)
            if cid is not None:
                out.append(cid)
            out.extend(S.pre_evolution(p))
            out.extend(S.energy_cards(p))
            out.extend(S.tools(p))
        out.extend(S.discard_ids(obs, opp))
        for c in S.stadium(obs):
            if int(S.get(c, "playerIndex", -1) or -1) == opp:
                cid = S.card_id(c)
                if cid is not None:
                    out.append(cid)
        return out

    def _choose_meta_deck(self, seen: list[int]) -> list[int]:
        if not seen:
            return list(D.META_DECKS.get("dragapult", D.TOP2_DECK))
        seen_set = set(seen)
        best_name = None
        best_score = -1.0
        for name, deck in D.META_DECKS.items():
            deck_set = set(deck)
            overlap = len(seen_set & deck_set)
            # Weight the archetype-defining Pokémon more than generic staples.
            key = {D.DREEPY, D.DRAKLOAK, D.DRAGAPULT}
            key_hits = len(seen_set & deck_set & key)
            score = overlap + key_hits * 2.0
            if score > best_score:
                best_score = score
                best_name = name
        return list(D.META_DECKS.get(best_name or "dragapult", D.TOP2_DECK))

    @staticmethod
    def _remove_cards(pool: list[int], remove: list[int]) -> list[int]:
        out = list(pool)
        for cid in remove:
            try:
                out.remove(int(cid))
            except ValueError:
                pass
        return out

    @staticmethod
    def _likely_basic(deck: list[int]) -> int:
        for cid in deck:
            if D.is_basic(cid):
                return int(cid)
        return D.DREEPY

    # ------------------------------------------------------------------
    # Simulation + leaf evaluation
    # ------------------------------------------------------------------
    def _rollout(self, obs: dict, candidate: int, hidden, me: int) -> float | None:
        try:
            typed = to_observation_class(obs)
            state = search_begin(typed, *hidden)
            search_id = state.searchId
        except Exception:
            self.stats["errors"] += 1
            return None
        try:
            try:
                leaf = search_step(search_id, [int(candidate)])
            except Exception:
                self.stats["errors"] += 1
                return None
            # If the action opens another decision for us (search/discard/
            # target selection), follow the rule policy for a short rollout.
            for _depth in range(2):
                cur = S.current(leaf.observation)
                if cur is None or S.result(leaf.observation) >= 0:
                    break
                if int(S.get(cur, "yourIndex", -1) or -1) != me:
                    break
                if self.rules is None:
                    break
                saved_searcher = getattr(self.rules, "searcher", None)
                try:
                    # Do not recursively launch a new search inside a search
                    # rollout; the rule policy is enough for follow-up picks.
                    self.rules.searcher = None
                    action = self.rules.choose(leaf.observation)
                    leaf = search_step(search_id, action)
                except Exception:
                    break
                finally:
                    self.rules.searcher = saved_searcher
            # The leaf is usually the opponent's decision; view the value
            # features from *our* perspective.
            try:
                cur = S.current(leaf.observation)
                if cur is not None:
                    cur.yourIndex = me
            except Exception:
                pass
            return self.evaluate(leaf.observation, me)
        finally:
            try:
                search_end()
            except Exception:
                pass

    def evaluate(self, obs: Any, me: int) -> float:
        result = S.result(obs)
        if result >= 0:
            return WIN_SCORE if result == me else LOSS_SCORE

        my = S.player_state(obs, me)
        op = S.player_state(obs, 1 - me)
        if my is None or op is None:
            return 0.0

        score = 0.0
        my_prizes = S.prize_count(obs, me)
        op_prizes = S.prize_count(obs, 1 - me)
        score += (op_prizes - my_prizes) * 130.0

        my_active = S.active(obs, me)
        op_active = S.active(obs, 1 - me)
        score += S.hp(my_active) / 8.0
        score -= S.hp(op_active) / 8.0
        score += len(S.energies(my_active)) * 12.0
        score -= len(S.energies(op_active)) * 8.0

        score += len(S.bench(obs, me)) * 8.0
        score -= len(S.bench(obs, 1 - me)) * 5.0
        score += S.hand_count(obs, me) * 1.5
        score -= S.hand_count(obs, 1 - me) * 0.5

        # Board quality: a developed Dragapult line is worth more than paper HP.
        for p in S.all_field(obs, me):
            cid = S.card_id(p)
            if cid == D.DRAGAPULT:
                score += 45.0
            elif cid == D.DRAKLOAK:
                score += 22.0
            elif cid == D.DREEPY:
                score += 9.0
            elif cid == D.MUNKIDORI:
                score += 8.0
            score += (S.max_hp(p) - S.hp(p)) * -0.8
        for p in S.all_field(obs, 1 - me):
            cid = S.card_id(p)
            if cid == D.DRAGAPULT:
                score -= 35.0
            elif cid in (D.DRAKLOAK, D.DREEPY):
                score -= 8.0
            score += (S.max_hp(p) - S.hp(p)) * 1.2

        my_deck = S.deck_count(obs, me)
        op_deck = S.deck_count(obs, 1 - me)
        if my_deck <= 8:
            score -= (9 - my_deck) * 18.0
        if op_deck <= 8:
            score += (9 - op_deck) * 12.0

        # Can the opponent KO our active next turn?  Rough but useful.
        if op_active is not None and my_active is not None:
            max_dmg = self._max_damage_from(op_active)
            if max_dmg >= S.hp(my_active):
                score -= 60.0

        if self.value_fn is not None and self.value_weight != 0.0:
            try:
                score += self.value_weight * float(self.value_fn(obs, self.deck))
            except Exception:
                pass

        return score

    def _max_damage_from(self, pokemon: Any) -> int:
        cid = S.card_id(pokemon)
        c = D.card(cid)
        if c is None:
            return 0
        best = 0
        energy_count = len(S.energies(pokemon))
        for aid in getattr(c, "attacks", []) or []:
            atk = D.attack(aid)
            if atk is None:
                continue
            if len(getattr(atk, "energies", []) or []) <= energy_count:
                best = max(best, int(getattr(atk, "damage", 0) or 0))
        return best
