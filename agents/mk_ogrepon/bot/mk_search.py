"""Determinised 1-ply search over our MAIN options.

The engine search API is used only at game time.  Hidden cards are sampled from
our known 60-card list and public archetype lists; no opponent deck list is
used as a state feature or persisted into the training data.
"""

from __future__ import annotations

import random
import time
from typing import Any

import mk_data as D
import mk_meta
import mk_state as S

try:
    from cg.api import search_begin, search_end, search_step, to_observation_class
except Exception:  # pragma: no cover - bundle without the engine
    search_begin = search_step = search_end = to_observation_class = None


def _subtract(deck: list[int], counts: dict[int, int]) -> list[int]:
    remaining = list(deck)
    for cid, count in counts.items():
        for _ in range(int(count)):
            try:
                remaining.remove(int(cid))
            except ValueError:
                pass
    return remaining


class SearchEngine:
    def __init__(self, deck: list[int], budget: float = 0.18,
                 rollouter=None, rng: random.Random | None = None,
                 value_model=None, value_weight: float = 0.0) -> None:
        self.deck = list(deck)
        self.budget = float(budget)
        self.rollouter = rollouter
        self.rng = rng or random.Random(20261103)
        self.value_model = value_model
        self.value_weight = float(value_weight)
        self.overrides = 0
        self.searches = 0
        self.total_seconds = 0.0

    def choose(self, obs: dict, scores: list[tuple[float, int]]) -> int | None:
        if search_begin is None or not scores:
            return None
        if S.select_type(obs) != S.S_MAIN or len(S.options(obs)) < 2:
            return None
        if max(s for s, _ in scores) <= 0:
            return None
        started = time.perf_counter()
        deadline = started + self.budget
        me = S.you(obs)
        options = S.options(obs)
        # Candidate list: keep the rule's best options; END is never a
        # candidate when the rule already rejected it.
        candidates = [i for _s, i in sorted(scores, reverse=True)[:4]]
        candidates = [i for i in candidates if S.option_type(options[i]) != S.O_END]
        if len(candidates) < 2:
            return None

        rule_best = max(scores, key=lambda x: (x[0], -x[1]))[1]
        totals: dict[int, float] = {i: 0.0 for i in candidates}
        counts = {i: 0 for i in candidates}
        samples = 2 if self.budget >= 0.08 else 1
        try:
            for _ in range(samples):
                if time.perf_counter() >= deadline:
                    break
                sample = self._sample(obs)
                if sample is None:
                    continue
                try:
                    root = search_begin(
                        to_observation_class(obs),
                        sample["your_deck"], sample["your_prize"],
                        sample["opp_deck"], sample["opp_prize"],
                        sample["opp_hand"], sample["opp_active"], False,
                    )
                except Exception:
                    continue
                try:
                    for action in candidates:
                        if time.perf_counter() >= deadline:
                            break
                        try:
                            nxt = search_step(root.searchId, [action])
                        except Exception:
                            continue
                        value = self._rollout_value(nxt.observation, me, deadline, nxt.searchId)
                        totals[action] += value
                        counts[action] += 1
                finally:
                    try:
                        search_end()
                    except Exception:
                        pass
        finally:
            self.total_seconds += time.perf_counter() - started

        usable = [i for i in candidates if counts[i] > 0]
        if len(usable) < 2:
            return None
        means = {i: totals[i] / counts[i] for i in usable}
        best = max(usable, key=lambda i: (means[i], -i))
        if best == rule_best:
            return None
        if means[best] - means.get(rule_best, means[best]) < 90.0:
            return None
        self.overrides += 1
        self.searches += 1
        return int(best)

    # ------------------------------------------------------------------
    # Hidden-information determinisation
    # ------------------------------------------------------------------
    def _sample(self, obs: dict) -> dict | None:
        if S.deck_cards(obs):
            # Search from a "look at deck" decision is rare; skip it rather
            # than pass an inconsistent deck/prize split.
            return None
        me = S.you(obs)
        opp = 1 - me
        known_me = S.known_counts(obs, me)
        remaining = _subtract(self.deck, known_me)
        # Our prize cards are face-down, so they are part of the remaining pool.
        want_me = S.deck_count(obs, me) + S.prize_count(obs, me)
        if len(remaining) < want_me:
            remaining.extend([D.GRASS] * (want_me - len(remaining)))
        self.rng.shuffle(remaining)
        your_prize = remaining[:S.prize_count(obs, me)]
        your_deck = remaining[S.prize_count(obs, me):S.prize_count(obs, me) + S.deck_count(obs, me)]

        arche = self._archetype(obs)
        base = list(mk_meta.META_DECKS.get(arche) or mk_meta.META_DECKS.get("dragapult")
                    or self.deck)
        known_opp = S.known_counts(obs, opp)
        opp_remaining = _subtract(base, known_opp)
        want_opp = (S.deck_count(obs, opp) + S.hand_count(obs, opp)
                    + S.prize_count(obs, opp))
        if len(opp_remaining) < want_opp:
            opp_remaining.extend([D.GRASS] * (want_opp - len(opp_remaining)))
        self.rng.shuffle(opp_remaining)
        opp_prize = opp_remaining[:S.prize_count(obs, opp)]
        opp_hand = opp_remaining[S.prize_count(obs, opp):S.prize_count(obs, opp) + S.hand_count(obs, opp)]
        opp_deck = opp_remaining[S.prize_count(obs, opp) + S.hand_count(obs, opp):
                                 S.prize_count(obs, opp) + S.hand_count(obs, opp) + S.deck_count(obs, opp)]

        opp_active: list[int] = []
        if len(S.zone_cards(obs, opp, S.ACTIVE)) > 0 and S.active(obs, opp) is None:
            basics = [c for c in base if D.is_basic(c)]
            if basics:
                opp_active = [self.rng.choice(basics)]

        return {
            "your_deck": your_deck,
            "your_prize": your_prize,
            "opp_deck": opp_deck,
            "opp_prize": opp_prize,
            "opp_hand": opp_hand,
            "opp_active": opp_active,
        }

    def _archetype(self, obs: dict) -> str:
        # Same public classifier as the rule layer, kept local to avoid a
        # circular import.
        opp = 1 - S.you(obs)
        counts: dict[int, int] = {}
        for p in S.field(obs, opp):
            cid = S.pokemon_id(p)
            if cid is not None:
                counts[cid] = counts.get(cid, 0) + 1
            for x in S.pre_evolution(p):
                counts[x] = counts.get(x, 0) + 1
        for cid in S.discard_ids(obs, opp):
            counts[cid] = counts.get(cid, 0) + 1
        best = "Unknown"
        best_score = 0
        for name, keys in D.ARCHETYPE_KEYS:
            if not keys:
                continue
            score = sum(min(2, counts.get(k, 0)) for k in keys)
            if score > best_score:
                best_score = score
                best = name
        mapping = {
            "Dragapult": "dragapult",
            "Alakazam": "alakazam",
            "Mega Kangaskhan": "mega_kangaskhan",
            "Hydrapple": "hydrapple",
            "Archaludon": "archaludon",
            "Banette": "banette",
            "Festival Lead": "festival_lead",
            "Mega Excadrill": "mega_excadrill",
            "Mega Abomasnow": "mega_abomasnow",
            "Ceruledge": "ceruledge",
            "Mega Starmie": "mega_starmie",
            "Mega Froslass": "mega_froslass",
            "Marnie's Grimmsnarl": "dragapult",
            "Cynthia's Garchomp": "dragapult",
            "Teal Mask Ogerpon": "teal_mask_ogerpon",
            "N's Zorua": "ns_zorua",
            "Slowking": "slowking_kangaskhan",
            "Crustle": "crustle_fire",
            "Munkidori-Crustle": "munkidori_crustle",
        }
        return mapping.get(best, "dragapult")

    # ------------------------------------------------------------------
    # Rollout + leaf evaluation
    # ------------------------------------------------------------------
    def _rollout_value(self, obs: Any, me: int, deadline: float,
                       search_id: int) -> float:
        if obs is None:
            return -1e9
        # A few greedy actions of our own; opponent decisions are evaluated as
        # they are (no hidden opponent hand is had-crafted).
        steps = 0
        while (steps < 3 and time.perf_counter() < deadline
               and S.result(obs) < 0 and S.select(obs) is not None
               and S.you(obs) == me and S.select_type(obs) == S.S_MAIN):
            if self.rollouter is None:
                break
            try:
                action = self.rollouter.choose(obs)
                nxt = search_step(search_id, action)
            except Exception:
                break
            if nxt is None:
                break
            obs = nxt.observation
            search_id = nxt.searchId
            steps += 1
        return self.evaluate(obs, me)

    def evaluate(self, obs: Any, me: int) -> float:
        if obs is None or S.current(obs) is None:
            return -1e9
        result = S.result(obs)
        if result >= 0:
            if result == 2:
                return 0.0
            return 1e8 if result == me else -1e8

        opp = 1 - me
        my_prizes = S.prize_count(obs, me)
        opp_prizes = S.prize_count(obs, opp)
        score = 1200.0 * (opp_prizes - my_prizes)
        score += 0.35 * (sum(S.hp(p) for p in S.field(obs, me))
                         - sum(S.hp(p) for p in S.field(obs, opp)))
        score += 45.0 * (sum(S.energy_count(p) for p in S.field(obs, me))
                         - sum(S.energy_count(p) for p in S.field(obs, opp)))
        score += 55.0 * (len(S.bench(obs, me)) - len(S.bench(obs, opp)))
        score += 18.0 * (S.hand_count(obs, me) - S.hand_count(obs, opp))
        my_deck = S.deck_count(obs, me)
        if my_deck <= 3:
            score -= (4 - my_deck) * 900.0
        active = S.active(obs, me)
        if active is not None:
            score += 80.0 * S.energy_count(active)
            score += 0.2 * S.hp(active)
        if self.value_model is not None and self.value_weight:
            try:
                p = float(self.value_model.win_prob(obs))
                score += self.value_weight * (p - 0.5) * 2000.0
            except Exception:
                pass
        return score
