"""Self-contained public-information belief tracker for the Kaggle bundle.

This is the *observation-only* version: it prints one compact JSON line per
MAIN decision and never changes the policy's chosen action.  The deck library
is generated into belief_data.py by tools/build_belief_data.py.

Activate/disable logging with PTCG_BELIEF_LOG=0 in local runs.  Kaggle's agent
logs capture stdout, so the lines can be downloaded with:

    kaggle competitions logs <episode_id> <agent_index>
"""

from __future__ import annotations

import json
import math
from collections import Counter

try:
    from belief_data import ARCHETYPE_DECKS, PRIORS
except Exception:  # pragma: no cover - bundle fallback
    ARCHETYPE_DECKS = {}
    PRIORS = {}


def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def _p_at_least_one(unseen: int, copies: int, draws: int) -> float:
    if copies <= 0 or draws <= 0 or unseen <= 0:
        return 0.0
    copies = min(int(copies), int(unseen))
    draws = min(int(draws), int(unseen))
    if unseen - copies < draws:
        return 1.0
    miss = _comb(unseen - copies, draws) / _comb(unseen, draws)
    return max(0.0, min(1.0, 1.0 - miss))


def _card_id(card):
    return int(_get(card, "id", 0) or 0)


def _card_serial(card):
    return _get(card, "serial", None)


def _consume_card(card, remaining: Counter) -> None:
    """Remove a known own card (and its attachments) from the unseen pool."""
    if card is None:
        return
    cid = _card_id(card)
    if cid:
        remaining[cid] = max(0, remaining.get(cid, 0) - 1)
    for group in ("energyCards", "tools", "preEvolution"):
        for child in _get(card, group, None) or []:
            _consume_card(child, remaining)


class BeliefObserver:
    """Tracks both sides' public information and hidden-card probabilities."""

    def __init__(self, my_deck: list[int] | None = None):
        self.my_deck = list(my_deck or [])
        self.deck_library = dict(ARCHETYPE_DECKS)
        self.priors = dict(PRIORS)
        self.reset()

    def reset(self) -> None:
        self.opp_seen_serials: set[int] = set()
        self.opp_seen_counts: Counter = Counter()
        self.posterior: dict[str, float] = {}
        self.turn = 0
        self.phase = "early"
        self.snapshot_data: dict = {}
        self.last_log_turn = -1

    # ── public opponent cards ─────────────────────────────────────────────
    def _record_opp_id_serial(self, card_id, serial) -> None:
        if card_id is None or serial is None:
            return
        serial = int(serial)
        if serial in self.opp_seen_serials:
            return
        self.opp_seen_serials.add(serial)
        self.opp_seen_counts[int(card_id)] += 1

    def _record_opp_card(self, card) -> None:
        if card is None:
            return
        self._record_opp_id_serial(_get(card, "id"), _get(card, "serial"))
        for group in ("energyCards", "tools", "preEvolution"):
            for child in _get(card, group, None) or []:
                self._record_opp_card(child)

    # ── main update ───────────────────────────────────────────────────────
    def update(self, obs, my_index: int) -> dict | None:
        state = _get(obs, "current")
        players = _get(state, "players", []) if state is not None else []
        if state is None or len(players) < 2:
            return None

        self.turn = int(_get(state, "turn", 0) or 0)
        if self.turn < self.last_log_turn:
            # Same observer reused for a new episode without reset().
            self.reset()
            self.turn = int(_get(state, "turn", 0) or 0)
        my = players[my_index]
        op = players[1 - my_index]
        my_hand = _get(my, "hand", []) or []
        my_discard = _get(my, "discard", []) or []
        my_active = _get(my, "active", []) or []
        my_bench = _get(my, "bench", []) or []
        my_stadium = _get(state, "stadium", []) or []
        my_prizes = len(_get(my, "prize", []) or [])
        op_prizes = len(_get(op, "prize", []) or [])
        my_deck_count = int(_get(my, "deckCount", 0) or 0)
        op_hand_count = int(_get(op, "handCount", 0) or 0)
        op_deck_count = int(_get(op, "deckCount", 0) or 0)

        for zone in (_get(op, "active", []) or [],
                     _get(op, "bench", []) or [],
                     _get(op, "discard", []) or []):
            for card in zone:
                self._record_opp_card(card)
        for card in my_stadium:
            if _get(card, "playerIndex", -1) != my_index:
                self._record_opp_card(card)
        for log in _get(obs, "logs", []) or []:
            if _get(log, "playerIndex", -1) == my_index:
                continue
            card_id = _get(log, "cardId")
            serial = _get(log, "serial")
            if card_id is not None and serial is not None:
                self._record_opp_id_serial(card_id, serial)

        # Own unseen pool: start from the deck list, remove known hand/field/
        # discard.  Prizes stay in the pool because their identities are hidden.
        remaining = Counter(self.my_deck)
        for card in my_hand + my_discard + my_active + my_bench:
            _consume_card(card, remaining)
        for card in my_stadium:
            if _get(card, "playerIndex", -1) == my_index:
                _consume_card(card, remaining)
        my_unseen = max(0, sum(remaining.values()))
        if my_unseen <= 0:
            my_unseen = max(0, my_deck_count + my_prizes)

        # Posterior over known archetype templates.  A seen tech card must not
        # eliminate every deck (top lists change weekly), so unseen-in-template
        # cards get a small smoothing mass instead of a hard -inf.
        scores = {}
        for name, deck in self.deck_library.items():
            counts = Counter(deck)
            logp = math.log(max(float(self.priors.get(name, 1)), 1e-9))
            for cid, seen in self.opp_seen_counts.items():
                n = counts.get(cid, 0)
                logp += math.log(max(n, 0.25)) * seen
            scores[name] = logp
        top = max(scores.values()) if scores else 0.0
        exp = {name: math.exp(score - top) for name, score in scores.items()}
        total = sum(exp.values()) or 1.0
        self.posterior = {name: value / total for name, value in exp.items()}

        best_name, _best_deck = self.best_deck()
        opp_unseen = max(0, 60 - sum(self.opp_seen_counts.values()))

        turns_left = max(1, min(my_prizes, op_prizes))
        p_setup = self._p_any(remaining, my_unseen, (119, 120, 121, 1079), turns_left)
        p_energy = self._p_any(remaining, my_unseen, (2, 5), turns_left)
        p_my_boss = self._p_any(remaining, my_unseen, (1182,), turns_left)

        def p_opp(cid: int) -> float:
            # Mixture over archetype posterior, not just the single best deck.
            total = 0.0
            for name, prob in self.posterior.items():
                deck = self.deck_library.get(name, [])
                copies = sum(1 for card_id in deck if card_id == cid)
                remaining_copies = max(0, copies - self.opp_seen_counts.get(cid, 0))
                if remaining_copies:
                    total += prob * _p_at_least_one(
                        opp_unseen, remaining_copies, op_hand_count
                    )
            return total

        self.phase = self._phase(self.turn, my_prizes, op_prizes, my_deck_count)
        top_arch = sorted(self.posterior.items(), key=lambda kv: kv[1], reverse=True)[:3]
        self.snapshot_data = {
            "t": self.turn,
            "ph": self.phase,
            "seen": len(self.opp_seen_serials),
            "me": {
                "h": len(my_hand),
                "d": my_deck_count,
                "p": my_prizes,
                "tl": turns_left,
                "setup": round(p_setup, 3),
                "energy": round(p_energy, 3),
                "boss": round(p_my_boss, 3),
            },
            "op": {
                "arch": [[name, round(prob, 3)] for name, prob in top_arch],
                "h": op_hand_count,
                "d": op_deck_count,
                "p": op_prizes,
                "boss": round(p_opp(1182), 3),
                "judge": round(p_opp(1213), 3),
                "lillie": round(p_opp(1227), 3),
                "unfair": round(p_opp(1080), 3),
                "best": best_name,
            },
        }
        return self.snapshot_data

    @staticmethod
    def _phase(turn: int, my_prizes: int, op_prizes: int, my_deck: int) -> str:
        if my_prizes <= 2 or op_prizes <= 2 or my_deck <= 10:
            return "late"
        if turn <= 4 and my_prizes >= 5 and op_prizes >= 5:
            return "early"
        return "mid"

    @staticmethod
    def _p_any(remaining: Counter, unseen: int, ids, turns: int) -> float:
        copies = sum(max(0, int(remaining.get(cid, 0))) for cid in ids)
        return _p_at_least_one(unseen, copies, turns)

    def best_deck(self) -> tuple[str, list[int]]:
        if not self.posterior:
            return "unknown", []
        name = max(self.posterior, key=self.posterior.get)
        return name, self.deck_library.get(name, [])

    def emit(self) -> None:
        if not self.snapshot_data or self.turn == self.last_log_turn:
            return
        self.last_log_turn = self.turn
        print("BELIEF " + json.dumps(self.snapshot_data, separators=(",", ":")), flush=True)
