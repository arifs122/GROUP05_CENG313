"""Public-information belief tracker for PTCG.

No machine learning: this module only counts revealed cards, compares them with
known deck lists, and computes hypergeometric probabilities.  It is the feature
layer that a rule-based policy (and later a value/search model) can use.

Example:
    python3 tools/belief.py replays/ladder/episode-125106581-replay.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

from cg.api import all_card_data  # noqa: E402

CARD_CSV = ROOT / "data" / "EN_Card_Data_R2_full.csv"


def load_cards() -> dict[int, str]:
    if CARD_CSV.is_file():
        names = {}
        with CARD_CSV.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                names.setdefault(int(row["Card ID"]), row["Card Name"])
        return names
    return {card.cardId: card.name for card in all_card_data()}


CARDS = load_cards()


def _comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def hypergeom_at_least_one(unseen: int, copies: int, draws: int) -> float:
    """P(at least one of `copies` in `unseen` appears in `draws`)."""
    if copies <= 0 or draws <= 0 or unseen <= 0:
        return 0.0
    copies = min(copies, unseen)
    draws = min(draws, unseen)
    miss = _comb(unseen - copies, draws) / _comb(unseen, draws)
    return max(0.0, min(1.0, 1.0 - miss))


def hypergeom_count(unseen: int, copies: int, draws: int, at_least: int) -> float:
    """P(at least `at_least` copies appear in `draws`)."""
    if at_least <= 0:
        return 1.0
    if copies < at_least or unseen <= 0:
        return 0.0
    total = _comb(unseen, draws)
    if total == 0:
        return 0.0
    good = sum(
        _comb(copies, k) * _comb(unseen - copies, draws - k)
        for k in range(at_least, min(copies, draws) + 1)
    )
    return max(0.0, min(1.0, good / total))


def load_deck(path: Path) -> list[int]:
    cards = [int(x) for x in path.read_text(encoding="utf-8").split() if x.strip()]
    if len(cards) != 60:
        raise ValueError(f"{path}: expected 60 cards, got {len(cards)}")
    return cards


def deck_library() -> dict[str, list[int]]:
    library = {}
    for folder in (ROOT / "decks" / "meta", ROOT / "decks" / "lab", ROOT / "benchmarks" / "meta_decks"):
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.csv")):
            try:
                library[path.stem] = load_deck(path)
            except ValueError:
                pass
        for path in sorted(folder.glob("*/deck.csv")):
            try:
                library[path.parent.name] = load_deck(path)
            except ValueError:
                pass
    return library


@dataclass
class Belief:
    """Belief about one opponent's hidden cards, updated from public info."""

    opponent_index: int = 1
    deck_library: dict[str, list[int]] = field(default_factory=dict)
    priors: dict[str, float] = field(default_factory=dict)
    seen_serials: set[int] = field(default_factory=set)
    seen_counts: Counter = field(default_factory=Counter)
    posterior: dict[str, float] = field(default_factory=dict)
    turn: int = 0
    hand_count: int = 0
    deck_count: int = 0
    prize_count: int = 0
    my_prizes: int = 0

    def __post_init__(self):
        if not self.deck_library:
            self.deck_library = deck_library()
        if not self.priors:
            self.priors = {name: 1.0 for name in self.deck_library}

    def _record(self, card: dict | None):
        if not card or "serial" not in card:
            return
        serial = int(card["serial"])
        if serial in self.seen_serials:
            return
        self.seen_serials.add(serial)
        self.seen_counts[int(card["id"])] += 1

    def _record_pokemon(self, pokemon: dict | None):
        if not pokemon:
            return
        self._record({"id": pokemon["id"], "serial": pokemon["serial"]})
        for group in ("energyCards", "tools", "preEvolution"):
            for card in pokemon.get(group) or []:
                self._record(card)

    def update(self, obs: dict) -> None:
        current = obs.get("current")
        if not current:
            return
        self.turn = int(current.get("turn", 0))
        players = current.get("players") or []
        if len(players) < 2:
            return
        opp = players[self.opponent_index]
        me = players[1 - self.opponent_index]
        self.hand_count = int(opp.get("handCount", 0))
        self.deck_count = int(opp.get("deckCount", 0))
        self.prize_count = len(opp.get("prize") or [])
        self.my_prizes = len(me.get("prize") or [])

        for pokemon in opp.get("active") or []:
            self._record_pokemon(pokemon)
        for pokemon in opp.get("bench") or []:
            self._record_pokemon(pokemon)
        for card in opp.get("discard") or []:
            self._record(card)
        for card in current.get("stadium") or []:
            if card and card.get("playerIndex") == self.opponent_index:
                self._record(card)
        for card in current.get("looking") or []:
            # Looking cards are public to the player who is looking; when the
            # opponent is looking, only revealed cards should be counted.
            if card and card.get("playerIndex") == self.opponent_index:
                self._record(card)

        # Logs are cumulative per selection; unique serials make this idempotent.
        for log in obs.get("logs") or []:
            if int(log.get("playerIndex", -1)) != self.opponent_index:
                continue
            if "cardId" in log and "serial" in log:
                self._record({"id": log["cardId"], "serial": log["serial"]})

        self._update_posterior()

    def _update_posterior(self) -> None:
        scores = {}
        for name, deck in self.deck_library.items():
            counts = Counter(deck)
            logp = math.log(max(self.priors.get(name, 1.0), 1e-9))
            possible = True
            for cid, seen in self.seen_counts.items():
                n = counts.get(cid, 0)
                if seen > n:
                    possible = False
                    break
                # A simple likelihood: seeing a card that is common in this
                # deck is more likely.  This is intentionally conservative.
                logp += math.log(max(n, 0.5)) * seen
            if possible:
                scores[name] = logp
        if not scores:
            self.posterior = {name: 1.0 / max(1, len(self.deck_library)) for name in self.deck_library}
            return
        top = max(scores.values())
        exp = {name: math.exp(score - top) for name, score in scores.items()}
        total = sum(exp.values())
        self.posterior = {name: value / total for name, value in exp.items()}

    def top_decks(self, k: int = 3) -> list[tuple[str, float]]:
        return sorted(self.posterior.items(), key=lambda item: item[1], reverse=True)[:k]

    def best_deck(self) -> tuple[str, list[int]]:
        if not self.posterior:
            return "unknown", []
        name = max(self.posterior, key=self.posterior.get)
        return name, self.deck_library[name]

    def remaining_counts(self) -> Counter:
        _, deck = self.best_deck()
        remaining = Counter(deck)
        for cid, seen in self.seen_counts.items():
            remaining[cid] = max(0, remaining.get(cid, 0) - seen)
        return +remaining

    def unseen_total(self) -> int:
        return max(0, 60 - sum(self.seen_counts.values()))

    def p_in_hand(self, card_id: int) -> float:
        """P(specific remaining copy is in the opponent's hand)."""
        remaining = self.remaining_counts().get(card_id, 0)
        return hypergeom_at_least_one(self.unseen_total(), remaining, self.hand_count)

    def p_in_prizes(self, card_id: int) -> float:
        remaining = self.remaining_counts().get(card_id, 0)
        return hypergeom_at_least_one(self.unseen_total(), remaining, self.prize_count)

    def p_next_draw(self, card_id: int) -> float:
        remaining = self.remaining_counts().get(card_id, 0)
        if self.deck_count <= 0:
            return 0.0
        return min(1.0, remaining / self.deck_count)

    def p_draw_within(self, card_id: int, turns: int) -> float:
        remaining = self.remaining_counts().get(card_id, 0)
        return hypergeom_at_least_one(self.deck_count, remaining, max(0, turns))

    def phase(self) -> str:
        if self.my_prizes <= 2 or self.prize_count <= 2 or self.deck_count <= 10:
            return "late"
        if self.turn <= 4 and self.my_prizes >= 5 and self.prize_count >= 5:
            return "early"
        return "mid"


def summarize(obs: dict, opponent_index: int, belief: Belief | None = None) -> Belief:
    belief = belief or Belief(opponent_index=opponent_index)
    belief.update(obs)
    print(f"turn={belief.turn} phase={belief.phase()} hand={belief.hand_count} "
          f"deck={belief.deck_count} prizes={belief.prize_count} seen={sum(belief.seen_counts.values())}")
    for name, prob in belief.top_decks(5):
        print(f"  {prob:6.1%} {name}")
    for cid in (1182, 1121, 1198, 722, 723, 678, 293):
        if belief.remaining_counts().get(cid, 0):
            print(f"  P(hand has {CARDS.get(cid, cid)}): {belief.p_in_hand(cid):5.1%}  "
                  f"P(next draw): {belief.p_next_draw(cid):5.1%}")
    return belief


def main() -> int:
    parser = argparse.ArgumentParser(description="Belief/card-count tracker")
    parser.add_argument("replay", type=Path)
    parser.add_argument("--every", type=int, default=10)
    args = parser.parse_args()

    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    steps = replay.get("steps", [])
    print(f"episode {replay.get('info', {}).get('EpisodeId')} steps={len(steps)}")
    beliefs = {0: Belief(opponent_index=0), 1: Belief(opponent_index=1)}
    for index, step in enumerate(steps):
        for player, agent in enumerate(step):
            if agent.get("status") != "ACTIVE":
                continue
            obs = agent.get("observation") or {}
            current = obs.get("current")
            if not current:
                continue
            opponent = 1 - int(current["yourIndex"])
            belief = beliefs[opponent]
            belief.update(obs)
            if index % args.every == 0:
                print(f"\nstep {index}")
                summarize(obs, opponent, belief)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
