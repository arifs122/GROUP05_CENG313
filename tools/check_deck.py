#!/usr/bin/env python3
"""Validate a 60-card deck against the competition card pool.

Usage:
    python3 tools/check_deck.py decks/meta/banette.csv
    python3 tools/check_deck.py bots/banette/deck.csv --engine

Checks:
* exactly 60 cards,
* every card id exists in the local card database,
* at most 4 copies of a card name (Basic Energy is unlimited),
* at most 1 ACE SPEC,
* at least 1 Basic Pokémon,
* optionally asks the engine to start a self-play game.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

from cg.api import CardType, all_card_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a PTCG deck list")
    parser.add_argument("deck", type=Path)
    parser.add_argument("--engine", action="store_true", help="run battle_start self-play")
    args = parser.parse_args()

    if not args.deck.is_file():
        raise SystemExit(f"deck not found: {args.deck}")

    card_ids = [
        int(line.strip())
        for line in args.deck.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cards = {card.cardId: card for card in all_card_data()}
    errors: list[str] = []

    if len(card_ids) != 60:
        errors.append(f"deck has {len(card_ids)} cards, expected 60")

    missing = sorted({cid for cid in card_ids if cid not in cards})
    if missing:
        errors.append(f"unknown card ids: {missing}")

    by_name: Counter = Counter()
    ace_specs = []
    basic_count = 0
    for cid, count in Counter(card_ids).items():
        card = cards.get(cid)
        if card is None:
            continue
        by_name[card.name] += count
        if card.aceSpec:
            ace_specs.append((cid, card.name, count))
        if card.basic:
            basic_count += count

    for name, count in by_name.items():
        card = next(card for card in cards.values() if card.name == name)
        if card.cardType == CardType.BASIC_ENERGY:
            continue
        if count > 4:
            errors.append(f"more than 4 copies of {name}: {count}")

    ace_spec_total = sum(count for _, _, count in ace_specs)
    if ace_spec_total > 1:
        errors.append(f"more than 1 ACE SPEC: {ace_specs}")

    if basic_count < 1:
        errors.append("deck has no Basic Pokémon")

    print(f"deck: {args.deck}")
    print(f"cards: {len(card_ids)}  unique: {len(set(card_ids))}")
    print(f"basic Pokémon: {basic_count}")
    print(f"ACE SPEC: {ace_specs or 'none'}")

    # Show the largest stacks, useful for a quick visual check.
    print("\nstacks:")
    for cid, count in Counter(card_ids).most_common():
        card = cards.get(cid)
        print(f"  {count}x {cid:>4} {card.name if card else '?'}")

    if args.engine:
        from cg.game import battle_finish, battle_start

        obs, start_data = battle_start(card_ids, card_ids)
        if obs is None:
            errors.append(
                f"engine rejected deck: player={start_data.errorPlayer} type={start_data.errorType}"
            )
        else:
            battle_finish()
            print("\nengine: deck accepted")

    if errors:
        print("\nFAIL")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("\nOK: deck looks legal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
