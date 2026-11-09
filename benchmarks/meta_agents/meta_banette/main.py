"""Template bot for the local arena.

Copy this folder to ``bots/<your_bot>/`` and edit ``agent()`` and ``deck.csv``.
The only required interface is::

    def agent(obs_dict: dict) -> list[int]

``obs_dict["select"] is None`` means the initial deck selection; return the
60 card ids.  A deck is also stored next to this file so the arena can load it
for ``battle_start``.
"""

from __future__ import annotations

from pathlib import Path

from cg.api import OptionType, SelectContext, SelectType

_ATTACK_DAMAGE: dict[int, int] | None = None
_DECK: list[int] | None = None

# Lower number = higher priority during the MAIN selection.
_MAIN_PRIORITY = {
    int(OptionType.EVOLVE): 0,
    int(OptionType.ABILITY): 1,
    int(OptionType.ATTACH): 2,
    int(OptionType.PLAY): 3,
    int(OptionType.ATTACK): 4,
    int(OptionType.RETREAT): 5,
    int(OptionType.DISCARD): 6,
    int(OptionType.END): 7,
}

_DISCARD_CONTEXTS = {
    int(SelectContext.DISCARD),
    int(SelectContext.DISCARD_ENERGY_CARD),
    int(SelectContext.DISCARD_TOOL_CARD),
    int(SelectContext.DISCARD_CARD_OR_ATTACHED_CARD),
    int(SelectContext.DISCARD_ENERGY),
}

_MORE_IS_BETTER_COUNT_CONTEXTS = {
    int(SelectContext.DRAW_COUNT),
    int(SelectContext.DAMAGE_COUNTER_COUNT),
    int(SelectContext.REMOVE_DAMAGE_COUNTER_COUNT),
}


def _get(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _attack_damage(attack_id: int | None) -> int:
    global _ATTACK_DAMAGE
    if attack_id is None:
        return -1
    if _ATTACK_DAMAGE is None:
        from cg.api import all_attack

        _ATTACK_DAMAGE = {int(a.attackId): int(a.damage or 0) for a in all_attack()}
    return _ATTACK_DAMAGE.get(int(attack_id), 0)


def load_deck() -> list[int]:
    global _DECK
    if _DECK is None:
        path = Path(__file__).with_name("deck.csv")
        cards = [
            int(line.strip())
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(cards) != 60:
            raise ValueError(f"{path} contains {len(cards)} cards, expected 60")
        _DECK = cards
    return list(_DECK)


def agent(obs_dict: dict) -> list[int]:
    select = obs_dict.get("select")
    if select is None:
        return load_deck()

    options = select.get("option") or []
    if not options:
        return []

    min_count = max(0, int(select.get("minCount", 0)))
    max_count = min(int(select.get("maxCount", 0)), len(options))
    if max_count <= 0:
        return []

    select_type = int(select.get("type", -1))
    context = int(select.get("context", -1))

    if select_type == int(SelectType.MAIN):
        best = min(
            range(len(options)),
            key=lambda i: (_MAIN_PRIORITY.get(int(_get(options[i], "type", -1)), 99), i),
        )
        return [best]

    if select_type == int(SelectType.ATTACK):
        best = max(
            range(len(options)),
            key=lambda i: (_attack_damage(_get(options[i], "attackId")), -i),
        )
        return [best]

    if select_type == int(SelectType.YES_NO):
        for i, option in enumerate(options):
            if int(_get(option, "type", -1)) == int(OptionType.YES):
                return [i]
        return list(range(min(1, max_count)))

    if select_type == int(SelectType.COUNT):
        numbers = [int(_get(o, "number", 0)) for o in options]
        if context in _MORE_IS_BETTER_COUNT_CONTEXTS:
            return [max(range(len(numbers)), key=lambda i: (numbers[i], -i))]
        return [min(range(len(numbers)), key=lambda i: (numbers[i], i))]

    count = min_count if context in _DISCARD_CONTEXTS else max_count
    count = max(min_count, min(count, max_count))
    return list(range(count))
