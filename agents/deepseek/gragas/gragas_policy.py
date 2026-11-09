"""Gragas — Grass/Ogerpon policy: learned prior + legal rule fallback."""

from __future__ import annotations

import os
from pathlib import Path

import gragas_features as F
import gragas_net
import gragas_state as S
from gragas_rules import (
    ATTACH,
    ABILITY,
    DISCARD_CONTEXTS,
    GrassRules,
    IS_FIRST,
    MAIN,
    COUNT,
    YES_NO,
    SPECIAL_CONDITION,
)

BOT_DIR = Path(__file__).resolve().parent
_DECK: list[int] | None = None
_RULES: GrassRules | None = None
_PRIOR = None
_RULE_WEIGHT = float(os.environ.get("GRAGAS_RULE_WEIGHT", "0.00025"))
_PRIOR_WEIGHT = float(os.environ.get("GRAGAS_PRIOR_WEIGHT", "1.0"))
_FORCE_ATTACH = os.environ.get("GRAGAS_FORCE_ATTACH", "1") not in ("0", "false", "False")
_FORCE_ABILITY = os.environ.get("GRAGAS_FORCE_ABILITY", "1") not in ("0", "false", "False")


def load_deck(path: Path | None = None) -> list[int]:
    global _DECK
    if _DECK is not None:
        return list(_DECK)
    candidates = [BOT_DIR / "deck.csv", Path.cwd() / "deck.csv",
                  Path("/kaggle_simulations/agent/deck.csv")]
    for candidate in candidates:
        if candidate.is_file():
            cards = [int(x) for x in candidate.read_text(encoding="utf-8").split() if x.strip()]
            if len(cards) == 60:
                _DECK = cards
                return list(_DECK)
    raise FileNotFoundError("gragas deck.csv not found")


def get_rules() -> GrassRules:
    global _RULES, _PRIOR
    if _RULES is None:
        deck = load_deck()
        try:
            _PRIOR = gragas_net.get_prior()
        except Exception:
            _PRIOR = None
        _RULES = GrassRules(deck)
    return _RULES


def _legalize(choice: list[int], n: int, mn: int, mx: int) -> list[int]:
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


def _prior_logits(obs: dict, deck: list[int]):
    if _PRIOR is None:
        return None
    try:
        return _PRIOR.logits(obs, deck, profile="grass")
    except Exception:
        return None


def choose(obs: dict) -> list[int]:
    rules = get_rules()
    deck = rules.deck
    sel = S.select(obs)
    if sel is None:
        return list(deck)
    options = list(S.get(sel, "option", []) or [])
    if not options:
        return []
    mn, mx = S.min_count(obs), S.max_count(obs)
    mx = min(mx, len(options))
    if mx <= 0:
        return []

    stype = S.select_type(obs)
    ctx = S.select_context(obs)

    # Turn order is a hard choice: always go first.
    if stype == YES_NO and ctx == IS_FIRST:
        for i, option in enumerate(options):
            if S.option_type(option) == 1:  # YES
                return [i]
        return [0]

    if stype == MAIN:
        rule_scores = []
        for i, option in enumerate(options):
            try:
                rule_scores.append(rules.main_score(obs, option))
            except Exception:
                rule_scores.append(0.0)
        lethal = [(s, i) for i, s in enumerate(rule_scores) if s >= 20000.0]
        if lethal:
            return [max(lethal, key=lambda x: (x[0], -x[1]))[1]]

        # The BC prior is strong at card sequencing but under-weights the
        # once-per-turn energy attachment and the draw abilities.  These are
        # hard, safe rules: use the best ability first, then attach energy.
        if _FORCE_ABILITY:
            abilities = [
                (rule_scores[i], i) for i, o in enumerate(options)
                if S.option_type(o) == ABILITY and rule_scores[i] >= 7500.0
            ]
            if abilities:
                return [max(abilities, key=lambda x: (x[0], -x[1]))[1]]

        if _FORCE_ATTACH:
            already = bool(S.get(S.current(obs), "energyAttached", False))
            if not already:
                attaches = [
                    (rule_scores[i], i) for i, o in enumerate(options)
                    if S.option_type(o) == ATTACH and rule_scores[i] >= 5000.0
                ]
                if attaches:
                    return [max(attaches, key=lambda x: (x[0], -x[1]))[1]]

        prior = _prior_logits(obs, deck)
        if prior is not None and len(prior) >= len(options):
            final = [
                _PRIOR_WEIGHT * float(prior[i]) + _RULE_WEIGHT * float(rule_scores[i])
                for i in range(len(options))
            ]
            return [int(max(range(len(options)), key=lambda i: (final[i], -i)))]
        return [int(max(range(len(options)), key=lambda i: (rule_scores[i], -i)))]

    # Multi-selects and discard/count/energy contexts stay rule-driven.
    if mx > 1 or stype in (COUNT, YES_NO, SPECIAL_CONDITION) or ctx in DISCARD_CONTEXTS:
        return rules.choose(obs)

    # Single-select non-MAIN: learned prior + rule bonus, legal fallback.
    prior = _prior_logits(obs, deck)
    if prior is not None and len(prior) >= len(options):
        final = []
        for i, option in enumerate(options):
            try:
                rule_score = rules.card_option_score(obs, option, ctx)
            except Exception:
                rule_score = 0.0
            final.append(_PRIOR_WEIGHT * float(prior[i]) + 0.00035 * float(rule_score))
        return [int(max(range(len(options)), key=lambda i: (final[i], -i)))]
    return rules.choose(obs)


def agent(obs_dict: dict) -> list[int]:
    return choose(obs_dict)
