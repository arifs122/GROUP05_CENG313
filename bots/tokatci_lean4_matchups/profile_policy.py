"""Confidence-gated archetype/deck profiles for the Dragapult policy.

The full belief posterior is good at identifying archetypes but the per-card
hand probabilities are overconfident.  Profiles therefore use the archetype
only, only after enough public evidence, and only as small score nudges.
"""

from __future__ import annotations

PROFILE_MAP = {
    "marnie's grimmsnarl": "grimmsnarl_custom",
    "mega_lopunny": "lopunny_custom",
    "dragapult": "mirror",
    "dragapult_top4": "mirror",
    "dragapult_blaziken": "mirror",
    "mega_lucario": "lucario_custom",
    "garchomp": "aggro",
    "bellibolt": "aggro",
    "ns_zorua": "aggro",
    "ns_zoroark": "aggro",
    "teal_mask_ogerpon": "teal_custom",
    "basic_box": "aggro",
    "ceruledge": "aggro",
    "festival_lead": "festival_custom",
    "hydrapple": "hydrapple_custom",
    "archaludon": "archaludon_custom",
    "alakazam": "alakazam_custom",
    "slowking_kangaskhan": "slowking_custom",
    "mega_excadrill": "setup",
    "mega_excadrill_top_a": "setup",
    "mega_excadrill_top_b": "setup",
    "mega_starmie": "starmie_custom",
    "mega_froslass": "spread",
    "crustle_fire": "crustle_custom",
    "munkidori_crustle": "crustle_custom",
    "banette": "banette_custom",
    "mega_kangaskhan": "kang_custom",
    "mega_abomasnow": "abomasnow_custom",
    "grimmsnarl": "tank",
}

# All values are deliberately small.  They are applied as score nudges, not as
# hard switches.  A wrong profile should not be able to dominate the policy.
PROFILE_WEIGHTS = {
    "default": {},
    "kang_custom": {"boss": 0.3, "budew": 0.5, "search": 0.05, "late": 0.0, "judge": 0.3, "stamp": 0.2},
    "lopunny_custom": {"boss": 0.5, "budew": 0.3, "search": 0.0, "late": 0.0},
    "banette_custom": {"boss": 0.0, "budew": 0.2, "search": -0.30, "late": 0.03},
    "hydrapple_custom": {"boss": 0.5, "budew": 0.0, "search": 0.1, "late": -0.03},
    "abomasnow_custom": {"boss": 0.2, "budew": 0.6, "search": 0.0, "late": 0.0},
    "festival_custom": {"boss": 0.5, "budew": 0.0, "search": 0.1, "late": -0.03},
    "starmie_custom": {"boss": 0.3, "budew": 0.2, "search": 0.0, "late": 0.0},
    "grimmsnarl_custom": {"boss": 0.2, "budew": 0.0, "search": 0.0, "late": -0.03},
    "archaludon_custom": {"boss": 0.5, "budew": 0.0, "search": 0.1, "late": -0.03},
    "slowking_custom": {"boss": 0.5, "budew": 0.0, "search": 0.1, "late": -0.03},
    "teal_custom": {"boss": 0.2, "budew": 0.6, "search": 0.0, "late": 0.0},
    "crustle_custom": {"boss": 0.0, "budew": 0.2, "search": -0.35, "late": 0.03},
    "alakazam_custom": {"boss": 0.4, "budew": 0.3, "search": 0.1, "late": -0.03, "judge": 3.0, "stamp": 0.5},
    "lucario_custom": {"boss": 0.3, "budew": 1.5, "search": 0.0, "late": 0.0, "hammer": 0.3},
    "aggro": {"boss": 0.2, "budew": 0.6, "search": 0.0, "late": 0.0},
    "setup": {"boss": 0.5, "budew": 0.0, "search": 0.1, "late": -0.03},
    "spread": {"boss": 0.3, "budew": 0.2, "search": 0.0, "late": 0.0},
    "lock": {"boss": 0.0, "budew": 0.2, "search": -0.35, "late": 0.03},
    "mirror": {"boss": 0.2, "budew": 0.4, "search": 0.0, "late": 0.0},
    "tank": {"boss": 0.2, "budew": 0.0, "search": 0.0, "late": -0.03},
}


def infer_profile(snapshot: dict | None) -> str:
    """Return a profile only when the top archetype is clear and supported."""
    if not snapshot:
        return "default"
    op = snapshot.get("op") or {}
    arch = op.get("arch") or []
    seen = int(snapshot.get("seen", 0) or 0)
    if len(arch) < 2 or seen < 6:
        return "default"
    name, p1 = arch[0]
    _, p2 = arch[1]
    if p1 < 0.66 or (p1 - p2) < 0.16:
        return "default"
    return PROFILE_MAP.get(name, "default")
