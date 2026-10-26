"""Confidence-gated archetype/deck profiles (v2, stronger matchup nudges).

Profiles are still advisory score nudges.  They only activate after enough
public cards have been seen and the top archetype is clear.
"""

from __future__ import annotations

PROFILE_MAP = {
    "dragapult": "mirror",
    "dragapult_top4": "mirror",
    "dragapult_blaziken": "mirror",
    "mega_lucario": "aggro",
    "garchomp": "aggro",
    "bellibolt": "aggro",
    "ns_zorua": "aggro",
    "ns_zoroark": "aggro",
    "teal_mask_ogerpon": "aggro",
    "basic_box": "aggro",
    "ceruledge": "aggro",
    "grimmsnarl": "aggro",
    "festival_lead": "setup",
    "hydrapple": "setup",
    "archaludon": "setup",
    "alakazam": "setup",
    "slowking_kangaskhan": "setup",
    "mega_excadrill": "setup",
    "mega_excadrill_top_a": "setup",
    "mega_excadrill_top_b": "setup",
    "mega_starmie": "spread",
    "mega_froslass": "spread",
    "crustle_fire": "lock",
    "munkidori_crustle": "lock",
    "banette": "lock",
    "mega_kangaskhan": "tank",
    "mega_abomasnow": "tank",
}

PROFILE_WEIGHTS = {
    "default": {},
    "aggro": {"boss": 0.30, "budew": 0.90, "search": 0.0, "late": 0.0},
    "setup": {"boss": 0.80, "budew": 0.0, "search": 0.15, "late": -0.02},
    "spread": {"boss": 0.40, "budew": 0.30, "search": 0.0, "late": 0.0},
    "lock": {"boss": 0.0, "budew": 0.30, "search": -0.60, "late": 0.10},
    "mirror": {"boss": 0.30, "budew": 0.70, "search": 0.0, "late": 0.0},
    "tank": {"boss": 0.30, "budew": 0.0, "search": 0.0, "late": -0.01},
}


def infer_profile(snapshot: dict | None) -> str:
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
