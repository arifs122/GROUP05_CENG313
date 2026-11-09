"""Confidence-gated archetype/deck profiles for the Dragapult policy.

The full belief posterior is good at identifying archetypes but the per-card
hand probabilities are overconfident.  Profiles therefore use the archetype
only, only after enough public evidence, and only as small score nudges.
"""

from __future__ import annotations

PROFILE_MAP = {
    "dragapult": "tokatci_mir_stamp_profile",
    "dragapult_top4": "tokatci_mir_stamp_profile",
    "dragapult_blaziken": "tokatci_mir_stamp_profile",
    "mega_lucario": "lucario_custom",
    "garchomp": "aggro",
    "bellibolt": "aggro",
    "ns_zorua": "aggro",
    "ns_zoroark": "aggro",
    "teal_mask_ogerpon": "aggro",
    "basic_box": "aggro",
    "ceruledge": "aggro",
    "festival_lead": "setup",
    "hydrapple": "setup",
    "archaludon": "setup",
    "alakazam": "alakazam_custom",
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
    "grimmsnarl": "tank",
}

# All values are deliberately small.  They are applied as score nudges, not as
# hard switches.  A wrong profile should not be able to dominate the policy.
PROFILE_WEIGHTS = {
    "default": {},
    "tokatci_mir_stamp_profile": {"boss": 0.2, "budew": 0.4, "search": 0.0, "late": 0.0, "stamp": 2.0},
    "alakazam_custom": {"boss": 0.4, "budew": 0.3, "search": 0.1, "late": -0.03, "judge": 3.0, "stamp": 0.5},
    "lucario_custom": {"boss": 0.3, "budew": 1.5, "search": 0.0, "late": 0.0, "hammer": 1.5},
    "dragapult_custom": {"boss": 0.2, "budew": 0.4, "search": 0.0, "late": 0.0, "judge": 1.5, "stamp": 0.5},
    "aggro": {"boss": 0.2, "budew": 0.6, "search": 0.0, "late": 0.0},
    "setup": {"boss": 0.5, "budew": 0.0, "search": 0.1, "late": -0.03},
    "spread": {"boss": 0.3, "budew": 0.2, "search": 0.0, "late": 0.0},
    "lock": {"boss": 0.0, "budew": 0.2, "search": -0.35, "late": 0.03},
    "mirror": {"boss": 0.2, "budew": 0.4, "search": 0.0, "late": 0.0},
    "tank": {"boss": 0.2, "budew": 0.0, "search": 0.0, "late": -0.03},
}


def infer_profile(snapshot: dict | None) -> str:
    """Return a profile only after 5 public opponent cards."""
    if not snapshot:
        return "default"
    seen = int(snapshot.get("seen", 0) or 0)
    if seen < 5:
        return "default"
    op = snapshot.get("op") or {}
    arch = op.get("arch") or []
    if len(arch) < 2:
        return "default"
    name, p1 = arch[0]
    _, p2 = arch[1]
    if p1 < 0.66 or (p1 - p2) < 0.16:
        return "default"
    return PROFILE_MAP.get(name, "default")
