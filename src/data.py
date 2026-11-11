"""Data loading and preprocessing for the PTCG AI Battle Challenge.

The competition is a *simulation* competition: there is no ``train.csv``.
The observations we can analyse are episode replays downloaded from Kaggle.
Two files are used by the notebooks:

``data/EN_Card_Data_R2_full.csv``
    Official card metadata that ships with the competition data page.

``data/analysis_matches.csv``
    One row per episode (match), built by ``tools/build_analysis_dataset.py``.
    It contains the two 60-card decks, their archetypes, the result and the
    player who went first.  The file is derived competition data and is
    therefore **not committed** (see ``.gitignore``).

Everything in this module is deterministic and side-effect free except for
the explicit ``build_*`` functions.  Keeping the transformations here means
that the EDA notebook and the model code use exactly the same definitions.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPLAYS_DIR = ROOT / "replays"

CARD_CSV = DATA_DIR / "EN_Card_Data_R2_full.csv"
MASTER_CSV = DATA_DIR / "master_replays.csv"
MATCH_CSV = DATA_DIR / "analysis_matches.csv"

# The raw card file uses long, human readable column names.
CARD_RENAMES = {
    "Card ID": "card_id",
    "Card Name": "name",
    "Expansion": "expansion",
    "Stage (Pokémon)/Type (Energy and Trainer)": "kind",
    "Rule": "rule",
    "Category": "category",
    "HP": "hp",
    "Type": "type",
    "Weakness": "weakness",
    "Resistance (Type)": "resistance",
    "Retreat": "retreat",
    "Move Name": "move_name",
    "Cost": "cost",
    "Damage": "damage",
    "Effect Explanation": "effect",
    "Species": "species",
}

# Coarse card groups used by the baseline.  They are intentionally simple so
# that the coefficients stay interpretable.
KIND_GROUPS = {
    "Basic Pokémon": "pokemon",
    "Stage 1 Pokémon": "pokemon",
    "Stage 2 Pokémon": "pokemon",
    "Item": "item",
    "Supporter": "supporter",
    "Pokémon Tool": "tool",
    "Stadium": "stadium",
    "Basic Energy": "basic_energy",
    "Special Energy": "special_energy",
}

KIND_FEATURES = [
    "pokemon",
    "item",
    "supporter",
    "tool",
    "stadium",
    "basic_energy",
    "special_energy",
    "other",
]

FIRST_PLAYER_RE = re.compile(r'"firstPlayer"\s*:\s*(-?\d+)')
EPISODE_RE = re.compile(r'"EpisodeId"\s*:\s*(\d+)')


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------
def load_cards(path: str | Path = CARD_CSV) -> pd.DataFrame:
    """Load card metadata and add ``kind_group`` and ``ace_spec`` columns."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Card metadata not found: {path}\n"
            "Run `bash scripts/setup_assets.sh /path/to/sample_submission/cg` "
            "or copy EN_Card_Data_R2_full.csv into data/."
        )
    cards = pd.read_csv(path, encoding="utf-8-sig")
    cards = cards.rename(columns=CARD_RENAMES)
    cards["card_id"] = cards["card_id"].astype(int)
    cards["kind"] = cards["kind"].fillna("Other")
    cards["kind_group"] = cards["kind"].map(KIND_GROUPS).fillna("other")
    cards["ace_spec"] = cards["rule"].fillna("").str.contains("ACE SPEC", case=False)
    cards = cards.drop_duplicates("card_id").reset_index(drop=True)
    return cards


# ---------------------------------------------------------------------------
# Match table
# ---------------------------------------------------------------------------
def parse_deck(value: object) -> list[int]:
    """Parse a space separated deck string into a list of card ids."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(x) for x in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [int(x) for x in str(value).split()]


def first_player_from_replay(path: str | Path) -> int | None:
    """Read the first player from the beginning of a replay file.

    ``firstPlayer`` is written repeatedly in the observation state.  The
    initial value is ``-1`` while the setup decisions are still running, so we
    return the last non-negative value found in the first 2 MiB.  This is
    enough for every replay in our corpus and avoids loading multi-megabyte
    JSON files into memory.
    """
    path = Path(path)
    try:
        with path.open("rb") as handle:
            text = handle.read(2_000_000).decode("utf-8", "ignore")
    except OSError:
        return None
    values = [int(x) for x in FIRST_PLAYER_RE.findall(text)]
    values = [value for value in values if value >= 0]
    return values[-1] if values else None


def _resolve_replay_path(value: str | Path) -> Path | None:
    """Find a replay even if an absolute path from another machine was saved."""
    path = Path(str(value))
    if path.is_file():
        return path
    candidate = REPLAYS_DIR / path.name
    if candidate.is_file():
        return candidate
    # The build script stores the path relative to replays/ when possible.
    candidate = REPLAYS_DIR / str(value)
    return candidate if candidate.is_file() else None


def _source_group(source: str) -> str:
    """Collapse the many local download folders into three useful groups."""
    source = str(source)
    if source == "ours":
        return "our matches"
    if source in {"top20", "top2_more", "top2_policy", "top5_nov1", "top_oct24"}:
        return "top-team replays"
    return "our bot logs"


def build_match_table(
    master_path: str | Path = MASTER_CSV,
    replays_dir: str | Path = REPLAYS_DIR,
    cards: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the one-row-per-match analysis table.

    Parameters
    ----------
    master_path:
        Output of ``tools/build_master_replays.py`` (``data/master_replays.csv``).
    replays_dir:
        Directory containing the replay JSON files.  Only the first 2 MiB of
        each file is read, to recover the first player.
    cards:
        Optional pre-loaded card metadata.
    """
    master_path = Path(master_path)
    if not master_path.is_file():
        raise FileNotFoundError(
            f"Replay metadata not found: {master_path}\n"
            "Build it first with:\n"
            "  python tools/build_master_replays.py --out-dir data\n"
            "then rename data/master_replays.csv if needed."
        )
    cards = load_cards() if cards is None else cards
    unique_cards = cards.drop_duplicates("card_id")
    card_kind = unique_cards.set_index("card_id")["kind_group"]
    card_ace = unique_cards.set_index("card_id")["ace_spec"]

    matches = pd.read_csv(master_path)
    matches = matches.drop_duplicates("episode_id").copy()
    matches["winner"] = pd.to_numeric(matches["winner"], errors="coerce")
    matches["reward0"] = pd.to_numeric(matches["reward0"], errors="coerce")
    matches["reward1"] = pd.to_numeric(matches["reward1"], errors="coerce")
    matches["source_group"] = matches["source"].map(_source_group)

    rows = []
    for row in matches.itertuples(index=False):
        decks = [parse_deck(row.deck0), parse_deck(row.deck1)]
        counts = [Counter(deck) for deck in decks]
        first = first_player_from_replay(_resolve_replay_path(row.path) or row.path)
        record = {
            "episode_id": int(row.episode_id),
            "source": row.source,
            "source_group": row.source_group,
            "source_team": "" if pd.isna(row.source_team) else row.source_team,
            "team0": row.team0,
            "team1": row.team1,
            "arch0": row.arch0,
            "arch1": row.arch1,
            "reward0": row.reward0,
            "reward1": row.reward1,
            "winner": None if pd.isna(row.winner) else int(row.winner),
            "first_player": first,
            "replay_path": str(_resolve_replay_path(row.path) or row.path),
            "deck0": " ".join(map(str, decks[0])),
            "deck1": " ".join(map(str, decks[1])),
        }
        for side, deck_counts in enumerate(counts):
            kind_counts = Counter()
            for card_id, count in deck_counts.items():
                kind_counts[card_kind.get(card_id, "other")] += count
            ace = sum(count for card_id, count in deck_counts.items() if card_ace.get(card_id, False))
            record[f"deck{side}_len"] = len(decks[side])
            record[f"unique{side}"] = len(deck_counts)
            record[f"ace{side}"] = int(ace)
            for kind in KIND_FEATURES:
                record[f"kind{side}_{kind}"] = int(kind_counts.get(kind, 0))
        rows.append(record)

    table = pd.DataFrame(rows)
    table["first_player"] = table["first_player"].astype("Int64")
    return table


def load_match_table(path: str | Path = MATCH_CSV) -> pd.DataFrame:
    """Load the analysis table, building it from replays when necessary."""
    path = Path(path)
    if path.is_file():
        table = pd.read_csv(path)
    elif MASTER_CSV.is_file():
        table = build_match_table(MASTER_CSV)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
    else:
        raise FileNotFoundError(
            f"No analysis table at {path} and no {MASTER_CSV}.\n"
            "Download the competition replays into replays/, then run:\n"
            "  python tools/build_analysis_dataset.py"
        )
    if "first_player" in table:
        table["first_player"] = pd.to_numeric(table["first_player"], errors="coerce")
    return table


# ---------------------------------------------------------------------------
# Modelling frame
# ---------------------------------------------------------------------------
def to_player_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """Expand each match into two player-perspective rows.

    Every match contributes one row for player 0 and one for player 1.  This
    removes the arbitrary side label: the model sees ``own`` and ``opponent``
    features and the target is always "did this player win?".  Splitting must
    be done by ``episode_id`` so that the two rows of a match never end up in
    different folds.
    """
    if matches["winner"].isna().any():
        matches = matches[matches["winner"].notna()].copy()

    rows = []
    for side, other in ((0, 1), (1, 0)):
        part = pd.DataFrame(
            {
                "episode_id": matches["episode_id"].astype(int),
                "player": side,
                "win": (matches["winner"].astype(int) == side).astype(int),
                "own_arch": matches[f"arch{side}"],
                "opp_arch": matches[f"arch{other}"],
                "own_deck": matches[f"deck{side}"],
                "own_len": matches[f"deck{side}_len"],
                "own_unique": matches[f"unique{side}"],
                "own_ace": matches[f"ace{side}"],
                "is_first": (matches["first_player"] == side).astype("float"),
                "source_group": matches["source_group"],
            }
        )
        for kind in KIND_FEATURES:
            part[f"own_{kind}"] = matches[f"kind{side}_{kind}"] / matches[f"deck{side}_len"].clip(lower=1)
        rows.append(part)

    frame = pd.concat(rows, ignore_index=True)
    frame["is_first"] = frame["is_first"].where(matches["first_player"].notna().repeat(2).to_numpy())
    frame["is_first"] = frame["is_first"].fillna(0.5)
    frame["matchup"] = frame["own_arch"] + " vs " + frame["opp_arch"]
    return frame


def add_key_card_features(
    frame: pd.DataFrame,
    cards: pd.DataFrame | None = None,
    top_n: int = 20,
) -> tuple[pd.DataFrame, list[str]]:
    """Add counts of the most common non-energy cards as model features.

    This is a convenience wrapper around :func:`top_non_energy_cards` and
    :func:`add_card_count_columns`.  For a strict pipeline, learn the card
    vocabulary on the training split and then apply it to both splits.
    """
    cards = load_cards() if cards is None else cards
    top_ids = top_non_energy_cards(frame, cards=cards, top_n=top_n)
    return add_card_count_columns(frame, top_ids, cards=cards)


def top_non_energy_cards(
    frame: pd.DataFrame,
    cards: pd.DataFrame | None = None,
    top_n: int = 20,
) -> list[int]:
    """Return the ``top_n`` most frequently played non-energy card ids."""
    cards = load_cards() if cards is None else cards
    energy_ids = set(cards.loc[cards["kind_group"].isin(["basic_energy", "special_energy"]), "card_id"])
    counter: Counter[int] = Counter()
    for value in frame["own_deck"]:
        for card_id, count in Counter(parse_deck(value)).items():
            if card_id not in energy_ids:
                counter[card_id] += count
    return [card_id for card_id, _ in counter.most_common(top_n)]


def add_card_count_columns(
    frame: pd.DataFrame,
    card_ids: list[int],
    cards: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Add one count column per card id (in place on a copy)."""
    cards = load_cards() if cards is None else cards
    name_map = cards.drop_duplicates("card_id").set_index("card_id")["name"].to_dict()
    frame = frame.copy()
    features = []
    for card_id in card_ids:
        column = f"card_{card_id}"
        frame[column] = frame["own_deck"].map(
            lambda value, cid=card_id: Counter(parse_deck(value)).get(cid, 0)
        )
        features.append(column)
    frame.attrs["key_card_names"] = {f"card_{cid}": name_map.get(cid, str(cid)) for cid in card_ids}
    return frame, features


def split_by_episode(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    target: str | None = "win",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group-aware train/test split (episodes are the groups).

    Pass ``target=None`` for a match-level frame that has no ``win`` column.
    """
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_index, test_index = next(splitter.split(frame, groups=frame["episode_id"]))
    train = frame.iloc[train_index].copy()
    test = frame.iloc[test_index].copy()
    if target is not None and target in frame:
        # Guard against a pathological split: both classes must be present.
        if train[target].nunique() < 2 or test[target].nunique() < 2:
            raise ValueError("Split produced a single-class train or test set; change random_state.")
    return train, test


def summarise_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Small, notebook-friendly summary of the raw table."""
    summary = pd.DataFrame(
        {
            "value": [
                len(matches),
                matches["episode_id"].nunique(),
                matches["source_group"].nunique(),
                matches["arch0"].nunique(),
                matches["winner"].notna().sum(),
                matches["winner"].isna().sum(),
                matches["deck0"].map(lambda value: len(parse_deck(value))).eq(60).sum(),
                matches["deck1"].map(lambda value: len(parse_deck(value))).eq(60).sum(),
                matches["first_player"].notna().sum(),
            ]
        },
        index=[
            "matches",
            "unique episode ids",
            "replay source groups",
            "archetypes (player 0)",
            "matches with a decisive result",
            "matches without a decisive result",
            "valid 60-card decks (player 0)",
            "valid 60-card decks (player 1)",
            "matches with first-player information",
        ],
    )
    return summary
