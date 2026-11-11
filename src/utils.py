"""Small shared helpers: confidence intervals, plotting and submission proofs.

The plotting helpers return ``(fig, ax)`` so the notebook stays readable and
the same figure style is used everywhere.  They deliberately avoid seaborn so
the notebook only needs matplotlib.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
]


def set_style() -> None:
    """Apply a compact, colour-blind friendly style."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
        }
    )


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (better for small n)."""
    if n <= 0:
        return (np.nan, np.nan)
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(
    values: np.ndarray | pd.Series,
    n_boot: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a mean."""
    rng = np.random.default_rng(random_state)
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (np.nan, np.nan)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return (float(np.quantile(draws, alpha)), float(np.quantile(draws, 1 - alpha)))


def save_figure(fig: plt.Figure, path: str | Path) -> Path:
    """Save and return the path (parent directories are created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------
def plot_archetype_share(matches: pd.DataFrame, top_n: int = 15, ax=None):
    """Horizontal bar chart of deck appearances (both players)."""
    counts = pd.concat([matches["arch0"], matches["arch1"]]).value_counts().head(top_n)
    ax = ax or plt.subplots(figsize=(8, 4.5))[1]
    y = np.arange(len(counts))[::-1]
    ax.barh(y, counts.values, color=PALETTE[0])
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index)
    for y_pos, value in zip(y, counts.values):
        ax.text(value, y_pos, f" {value}", va="center", fontsize=9)
    ax.set_xlabel("Deck appearances (both players)")
    ax.set_title(f"Top {top_n} archetypes in the replay corpus")
    return ax.figure, ax


def plot_winrate_ci(matches: pd.DataFrame, min_games: int = 30, top_n: int = 15, ax=None):
    """Win rate per archetype with Wilson 95% intervals."""
    rows = []
    for side, other in ((0, 1), (1, 0)):
        frame = matches[matches["winner"].notna()][[f"arch{side}", "winner"]].copy()
        frame.columns = ["archetype", "winner"]
        frame["win"] = (frame["winner"] == side).astype(int)
        rows.append(frame[["archetype", "win"]])
    frame = pd.concat(rows, ignore_index=True)
    summary = (
        frame.groupby("archetype")["win"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "win_rate", "size": "games"})
    )
    summary = summary[summary["games"] >= min_games].sort_values("win_rate", ascending=False).head(top_n)
    intervals = [wilson_ci(int(round(row.win_rate * row.games)), int(row.games)) for row in summary.itertuples()]
    lower = np.array([item[0] for item in intervals])
    upper = np.array([item[1] for item in intervals])

    ax = ax or plt.subplots(figsize=(8, 4.5))[1]
    y = np.arange(len(summary))[::-1]
    ax.errorbar(
        summary["win_rate"].values,
        y,
        xerr=[summary["win_rate"].values - lower, upper - summary["win_rate"].values],
        fmt="o",
        color=PALETTE[2],
        ecolor=PALETTE[7],
        capsize=3,
        markersize=6,
    )
    ax.axvline(0.5, color="#888888", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{name} (n={int(n)})" for name, n in zip(summary.index, summary["games"])])
    ax.set_xlim(0.25, 0.75)
    ax.set_xlabel("Win rate (95% Wilson interval)")
    ax.set_title("Archetype win rates (min. 30 appearances)")
    return ax.figure, ax


def plot_matchup_heatmap(
    matches: pd.DataFrame,
    min_games: int = 20,
    top_n: int = 12,
    ax=None,
):
    """Win rate of row archetype against column archetype."""
    popular = pd.concat([matches["arch0"], matches["arch1"]]).value_counts().head(top_n).index
    data = matches[matches["winner"].notna()].copy()
    data = data[data["arch0"].isin(popular) & data["arch1"].isin(popular)]
    matrix = pd.DataFrame(np.nan, index=popular, columns=popular, dtype=float)
    counts = pd.DataFrame(0, index=popular, columns=popular, dtype=int)
    for row in data.itertuples():
        counts.loc[row.arch0, row.arch1] += 1
        counts.loc[row.arch1, row.arch0] += 1

    wins = pd.DataFrame(0, index=popular, columns=popular, dtype=int)
    for row in data.itertuples():
        if row.winner == 0:
            wins.loc[row.arch0, row.arch1] += 1
        else:
            wins.loc[row.arch1, row.arch0] += 1
    win_rate = wins / counts.replace(0, np.nan)
    win_rate = win_rate.mask(counts < min_games)

    ax = ax or plt.subplots(figsize=(9, 7))[1]
    image = ax.imshow(win_rate.values, cmap="coolwarm", vmin=0.3, vmax=0.7)
    ax.set_xticks(range(len(popular)))
    ax.set_xticklabels(popular, rotation=45, ha="right")
    ax.set_yticks(range(len(popular)))
    ax.set_yticklabels(popular)
    ax.set_title(f"Matchup win rate (row vs column, n >= {min_games})")
    ax.grid(False)
    for i in range(len(popular)):
        for j in range(len(popular)):
            value = win_rate.values[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="row win rate")
    return ax.figure, ax


def plot_card_kind_mix(matches: pd.DataFrame, top_n: int = 10, ax=None):
    """Average deck composition (card kinds) for the most common archetypes."""
    from .data import KIND_FEATURES

    counts = pd.concat([matches["arch0"], matches["arch1"]]).value_counts().head(top_n)
    rows = []
    for side in (0, 1):
        frame = matches[[f"arch{side}"] + [f"kind{side}_{kind}" for kind in KIND_FEATURES]].copy()
        frame.columns = ["archetype"] + KIND_FEATURES
        rows.append(frame)
    frame = pd.concat(rows, ignore_index=True)
    frame = frame[frame["archetype"].isin(counts.index)]
    means = frame.groupby("archetype")[KIND_FEATURES].mean().loc[counts.index]

    ax = ax or plt.subplots(figsize=(9, 4.8))[1]
    bottom = np.zeros(len(means))
    for index, kind in enumerate(KIND_FEATURES):
        values = means[kind].values
        ax.bar(means.index, values, bottom=bottom, label=kind.replace("_", " "), color=PALETTE[index % len(PALETTE)])
        bottom += values
    ax.set_ylabel("Average cards per deck")
    ax.set_title("Deck composition by card kind")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig = ax.figure
    fig.tight_layout()
    return fig, ax


def plot_top_cards(cards: pd.DataFrame, top_cards: pd.DataFrame, ax=None):
    """Bar chart of the most used non-energy cards."""
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    y = np.arange(len(top_cards))[::-1]
    ax.barh(y, top_cards["copies"].values, color=PALETTE[4])
    ax.set_yticks(y)
    ax.set_yticklabels(top_cards["name"].values)
    ax.set_xlabel("Total copies in the corpus")
    ax.set_title("Most played non-energy cards")
    return ax.figure, ax


def plot_calibration(table: pd.DataFrame, ax=None):
    """Calibration curve for the baseline model."""
    ax = ax or plt.subplots(figsize=(5, 4.5))[1]
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="perfect")
    ax.plot(table["mean_probability"], table["win_rate"], marker="o", color=PALETTE[3], label="baseline")
    ax.set_xlabel("Predicted P(win)")
    ax.set_ylabel("Observed win rate")
    ax.set_title("Calibration on the held-out episodes")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    return ax.figure, ax


# ---------------------------------------------------------------------------
# Submission proof
# ---------------------------------------------------------------------------
SUBMISSION_PROOF_COLUMNS = [
    "ref",
    "fileName",
    "date",
    "description",
    "status",
    "publicScore",
]


def load_submission_history(path: str | Path = "submissions/kaggle_submissions.csv") -> pd.DataFrame:
    """Load the small, committed summary of Kaggle submissions."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Submission history not found: {path}")
    frame = pd.read_csv(path)
    return frame


def render_submission_proof(
    history: pd.DataFrame,
    output: str | Path = "submissions/kaggle_submissions.png",
    title: str = "Kaggle submissions — Attention Is All You Mew (GROUP05)",
) -> Path:
    """Render a readable proof image from the Kaggle CLI history.

    This is a faithful table of the ``My Submissions`` page, captured with the
    Kaggle CLI.  It is committed so the grader can see the score without
    logging into the team account.
    """
    frame = history.copy()
    frame["publicScore"] = frame["publicScore"].map(lambda value: f"{value:.1f}" if pd.notna(value) and value != "" else "—")
    frame["description"] = frame["description"].map(lambda value: textwrap.shorten(str(value), width=48, placeholder="…"))
    frame = frame.head(10)
    fig, ax = plt.subplots(figsize=(13, 0.38 * len(frame) + 1.4))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
    table = ax.table(
        cellText=frame[SUBMISSION_PROOF_COLUMNS].values,
        colLabels=["ref", "file", "date", "description", "status", "public"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.09, 0.20, 0.09, 0.39, 0.10, 0.08],
        loc="upper left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.35)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if row == 0:
            cell.set_facecolor("#F2F2F2")
            cell.set_text_props(weight="bold", fontsize=7.5)
        elif column == 5:
            cell.set_text_props(weight="bold", color="#1A6B3C")
    fig.text(
        0.01,
        0.01,
        "Source: kaggle competitions submissions -c the-pokemon-company-ptcg-ai-battle-challenge-playground",
        fontsize=7,
        color="#666666",
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=180)
    return output
