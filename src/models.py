"""Baseline models and evaluation helpers.

The competition metric is a match result (win / loss in a best-of-three
series).  For the tabular baseline we therefore model

    P(the player wins | own deck, opponent deck, first/second)

from historical episodes.  Both player perspectives are used, and the
train/test split is grouped by episode so a match cannot leak across folds.

The second baseline is the rule-based agent in ``src/agent/policy.py``.  It is
evaluated separately with the local arena because an agent's score is a win
rate, not a row-level prediction metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_FEATURES = ["own_arch", "opp_arch", "matchup"]
NUMERIC_FEATURES = [
    "is_first",
    "own_unique",
    "own_ace",
    "own_pokemon",
    "own_item",
    "own_supporter",
    "own_tool",
    "own_stadium",
    "own_basic_energy",
    "own_special_energy",
    "own_other",
]


def feature_columns(frame: pd.DataFrame, key_card_features: list[str] | None = None) -> tuple[list[str], list[str]]:
    """Return ``(categorical, numeric)`` columns that actually exist."""
    categorical = [column for column in CATEGORICAL_FEATURES if column in frame]
    numeric = [column for column in NUMERIC_FEATURES if column in frame]
    if key_card_features:
        numeric = numeric + [column for column in key_card_features if column in frame]
    return categorical, numeric


def make_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    """One-hot encode deck/matchup labels and scale the numeric block."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10),
                categorical,
            ),
            ("numeric", StandardScaler(), numeric),
        ],
        remainder="drop",
    )


def build_logistic(
    frame: pd.DataFrame,
    key_card_features: list[str] | None = None,
) -> Pipeline:
    """Simple, regularised logistic-regression baseline."""
    categorical, numeric = feature_columns(frame, key_card_features)
    return Pipeline(
        [
            ("preprocess", make_preprocessor(categorical, numeric)),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def build_dummy(frame: pd.DataFrame) -> Pipeline:
    """Reference model that always predicts the training prior."""
    categorical, numeric = feature_columns(frame)
    return Pipeline(
        [
            ("preprocess", make_preprocessor(categorical, numeric)),
            ("model", DummyClassifier(strategy="prior")),
        ]
    )


def predict_probability(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Positive-class probability, robust to the dummy classifier's output."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)
        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            return probabilities[:, 1]
        return np.asarray(probabilities).ravel()
    return np.asarray(model.predict(frame), dtype=float)


@dataclass
class Evaluation:
    """Container for row-level metrics and the raw predictions."""

    name: str
    accuracy: float
    roc_auc: float
    log_loss: float
    brier: float
    match_accuracy: float
    n_rows: int
    n_matches: int
    predictions: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def as_dict(self) -> dict:
        return {
            "model": self.name,
            "row accuracy": self.accuracy,
            "ROC-AUC": self.roc_auc,
            "log loss": self.log_loss,
            "Brier": self.brier,
            "match winner accuracy": self.match_accuracy,
            "rows": self.n_rows,
            "matches": self.n_matches,
        }


def match_level_accuracy(predictions: pd.DataFrame) -> float:
    """Pick the side with the higher predicted win probability per match.

    ``predictions`` needs ``episode_id``, ``player``, ``win`` and ``p_win``.
    The two perspectives of a match normally sum to one; ties (exactly 0.5 or
    a missing perspective) score 0.5.
    """
    scores: list[float] = []
    for _, group in predictions.groupby("episode_id", sort=False):
        if len(group) != 2:
            # Only one perspective available: fall back to the row prediction.
            scores.extend((group["p_win"] >= 0.5).astype(float).eq(group["win"]).astype(float))
            continue
        group = group.sort_values("player")
        p0 = float(group.iloc[0]["p_win"])
        p1 = float(group.iloc[1]["p_win"])
        winner = 0 if int(group.iloc[0]["win"]) == 1 else 1
        if abs(p0 - p1) < 1e-12:
            scores.append(0.5)
        else:
            predicted_winner = 0 if p0 > p1 else 1
            scores.append(float(predicted_winner == winner))
    return float(np.mean(scores)) if scores else float("nan")


def evaluate_model(
    model: Pipeline,
    frame: pd.DataFrame,
    name: str,
    threshold: float = 0.5,
) -> Evaluation:
    """Score a fitted model on a held-out player-perspective frame."""
    probabilities = predict_probability(model, frame)
    predictions = frame[["episode_id", "player", "win"]].copy()
    predictions["p_win"] = probabilities
    predictions["predicted"] = (probabilities >= threshold).astype(int)

    try:
        auc = float(roc_auc_score(frame["win"], probabilities))
    except ValueError:
        auc = float("nan")
    try:
        loss = float(log_loss(frame["win"], probabilities, labels=[0, 1]))
    except ValueError:
        loss = float("nan")

    return Evaluation(
        name=name,
        accuracy=float(accuracy_score(frame["win"], predictions["predicted"])),
        roc_auc=auc,
        log_loss=loss,
        brier=float(brier_score_loss(frame["win"], probabilities)),
        match_accuracy=match_level_accuracy(predictions),
        n_rows=len(frame),
        n_matches=int(frame["episode_id"].nunique()),
        predictions=predictions,
    )


def fit_and_evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    key_card_features: list[str] | None = None,
) -> dict[str, Evaluation]:
    """Fit the dummy and logistic baselines and return their evaluations."""
    dummy = build_dummy(train)
    dummy.fit(train, train["win"])
    logistic = build_logistic(train, key_card_features)
    logistic.fit(train, train["win"])
    return {
        "dummy": evaluate_model(dummy, test, "Dummy (prior)"),
        "logistic": evaluate_model(logistic, test, "Logistic regression"),
        "fitted": logistic,
    }


def coefficient_table(model: Pipeline, top_n: int = 15) -> pd.DataFrame:
    """Readable one-hot coefficients from the fitted logistic pipeline."""
    preprocess: ColumnTransformer = model.named_steps["preprocess"]
    classifier: LogisticRegression = model.named_steps["model"]
    names = preprocess.get_feature_names_out()
    coefficients = classifier.coef_[0]
    table = pd.DataFrame({"feature": names, "coefficient": coefficients})
    table["abs_coefficient"] = table["coefficient"].abs()
    return table.sort_values("abs_coefficient", ascending=False).head(top_n).reset_index(drop=True)


def calibration_table(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Bin predictions and compare mean probability with observed win rate."""
    frame = predictions.copy()
    frame["bin"] = pd.cut(frame["p_win"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    table = (
        frame.groupby("bin", observed=True)
        .agg(mean_probability=("p_win", "mean"), win_rate=("win", "mean"), n=("win", "size"))
        .reset_index()
    )
    table["bin_mid"] = table["bin"].apply(lambda interval: (interval.left + interval.right) / 2)
    return table
