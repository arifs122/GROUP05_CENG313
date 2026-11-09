"""Pure-NumPy inference for the offline-trained state value net."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import top2_core_features as F

BOT_DIR = Path(__file__).resolve().parent
_MODEL = None
_MISSING = False


def _candidates() -> list[Path]:
    out = []
    env = os.environ.get("TOP2_VALUE_MODEL")
    if env:
        out.append(Path(env))
    out.extend(
        [
            BOT_DIR / "value_model.npz",
            BOT_DIR.parent / "models" / "value_model.npz",
            Path("/kaggle_simulations/agent/value_model.npz"),
        ]
    )
    return out


class ValueModel:
    def __init__(self, path: Path) -> None:
        data = np.load(path)
        self.mean = data["state_mean"].astype(np.float32)
        self.std = data["state_std"].astype(np.float32)
        self.w0 = data["w0"].astype(np.float32)
        self.b0 = data["b0"].astype(np.float32)
        self.w1 = data["w1"].astype(np.float32)
        self.b1 = data["b1"].astype(np.float32)
        self.w2 = data["w2"].astype(np.float32)
        self.b2 = data["b2"].astype(np.float32)
        self.path = path

    def logit(self, state: np.ndarray) -> float:
        if state.shape[0] != self.mean.shape[0]:
            return 0.0
        x = (state - self.mean) / (self.std + 1e-8)
        h = np.maximum(0.0, x @ self.w0.T + self.b0)
        h = np.maximum(0.0, h @ self.w1.T + self.b1)
        return float((h @ self.w2.T + self.b2).reshape(-1)[0])

    def prob(self, state: np.ndarray) -> float:
        z = self.logit(state)
        if z >= 0:
            return float(1.0 / (1.0 + np.exp(-z)))
        e = float(np.exp(z))
        return e / (1.0 + e)


def get_model() -> ValueModel | None:
    global _MODEL, _MISSING
    if _MISSING:
        return None
    if _MODEL is None:
        for path in _candidates():
            if path.is_file():
                try:
                    _MODEL = ValueModel(path)
                    return _MODEL
                except Exception:
                    continue
        _MISSING = True
    return _MODEL


def value_logit(obs: dict, deck: list[int]) -> float:
    model = get_model()
    if model is None:
        return 0.0
    try:
        return model.logit(F.state_vector(obs, deck))
    except Exception:
        return 0.0


def value_prob(obs: dict, deck: list[int]) -> float:
    model = get_model()
    if model is None:
        return 0.5
    try:
        return model.prob(F.state_vector(obs, deck))
    except Exception:
        return 0.5
