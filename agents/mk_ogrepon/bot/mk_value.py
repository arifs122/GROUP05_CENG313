"""Numpy inference for the public-state value net."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import mk_features as F

BOT_DIR = Path(__file__).resolve().parent


class ValueModel:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (BOT_DIR / "value_model.npz")
        data = np.load(self.path)
        self.w0 = np.asarray(data["net.0.weight"], dtype=np.float32)
        self.b0 = np.asarray(data["net.0.bias"], dtype=np.float32)
        self.w3 = np.asarray(data["net.3.weight"], dtype=np.float32)
        self.b3 = np.asarray(data["net.3.bias"], dtype=np.float32)
        self.w6 = np.asarray(data["net.6.weight"], dtype=np.float32)
        self.b6 = np.asarray(data["net.6.bias"], dtype=np.float32)

    def win_prob(self, obs: dict) -> float:
        x = np.asarray(F.state_features(obs), dtype=np.float32)
        h = np.maximum(0.0, self.w0 @ x + self.b0)
        h = np.maximum(0.0, self.w3 @ h + self.b3)
        logit = float(self.w6 @ h + self.b6)
        if logit >= 0:
            z = np.exp(-logit)
            return float(1.0 / (1.0 + z))
        z = np.exp(logit)
        return float(z / (1.0 + z))
