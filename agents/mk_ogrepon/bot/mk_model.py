"""Numpy inference for the small Top1 policy prior."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import mk_features as F
import mk_state as S

BOT_DIR = Path(__file__).resolve().parent


class PolicyPrior:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (BOT_DIR / "policy_prior.npz")
        data = np.load(self.path)
        self.w0 = np.asarray(data["net.0.weight"], dtype=np.float32)
        self.b0 = np.asarray(data["net.0.bias"], dtype=np.float32)
        self.w3 = np.asarray(data["net.3.weight"], dtype=np.float32)
        self.b3 = np.asarray(data["net.3.bias"], dtype=np.float32)
        self.w6 = np.asarray(data["net.6.weight"], dtype=np.float32)
        self.b6 = np.asarray(data["net.6.bias"], dtype=np.float32)

    def logits(self, obs: dict) -> np.ndarray:
        state = np.asarray(F.state_features(obs), dtype=np.float32)
        opts = [np.asarray(F.option_features(obs, o), dtype=np.float32)
                for o in S.options(obs)]
        if not opts:
            return np.zeros(0, dtype=np.float32)
        options = np.stack(opts)
        st = np.repeat(state[None, :], len(opts), axis=0)
        x = np.concatenate([st, options], axis=-1)
        h1 = np.maximum(0.0, x @ self.w0.T + self.b0)
        h2 = np.maximum(0.0, h1 @ self.w3.T + self.b3)
        return (h2 @ self.w6.T + self.b6).reshape(-1)
