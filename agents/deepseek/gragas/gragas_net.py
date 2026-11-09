"""Pure-NumPy inference for the offline-trained option-ranking policy prior."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import gragas_features as F

BOT_DIR = Path(__file__).resolve().parent
_MODEL = None
_MISSING = False
NEG_INF = -1e9


def _candidates() -> list[Path]:
    out = []
    env = os.environ.get("TOP2_POLICY_MODEL")
    if env:
        out.append(Path(env))
    out.extend(
        [
            BOT_DIR / "policy_model.npz",
            BOT_DIR.parent / "models" / "policy_model.npz",
            Path("/kaggle_simulations/agent/policy_model.npz"),
        ]
    )
    return out


class PolicyPrior:
    def __init__(self, path: Path) -> None:
        data = np.load(path)
        self.mean = data["state_mean"].astype(np.float32)
        self.std = data["state_std"].astype(np.float32)
        self.omean = data["option_mean"].astype(np.float32)
        self.ostd = data["option_std"].astype(np.float32)
        self.embedding = data["card_embedding.weight"].astype(np.float32)
        self.w_state = data["state_proj.weight"].astype(np.float32)
        self.b_state = data["state_proj.bias"].astype(np.float32)
        self.w_option = data["option_proj.weight"].astype(np.float32)
        self.b_option = data["option_proj.bias"].astype(np.float32)
        self.w_q = data["q.weight"].astype(np.float32)
        self.b_q = data["q.bias"].astype(np.float32)
        self.w_k = data["k.weight"].astype(np.float32)
        self.b_k = data["k.bias"].astype(np.float32)
        self.w_v = data["v.weight"].astype(np.float32)
        self.b_v = data["v.bias"].astype(np.float32)
        self.w_head = data["head.0.weight"].astype(np.float32)
        self.b_head = data["head.0.bias"].astype(np.float32)
        self.w_out = data["head.3.weight"].astype(np.float32)
        self.b_out = data["head.3.bias"].astype(np.float32)
        self.path = path

    def logits(self, obs: dict, deck: list[int], profile: str = "grass") -> np.ndarray:
        state = F.state_vector(obs, deck, profile=profile)
        options, mask = F.options_matrix(obs)
        card_ids = F.card_ids_matrix(obs)
        return self.logits_arrays(state, options, card_ids, mask)

    def logits_arrays(self, state: np.ndarray, options: np.ndarray,
                      card_ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if state.shape[0] != self.mean.shape[0] or options.shape[1] != self.omean.shape[0]:
            return np.full(options.shape[0], NEG_INF, dtype=np.float32)

        s = (state - self.mean) / (self.std + 1e-8)
        o = (options - self.omean) / (self.ostd + 1e-8)
        actor = self.embedding[card_ids[:, 0].astype(np.int64)]
        target = self.embedding[card_ids[:, 1].astype(np.int64)]
        inp = np.concatenate([o, actor, target], axis=-1)
        tokens = inp @ self.w_option.T + self.b_option
        state_token = s @ self.w_state.T + self.b_state
        q = state_token @ self.w_q.T + self.b_q
        k = tokens @ self.w_k.T + self.b_k
        v = tokens @ self.w_v.T + self.b_v

        scores = (q @ k.T) / np.sqrt(tokens.shape[-1])
        scores = np.where(mask > 0, scores, NEG_INF)
        scores = scores - scores.max()
        weights = np.exp(scores)
        weights = np.where(mask > 0, weights, 0.0)
        total = weights.sum()
        weights = weights / total if total > 0 else weights
        context = weights @ v
        joint = np.concatenate([tokens, np.broadcast_to(context[None, :], tokens.shape)], axis=-1)
        hidden = np.maximum(0.0, joint @ self.w_head.T + self.b_head)
        logits = (hidden @ self.w_out.T + self.b_out).reshape(-1)
        logits = np.where(mask > 0, logits, NEG_INF)
        return logits.astype(np.float32)


def get_prior() -> PolicyPrior | None:
    global _MODEL, _MISSING
    if _MISSING:
        return None
    if _MODEL is None:
        for path in _candidates():
            if path.is_file():
                try:
                    _MODEL = PolicyPrior(path)
                    return _MODEL
                except Exception:
                    continue
        _MISSING = True
    return _MODEL


def prior_logits(obs: dict, deck: list[int], profile: str = "grass") -> np.ndarray | None:
    model = get_prior()
    if model is None:
        return None
    try:
        return model.logits(obs, deck, profile=profile)
    except Exception:
        return None
