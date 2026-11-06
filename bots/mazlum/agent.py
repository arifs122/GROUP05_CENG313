"""Mazlum — pure NumPy inference for the attention-based BC policy.

No PyTorch here: the weights come from ``data/mazlum/model.npz`` and the forward
pass is written by hand.

Pipeline per decision:

1. ``features.state_vector`` / ``options_matrix`` / ``card_ids_matrix``.
2. Standardise with the training statistics: ``(x - mean) / (std + 1e-8)``.
3. Dot-product attention: state query × option keys -> masked softmax ->
   context vector, then a small ReLU head gives one logit per option.
4. **Lethal filter**: if a legal option has ``attack_lethal == 1`` it wins
   outright, regardless of the network logits.
5. ``np.where(mask == 0, -1e9, logits)`` -> argmax (or the lethal index).

The bot folder is self-contained except for ``features.py``: it is imported
from ``src/mazlum`` when running inside the repo, or from the bundle root when
packaged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

BOT_DIR = Path(__file__).resolve().parent
ROOT = BOT_DIR.parents[1]

MODEL_ENV = "MAZLUM_MODEL"
NORMALIZER_KEYS = ("state_mean", "state_std", "option_mean", "option_std")
NEG_INF = -1e9


def _import_features():
    """Import the shared featurizer from the repo or from a bundle."""
    candidates = (BOT_DIR, ROOT / "src" / "mazlum", Path("/kaggle_simulations/agent"))
    for candidate in candidates:
        if (candidate / "features.py").is_file() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    import features  # noqa: PLC0415 - resolved after the path setup above

    return features


def _model_candidates() -> list[Path]:
    candidates = []
    env_path = os.environ.get(MODEL_ENV)
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(BOT_DIR / "model.npz")
    candidates.append(ROOT / "data" / "mazlum" / "model.npz")
    return candidates


def load_deck(path: Path | None = None) -> list[int]:
    """Read the 60-card deck next to this file."""
    path = path or (BOT_DIR / "deck.csv")
    cards = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cards) != 60:
        raise ValueError(f"{path}: expected 60 cards, found {len(cards)}")
    return cards


def masked_softmax(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Masked softmax along the last axis; padding gets probability 0."""
    valid = mask > 0
    if not valid.any():
        return np.zeros_like(scores, dtype=np.float32)
    shifted = np.where(valid, scores, -np.inf)
    shifted = shifted - np.max(shifted[..., valid], axis=-1, keepdims=True)
    weights = np.exp(shifted)
    weights[..., ~valid] = 0.0
    total = weights.sum(axis=-1, keepdims=True)
    return np.divide(weights, np.maximum(total, 1e-12), out=np.zeros_like(weights), where=total > 0)


class MazlumPolicy:
    """NumPy port of the Mazlum policies (v9 2x2 and v19 4x4 Pre-LN) + lethal override."""

    def __init__(self, model_path: Path, features_module, deck: list[int]) -> None:
        data = np.load(model_path)
        self.state_mean = data["state_mean"].astype(np.float32)
        self.state_std = data["state_std"].astype(np.float32)
        self.option_mean = data["option_mean"].astype(np.float32)
        self.option_std = data["option_std"].astype(np.float32)
        self.weights = {
            name: data[name].astype(np.float32)
            for name in data.files
            if name not in NORMALIZER_KEYS and name != "arch"
        }
        self.features = features_module
        # Feature schema follows the checkpoint: v2 checkpoints have wider
        # projections than the legacy 37/41 dims.
        state_dim = int(self.weights["state_proj.weight"].shape[1])
        option_dim = int(self.weights["option_proj.weight"].shape[1])
        if state_dim > features_module.STATE_FEATURE_SIZE or option_dim > features_module.OPTION_FEATURE_SIZE:
            features_module.configure(2)
        self.deck = list(deck)
        self.model_path = model_path
        self.lethal_index = features_module.OPTION_FEATURE_NAMES.index("attack_lethal")

        self.is_big = "blocks.0.q.weight" in self.weights
        if self.is_big:
            self.attn_dim = int(self.weights["blocks.0.q.weight"].shape[0])
            arch = data["arch"] if "arch" in data.files else None
            self.heads = int(arch[1]) if arch is not None else 4
            self.num_blocks = len({int(n.split(".")[1]) for n in self.weights if n.startswith("blocks.")})
        else:
            self.heads = 2
            self.attn_dim = int(self.weights["q1.weight"].shape[0])
            self.num_blocks = 2
        self.head_dim = self.attn_dim // self.heads
        self.attn_scale = float(np.sqrt(self.head_dim))
        # Hybrid rule; can be disabled for ablations.
        self.use_lethal = True
        self.lethal_overrides = 0

    # -- forward pass ------------------------------------------------------
    @staticmethod
    def _linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        return x @ w.T + b

    def _attention(self, query: np.ndarray, keys: np.ndarray, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Multi-head dot-product attention; returns a flat ``(attn_dim,)`` context."""
        q = query.reshape(self.heads, self.head_dim)
        k = keys.reshape(keys.shape[0], self.heads, self.head_dim).transpose(1, 0, 2)
        v = values.reshape(values.shape[0], self.heads, self.head_dim).transpose(1, 0, 2)

        scores = np.einsum("hd,hnd->hn", q, k) / self.attn_scale  # (H, N)
        attention = masked_softmax(scores, mask)  # (H, N)
        context = np.einsum("hn,hnd->hd", attention, v)  # (H, dh)
        return context.reshape(self.attn_dim)

    def _layer_norm(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        variance = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(variance + 1e-5) * weight + bias

    def _logits_big(self, option_input: np.ndarray, state: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """v19 forward: 4 Pre-LN blocks x 4 heads, d=80."""
        w = self.weights
        tokens = self._linear(option_input, w["option_proj.weight"], w["option_proj.bias"])
        state_token = self._linear(state, w["state_proj.weight"], w["state_proj.bias"])
        carry = np.zeros_like(state_token)

        for index in range(self.num_blocks):
            p = f"blocks.{index}."
            token_norm = self._layer_norm(tokens, w[p + "ln_1.weight"], w[p + "ln_1.bias"])
            query = self._linear(
                self._layer_norm(state_token + carry, w[p + "ln_q.weight"], w[p + "ln_q.bias"]),
                w[p + "q.weight"],
                w[p + "q.bias"],
            )
            context = self._attention(
                query,
                self._linear(token_norm, w[p + "k.weight"], w[p + "k.bias"]),
                self._linear(token_norm, w[p + "v.weight"], w[p + "v.bias"]),
                mask,
            )
            carry = carry + self._linear(context, w[p + "o.weight"], w[p + "o.bias"])

            ffn_input = self._layer_norm(tokens, w[p + "ln_2.weight"], w[p + "ln_2.bias"])
            ffn_hidden = np.maximum(0.0, self._linear(ffn_input, w[p + "ffn.0.weight"], w[p + "ffn.0.bias"]))
            tokens = tokens + self._linear(ffn_hidden, w[p + "ffn.2.weight"], w[p + "ffn.2.bias"])

        joint = np.concatenate([tokens, np.broadcast_to(carry, tokens.shape)], axis=1)
        hidden = np.maximum(0.0, self._linear(joint, w["head.0.weight"], w["head.0.bias"]))
        return self._linear(hidden, w["head.2.weight"], w["head.2.bias"]).ravel()

    def logits(self, state: np.ndarray, options: np.ndarray, card_ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return one score per option (no final masking applied)."""
        state = (state - self.state_mean) / (self.state_std + 1e-8)
        options = (options - self.option_mean) / (self.option_std + 1e-8)

        w = self.weights
        table = w["card_embedding.weight"]
        actor_emb = table[card_ids[:, 0].astype(np.int64)]
        target_emb = table[card_ids[:, 1].astype(np.int64)]
        option_input = np.concatenate([options, actor_emb, target_emb], axis=1)  # (N, 104)

        if self.is_big:
            return self._logits_big(option_input, state, mask)

        tokens = self._linear(option_input, w["option_proj.weight"], w["option_proj.bias"])  # (N,64)
        state_token = self._linear(state, w["state_proj.weight"], w["state_proj.bias"])  # (64,)

        # Block 1.
        context1 = self._attention(
            self._linear(state_token, w["q1.weight"], w["q1.bias"]),
            self._linear(tokens, w["k1.weight"], w["k1.bias"]),
            self._linear(tokens, w["v1.weight"], w["v1.bias"]),
            mask,
        )

        # Per-token FFN with residual.
        ffn_hidden = np.maximum(0.0, self._linear(tokens, w["ffn.0.weight"], w["ffn.0.bias"]))
        tokens = tokens + self._linear(ffn_hidden, w["ffn.2.weight"], w["ffn.2.bias"])

        # Block 2.
        query2 = self._linear(
            np.concatenate([state_token, context1]), w["q2.weight"], w["q2.bias"]
        )
        context2 = self._attention(
            query2,
            self._linear(tokens, w["k2.weight"], w["k2.bias"]),
            self._linear(tokens, w["v2.weight"], w["v2.bias"]),
            mask,
        )

        joint = np.concatenate(
            [
                tokens,
                np.broadcast_to(context1, tokens.shape),
                np.broadcast_to(context2, tokens.shape),
            ],
            axis=1,
        )  # (N, 192)
        hidden = np.maximum(0.0, self._linear(joint, w["head.0.weight"], w["head.0.bias"]))
        return self._linear(hidden, w["head.2.weight"], w["head.2.bias"]).ravel()

    # -- decision ----------------------------------------------------------
    def act(self, obs_dict: dict) -> list[int]:
        select = obs_dict.get("select")
        if select is None:
            # Initial deck selection.
            return list(self.deck)

        options = select.get("option") or []
        if not options:
            return []

        min_count = max(0, int(select.get("minCount", 0)))
        max_count = min(int(select.get("maxCount", 0)), len(options))
        if max_count <= 0:
            return []

        state = self.features.state_vector(obs_dict, self.deck)
        option_matrix, mask = self.features.options_matrix(obs_dict)
        card_ids = self.features.card_ids_matrix(obs_dict)

        logits = self.logits(state, option_matrix, card_ids, mask)
        logits = np.where(mask == 0, NEG_INF, logits)

        count = max(1, min_count)
        count = min(count, max_count)

        # Hybrid policy: a legal lethal attack overrides the network entirely.
        if count == 1 and self.use_lethal:
            lethal = (option_matrix[:, self.lethal_index] > 0.5) & (mask > 0)
            lethal_indices = np.nonzero(lethal)[0]
            if lethal_indices.size:
                best = lethal_indices[int(np.argmax(logits[lethal_indices]))]
                self.lethal_overrides += 1
                return [int(best)]

        chosen: list[int] = []
        for index in np.argsort(-logits):
            if mask[index] == 0:
                break
            chosen.append(int(index))
            if len(chosen) >= count:
                break

        if len(chosen) < min_count:
            for index in range(len(options)):
                if index not in chosen:
                    chosen.append(index)
                if len(chosen) >= min_count:
                    break
        return chosen[:max_count]


_POLICY: MazlumPolicy | None = None


def get_policy() -> MazlumPolicy:
    """Load the model and deck once per process."""
    global _POLICY
    if _POLICY is None:
        model_path = next((path for path in _model_candidates() if path.is_file()), None)
        if model_path is None:
            searched = "\n  ".join(str(path) for path in _model_candidates())
            raise FileNotFoundError(
                "Mazlum model.npz not found. Run src/mazlum/train_bc.py or set MAZLUM_MODEL.\n"
                f"Searched:\n  {searched}"
            )
        _POLICY = MazlumPolicy(model_path, _import_features(), load_deck())
    return _POLICY


def agent(obs_dict: dict) -> list[int]:
    """Return the indices of the options Mazlum selects for this decision."""
    return get_policy().act(obs_dict)
