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
    """NumPy port of ``model.MazlumNet`` (2 heads, 2 blocks) + lethal override."""

    def __init__(self, model_path: Path, features_module, deck: list[int]) -> None:
        data = np.load(model_path)
        self.state_mean = data["state_mean"].astype(np.float32)
        self.state_std = data["state_std"].astype(np.float32)
        self.option_mean = data["option_mean"].astype(np.float32)
        self.option_std = data["option_std"].astype(np.float32)
        self.weights = {
            name: data[name].astype(np.float32)
            for name in data.files
            if name not in NORMALIZER_KEYS
        }
        self.features = features_module
        self.deck = list(deck)
        self.model_path = model_path
        self.lethal_index = features_module.OPTION_FEATURE_NAMES.index("attack_lethal")
        # Architecture constants shared with model.MazlumNet (2 heads, d=64).
        self.heads = 2
        self.attn_dim = int(self.weights["q1.weight"].shape[0])
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

    def logits(self, state: np.ndarray, options: np.ndarray, card_ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return one score per option (no final masking applied)."""
        state = (state - self.state_mean) / (self.state_std + 1e-8)
        options = (options - self.option_mean) / (self.option_std + 1e-8)

        w = self.weights
        table = w["card_embedding.weight"]
        actor_emb = table[card_ids[:, 0].astype(np.int64)]
        target_emb = table[card_ids[:, 1].astype(np.int64)]
        option_input = np.concatenate([options, actor_emb, target_emb], axis=1)  # (N, 104)

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
    def _own_context(self, obs_dict):
        current = obs_dict.get("current") or {}
        player = int(current.get("yourIndex", 0) or 0)
        me = (current.get("players") or [{}, {}])[player] or {}
        bench = me.get("bench") or []
        damaged = 0
        dark = False
        for p in (me.get("active") or []) + bench:
            if not p:
                continue
            hp, mx = int(p.get("hp", 0) or 0), int(p.get("maxHp", 0) or 0)
            if mx and hp < mx:
                damaged += 1
            if int(p.get("id", 0) or 0) == 112:
                energies = [int(x) for x in (p.get("energies") or [])]
                if 7 in energies or any(int((c or {}).get("id", 0) or 0) == 7 for c in (p.get("energyCards") or [])):
                    dark = True
        return int(current.get("turn", 0) or 0), len(bench), damaged, dark

    def _rule_prior(self, obs_dict, options):
        turn, bench, damaged, dark = self._own_context(obs_dict)
        setup = {1086, 1152, 1121, 1227, 119, 120, 121, 1198, 1079}
        tech = {112, 1260, 1182, 1213, 235, 1071, 140, 31}
        prior = np.zeros(len(options), dtype=np.float32)
        player = int((obs_dict.get("current") or {}).get("yourIndex", 0) or 0)
        for i, o in enumerate(options):
            try:
                actor, _ = self.features._option_cards(obs_dict, player, o)
                typ = int(o.get("type", -1))
            except Exception:
                continue
            if turn <= 3:
                if actor in setup:
                    prior[i] += 0.25
                if actor in tech:
                    prior[i] -= 0.25
                if typ == 12:  # retreat
                    prior[i] -= 0.15
                if actor == 112 and (not dark or damaged == 0):
                    prior[i] -= 0.35
                if actor == 1260 and (turn <= 2 or bench < 3):
                    prior[i] -= 0.45
                if actor == 1182 and turn <= 3:
                    prior[i] -= 0.15
                if actor == 120 and typ == 10:
                    prior[i] += 0.25
        return prior

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
        prior = self._rule_prior(obs_dict, options)
        valid = mask > 0
        if int(valid.sum()) >= 2:
            vals = np.sort(logits[valid])[::-1]
            if float(vals[0] - vals[1]) >= 1.0:
                prior[:] = 0.0
        logits[:len(prior)] += prior
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
