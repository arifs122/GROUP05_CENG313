"""Small PyTorch models used only for offline training.

The exported weights are consumed by pure-NumPy runtime modules in
``agents/deepseek/top2_core``.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class PolicyNet(nn.Module):
    """State-conditioned pointer scorer with one attention block."""

    def __init__(
        self,
        state_dim: int,
        option_dim: int,
        embed_dim: int = 32,
        attn_dim: int = 64,
        hidden: int = 128,
        card_vocab: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.card_embedding = nn.Embedding(card_vocab, embed_dim, padding_idx=0)
        self.state_proj = nn.Linear(state_dim, attn_dim)
        self.option_proj = nn.Linear(option_dim + 2 * embed_dim, attn_dim)
        self.q = nn.Linear(attn_dim, attn_dim)
        self.k = nn.Linear(attn_dim, attn_dim)
        self.v = nn.Linear(attn_dim, attn_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * attn_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.attn_scale = math.sqrt(attn_dim)

    def forward(
        self,
        state: torch.Tensor,
        options: torch.Tensor,
        card_ids: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if card_ids is None:
            card_ids = torch.zeros(
                (options.shape[0], options.shape[1], 2),
                dtype=torch.long,
                device=options.device,
            )
        actor = self.card_embedding(card_ids[..., 0].long())
        target = self.card_embedding(card_ids[..., 1].long())
        tokens = self.dropout(
            self.option_proj(torch.cat([options, actor, target], dim=-1))
        )
        state_token = self.state_proj(state)
        scores = torch.matmul(self.q(state_token).unsqueeze(1), self.k(tokens).transpose(-1, -2)).squeeze(1)
        scores = scores / self.attn_scale
        if mask is not None:
            scores = scores.masked_fill(mask <= 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention.unsqueeze(1), self.v(tokens)).squeeze(1)
        joint = torch.cat([tokens, context.unsqueeze(1).expand(-1, tokens.size(1), -1)], dim=-1)
        logits = self.head(joint).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(mask <= 0, -1e9)
        return logits


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    has_label = (labels >= 0) & (mask.sum(dim=-1) > 0)
    if not bool(has_label.any()):
        return logits.sum() * 0.0
    logits = logits[has_label]
    labels = labels[has_label]
    if weight is None:
        return nn.functional.cross_entropy(logits, labels)
    losses = nn.functional.cross_entropy(logits, labels, reduction="none")
    w = weight[has_label].to(dtype=losses.dtype)
    if float(w.sum()) <= 0:
        return losses.mean()
    return (losses * w).sum() / w.sum().clamp_min(1e-8)


def compute_normalizer(state, options, mask, train_index):
    """Return state/option mean/std computed on training rows only."""
    import numpy as np

    s = state[train_index].astype(np.float32)
    s_mean = s.mean(axis=0)
    s_std = s.std(axis=0)
    s_std[s_std < 1e-4] = 1.0

    valid = np.asarray(mask[train_index] > 0)
    o = options[train_index][valid].astype(np.float32)
    if o.size == 0:
        o_mean = np.zeros(options.shape[-1], dtype=np.float32)
        o_std = np.ones(options.shape[-1], dtype=np.float32)
    else:
        o_mean = o.mean(axis=0)
        o_std = o.std(axis=0)
        o_std[o_std < 1e-4] = 1.0
    return {
        "state_mean": s_mean.astype(np.float32),
        "state_std": s_std.astype(np.float32),
        "option_mean": o_mean.astype(np.float32),
        "option_std": o_std.astype(np.float32),
    }
