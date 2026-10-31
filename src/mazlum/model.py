"""MazlumNet v9 — two-head attention, two blocks, ~147k params (~576 KB raw).

Architecture:

1. ``card_ids`` (40, 2) -> ``nn.Embedding`` (32) for actor + target card
2. ``state`` (73) -> ``state_proj`` (64) -> query for block 1
3. ``options`` (41) + embeddings (64) -> ``option_proj`` (64) -> tokens
4. **Block 1 (2 heads, d=32/head):** state query x option keys -> masked
   softmax -> context 1 (64)
5. Per-token FFN with residual: ``tokens = tokens + FFN(tokens)``
6. **Block 2 (2 heads):** query from ``[state_token, context1]`` over the
   enriched tokens -> context 2 (64)
7. Head: ``[token, context1, context2]`` (192) -> 128 ReLU -> one logit

Every operation is plain matmul/einsum, so ``bots/mazlum/agent.py`` reproduces
it with NumPy exactly. Parameter budget stays inside the 600 KB submission
limit.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from features import (  # noqa: E402
    CARD_ID_LIMIT,
    MAX_OPTIONS,
    OPTION_FEATURE_SIZE,
    STATE_FEATURE_SIZE,
)

NEG_INF = -1e9


def mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Push invalid options to -1e9 so they get ~0 probability."""
    valid = mask.to(dtype=torch.bool)
    return logits.masked_fill(~valid, NEG_INF)


def _split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    """(B, N, D) -> (B, heads, N, D/heads)."""
    batch, count, dim = x.shape
    return x.view(batch, count, heads, dim // heads).permute(0, 2, 1, 3)


class MazlumNet(nn.Module):
    """Two-block, two-head attention policy: one logit per legal option."""

    def __init__(
        self,
        state_dim: int = STATE_FEATURE_SIZE,
        option_dim: int = OPTION_FEATURE_SIZE,
        embed_dim: int = 32,
        attn_dim: int = 64,
        heads: int = 2,
        hidden: int = 128,
        card_vocab: int = CARD_ID_LIMIT,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.option_dim = option_dim
        self.attn_dim = attn_dim
        self.heads = heads
        self.head_dim = attn_dim // heads

        self.card_embedding = nn.Embedding(card_vocab, embed_dim, padding_idx=0)

        inner = option_dim + 2 * embed_dim
        self.state_proj = nn.Linear(state_dim, attn_dim)
        self.option_proj = nn.Linear(inner, attn_dim)

        # Block 1: state query over option tokens.
        self.q1 = nn.Linear(attn_dim, attn_dim)
        self.k1 = nn.Linear(attn_dim, attn_dim)
        self.v1 = nn.Linear(attn_dim, attn_dim)

        # Per-token feed-forward with residual.
        self.ffn = nn.Sequential(
            nn.Linear(attn_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, attn_dim),
        )

        # Block 2: query from [state_token, context1] over enriched tokens.
        self.q2 = nn.Linear(2 * attn_dim, attn_dim)
        self.k2 = nn.Linear(attn_dim, attn_dim)
        self.v2 = nn.Linear(attn_dim, attn_dim)

        # Head over [token, context1, context2].
        self.head = nn.Sequential(
            nn.Linear(3 * attn_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _attention(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Multi-head dot-product attention; returns ``(B, D)`` context."""
        q = _split_heads(query.unsqueeze(1), self.heads)  # (B, H, 1, dh)
        k = _split_heads(keys, self.heads)  # (B, H, N, dh)
        v = _split_heads(values, self.heads)  # (B, H, N, dh)

        scores = torch.matmul(q, k.transpose(-1, -2)).squeeze(-2)  # (B, H, N)
        scores = scores / math.sqrt(self.head_dim)
        if mask is not None:
            scores = mask_logits(scores, mask.unsqueeze(1))
        attention = torch.softmax(scores, dim=-1)  # (B, H, N)

        context = torch.matmul(attention.unsqueeze(-2), v).squeeze(-2)  # (B, H, dh)
        return context.reshape(context.size(0), self.attn_dim)

    def forward(
        self,
        state: torch.Tensor,
        options: torch.Tensor,
        card_ids: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits of shape ``(batch, num_options)``."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if options.dim() == 2:
            options = options.unsqueeze(0)
        if card_ids is not None and card_ids.dim() == 2:
            card_ids = card_ids.unsqueeze(0)
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)

        if card_ids is None:
            card_ids = torch.zeros(
                (options.shape[0], options.shape[1], 2),
                dtype=torch.long,
                device=options.device,
            )
        card_ids = card_ids.to(dtype=torch.long, device=options.device)

        actor_emb = self.card_embedding(card_ids[..., 0])
        target_emb = self.card_embedding(card_ids[..., 1])
        option_input = torch.cat([options, actor_emb, target_emb], dim=-1)  # (B,N,104)

        tokens = self.option_proj(option_input)  # (B,N,64)
        state_token = self.state_proj(state)  # (B,64)

        # Block 1: state attends over option tokens.
        context1 = self._attention(
            self.q1(state_token), self.k1(tokens), self.v1(tokens), mask
        )

        # Per-token FFN with residual.
        tokens = tokens + self.ffn(tokens)

        # Block 2: richer query attends over the enriched tokens.
        query2 = self.q2(torch.cat([state_token, context1], dim=-1))
        context2 = self._attention(query2, self.k2(tokens), self.v2(tokens), mask)

        c1 = context1.unsqueeze(1).expand(-1, tokens.size(1), -1)
        c2 = context2.unsqueeze(1).expand(-1, tokens.size(1), -1)
        joint = torch.cat([tokens, c1, c2], dim=-1)  # (B,N,192)

        logits = self.head(joint).squeeze(-1)  # (B,N)
        if mask is not None:
            logits = mask_logits(logits, mask)
        return logits


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross entropy over masked logits, ignoring rows without legal options."""
    has_option = mask.sum(dim=-1) > 0
    if not bool(has_option.any()):
        return logits.sum() * 0.0
    logits = logits[has_option]
    labels = labels[has_option]
    if weight is None:
        return nn.functional.cross_entropy(logits, labels)
    losses = nn.functional.cross_entropy(logits, labels, reduction="none")
    w = weight[has_option].to(dtype=losses.dtype)
    return (losses * w).sum() / w.sum().clamp_min(1e-8)


def main() -> int:
    torch.manual_seed(0)

    batch = 4
    state = torch.randn(batch, STATE_FEATURE_SIZE)
    options = torch.randn(batch, MAX_OPTIONS, OPTION_FEATURE_SIZE)
    valid_counts = torch.tensor([3, 1, 9, 33])
    mask = torch.zeros(batch, MAX_OPTIONS)
    for row, count in enumerate(valid_counts.tolist()):
        mask[row, :count] = 1.0
    card_ids = torch.randint(1, CARD_ID_LIMIT, (batch, MAX_OPTIONS, 2))
    card_ids[mask == 0] = 0

    model = MazlumNet().eval()
    with torch.no_grad():
        logits = model(state, options, card_ids, mask)
        probs = torch.softmax(logits, dim=-1)

    argmax = logits.argmax(dim=-1)
    params = sum(p.numel() for p in model.parameters())
    print(f"logits.shape   = {tuple(logits.shape)}")
    print(f"heads          = {model.heads} x {model.head_dim}")
    print(f"valid options  = {valid_counts.tolist()}")
    print(f"argmax         = {argmax.tolist()}")
    print(f"all legal      = {bool((argmax < valid_counts).all())}")
    print(f"padding max    = {logits[mask == 0].max().item():.1f}")
    print(f"probs sums     = {probs.sum(dim=-1).tolist()}")
    print(f"parameters     = {params}  (~{params * 4 / 1024:.0f} KB float32)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
