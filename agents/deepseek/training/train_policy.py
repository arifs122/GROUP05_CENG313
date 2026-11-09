#!/usr/bin/env python3
"""Train the Top2 option-ranking policy prior.

Only winning-game, single-selection decisions are used.  ATTACH/target choices
are masked out of the loss because they are better handled by the rule/search
layers (the prior is for candidate ordering, not for playing alone).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from nets import PolicyNet, compute_normalizer, masked_cross_entropy  # noqa: E402


def normalized(state, options, mean, std, omean, ostd):
    s = (state - mean) / std
    o = (options - omean) / ostd
    return s.astype(np.float32), o.astype(np.float32)


@torch.no_grad()
def evaluate(net, state, options, card_ids, mask, label, norm, index, batch=1024):
    net.eval()
    total = 0.0
    correct = 0
    top3 = 0
    count = 0
    for start in range(0, len(index), batch):
        idx = index[start:start + batch]
        s, o = normalized(state[idx], options[idx], norm["state_mean"], norm["state_std"],
                          norm["option_mean"], norm["option_std"])
        logits = net(
            torch.from_numpy(s),
            torch.from_numpy(o),
            torch.from_numpy(card_ids[idx].astype(np.int64)),
            torch.from_numpy(mask[idx]),
        )
        y = torch.from_numpy(label[idx].astype(np.int64))
        m = torch.from_numpy(mask[idx])
        loss = masked_cross_entropy(logits, y, m)
        total += float(loss.item()) * len(idx)
        count += len(idx)
        pred = logits.argmax(dim=-1).numpy()
        correct += int((pred == label[idx]).sum())
        k = min(3, logits.shape[1])
        top = np.argsort(-logits.numpy(), axis=1)[:, :k]
        top3 += int(np.any(top == label[idx, None], axis=1).sum())
    return total / max(1, count), correct / max(1, count), top3 / max(1, count)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "dataset_top2_200.npz")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "models" / "policy_model.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = np.load(args.dataset)
    state = data["state"].astype(np.float32)
    options = data["options"].astype(np.float32)
    card_ids = data["card_ids"].astype(np.int16)
    mask = data["mask"].astype(np.float32)
    label = data["label"].astype(np.int64)
    game_win = data["game_win"].astype(np.float32)
    split = data["split"].astype(np.int8)
    selected_type = data["selected_type"].astype(np.int16)
    context = data["context"].astype(np.int16)
    max_count = data["max_count"].astype(np.int16)
    action_len = data["action_length"].astype(np.int16)

    label_in_range = (label >= 0) & (label < options.shape[1])
    label_valid = np.zeros(len(label), dtype=bool)
    valid_rows = np.nonzero(label_in_range)[0]
    if len(valid_rows):
        label_valid[valid_rows] = mask[valid_rows, label[valid_rows]] > 0
    usable = (
        (split >= 0)
        & label_in_range
        & label_valid
        & (max_count == 1)
        & (action_len == 1)
        & (game_win >= 0.5)
        & (mask.sum(axis=1) > 1)
    )
    # Policy prior is for ordering, not for ATTACH/target micro-decisions.
    zero_weight = (selected_type == 8) | np.isin(context, [21, 22, 26, 27, 28, 29, 30])
    train_idx = np.nonzero(usable & (split == 0))[0]
    val_idx = np.nonzero(usable & (split == 1))[0]
    print(f"usable train={len(train_idx)} val={len(val_idx)} rows={usable.sum()}")

    norm = compute_normalizer(state, options, mask, train_idx)
    net = PolicyNet(state.shape[1], options.shape[2], dropout=args.dropout)
    opt = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    weight = np.where(zero_weight, 0.0, 1.0).astype(np.float32)
    best = (-1.0, 1e9, None, 0)
    bad = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        net.train()
        order = train_idx.copy()
        np.random.shuffle(order)
        total = 0.0
        count = 0
        for start in range(0, len(order), args.batch):
            idx = order[start:start + args.batch]
            s, o = normalized(state[idx], options[idx], norm["state_mean"], norm["state_std"],
                              norm["option_mean"], norm["option_std"])
            logits = net(
                torch.from_numpy(s),
                torch.from_numpy(o),
                torch.from_numpy(card_ids[idx].astype(np.int64)),
                torch.from_numpy(mask[idx]),
            )
            loss = masked_cross_entropy(
                logits,
                torch.from_numpy(label[idx]),
                torch.from_numpy(mask[idx]),
                torch.from_numpy(weight[idx]),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
            count += len(idx)

        val_loss, val_top1, val_top3 = evaluate(
            net, state, options, card_ids, mask, label, norm, val_idx
        )
        print(f"epoch {epoch:2d} train_ce={total / max(1, count):.4f} "
              f"val_ce={val_loss:.4f} top1={val_top1:.4f} top3={val_top3:.4f}")
        improved = (val_top1 > best[0] + 1e-12) or (
            abs(val_top1 - best[0]) <= 1e-12 and val_loss < best[1] - 1e-12
        )
        if improved:
            best = (val_top1, val_loss, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, epoch)
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                break

    net.load_state_dict(best[2])
    val_loss, val_top1, val_top3 = evaluate(
        net, state, options, card_ids, mask, label, norm, val_idx
    )
    print(f"best epoch {best[3]} val_ce={val_loss:.4f} top1={val_top1:.4f} top3={val_top3:.4f} "
          f"({time.time() - start_time:.1f}s)")

    # NumPy parity on 1000 validation rows.
    sample = val_idx[:1000]
    s, o = normalized(state[sample], options[sample], norm["state_mean"], norm["state_std"],
                      norm["option_mean"], norm["option_std"])
    with torch.no_grad():
        logits_t = net(
            torch.from_numpy(s), torch.from_numpy(o),
            torch.from_numpy(card_ids[sample].astype(np.int64)),
            torch.from_numpy(mask[sample]),
        ).numpy()
    w = {k: v.detach().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    emb = w["card_embedding.weight"]
    actor = emb[card_ids[sample, :, 0].astype(np.int64)]
    target = emb[card_ids[sample, :, 1].astype(np.int64)]
    # Explicit concat path exactly mirrors torch.
    inp = np.concatenate([o, actor, target], axis=-1)
    tokens = inp @ w["option_proj.weight"].T + w["option_proj.bias"]
    state_token = s @ w["state_proj.weight"].T + w["state_proj.bias"]
    q = state_token @ w["q.weight"].T + w["q.bias"]
    k = tokens @ w["k.weight"].T + w["k.bias"]
    v = tokens @ w["v.weight"].T + w["v.bias"]
    scores = np.einsum("bd,bnd->bn", q, k) / np.sqrt(tokens.shape[-1])
    scores = np.where(mask[sample] > 0, scores, -1e9)
    scores = scores - scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights = np.where(mask[sample] > 0, weights, 0.0)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    context = np.einsum("bn,bnd->bd", weights, v)
    joint = np.concatenate([tokens, np.broadcast_to(context[:, None, :], tokens.shape)], axis=-1)
    hidden = np.maximum(0.0, joint @ w["head.0.weight"].T + w["head.0.bias"])
    logits_np = (hidden @ w["head.3.weight"].T + w["head.3.bias"]).squeeze(-1)
    logits_np = np.where(mask[sample] > 0, logits_np, -1e9)
    diff = float(np.max(np.abs(logits_np - logits_t)))
    match = bool(np.mean(np.argmax(logits_np, axis=1) == np.argmax(logits_t, axis=1)) >= 0.999)
    print(f"numpy parity max_abs_diff={diff:.3e} argmax_match={match}")
    if diff > 2e-4 or not match:
        raise SystemExit("NumPy parity failed")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_mean": norm["state_mean"],
        "state_std": norm["state_std"],
        "option_mean": norm["option_mean"],
        "option_std": norm["option_std"],
    }
    payload.update(w)
    np.savez_compressed(args.out, **payload)
    manifest = {
        "dataset": str(args.dataset),
        "rows": int(len(state)),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "best_epoch": int(best[3]),
        "val_ce": val_loss,
        "val_top1": val_top1,
        "val_top3": val_top3,
        "numpy_parity_max_abs_diff": diff,
        "state_dim": int(state.shape[1]),
        "option_dim": int(options.shape[2]),
        "out": str(args.out),
    }
    (args.out.parent / "policy_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
