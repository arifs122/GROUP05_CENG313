#!/usr/bin/env python3
"""Train the small state -> win-probability value net (CPU friendly)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from nets import ValueNet, compute_normalizer  # noqa: E402


def metrics(y: np.ndarray, logit: np.ndarray):
    p = 1.0 / (1.0 + np.exp(-logit))
    bce = float(np.mean(np.logaddexp(0.0, logit) - y * logit))
    acc = float(((p >= 0.5) == (y >= 0.5)).mean())
    brier = float(np.mean((p - y) ** 2))
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    pos = float(y.sum())
    neg = float(len(y) - pos)
    if pos > 0 and neg > 0:
        auc = float((ranks[y >= 0.5].sum() - pos * (pos + 1) / 2) / (pos * neg))
    else:
        auc = 0.5
    return bce, acc, auc, brier


@torch.no_grad()
def infer(net, state, mean, std, index, batch=4096):
    net.eval()
    out = []
    for start in range(0, len(index), batch):
        idx = index[start:start + batch]
        x = torch.from_numpy(((state[idx] - mean) / std).astype(np.float32))
        out.append(net(x).numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "dataset_top2_200.npz")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "models" / "value_model.npz")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = np.load(args.dataset)
    state = data["state"].astype(np.float32)
    y = data["game_win"].astype(np.float32)
    split = data["split"].astype(np.int8)
    train_idx = np.nonzero(split == 0)[0]
    val_idx = np.nonzero(split == 1)[0]
    if len(val_idx) == 0:
        raise SystemExit("no validation rows")
    print(f"rows={len(state)} train={len(train_idx)} val={len(val_idx)} win={y.mean():.3f}")

    mean = state[train_idx].mean(axis=0).astype(np.float32)
    std = state[train_idx].std(axis=0).astype(np.float32)
    std[std < 1e-4] = 1.0

    net = ValueNet(state.shape[1], hidden=args.hidden, dropout=args.dropout)
    opt = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    best = (1e9, None, 0)
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
            x = torch.from_numpy(((state[idx] - mean) / std).astype(np.float32))
            yy = torch.from_numpy(y[idx])
            opt.zero_grad(set_to_none=True)
            logit = net(x)
            loss = loss_fn(logit, yy)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
            count += len(idx)

        val_logit = infer(net, state, mean, std, val_idx)
        m = metrics(y[val_idx], val_logit)
        print(f"epoch {epoch:2d} train_bce={total / max(1, count):.4f} "
              f"val_bce={m[0]:.4f} acc={m[1]:.4f} auc={m[2]:.4f} brier={m[3]:.4f}")
        if m[0] < best[0] - 1e-12:
            best = (m[0], {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, epoch)
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                break

    net.load_state_dict(best[1])
    val_logit = infer(net, state, mean, std, val_idx)
    print(f"best epoch {best[2]} val {metrics(y[val_idx], val_logit)}  ({time.time() - start_time:.1f}s)")

    # Parity: NumPy forward must match the torch logits on 1000 rows.
    sample = val_idx[:1000]
    torch_logit = infer(net, state, mean, std, sample)
    x = (state[sample] - mean) / std
    w0 = net.net[0].weight.detach().numpy().astype(np.float32)
    b0 = net.net[0].bias.detach().numpy().astype(np.float32)
    w1 = net.net[3].weight.detach().numpy().astype(np.float32)
    b1 = net.net[3].bias.detach().numpy().astype(np.float32)
    w2 = net.net[6].weight.detach().numpy().astype(np.float32)
    b2 = net.net[6].bias.detach().numpy().astype(np.float32)
    h = np.maximum(0.0, x @ w0.T + b0)
    h = np.maximum(0.0, h @ w1.T + b1)
    numpy_logit = (h @ w2.T + b2).reshape(-1)
    diff = float(np.max(np.abs(numpy_logit - torch_logit))) if len(sample) else 0.0
    print(f"numpy parity max_abs_diff={diff:.3e} argmax_match={bool(np.allclose(numpy_logit, torch_logit, atol=2e-5, rtol=2e-5))}")
    if diff > 1e-4:
        raise SystemExit("NumPy parity failed")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        state_mean=mean,
        state_std=std,
        w0=w0, b0=b0, w1=w1, b1=b1, w2=w2, b2=b2,
    )
    manifest = {
        "dataset": str(args.dataset),
        "rows": int(len(state)),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "best_epoch": int(best[2]),
        "val": dict(zip(("bce", "acc", "auc", "brier"), metrics(y[val_idx], val_logit))),
        "numpy_parity_max_abs_diff": diff,
        "state_dim": int(state.shape[1]),
        "out": str(args.out),
    }
    (args.out.parent / "value_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
