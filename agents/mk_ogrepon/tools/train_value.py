#!/usr/bin/env python3
"""Train the small public-state value net (win probability)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BOT = ROOT / "agents" / "mk_ogrepon" / "bot"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "agents" / "mk_ogrepon" / "data" / "value_dataset.npz")
    parser.add_argument("--out", type=Path, default=BOT / "value_model.npz")
    parser.add_argument("--report", type=Path, default=ROOT / "agents" / "mk_ogrepon" / "reports" / "value_model.json")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    d = np.load(args.data)
    state = d["state"].astype(np.float32)
    label = d["label"].astype(np.float32)
    episode = d["episode"]
    uniq = np.unique(episode)
    rng = np.random.default_rng(42)
    rng.shuffle(uniq)
    cut = max(1, int(len(uniq) * 0.8))
    train_eps = set(uniq[:cut].tolist())
    train = np.array([e in train_eps for e in episode])
    val = ~train
    print(f"samples={len(state)} episodes={len(uniq)} train={train.sum()} val={val.sum()}")

    import torch
    from torch import nn

    device = torch.device("cpu")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state.shape[1], args.hidden),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden, args.hidden // 2),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden // 2, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = Net().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    xt = torch.from_numpy(state[train])
    yt = torch.from_numpy(label[train])
    xv = torch.from_numpy(state[val])
    yv = torch.from_numpy(label[val])

    def metrics(logits, y):
        p = torch.sigmoid(logits)
        bce = float(torch.mean(torch.logaddexp(torch.zeros_like(logits), logits) - y * logits).item())
        acc = float(((p >= 0.5).float() == y).float().mean().item())
        brier = float(torch.mean((p - y) ** 2).item())
        return bce, acc, brier

    best = (1e9, None, 0.0)
    bad = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(xt))
        total = 0.0
        for start in range(0, len(order), 512):
            idx = order[start:start + 512]
            logits = model(xt[idx])
            loss = loss_fn(logits, yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
        model.eval()
        with torch.no_grad():
            vlog = model(xv)
            vloss, vacc, vbrier = metrics(vlog, yv)
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} train={total/len(xt):.4f} val={vloss:.4f} acc={vacc:.3f} brier={vbrier:.4f}")
        if vloss < best[0] - 1e-5:
            best = (vloss, {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}, vacc)
            bad = 0
        else:
            bad += 1
            if bad >= 20:
                break

    weights = best[1]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **weights)
    print(f"best val={best[0]:.4f} acc={best[2]:.3f} -> {args.out} ({args.out.stat().st_size/1024:.1f} KB)")

    def numpy_logits(x):
        h = np.maximum(0.0, x @ weights["net.0.weight"].T + weights["net.0.bias"])
        h = np.maximum(0.0, h @ weights["net.3.weight"].T + weights["net.3.bias"])
        return (h @ weights["net.6.weight"].T + weights["net.6.bias"]).reshape(-1)

    n = min(1000, int(val.sum()))
    with torch.no_grad():
        torch_logits = model(xv[:n]).cpu().numpy()
    np_logits = numpy_logits(state[val][:n])
    max_diff = float(np.max(np.abs(torch_logits - np_logits)))
    print(f"parity max_diff={max_diff:.8f}")
    if max_diff > 1e-5:
        raise SystemExit("parity failed")

    report = {
        "samples": int(len(state)),
        "train": int(train.sum()),
        "val": int(val.sum()),
        "val_loss": round(float(best[0]), 5),
        "val_acc": round(float(best[2]), 4),
        "parity_max_diff": max_diff,
        "state_dim": int(state.shape[1]),
        "hidden": args.hidden,
        "dropout": args.dropout,
        "weights": str(args.out.relative_to(ROOT)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
