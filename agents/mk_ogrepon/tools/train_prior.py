#!/usr/bin/env python3
"""Train the small observation/option policy prior from Top1 winning replays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BOT = ROOT / "agents" / "mk_ogrepon" / "bot"
sys.path.insert(0, str(BOT))
sys.path.insert(0, str(ROOT / "tools"))

import mk_features as F  # noqa: E402
import replay_meta as rm  # noqa: E402

TEAM_ID = "17014244"
TEAM_NAME = "やる気元気ミワハルキ 2nd ver."
MAX_OPTIONS = 24


def load_samples() -> tuple[list[dict], list[str]]:
    base = ROOT / "replays" / "top5_nov1" / TEAM_ID
    files = sorted(base.glob("*.json"))
    samples: list[dict] = []
    used_files: list[str] = []
    for path in files:
        try:
            episode = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        names = rm.team_names(episode)[:2]
        ours = [i for i, n in enumerate(names) if TEAM_NAME in str(n)]
        if len(ours) != 1:
            continue
        me = ours[0]
        if rm.winner_index(episode) != me:
            continue
        used_files.append(path.name)
        for player, si, obs, sel, action in rm.iter_decisions(episode):
            if player != me or sel is None:
                continue
            if len(action) != 1:
                continue
            idx = int(action[0])
            opts = sel.get("option") or []
            if not opts or idx < 0 or idx >= len(opts):
                continue
            if len(opts) < 2 or len(opts) > MAX_OPTIONS:
                continue
            try:
                state, option_vecs = F.decision_features(obs)
            except Exception:
                continue
            samples.append({
                "file": path.name,
                "step": si,
                "state": state,
                "options": option_vecs,
                "label": idx,
                "type": int(sel.get("type", -1)),
                "context": int(sel.get("context", -1)),
            })
    return samples, used_files


def make_arrays(samples: list[dict]):
    n = len(samples)
    state = np.zeros((n, F.STATE_DIM), dtype=np.float32)
    options = np.zeros((n, MAX_OPTIONS, F.OPTION_DIM), dtype=np.float32)
    mask = np.zeros((n, MAX_OPTIONS), dtype=np.float32)
    label = np.zeros(n, dtype=np.int64)
    for i, s in enumerate(samples):
        state[i] = np.asarray(s["state"], dtype=np.float32)
        for j, vec in enumerate(s["options"]):
            options[i, j] = np.asarray(vec, dtype=np.float32)
            mask[i, j] = 1.0
        label[i] = int(s["label"])
    return state, options, mask, label


class Model:
    def __init__(self, state_dim: int, option_dim: int, hidden: int = 128, dropout: float = 0.1):
        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(state_dim + option_dim, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden // 2, 1),
                )

            def forward(self, state, options):
                # state: B,S  options: B,O,D
                b, o, d = options.shape
                st = state[:, None, :].expand(b, o, state.shape[1])
                x = torch.cat([st, options], dim=-1)
                return self.net(x).squeeze(-1)

        self.net = Net()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=BOT / "policy_prior.npz")
    parser.add_argument("--report", type=Path, default=ROOT / "agents" / "mk_ogrepon" / "reports" / "policy_prior.json")
    args = parser.parse_args()

    samples, files = load_samples()
    if len(samples) < 100:
        raise SystemExit(f"too few samples: {len(samples)}")
    # Group split by replay so the validation set is unseen games.
    train_files = set(files[: max(1, int(len(files) * 0.8))])
    train = [s for s in samples if s["file"] in train_files]
    val = [s for s in samples if s["file"] not in train_files]
    print(f"samples={len(samples)} train={len(train)} val={len(val)} files={len(files)}")

    import torch
    from torch import nn

    st, op, mask, lab = make_arrays(train)
    vst, vop, vmask, vlab = make_arrays(val)
    device = torch.device("cpu")
    model = Model(F.STATE_DIM, F.OPTION_DIM, args.hidden, args.dropout).net.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    def evaluate(state, options, mask, label):
        model.eval()
        with torch.no_grad():
            logits = model(
                torch.from_numpy(state).to(device),
                torch.from_numpy(options).to(device),
            )
            logits = logits.masked_fill(torch.from_numpy(mask == 0).to(device), -1e9)
            pred = logits.argmax(dim=1).cpu().numpy()
            acc = float((pred == label).mean())
            loss = float(nn.functional.cross_entropy(logits, torch.from_numpy(label).to(device)).item())
        return loss, acc

    best = (1e9, None, 0.0)
    bad = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.arange(len(st))
        np.random.shuffle(order)
        total = 0.0
        for start in range(0, len(order), 256):
            idx = order[start:start + 256]
            s = torch.from_numpy(st[idx]).to(device)
            o = torch.from_numpy(op[idx]).to(device)
            m = torch.from_numpy(mask[idx]).to(device)
            y = torch.from_numpy(lab[idx]).to(device)
            logits = model(s, o).masked_fill(m == 0, -1e9)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
        vloss, vacc = evaluate(vst, vop, vmask, vlab)
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} train_loss={total / max(1, len(st)):.4f} val_loss={vloss:.4f} val_acc={vacc:.3f}")
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
    print(f"best val_loss={best[0]:.4f} val_acc={best[2]:.3f} -> {args.out} ({args.out.stat().st_size/1024:.1f} KB)")

    # Numpy / torch parity check on 1000 validation rows.
    def numpy_logits(state, options):
        x = np.concatenate([np.repeat(state[:, None, :], options.shape[1], axis=1), options], axis=-1)
        h = np.maximum(0.0, x @ weights["net.0.weight"].T + weights["net.0.bias"])
        h = np.maximum(0.0, h @ weights["net.3.weight"].T + weights["net.3.bias"])
        return (h @ weights["net.6.weight"].T + weights["net.6.bias"]).squeeze(-1)

    n = min(1000, len(vst))
    with torch.no_grad():
        torch_logits = model(
            torch.from_numpy(vst[:n]),
            torch.from_numpy(vop[:n]),
        ).cpu().numpy()
    np_logits = numpy_logits(vst[:n], vop[:n])
    max_diff = float(np.max(np.abs(torch_logits - np_logits)))
    print(f"parity max_diff={max_diff:.8f}")
    if max_diff > 1e-5:
        raise SystemExit("parity failed")

    report = {
        "samples": len(samples),
        "train": len(train),
        "val": len(val),
        "files": len(files),
        "val_acc": round(best[2], 4),
        "val_loss": round(float(best[0]), 5),
        "parity_max_diff": max_diff,
        "state_dim": F.STATE_DIM,
        "option_dim": F.OPTION_DIM,
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
