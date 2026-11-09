#!/usr/bin/env python3
"""Train a small behavior-cloning prior for MAIN decisions from Mega Lucario replays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
LUCARIO = HERE.parent
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(LUCARIO))

import lucario_features as F  # noqa: E402
import replay_meta as rm  # noqa: E402

MAX_OPTIONS = 24


def load_samples() -> tuple[list[dict], list[str]]:
    samples = []
    files = []
    seen = set()
    for path in sorted((ROOT / "replays").rglob("*.json")):
        try:
            e = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        eid = (e.get("info") or {}).get("EpisodeId") or e.get("id")
        if eid in seen:
            continue
        seen.add(eid)
        decks = rm.extract_decks(e)
        if not decks:
            continue
        players = [i for i, d in enumerate(decks) if 678 in d]
        if len(players) != 1:
            continue
        me = players[0]
        if rm.winner_index(e) != me:
            # Still keep winning decisions? Start with all; the report can
            # filter later.  Winning-only is the stronger signal.
            continue
        files.append(path.name)
        for player, si, obs, sel, action in rm.iter_decisions(e):
            if player != me or not sel or int(sel.get("type", -1)) != 0:
                continue
            if len(action) != 1:
                continue
            opts = sel.get("option") or []
            idx = int(action[0])
            if not (0 <= idx < len(opts)) or not (2 <= len(opts) <= MAX_OPTIONS):
                continue
            try:
                state, option_vecs = F.decision_features(obs)
            except Exception:
                continue
            samples.append({"file": path.name, "state": state, "options": option_vecs,
                            "label": idx})
    return samples, files


def make_arrays(samples):
    n = len(samples)
    state = np.zeros((n, F.STATE_DIM), dtype=np.float32)
    options = np.zeros((n, MAX_OPTIONS, F.OPTION_DIM), dtype=np.float32)
    mask = np.zeros((n, MAX_OPTIONS), dtype=np.float32)
    label = np.zeros(n, dtype=np.int64)
    for i, s in enumerate(samples):
        state[i] = np.asarray(s["state"], dtype=np.float32)
        for j, v in enumerate(s["options"]):
            options[i, j] = np.asarray(v, dtype=np.float32)
            mask[i, j] = 1.0
        label[i] = int(s["label"])
    return state, options, mask, label


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--out", type=Path, default=LUCARIO / "policy_prior.npz")
    parser.add_argument("--report", type=Path, default=LUCARIO / "reports" / "policy_prior.json")
    args = parser.parse_args()

    samples, files = load_samples()
    if len(samples) < 200:
        raise SystemExit(f"too few samples: {len(samples)}")
    train_files = set(files[: max(1, int(len(files) * 0.8))])
    train = [s for s in samples if s["file"] in train_files]
    val = [s for s in samples if s["file"] not in train_files]
    print(f"samples={len(samples)} train={len(train)} val={len(val)} files={len(files)}")

    import torch
    from torch import nn

    st, op, mask, lab = make_arrays(train)
    vst, vop, vmask, vlab = make_arrays(val)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(F.STATE_DIM + F.OPTION_DIM, args.hidden),
                nn.ReLU(), nn.Dropout(args.dropout),
                nn.Linear(args.hidden, args.hidden // 2),
                nn.ReLU(), nn.Dropout(args.dropout),
                nn.Linear(args.hidden // 2, 1),
            )

        def forward(self, state, options):
            b, o, d = options.shape
            st = state[:, None, :].expand(b, o, state.shape[1])
            return self.net(torch.cat([st, options], dim=-1)).squeeze(-1)

    model = Net()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    def evaluate(state, options, mask, label):
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(state), torch.from_numpy(options))
            logits = logits.masked_fill(torch.from_numpy(mask == 0), -1e9)
            pred = logits.argmax(dim=1).numpy()
            acc = float((pred == label).mean())
            loss = float(nn.functional.cross_entropy(logits, torch.from_numpy(label)).item())
        return loss, acc

    best = (1e9, None, 0.0)
    bad = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.arange(len(st))
        np.random.shuffle(order)
        for start in range(0, len(order), 256):
            idx = order[start:start + 256]
            logits = model(torch.from_numpy(st[idx]), torch.from_numpy(op[idx]))
            logits = logits.masked_fill(torch.from_numpy(mask[idx] == 0), -1e9)
            loss = loss_fn(logits, torch.from_numpy(lab[idx]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        vl, va = evaluate(vst, vop, vmask, vlab)
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} val={vl:.4f} acc={va:.3f}")
        if vl < best[0] - 1e-5:
            best = (vl, {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}, va)
            bad = 0
        else:
            bad += 1
            if bad >= 20:
                break

    weights = best[1]
    np.savez_compressed(args.out, **weights)
    print(f"best val={best[0]:.4f} acc={best[2]:.3f} -> {args.out}")

    def numpy_logits(state, options):
        x = np.concatenate([np.repeat(state[:, None, :], options.shape[1], axis=1), options], axis=-1)
        h = np.maximum(0.0, x @ weights["net.0.weight"].T + weights["net.0.bias"])
        h = np.maximum(0.0, h @ weights["net.3.weight"].T + weights["net.3.bias"])
        return (h @ weights["net.6.weight"].T + weights["net.6.bias"]).squeeze(-1)

    n = min(1000, len(vst))
    with torch.no_grad():
        tl = model(torch.from_numpy(vst[:n]), torch.from_numpy(vop[:n])).numpy()
    nl = numpy_logits(vst[:n], vop[:n])
    diff = float(np.max(np.abs(tl - nl)))
    print(f"parity max_diff={diff:.8f}")
    if diff > 1e-5:
        raise SystemExit("parity failed")

    report = {"samples": len(samples), "train": len(train), "val": len(val),
              "files": len(files), "val_acc": round(best[2], 4),
              "val_loss": round(float(best[0]), 5), "parity": diff,
              "state_dim": F.STATE_DIM, "option_dim": F.OPTION_DIM}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
