"""Mazlum â€” behavioral-cloning training loop with DAgger weights + arena early stopping.

* Dataset: episode-split (``split``) rows built by ``build_dataset.py`` plus
  DAgger rows appended by ``tools/mazlum/dagger_collect.py``.
* Sample weights: replay rows 1.0, DAgger rows ``--dagger-weight`` (0.3).
* Early stopping: every ``--arena-every`` epochs a **100-game mini arena** against
  ``bots/banette`` is played and the best win rate wins (val metrics are only
  reported). This is the metric that actually matters, and val top-1 proved to
  be misleading for DAgger data.
* Export: the best-by-arena weights as a plain NumPy archive for ``bots/mazlum``.

    data/mazlum/dataset.npz  ->  data/mazlum/model.npz

Usage::

    python src/mazlum/train_bc.py
    python src/mazlum/train_bc.py --epochs 80 --patience 2 --arena-games 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import model as M  # noqa: E402

ROOT = HERE.parents[1]
DEFAULT_DATASET = ROOT / "data" / "mazlum" / "dataset.npz"
DEFAULT_OUT = ROOT / "data" / "mazlum" / "model.npz"
CANDIDATE_MODEL = ROOT / "data" / "mazlum" / "_arena_candidate.npz"

NORMALIZER_KEYS = ("state_mean", "state_std", "option_mean", "option_std")
EVAL_CHUNK = 4096


def load_dataset(path: Path):
    """Load the .npz dataset (features are float16 on disk)."""
    data = np.load(path)
    state = torch.from_numpy(data["state"])
    options = torch.from_numpy(data["options"])
    card_ids = torch.from_numpy(data["card_ids"].astype(np.int64))
    mask = torch.from_numpy(data["mask"].astype(np.float32))
    label = torch.from_numpy(data["label"].astype(np.int64))
    split = torch.from_numpy(data["split"].astype(np.int8)) if "split" in data.files else None
    source = torch.from_numpy(data["source"].astype(np.int8)) if "source" in data.files else None
    episodes = int(np.unique(data["episode_id"]).size)
    return state, options, card_ids, mask, label, split, source, episodes


def _safe_stats(mean: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Features that are constant in train (std ~ 0) are left unnormalised."""
    weak = std < 1e-4
    mean = torch.where(weak, torch.zeros_like(mean), mean)
    std = torch.where(weak, torch.ones_like(std), std)
    return mean, std


def compute_normalizer(state, options, mask, index=None) -> dict[str, torch.Tensor]:
    """Z-score statistics in float32; option stats use valid options only.

    Option statistics are accumulated in chunks so a multi-GB dataset never
    needs a second full-size float32 copy.
    """
    if index is None:
        state_rows, option_rows, mask_rows = state, options, mask
    else:
        state_rows, option_rows, mask_rows = state[index], options[index], mask[index]

    state_rows = state_rows.float()
    state_mean, state_std = _safe_stats(state_rows.mean(dim=0), state_rows.std(dim=0))

    dim = option_rows.shape[-1]
    total = torch.zeros(dim, dtype=torch.float64)
    total_sq = torch.zeros(dim, dtype=torch.float64)
    count = 0
    rows = option_rows.shape[0]
    for start in range(0, rows, 65536):
        stop = min(start + 65536, rows)
        chunk_mask = mask_rows[start:stop].bool()
        values = option_rows[start:stop][chunk_mask].float().double()
        if values.numel() == 0:
            continue
        total += values.sum(dim=0)
        total_sq += (values * values).sum(dim=0)
        count += values.shape[0]

    count = max(1, count)
    mean = (total / count).float()
    variance = (total_sq / count - (total / count) ** 2).clamp_min(0)
    std = variance.sqrt().float()
    option_mean, option_std = _safe_stats(mean, std)

    return {
        "state_mean": state_mean,
        "state_std": state_std,
        "option_mean": option_mean,
        "option_std": option_std,
    }


def normalize(state, options, normalizer) -> tuple[torch.Tensor, torch.Tensor]:
    state = (state.float() - normalizer["state_mean"]) / normalizer["state_std"]
    options = (options.float() - normalizer["option_mean"]) / normalizer["option_std"]
    return state, options


@torch.no_grad()
def evaluate(net, state, options, card_ids, mask, label, normalizer, index=None) -> tuple[float, float]:
    """Return (loss, top-1 accuracy) over ``index`` (or everything), chunked."""
    was_training = net.training
    net.eval()
    rows = state.shape[0] if index is None else int(index.numel())
    total_loss = 0.0
    correct = 0
    for start in range(0, rows, EVAL_CHUNK):
        stop = min(start + EVAL_CHUNK, rows)
        if index is None:
            s_raw, o_raw = state[start:stop], options[start:stop]
            chunk_ids = card_ids[start:stop]
            chunk_mask, chunk_label = mask[start:stop], label[start:stop]
        else:
            chunk = index[start:stop]
            s_raw, o_raw = state[chunk], options[chunk]
            chunk_ids = card_ids[chunk]
            chunk_mask, chunk_label = mask[chunk], label[chunk]
        s, o = normalize(s_raw, o_raw, normalizer)
        logits = net(s, o, chunk_ids, chunk_mask)
        total_loss += M.masked_cross_entropy(logits, chunk_label, chunk_mask).item() * (stop - start)
        correct += int((logits.argmax(dim=-1) == chunk_label).sum())
    net.train(was_training)
    return total_loss / max(1, rows), correct / max(1, rows)


def save_numpy(net, normalizer, path: Path, fp16: bool = False) -> dict[str, np.ndarray]:
    """Convert ``state_dict`` + normaliser tensors to NumPy and store them."""
    dtype = np.float16 if fp16 else np.float32
    arrays = {
        name: tensor.detach().cpu().numpy().astype(dtype)
        for name, tensor in net.state_dict().items()
    }
    arrays.update(
        {name: tensor.detach().cpu().numpy().astype(dtype) for name, tensor in normalizer.items()}
    )
    if hasattr(net, "blocks"):
        arrays["arch"] = np.asarray([len(net.blocks), net.heads, net.attn_dim], dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return arrays


def build_mini_arena(games: int, model_path: Path):
    """Return ``win_rate(net, normalizer) -> float`` playing ``games`` matches."""
    for path in (ROOT / "vendor", ROOT / "tools", ROOT / "bots" / "mazlum"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import arena  # noqa: PLC0415 - needs the engine path above
    import agent as bot_agent  # noqa: PLC0415

    features = bot_agent._import_features()
    deck = bot_agent.load_deck()
    expert = arena.load_bot(ROOT / "bots" / "banette")

    def win_rate(net: M.MazlumNet, normalizer: dict[str, torch.Tensor]) -> float:
        save_numpy(net, normalizer, model_path)
        policy = bot_agent.MazlumPolicy(model_path, features, deck)
        wins = 0
        for game_index in range(games):
            result = arena.play_game(
                policy.act,
                expert.agent,
                deck,
                expert.deck,
                reverse=(game_index % 2 == 1),
                max_steps=5000,
            )
            wins += int(result["winner"] == 0)
        return wins / max(1, games)

    return win_rate


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MazlumNet with DAgger weights and arena early stopping")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dagger-weight", type=float, default=0.3, help="sample weight for DAgger rows")
    parser.add_argument("--arena-every", type=int, default=5, help="run the mini arena every N epochs")
    parser.add_argument("--arena-games", type=int, default=100, help="matches per mini arena")
    parser.add_argument("--patience", type=int, default=2, help="arena evaluations without improvement")
    parser.add_argument("--arena-start", type=int, default=0, help="first epoch that runs the mini arena")
    parser.add_argument("--save-each", type=Path, default=None, help="directory for per-epoch checkpoints")
    parser.add_argument("--arch", choices=["v9", "big"], default="v9",
                        help="v9 = 2x2 d=64 (v15); big = 4x4 d=80 Pre-LN (v19)")
    parser.add_argument("--fp16-save", action="store_true", help="store weights as float16 (~half size)")
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--init",
        type=Path,
        default=None,
        help="initialize the network from an existing model.npz (fine-tuning)",
    )
    args = parser.parse_args()

    if not args.dataset.is_file():
        print(f"error: dataset not found: {args.dataset}")
        print("run: python src/mazlum/build_dataset.py")
        return 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state, options, card_ids, mask, label, split, source, episodes = load_dataset(args.dataset)
    rows = state.shape[0]

    if split is None:
        split = torch.zeros(rows, dtype=torch.int8)
        print("warning: dataset has no 'split' array; training on all rows")

    # v11 reçetesi: anti-meta (source == 2) satırları 2.0 ağırlıkla öğrenilir.
    weight = torch.ones(rows, dtype=torch.float32)
    if source is not None:
        weight[source == 2] = 2.0
    dagger_rows = int((source == 1).sum()) if source is not None else 0
    antimeta_rows = int((source == 2).sum()) if source is not None else 0

    train_idx = torch.nonzero(split == 0, as_tuple=False).flatten()
    val_idx = torch.nonzero(split == 1, as_tuple=False).flatten()
    if train_idx.numel() == 0:
        print("error: empty training split")
        return 1
    if val_idx.numel() == 0:
        print("warning: empty validation split; val metrics only")

    print(f"dataset: {args.dataset}  rows={rows}  episodes={episodes}")
    print(
        f"split: train {train_idx.numel()} / val {val_idx.numel()}  "
        f"dagger rows {dagger_rows} (weight {args.dagger_weight})  "
        f"anti-meta rows {antimeta_rows} (weight 2.0)"
    )
    print(
        f"model: MazlumNet  Adam(lr={args.lr})  epochs<={args.epochs}  batch={args.batch_size}  "
        f"arena every {args.arena_every} ep x {args.arena_games} games  patience={args.patience}"
    )

    normalizer = compute_normalizer(state, options, mask, train_idx)

    net = M.MazlumNetBig() if args.arch == "big" else M.MazlumNet()
    if args.init is not None:
        if not args.init.is_file():
            print(f"error: init model not found: {args.init}")
            return 1
        with np.load(args.init) as init_data:
            init_state = {
                name: torch.from_numpy(init_data[name].astype(np.float32))
                for name in init_data.files
                if name not in NORMALIZER_KEYS
            }
        net.load_state_dict(init_state)
        print(f"initialized from: {args.init} ({len(init_state)} tensors)")
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    batch_size = max(1, min(args.batch_size, int(train_idx.numel())))
    win_rate = build_mini_arena(args.arena_games, CANDIDATE_MODEL) if args.arena_every > 0 else None

    best_metric = -1.0
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    bad_evals = 0

    for epoch in range(1, args.epochs + 1):
        net.train()
        order = train_idx[torch.randperm(train_idx.numel())]
        for start in range(0, order.numel(), batch_size):
            index = order[start:start + batch_size]
            s, o = normalize(state[index], options[index], normalizer)
            logits = net(s, o, card_ids[index], mask[index])
            loss = M.masked_cross_entropy(logits, label[index], mask[index], weight[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        needs_arena = (
            win_rate is not None
            and args.arena_every > 0
            and epoch % args.arena_every == 0
            and epoch >= args.arena_start
        )
        if epoch % args.print_every != 0 and epoch != args.epochs and not needs_arena:
            continue

        train_loss, train_acc = evaluate(net, state, options, card_ids, mask, label, normalizer, train_idx)
        val_loss, val_acc = evaluate(net, state, options, card_ids, mask, label, normalizer, val_idx)
        line = (
            f"epoch {epoch:3d}/{args.epochs}  train {train_loss:.4f}/{train_acc * 100:5.2f}%  "
            f"val {val_loss:.4f}/{val_acc * 100:5.2f}%"
        )

        improved = False
        if win_rate is not None and epoch % args.arena_every == 0 and epoch >= args.arena_start:
            arena_wr = win_rate(net, normalizer)
            # >= with a validation tie-break: equal arena scores no longer
            # freeze the first (often undertrained) checkpoint.
            delta = arena_wr - best_metric
            better = delta > 1e-12
            tie = abs(delta) <= 1e-12 and (
                val_acc > best_val_acc + 1e-12
                or (abs(val_acc - best_val_acc) <= 1e-12 and val_loss < best_val_loss - 1e-12)
            )
            improved = better or tie
            if improved:
                best_metric = arena_wr
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {name: tensor.detach().clone() for name, tensor in net.state_dict().items()}
                bad_evals = 0
            else:
                bad_evals += 1
            line += f"  arena {arena_wr * 100:5.1f}% ({int(round(arena_wr * args.arena_games))}/{args.arena_games})"
            if improved:
                line += f" * (val {val_acc * 100:.2f}%)"

        if args.save_each is not None:
            args.save_each.mkdir(parents=True, exist_ok=True)
            save_numpy(net, normalizer, args.save_each / f"epoch_{epoch:03d}.npz", args.fp16_save)

        print(line)

        if win_rate is not None and args.patience > 0 and bad_evals >= args.patience:
            print(f"early stop at epoch {epoch}: no arena improvement for {bad_evals} evaluations")
            break

    if best_state is not None:
        net.load_state_dict(best_state)

    train_loss, train_acc = evaluate(net, state, options, card_ids, mask, label, normalizer, train_idx)
    val_loss, val_acc = evaluate(net, state, options, card_ids, mask, label, normalizer, val_idx)

    print(f"\nbest epoch: {best_epoch}  best arena: {best_metric * 100:.1f}%  best val: {best_val_acc * 100:.2f}%  (val loss {best_val_loss:.4f})")
    print(f"train: loss {train_loss:.4f}  top1 {train_acc * 100:.2f}%")
    print(f"val:   loss {val_loss:.4f}  top1 {val_acc * 100:.2f}%")

    arrays = save_numpy(net, normalizer, args.out, args.fp16_save)
    size_kb = args.out.stat().st_size / 1024
    print(f"\nwrote: {args.out}  ({size_kb:.1f} KB, {len(arrays)} arrays)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
