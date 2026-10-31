"""Mazlum — pack a trained ``model_*.npz`` into the 600 KB int4 archive.

Usage::

    python tools/mazlum/quantize_model.py --inp data/mazlum/model_v12_8.npz \
        --out data/mazlum/model_v12_8_int4.npz

Prints the resulting size and the per-tensor quantization error so the
accuracy cost of 4 bits is visible before shipping.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_SEARCH = [Path(__file__).resolve().parents[1] / "src" / "mazlum", ROOT / "src" / "mazlum"]
for path in _SEARCH:
    if (path / "quantize.py").is_file() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from quantize import dequantize_array, save_quantized, should_quantize  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantize a Mazlum model to int4")
    parser.add_argument("--inp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.inp) as data:
        arrays = {name: data[name] for name in data.files}

    original = args.inp.stat().st_size
    payload = save_quantized(args.out, arrays)
    packed = args.out.stat().st_size

    worst = 0.0
    worst_name = ""
    total_err = 0.0
    total_abs = 0.0
    for name, value in arrays.items():
        array = np.asarray(value)
        if not should_quantize(array):
            continue
        restored = dequantize_array(
            payload[name], payload[name + ".shape"], payload[name + ".scale"]
        )
        err = float(np.abs(restored - array.astype(np.float32)).max())
        total_err += float(np.abs(restored - array.astype(np.float32)).sum())
        total_abs += float(np.abs(array.astype(np.float32)).sum())
        if err > worst:
            worst, worst_name = err, name

    print(f"input : {args.inp}  ({original / 1024:.1f} KB)")
    print(f"output: {args.out}  ({packed / 1024:.1f} KB)  budget 600 KB")
    print(f"arrays: {len(arrays)} -> {len(payload)} (packed matrices get .shape/.scale)")
    print(f"max |error| = {worst:.5f} on {worst_name}")
    print(f"relative L1 error = {100 * total_err / max(total_abs, 1e-9):.2f}%")
    if packed > 600 * 1024:
        print("warning: archive exceeds the 600 KB budget")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
