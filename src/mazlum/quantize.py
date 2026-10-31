"""Mazlum — 4-bit packing for the submission export.

The competition budget for ``model.npz`` is 600 KB. At v12.8 (1.16M
parameters) float16 needs ~2.0 MB, so the shipped archive stores every
weight matrix as **int4** (symmetric, per-tensor scale, two values per
byte) and keeps the small 1-D tensors (biases, LayerNorm, normaliser) in
float16.

Layout for a packed tensor ``W``::

    W        uint8    packed nibbles (C-order, low nibble = even index)
    W.shape  int32    original shape
    W.scale  float16  symmetric scale: W = (q - 8) * scale

``bots/mazlum/agent.py`` calls :func:`load_model` so inference never sees
the packed form.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

META_SUFFIXES = (".scale", ".shape")
QUANT_MIN_ELEMENTS = 256


def should_quantize(array: np.ndarray) -> bool:
    """Only matrices large enough to matter are packed."""
    return array.ndim >= 2 and array.size >= QUANT_MIN_ELEMENTS


def quantize_array(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.float16]:
    """Symmetric int4 packing with a per-tensor scale."""
    weights = np.asarray(array, dtype=np.float32)
    peak = float(np.abs(weights).max()) if weights.size else 0.0
    scale = peak / 7.0 if peak > 0 else 1.0
    quantised = np.clip(np.rint(weights / scale), -8, 7).astype(np.int8) + 8
    flat = quantised.reshape(-1).astype(np.uint8)
    if flat.size % 2:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])
    packed = (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)
    return packed, np.asarray(weights.shape, dtype=np.int32), np.float16(scale)


def dequantize_array(packed: np.ndarray, shape: np.ndarray, scale: np.float16) -> np.ndarray:
    """Inverse of :func:`quantize_array`."""
    low = (packed & 0x0F).astype(np.float32)
    high = ((packed >> 4) & 0x0F).astype(np.float32)
    flat = np.empty(low.size * 2, dtype=np.float32)
    flat[0::2] = low
    flat[1::2] = high
    count = int(np.prod(shape))
    return (flat[:count].reshape(tuple(int(s) for s in shape)) - 8.0) * float(scale)


def save_quantized(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Write the archive with int4 matrices and float16 scalars."""
    payload: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if should_quantize(array):
            packed, shape, scale = quantize_array(array)
            payload[name] = packed
            payload[name + ".shape"] = shape
            payload[name + ".scale"] = scale
        else:
            payload[name] = array.astype(np.float16)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return payload


def load_model(path: Path) -> dict[str, np.ndarray]:
    """Load an archive produced by :func:`save_quantized` (or a plain one)."""
    with np.load(path) as data:
        files = set(data.files)
        arrays: dict[str, np.ndarray] = {}
        for name in data.files:
            if name.endswith(META_SUFFIXES):
                continue
            if name + ".scale" in files:
                arrays[name] = dequantize_array(
                    data[name], data[name + ".shape"], data[name + ".scale"]
                )
            else:
                arrays[name] = data[name].astype(np.float32)
    return arrays
