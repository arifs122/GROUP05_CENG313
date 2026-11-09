"""Kaggle entry point for the Mega Kangaskhan / Ogerpon agent."""

from __future__ import annotations

import os
import sys

_CANDIDATES = []
if "__file__" in globals():
    _CANDIDATES.append(os.path.dirname(os.path.abspath(__file__)))
_CANDIDATES.extend([os.getcwd(), "/kaggle_simulations/agent"])
_CANDIDATES.extend([p for p in sys.path if p])
_HERE = None
for _path in _CANDIDATES:
    if _path and os.path.isfile(os.path.join(_path, "mk_policy.py")):
        _HERE = os.path.abspath(_path)
        break
if _HERE is None:
    _HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
for _path in (_HERE, os.path.join(_HERE, "..", "..", "vendor"), "/kaggle_simulations/agent"):
    if os.path.isdir(os.path.join(_path, "cg")) and _path not in sys.path:
        sys.path.insert(0, _path)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mk_policy as _policy  # noqa: E402


def agent(obs_dict: dict) -> list[int]:
    return _policy.agent(obs_dict)
