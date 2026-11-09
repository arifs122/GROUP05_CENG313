"""Pikachu ex rule pilot entry point."""
from __future__ import annotations
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
for _p in (_HERE, os.path.join(_HERE, "..", "..", "vendor"), "/kaggle_simulations/agent"):
    if os.path.isdir(os.path.join(_p, "cg")) and _p not in sys.path:
        sys.path.insert(0, _p)
from policy import act  # noqa: E402

def agent(obs_dict: dict) -> list[int]:
    return act(obs_dict)
