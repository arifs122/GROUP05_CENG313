"""Kaggle entry point for the from-scratch Top2 Dragapult agent."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
for _path in (
    _HERE,
    os.path.join(_HERE, "..", "..", "..", "vendor"),  # agents/deepseek/top2_core
    os.path.join(_HERE, "..", "..", "vendor"),
    "/kaggle_simulations/agent",
):
    if os.path.isdir(os.path.join(_path, "cg")) and _path not in sys.path:
        sys.path.insert(0, _path)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import top2_core_policy as _policy  # noqa: E402


def agent(obs_dict: dict) -> list[int]:
    return _policy.agent(obs_dict)
