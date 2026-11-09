"""Submission entry point for the Mazlum bot.

Kaggle imports this file and calls ``agent(observation)`` once per decision.
The logic lives in ``agent.py``; the bundle must also contain ``deck.csv``,
``model.npz``, ``features.py`` and the ``cg`` folder.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import agent as _agent  # noqa: E402


def agent(obs_dict: dict) -> list[int]:
    """Return the indices of the options selected for this decision."""
    return _agent.agent(obs_dict)
