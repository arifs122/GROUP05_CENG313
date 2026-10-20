"""Submission entry point.

Kaggle imports this file and calls ``agent(observation)`` once per decision.
The bundle must contain ``main.py`` and ``deck.csv`` at its root, plus the
``cg`` folder from the competition sample submission.

Keep this file small: the actual decision logic lives in ``policy.py``.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# Locally the engine lives in vendor/; in a submission bundle cg/ sits next
# to this file.  Kaggle may also expose the bundle at the path below.
for _path in (_HERE, os.path.join(_HERE, "..", "..", "vendor"), "/kaggle_simulations/agent"):
    if os.path.isdir(os.path.join(_path, "cg")) and _path not in sys.path:
        sys.path.insert(0, _path)

from policy import act  # noqa: E402


def agent(obs_dict: dict) -> list[int]:
    """Return the indices of the options selected for this decision."""
    return act(obs_dict)
