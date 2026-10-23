"""Contract checks for bot entry points.

The Kaggle validation runner calls the last callable it finds in ``main.py``.
If a helper such as ``reset()`` is defined after ``agent``, validation calls
the helper, gets ``None``, and the submission fails with
"Validation Episode failed." even though the agent itself is fine.

Keep ``agent`` as the last function in every bot module.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class BotContractTest(unittest.TestCase):
    def test_agent_is_last_function(self) -> None:
        bots = sorted((ROOT / "bots").glob("*/main.py"))
        self.assertTrue(bots, "no bots found")
        for path in bots:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            functions = [
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if "agent" not in functions:
                continue
            self.assertEqual(
                functions[-1],
                "agent",
                f"{path.relative_to(ROOT)}: last function is {functions[-1]!r}; "
                "the Kaggle validator calls the last callable, so agent must be last",
            )


if __name__ == "__main__":
    unittest.main()
