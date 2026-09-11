"""Run focused graph regression tests and persist their result."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


OUTPUT_PATH = Path(__file__).with_name("results") / "graph_edge_regression_test_result.txt"


def main() -> int:
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(Path(__file__).parent),
        pattern="test_graph_edge_regressions.py",
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())