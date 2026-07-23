"""Pytest configuration.

Ensures the ``src`` layout is importable even when the package has not been
installed with ``pip install -e .`` (e.g. a fresh checkout run directly with
``pytest``).
"""

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
