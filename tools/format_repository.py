#!/usr/bin/env python3
"""Check code style across the repository.

Runs, when available, the ``black`` formatter (in check mode by default) and the
``flake8`` linter over the library, tests, tools and example programs.  With
``--fix`` it lets ``black`` rewrite files in place.

Usage
-----
    python tools/format_repository.py          # report style issues
    python tools/format_repository.py --fix     # apply black formatting

The tools are optional; install them with ``pip install black flake8`` (or
``pip install -e .[dev]``).  Missing tools are reported and skipped, not treated
as failures.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _repo import ROOT, chapter_dirs  # noqa: E402


def _targets() -> list:
    targets = [os.path.join(ROOT, "src"),
               os.path.join(ROOT, "tests"),
               os.path.join(ROOT, "tools")]
    targets += chapter_dirs()
    return [t for t in targets if os.path.exists(t)]


def _have(tool: str) -> bool:
    try:
        subprocess.run([sys.executable, "-m", tool, "--version"],
                       capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check/apply code style.")
    parser.add_argument("--fix", action="store_true",
                        help="Let black rewrite files in place.")
    args = parser.parse_args()

    targets = _targets()
    status = 0

    if _have("black"):
        cmd = [sys.executable, "-m", "black"]
        if not args.fix:
            cmd += ["--check", "--diff"]
        print("Running black" + (" (fix)" if args.fix else " (check)") + " ...")
        result = subprocess.run(cmd + targets)
        status |= result.returncode
    else:
        print("black not installed -- skipping (pip install black)")

    if _have("flake8"):
        print("\nRunning flake8 ...")
        result = subprocess.run([sys.executable, "-m", "flake8"] + targets)
        status |= result.returncode
    else:
        print("flake8 not installed -- skipping (pip install flake8)")

    print("\n" + "=" * 72)
    print("Style check complete." if status == 0 else "Style issues found.")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
