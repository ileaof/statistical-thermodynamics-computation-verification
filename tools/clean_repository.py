#!/usr/bin/env python3
"""Remove generated artefacts, leaving the tracked source untouched.

Deletes Python caches, packaging build directories, and stray figure PNGs that
examples leave in a chapter's top level.  Curated figures under any ``figures/``
directory and the ``.gitkeep`` placeholders are preserved.

Usage
-----
    python tools/clean_repository.py            # show what would be removed
    python tools/clean_repository.py --force     # actually remove

By default the script performs a dry run and only lists the artefacts it would
delete; pass ``--force`` to delete them.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _repo import ROOT, relative  # noqa: E402

# Directory names removed wherever they occur.
_DIR_PATTERNS = ["__pycache__", ".pytest_cache", ".mypy_cache",
                 "*.egg-info", "build", "dist"]
# File patterns removed wherever they occur.
_FILE_PATTERNS = ["*.pyc", "*.pyo"]


def _matches(name: str, patterns) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _collect():
    """Return (dirs, files) of artefacts to remove."""
    dirs, files = [], []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        if os.sep + ".git" in dirpath + os.sep:
            continue
        for d in list(dirnames):
            if _matches(d, _DIR_PATTERNS):
                dirs.append(os.path.join(dirpath, d))
                dirnames.remove(d)  # do not descend into a doomed directory
        for f in filenames:
            if _matches(f, _FILE_PATTERNS):
                files.append(os.path.join(dirpath, f))
            # stray figures written to a chapter's top level (not in figures/)
            elif (fnmatch.fnmatch(f, "fig*.png")
                  and os.path.basename(dirpath).startswith("Chapter")):
                files.append(os.path.join(dirpath, f))
    return dirs, files


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean generated artefacts.")
    parser.add_argument("--force", action="store_true",
                        help="Actually delete (default is a dry run).")
    args = parser.parse_args()

    dirs, files = _collect()
    if not dirs and not files:
        print("Nothing to clean -- the repository is already tidy.")
        return 0

    verb = "Removing" if args.force else "Would remove"
    for path in sorted(dirs) + sorted(files):
        print(f"  {verb}: {relative(path)}")
        if args.force:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.exists(path):
                os.remove(path)

    print("\n" + "=" * 72)
    total = len(dirs) + len(files)
    if args.force:
        print(f"Removed {total} artefact(s).")
    else:
        print(f"{total} artefact(s) would be removed. Re-run with --force.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
