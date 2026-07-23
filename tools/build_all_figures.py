#!/usr/bin/env python3
"""Regenerate every figure and collect the PNGs into each chapter's ``figures/``.

Each example writes its figure (``figN_M.png``) to its working directory when
run.  This tool runs every example from within its chapter directory and then
moves the freshly produced PNG files into ``<chapter>/figures/`` so that the
generated artwork is tidily organised.

Usage
-----
    python tools/build_all_figures.py
    python tools/build_all_figures.py --chapter 6

Requires the same dependencies as the examples (NumPy, SciPy, Matplotlib).
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _repo import chapter_dirs, example_files, relative  # noqa: E402


def _pngs_in(directory: str) -> set:
    return set(glob.glob(os.path.join(directory, "*.png")))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build all figures.")
    parser.add_argument("--chapter", type=int, default=None,
                        help="Only build figures for this chapter number.")
    parser.add_argument("--timeout", type=int, default=900,
                        help="Per-example timeout in seconds (default 900).")
    args = parser.parse_args()

    examples = example_files()
    if args.chapter is not None:
        tag = f"{os.sep}Chapter{args.chapter:02d}_"
        examples = [e for e in examples if tag in e]

    built, failures = 0, []
    for path in examples:
        chapter = os.path.dirname(path)
        before = _pngs_in(chapter)
        print(f"  building {relative(path)} ...")
        try:
            proc = subprocess.run(
                [sys.executable, os.path.basename(path)],
                cwd=chapter, capture_output=True, text=True, timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            failures.append(relative(path))
            continue
        if proc.returncode != 0:
            failures.append(relative(path))
            continue

        new_pngs = _pngs_in(chapter) - before
        figures_dir = os.path.join(chapter, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        for png in sorted(new_pngs):
            dest = os.path.join(figures_dir, os.path.basename(png))
            shutil.move(png, dest)
            print(f"      -> {relative(dest)}")
            built += 1

    print("\n" + "=" * 72)
    print(f"Built {built} figure(s) across {len(chapter_dirs())} chapters.")
    if failures:
        print(f"{len(failures)} example(s) failed to run:")
        for rel in failures:
            print(f"  - {rel}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
