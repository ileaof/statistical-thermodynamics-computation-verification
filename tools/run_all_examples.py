#!/usr/bin/env python3
"""Run every example program and report which succeed.

Each example is executed as a separate process from within its own chapter
directory (so that its figure is written there), and its exit status and wall
time are recorded.  A non-zero exit code or a timeout counts as a failure.

Usage
-----
    python tools/run_all_examples.py            # run all examples
    python tools/run_all_examples.py --chapter 9
    python tools/run_all_examples.py --timeout 600

The script exits with status 1 if any example fails, so it is suitable for use
in continuous integration.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _repo import example_files, relative  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all example programs.")
    parser.add_argument("--chapter", type=int, default=None,
                        help="Only run examples from this chapter number.")
    parser.add_argument("--timeout", type=int, default=900,
                        help="Per-example timeout in seconds (default 900).")
    args = parser.parse_args()

    examples = example_files()
    if args.chapter is not None:
        tag = f"{os.sep}Chapter{args.chapter:02d}_"
        examples = [e for e in examples if tag in e]

    if not examples:
        print("No examples found.")
        return 1

    print(f"Running {len(examples)} example(s)...\n")
    failures = []
    for path in examples:
        rel = relative(path)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, os.path.basename(path)],
                cwd=os.path.dirname(path),
                capture_output=True, text=True, timeout=args.timeout,
            )
            dt = time.perf_counter() - t0
            if proc.returncode == 0:
                print(f"  PASS  {rel:<62} {dt:6.1f}s")
            else:
                print(f"  FAIL  {rel:<62} {dt:6.1f}s")
                failures.append((rel, proc.stderr.strip().splitlines()[-1:]))
        except subprocess.TimeoutExpired:
            print(f"  TIME  {rel:<62}  > {args.timeout}s")
            failures.append((rel, [f"timeout after {args.timeout}s"]))

    print("\n" + "=" * 72)
    if failures:
        print(f"{len(failures)} of {len(examples)} example(s) FAILED:")
        for rel, msg in failures:
            print(f"  - {rel}: {' '.join(msg)}")
        return 1
    print(f"All {len(examples)} example(s) ran successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
