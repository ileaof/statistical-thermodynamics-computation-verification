#!/usr/bin/env python3
"""Repository integrity check used by continuous integration.

Confirms that the project has the structure a reader (and the book) expects:

* the ten chapter directories exist, each with a README, three example programs,
  and the ``figures``/``data``/``results`` subdirectories;
* every required top-level file is present;
* the ``statistical_thermodynamics`` package imports and exposes its modules.

Exits non-zero (listing every problem) if anything is missing, so it doubles as a
pre-release checklist. Run it locally with ``python .github/scripts/check_integrity.py``.
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REQUIRED_FILES = [
    "README.md", "LICENSE", "requirements.txt", "pyproject.toml",
    "CITATION.cff", "CONTRIBUTING.md", "CHANGELOG.md",
    "CODE_OF_CONDUCT.md", "SECURITY.md", "AUTHORS.md", ".gitignore",
]

EXPECTED_CHAPTERS = {
    1: "Chapter01_Foundations",
    2: "Chapter02_Kinetic_Theory_and_Intermolecular_Forces",
    3: "Chapter03_Statistical_Distributions",
    4: "Chapter04_Monatomic_Ideal_Gas",
    5: "Chapter05_Diatomic_and_Polyatomic_Gases",
    6: "Chapter06_Quantum_Statistical_Thermodynamics",
    7: "Chapter07_Chemical_Equilibrium_and_Imperfect_Gases",
    8: "Chapter08_Statistical_Thermodynamics_of_Solids",
    9: "Chapter09_Phase_Transitions_and_Critical_Phenomena",
    10: "Chapter10_Computational_Statistical_Thermodynamics",
}

LIBRARY_MODULES = [
    "constants", "partition_functions", "probability", "thermodynamics",
    "kinetic_theory", "transport", "potentials", "quantum_statistics",
    "solids", "equilibrium", "numerical_methods", "plotting", "utilities",
]

_EXAMPLE_RE = re.compile(r"^ex\d+_\d+_.*\.py$")


def main() -> int:
    problems = []

    # --- required top-level files --------------------------------------
    for name in REQUIRED_FILES:
        if not os.path.isfile(os.path.join(ROOT, name)):
            problems.append(f"missing required file: {name}")

    # --- chapter directories -------------------------------------------
    for num, dirname in EXPECTED_CHAPTERS.items():
        cdir = os.path.join(ROOT, dirname)
        if not os.path.isdir(cdir):
            problems.append(f"missing chapter directory: {dirname}")
            continue
        if not os.path.isfile(os.path.join(cdir, "README.md")):
            problems.append(f"{dirname}: missing README.md")
        for sub in ("figures", "data", "results"):
            if not os.path.isdir(os.path.join(cdir, sub)):
                problems.append(f"{dirname}: missing {sub}/ subdirectory")
        examples = [f for f in os.listdir(cdir) if _EXAMPLE_RE.match(f)]
        if len(examples) != 3:
            problems.append(
                f"{dirname}: expected 3 example programs, found {len(examples)}"
            )

    # --- library import -------------------------------------------------
    sys.path.insert(0, os.path.join(ROOT, "src"))
    try:
        import statistical_thermodynamics as st
        for mod in LIBRARY_MODULES:
            if not hasattr(st, mod):
                problems.append(f"library: missing module '{mod}'")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"library import failed: {exc}")

    # --- report ---------------------------------------------------------
    if problems:
        print("Repository integrity check FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("Repository integrity check passed:")
    print(f"  {len(EXPECTED_CHAPTERS)} chapters, "
          f"{len(LIBRARY_MODULES)} library modules, "
          f"{len(REQUIRED_FILES)} required files -- all present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
