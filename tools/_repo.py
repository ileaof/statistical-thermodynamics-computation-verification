"""Shared helpers for the repository maintenance tools.

Provides discovery of the repository root, the chapter directories and the
example programs, so that every tool agrees on what "the examples" are.
"""

from __future__ import annotations

import os
import re
from typing import List

#: Repository root = the parent of this ``tools`` directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CHAPTER_RE = re.compile(r"^Chapter\d{2}_")
_EXAMPLE_RE = re.compile(r"^ex\d+_\d+_.*\.py$")


def chapter_dirs() -> List[str]:
    """Return the absolute paths of the chapter directories, sorted by number."""
    dirs = [
        os.path.join(ROOT, name)
        for name in os.listdir(ROOT)
        if _CHAPTER_RE.match(name) and os.path.isdir(os.path.join(ROOT, name))
    ]
    return sorted(dirs)


def example_files() -> List[str]:
    """Return the absolute paths of every example program, in chapter order."""
    examples: List[str] = []
    for chapter in chapter_dirs():
        for name in sorted(os.listdir(chapter)):
            if _EXAMPLE_RE.match(name):
                examples.append(os.path.join(chapter, name))
    return examples


def relative(path: str) -> str:
    """Return ``path`` relative to the repository root (for tidy printing)."""
    return os.path.relpath(path, ROOT)
