# Coding style

The code is meant to be *read* as much as run. It is written for a graduate
student meeting each method for the first time, so clarity always wins over
cleverness. These conventions keep the repository uniform.

## Language and dependencies

- **Python 3.9+**, using only **NumPy**, **SciPy** and **Matplotlib**.
- No framework magic, no hidden state, no dependency that a reader would have to
  learn before understanding the physics.

## Formatting

- **PEP 8** throughout, checked with `flake8` (line length 100).
- **[Black](https://black.readthedocs.io/)**-compatible formatting; run
  `python tools/format_repository.py --fix` to apply it.
- Meaningful names: `partition_sum`, `mean_free_path`, `blocking_errors` &mdash;
  never `f2`, `tmp2`, `data3`.

## Documentation

- Every module and public function has a **NumPy-style docstring** with
  `Parameters`, `Returns`, and (where useful) `Notes` or `Examples` sections.
- Each example program opens with a module docstring stating the *physics*, the
  *equations*, and *what is verified* &mdash; not just what the code does.
- Comments explain the **why**, especially the reason a particular numerical
  choice is safe (log-gamma to avoid overflow, `expm1` for small arguments,
  shifted-force cutoffs for clean energy conservation).

## Numerical hygiene

- Use `scipy.special.gammaln` for factorials and binomials so nothing overflows.
- Use `numpy.expm1` / `numpy.log1p` near zero; subtract the maximum before
  exponentiating Boltzmann weights.
- Prefer stable closed forms (e.g. the logistic form of the Fermi function) over
  naive expressions that overflow.

## Reproducibility

- All randomness goes through a single seeded generator,
  `numpy.random.default_rng(20260723)`.
- No example depends on wall-clock time, machine locale, or thread count for its
  numerical result.

## Figures

- One restrained, colour-blind-friendly palette, defined in
  [`plotting.py`](../src/statistical_thermodynamics/plotting.py).
- Every axis is labelled with units; every figure carries a title; legends have
  no frame.
- Figures are saved at 200 dpi with `bbox_inches="tight"`.

## Structure of an example program

```python
#!/usr/bin/env python3
"""Example N.M (role) -- one-line title.

Physics, equations, and what is verified.
"""

import numpy as np
import matplotlib.pyplot as plt
# ... scipy imports as needed

# constants and small, named helper functions
def some_quantity(...):
    """NumPy-style docstring."""
    ...

def main():
    # 1. compute
    # 2. VERIFY against exact / limit / independent method, printing the check
    # 3. plot and save the figure

if __name__ == "__main__":
    main()
```

## The library

Functions that recur across chapters live in
[`src/statistical_thermodynamics/`](../src/statistical_thermodynamics/) rather
than being copied. The library is unit-tested (`pytest`), fully documented, and
ships a `py.typed` marker so type checkers see its annotations. The per-chapter
example programs remain deliberately self-contained; the library exists for
*your* reuse, not to fragment the pedagogy.
