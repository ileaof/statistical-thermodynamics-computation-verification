# Chapter 9 &mdash; Phase Transitions and Critical Phenomena

> Companion code for Chapter 9 of *Statistical Thermodynamics: Theory,
> Computation, and Molecular Applications* by I. L. Ferreira.

## Overview

The Ising model as the paradigm of collective behaviour. Mean-field (Weiss) theory and its critical exponents, a vectorised Metropolis Monte Carlo simulation checked against Onsager's exact 2-D solution, and finite-size scaling with the Binder cumulant.

Following the book's pattern, this chapter contains three programs: an
**analytical** result with a plot, a **direct numerical** computation, and an
**advanced**, research-grade verification study. Every program checks its output
against an exact result, a limiting case, or an independent method, and prints a
verification table to the console.

## Programs

| Program | Role | Computes | Verifies against |
|---|---|---|---|
| [`ex9_1_meanfield.py`](ex9_1_meanfield.py) | analytical | Mean-field (Weiss) theory of the Ising ferromagnet | the critical exponents beta = 1/2 (magnetization) and gamma = 1 (susceptibility); the Landau free energy single-well to double-well |
| [`ex9_2_ising_mc.py`](ex9_2_ising_mc.py) | numerical | Monte Carlo of the 2-D Ising model (checkerboard Metropolis) | the critical temperature from the susceptibility peak against Onsager's T_c = 2.269; the magnetization against the exact Onsager curve |
| [`ex9_3_advanced.py`](ex9_3_advanced.py) | advanced | Finite-size scaling and critical exponents of the 2-D Ising model | T_c from the size-independent Binder-cumulant crossing; the exponent ratios gamma/nu = 7/4 and beta/nu = 1/8; a susceptibility data collapse |

Each program also writes a publication-quality figure (`fig9_1.png, fig9_2.png, fig9_3.png`) to its working
directory; `tools/build_all_figures.py` collects these into [`figures/`](figures/).

## Key concepts

- Mean-field theory
- Critical exponents
- Onsager's exact solution
- Binder cumulant
- Finite-size scaling

## Running

```bash
cd Chapter09_Phase_Transitions_and_Critical_Phenomena
python ex9_1_meanfield.py
```

Every program is self-contained and depends only on NumPy, SciPy and Matplotlib.
Stochastic programs fix their random seed, so results are reproducible to the
last digit.

## Reusable building blocks

The same physics is available as a documented, unit-tested library for your own
work:

- [`statistical_thermodynamics.numerical_methods`](../src/statistical_thermodynamics/numerical_methods.py)

## Directory contents

| Path | Purpose |
|---|---|
| `ex*.py` | The three example programs |
| [`figures/`](figures/) | Generated figures (built on demand) |
| [`data/`](data/) | Input or reference data, if any |
| [`results/`](results/) | Numerical output tables, if saved |

---

Back to the [main README](../README.md) &middot; browse all chapters in the
[repository structure](../docs/repository_structure.md).
