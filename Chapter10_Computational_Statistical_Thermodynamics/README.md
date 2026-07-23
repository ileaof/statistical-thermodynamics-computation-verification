# Chapter 10 &mdash; Computational Statistical Thermodynamics

> Companion code for Chapter 10 of *Statistical Thermodynamics: Theory,
> Computation, and Molecular Applications* by I. L. Ferreira.

## Overview

The methodology that underpins every simulation in the book: Monte Carlo integration and importance sampling, the Metropolis algorithm with honest autocorrelation-corrected error bars, and a complete verification, validation and reproducibility campaign.

Following the book's pattern, this chapter contains three programs: an
**analytical** result with a plot, a **direct numerical** computation, and an
**advanced**, research-grade verification study. Every program checks its output
against an exact result, a limiting case, or an independent method, and prints a
verification table to the console.

## Programs

| Program | Role | Computes | Verifies against |
|---|---|---|---|
| [`ex10_1_importance.py`](ex10_1_importance.py) | analytical | Monte Carlo thermal averages and the power of importance sampling | both estimators converge as 1/sqrt(N); importance sampling achieves the same accuracy with a far smaller variance |
| [`ex10_2_metropolis.py`](ex10_2_metropolis.py) | numerical | The Metropolis algorithm: sampling, autocorrelation and error bars | the sampled histogram against the exact Boltzmann distribution; the integrated autocorrelation time; block averaging to a plateau |
| [`ex10_3_advanced.py`](ex10_3_advanced.py) | advanced | A verification, validation and reproducibility campaign | validation against quadrature; blocking and bootstrap error estimates in agreement; the pull distribution proving the error bars are honest |

Each program also writes a publication-quality figure (`fig10_1.png, fig10_2.png, fig10_3.png`) to its working
directory; `tools/build_all_figures.py` collects these into [`figures/`](figures/).

## Key concepts

- Importance sampling
- Metropolis algorithm
- Autocorrelation time
- Blocking & bootstrap errors
- Verification & validation

## Running

```bash
cd Chapter10_Computational_Statistical_Thermodynamics
python ex10_1_importance.py
```

Every program is self-contained and depends only on NumPy, SciPy and Matplotlib.
Stochastic programs fix their random seed, so results are reproducible to the
last digit.

## Reusable building blocks

The same physics is available as a documented, unit-tested library for your own
work:

- [`statistical_thermodynamics.numerical_methods`](../src/statistical_thermodynamics/numerical_methods.py)
- [`statistical_thermodynamics.utilities`](../src/statistical_thermodynamics/utilities.py)

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
