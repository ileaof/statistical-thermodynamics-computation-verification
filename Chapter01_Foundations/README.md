# Chapter 1 &mdash; Foundations of Statistical Thermodynamics

> Companion code for Chapter 1 of *Statistical Thermodynamics: Theory,
> Computation, and Molecular Applications* by I. L. Ferreira.

## Overview

The microscopic origin of thermodynamics: microstates and macrostates, the multiplicity function, Boltzmann's entropy S = k_B ln Omega, and the emergence of the Boltzmann distribution from the postulate of equal a priori probabilities. The overarching lesson is why macroscopic states are so overwhelmingly sharp: fluctuations vanish as N^(-1/2).

Following the book's pattern, this chapter contains three programs: an
**analytical** result with a plot, a **direct numerical** computation, and an
**advanced**, research-grade verification study. Every program checks its output
against an exact result, a limiting case, or an independent method, and prints a
verification table to the console.

## Programs

| Program | Role | Computes | Verifies against |
|---|---|---|---|
| [`ex1_1_analytical.py`](ex1_1_analytical.py) | analytical | Multiplicity of a two-state (spin) system and its Gaussian limit | Gaussian (Stirling) approximation against exact binomial coefficients; the 1/sqrt(N) narrowing of the macrostate; entropy per spin to k_B ln 2 |
| [`ex1_2_einstein_solids.py`](ex1_2_einstein_solids.py) | numerical | Two Einstein solids exchanging energy quanta; the equilibrium distribution | the Vandermonde sum rule; the most-probable partition q* = qN_A/(N_A+N_B); equal statistical temperatures at the peak; sharpening as N^(-1/2) |
| [`ex1_3_advanced.py`](ex1_3_advanced.py) | advanced | Emergence of the Boltzmann distribution from equal a priori probabilities | reservoir-to-Boltzmann convergence at order N^(-1); a uniform-microstate Monte Carlo with the expected n^(-1/2) error; the recovered temperature |

Each program also writes a publication-quality figure (`fig1_1.png, fig1_2.png, fig1_3.png`) to its working
directory; `tools/build_all_figures.py` collects these into [`figures/`](figures/).

## Key concepts

- Microstates & macrostates
- Multiplicity and Boltzmann entropy
- Stirling's approximation
- Thermal contact and temperature
- Equal a priori probabilities

## Running

```bash
cd Chapter01_Foundations
python ex1_1_analytical.py
```

Every program is self-contained and depends only on NumPy, SciPy and Matplotlib.
Stochastic programs fix their random seed, so results are reproducible to the
last digit.

## Reusable building blocks

The same physics is available as a documented, unit-tested library for your own
work:

- [`statistical_thermodynamics.probability`](../src/statistical_thermodynamics/probability.py)
- [`statistical_thermodynamics.partition_functions`](../src/statistical_thermodynamics/partition_functions.py)

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
