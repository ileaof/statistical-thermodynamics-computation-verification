# Chapter 3 &mdash; Statistical Distributions and Partition Functions

> Companion code for Chapter 3 of *Statistical Thermodynamics: Theory,
> Computation, and Molecular Applications* by I. L. Ferreira.

## Overview

The canonical partition function as the bridge from microscopic energy levels to all thermodynamics. The two-level system and its Schottky anomaly, the three quantum statistics and their common classical limit, and the harmonic oscillator built explicitly by summation over levels.

Following the book's pattern, this chapter contains three programs: an
**analytical** result with a plot, a **direct numerical** computation, and an
**advanced**, research-grade verification study. Every program checks its output
against an exact result, a limiting case, or an independent method, and prints a
verification table to the console.

## Programs

| Program | Role | Computes | Verifies against |
|---|---|---|---|
| [`ex3_1_analytical.py`](ex3_1_analytical.py) | analytical | The two-level system: thermodynamics and the Schottky anomaly | the heat capacity computed three ways -- closed form, finite difference of U(T), and the energy-fluctuation formula |
| [`ex3_2_statistics.py`](ex3_2_statistics.py) | numerical | Maxwell-Boltzmann, Bose-Einstein and Fermi-Dirac occupations | the classical (dilute) limit: both quantum statistics collapse onto MB with a relative deviation that decays as e^(-x) |
| [`ex3_3_advanced.py`](ex3_3_advanced.py) | advanced | Harmonic-oscillator thermodynamics by direct summation of the partition function | geometric truncation convergence; the heat capacity three independent ways; the zero-point and equipartition limits |

Each program also writes a publication-quality figure (`fig3_1.png, fig3_2.png, fig3_3.png`) to its working
directory; `tools/build_all_figures.py` collects these into [`figures/`](figures/).

## Key concepts

- Canonical partition function
- Schottky anomaly
- Bose-Einstein & Fermi-Dirac statistics
- Energy-fluctuation heat capacity
- Zero-point energy

## Running

```bash
cd Chapter03_Statistical_Distributions
python ex3_1_analytical.py
```

Every program is self-contained and depends only on NumPy, SciPy and Matplotlib.
Stochastic programs fix their random seed, so results are reproducible to the
last digit.

## Reusable building blocks

The same physics is available as a documented, unit-tested library for your own
work:

- [`statistical_thermodynamics.partition_functions`](../src/statistical_thermodynamics/partition_functions.py)
- [`statistical_thermodynamics.quantum_statistics`](../src/statistical_thermodynamics/quantum_statistics.py)
- [`statistical_thermodynamics.probability`](../src/statistical_thermodynamics/probability.py)

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
