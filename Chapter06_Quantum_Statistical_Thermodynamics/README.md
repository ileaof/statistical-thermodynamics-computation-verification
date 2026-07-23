# Chapter 6 &mdash; Quantum Statistical Thermodynamics

> Companion code for Chapter 6 of *Statistical Thermodynamics: Theory,
> Computation, and Molecular Applications* by I. L. Ferreira.

## Overview

The three canonical quantum gases. Blackbody radiation as a photon gas (Planck, Stefan-Boltzmann, Wien), the ideal Fermi gas across the degenerate-to-classical crossover, and Bose-Einstein condensation with its characteristic cusp in the heat capacity.

Following the book's pattern, this chapter contains three programs: an
**analytical** result with a plot, a **direct numerical** computation, and an
**advanced**, research-grade verification study. Every program checks its output
against an exact result, a limiting case, or an independent method, and prints a
verification table to the console.

## Programs

| Program | Role | Computes | Verifies against |
|---|---|---|---|
| [`ex6_1_analytical.py`](ex6_1_analytical.py) | analytical | The photon gas: Planck's law, Stefan-Boltzmann and Wien displacement | the Stefan-Boltzmann constant recovered by integrating the Planck spectrum (matching CODATA); the Wien peaks by root finding |
| [`ex6_2_fermi.py`](ex6_2_fermi.py) | numerical | The ideal Fermi gas: chemical potential, energy and heat capacity | the low-T Sommerfeld expansion of mu(T); the linear C ~ (pi^2/2)T at low T and the classical 3/2 at high T |
| [`ex6_3_advanced.py`](ex6_3_advanced.py) | advanced | Bose-Einstein condensation of the ideal Bose gas | the Bose functions against zeta at z=1; the condensate fraction 1 - t^(3/2); the value C(T_c)/Nk = 1.926 and the slope discontinuity (cusp) |

Each program also writes a publication-quality figure (`fig6_1.png, fig6_2.png, fig6_3.png`) to its working
directory; `tools/build_all_figures.py` collects these into [`figures/`](figures/).

## Key concepts

- Planck spectrum
- Stefan-Boltzmann law
- Fermi-Dirac degeneracy
- Sommerfeld expansion
- Bose-Einstein condensation

## Running

```bash
cd Chapter06_Quantum_Statistical_Thermodynamics
python ex6_1_analytical.py
```

Every program is self-contained and depends only on NumPy, SciPy and Matplotlib.
Stochastic programs fix their random seed, so results are reproducible to the
last digit.

## Reusable building blocks

The same physics is available as a documented, unit-tested library for your own
work:

- [`statistical_thermodynamics.quantum_statistics`](../src/statistical_thermodynamics/quantum_statistics.py)
- [`statistical_thermodynamics.constants`](../src/statistical_thermodynamics/constants.py)

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
