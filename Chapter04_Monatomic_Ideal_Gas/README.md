# Chapter 4 &mdash; The Monatomic Ideal Gas

> Companion code for Chapter 4 of *Statistical Thermodynamics: Theory,
> Computation, and Molecular Applications* by I. L. Ferreira.

## Overview

The translational partition function, the absolute (Sackur-Tetrode) entropy tested against experiment, the justification for treating translation classically, and the role of the Gibbs 1/N! factor in making entropy extensive and resolving the Gibbs paradox.

Following the book's pattern, this chapter contains three programs: an
**analytical** result with a plot, a **direct numerical** computation, and an
**advanced**, research-grade verification study. Every program checks its output
against an exact result, a limiting case, or an independent method, and prints a
verification table to the console.

## Programs

| Program | Role | Computes | Verifies against |
|---|---|---|---|
| [`ex4_1_analytical.py`](ex4_1_analytical.py) | analytical | The Sackur-Tetrode absolute entropy of the noble gases | the purely theoretical molar entropy against experimental standard entropies for He...Xe; the (3/2)R ln M mass scaling |
| [`ex4_2_boxsum.py`](ex4_2_boxsum.py) | numerical | The translational partition function: discrete sum vs classical continuum | the sqrt(alpha/pi) relative-error law of the continuum approximation and its utter negligibility for a macroscopic box |
| [`ex4_3_advanced.py`](ex4_3_advanced.py) | advanced | Chemical potential, extensivity and the Gibbs paradox | extensivity along a fixed-density line; the entropy of mixing (2N k_B ln 2 vs zero); the chemical potential by finite difference; C_p - C_V = Nk_B, gamma = 5/3 |

Each program also writes a publication-quality figure (`fig4_1.png, fig4_2.png, fig4_3.png`) to its working
directory; `tools/build_all_figures.py` collects these into [`figures/`](figures/).

## Key concepts

- Thermal de Broglie wavelength
- Sackur-Tetrode entropy
- Indistinguishability & the 1/N! factor
- Extensivity
- The Gibbs paradox

## Running

```bash
cd Chapter04_Monatomic_Ideal_Gas
python ex4_1_analytical.py
```

Every program is self-contained and depends only on NumPy, SciPy and Matplotlib.
Stochastic programs fix their random seed, so results are reproducible to the
last digit.

## Reusable building blocks

The same physics is available as a documented, unit-tested library for your own
work:

- [`statistical_thermodynamics.thermodynamics`](../src/statistical_thermodynamics/thermodynamics.py)
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
