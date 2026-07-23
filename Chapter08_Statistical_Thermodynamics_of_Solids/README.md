# Chapter 8 &mdash; Statistical Thermodynamics of Solids

> Companion code for Chapter 8 of *Statistical Thermodynamics: Theory,
> Computation, and Molecular Applications* by I. L. Ferreira.

## Overview

The heat capacity of crystalline solids. Einstein's single-frequency model, Debye's elastic-continuum model with its low-temperature T^3 law, and the true phonon density of states computed from a lattice dispersion relation, complete with van Hove singularities.

Following the book's pattern, this chapter contains three programs: an
**analytical** result with a plot, a **direct numerical** computation, and an
**advanced**, research-grade verification study. Every program checks its output
against an exact result, a limiting case, or an independent method, and prints a
verification table to the console.

## Programs

| Program | Role | Computes | Verifies against |
|---|---|---|---|
| [`ex8_1_einstein.py`](ex8_1_einstein.py) | analytical | The Einstein model of a solid | the high-temperature Dulong-Petit limit 3R and the low-temperature exponential freeze-out; comparison with diamond heat-capacity data |
| [`ex8_2_debye.py`](ex8_2_debye.py) | numerical | The Debye model and the T^3 law | the low-T coefficient (12 pi^4/5) R (T/theta_D)^3 by numerical integration; the high-T Dulong-Petit limit; a comparison with the Einstein model |
| [`ex8_3_advanced.py`](ex8_3_advanced.py) | advanced | Phonon density of states and heat capacity from a lattice dispersion | the density of states normalisation; the long-wavelength sound speed and Debye T^3 law; the high-T Dulong-Petit limit per mode |

Each program also writes a publication-quality figure (`fig8_1.png, fig8_2.png, fig8_3.png`) to its working
directory; `tools/build_all_figures.py` collects these into [`figures/`](figures/).

## Key concepts

- Einstein model
- Debye model & the T^3 law
- Dulong-Petit limit
- Phonon density of states
- van Hove singularities

## Running

```bash
cd Chapter08_Statistical_Thermodynamics_of_Solids
python ex8_1_einstein.py
```

Every program is self-contained and depends only on NumPy, SciPy and Matplotlib.
Stochastic programs fix their random seed, so results are reproducible to the
last digit.

## Reusable building blocks

The same physics is available as a documented, unit-tested library for your own
work:

- [`statistical_thermodynamics.solids`](../src/statistical_thermodynamics/solids.py)
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
