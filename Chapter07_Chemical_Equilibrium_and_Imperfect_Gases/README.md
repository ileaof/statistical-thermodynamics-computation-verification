# Chapter 7 &mdash; Chemical Equilibrium and Imperfect Gases

> Companion code for Chapter 7 of *Statistical Thermodynamics: Theory,
> Computation, and Molecular Applications* by I. L. Ferreira.

## Overview

Statistical equilibrium constants from molecular partition functions, and the departure from ideality. Dissociation equilibrium of H2, the second virial coefficient of a Lennard-Jones gas, and hard-sphere virial coefficients evaluated by Monte Carlo cluster integrals.

Following the book's pattern, this chapter contains three programs: an
**analytical** result with a plot, a **direct numerical** computation, and an
**advanced**, research-grade verification study. Every program checks its output
against an exact result, a limiting case, or an independent method, and prints a
verification table to the console.

## Programs

| Program | Role | Computes | Verifies against |
|---|---|---|---|
| [`ex7_1_analytical.py`](ex7_1_analytical.py) | analytical | Chemical equilibrium H2 <-> 2H from molecular partition functions | the reaction enthalpy computed two ways -- the van't Hoff derivative of ln K_p and the direct molecular enthalpies -- in agreement |
| [`ex7_2_virial.py`](ex7_2_virial.py) | numerical | The second virial coefficient of a Lennard-Jones gas | the Boyle temperature T*_B = 3.418 by root finding; comparison of the reduced coefficient with experimental data for argon |
| [`ex7_3_advanced.py`](ex7_3_advanced.py) | advanced | Hard-sphere virial coefficients by Monte Carlo Mayer cluster integrals | the exact B2; the three-body B3 with a 1/sqrt(N) error bar; the truncated virial equation of state against the Carnahan-Starling formula |

Each program also writes a publication-quality figure (`fig7_1.png, fig7_2.png, fig7_3.png`) to its working
directory; `tools/build_all_figures.py` collects these into [`figures/`](figures/).

## Key concepts

- Statistical equilibrium constant
- van't Hoff equation
- Second virial coefficient
- Mayer cluster integrals
- Carnahan-Starling equation of state

## Running

```bash
cd Chapter07_Chemical_Equilibrium_and_Imperfect_Gases
python ex7_1_analytical.py
```

Every program is self-contained and depends only on NumPy, SciPy and Matplotlib.
Stochastic programs fix their random seed, so results are reproducible to the
last digit.

## Reusable building blocks

The same physics is available as a documented, unit-tested library for your own
work:

- [`statistical_thermodynamics.equilibrium`](../src/statistical_thermodynamics/equilibrium.py)
- [`statistical_thermodynamics.potentials`](../src/statistical_thermodynamics/potentials.py)
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
