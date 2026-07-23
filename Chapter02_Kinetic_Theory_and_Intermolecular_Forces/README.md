# Chapter 2 &mdash; Kinetic Theory of Gases and Intermolecular Forces

> Companion code for Chapter 2 of *Statistical Thermodynamics: Theory,
> Computation, and Molecular Applications* by I. L. Ferreira.

## Overview

From the Maxwell-Boltzmann velocity distribution to transport and real molecular interactions. Characteristic speeds, the sqrt(2) factor in the relative speed, the collision frequency and mean free path, and a full molecular-dynamics study of the Lennard-Jones fluid.

Following the book's pattern, this chapter contains three programs: an
**analytical** result with a plot, a **direct numerical** computation, and an
**advanced**, research-grade verification study. Every program checks its output
against an exact result, a limiting case, or an independent method, and prints a
verification table to the console.

## Programs

| Program | Role | Computes | Verifies against |
|---|---|---|---|
| [`ex2_1_analytical.py`](ex2_1_analytical.py) | analytical | The Maxwell-Boltzmann speed distribution and the three characteristic speeds | normalisation and the first two moments by quadrature against closed forms; the fixed ratios v_p : <v> : v_rms |
| [`ex2_2_meanfreepath.py`](ex2_2_meanfreepath.py) | numerical | Relative speed, collision frequency and mean free path | the Maxwell sqrt(2) factor by Monte Carlo (with the 1/sqrt(N) error law); the 1/P scaling of the mean free path of air |
| [`ex2_3_advanced.py`](ex2_3_advanced.py) | advanced | Molecular dynamics of a Lennard-Jones fluid (velocity-Verlet) | symplectic energy conservation of order dt^2; the emergence of a Maxwell-Boltzmann velocity distribution; equipartition; the radial distribution g(r) |

Each program also writes a publication-quality figure (`fig2_1.png, fig2_2.png, fig2_3.png`) to its working
directory; `tools/build_all_figures.py` collects these into [`figures/`](figures/).

## Key concepts

- Maxwell-Boltzmann distribution
- Mean free path & collisions
- Lennard-Jones potential
- Velocity-Verlet integration
- Radial distribution function

## Running

```bash
cd Chapter02_Kinetic_Theory_and_Intermolecular_Forces
python ex2_1_analytical.py
```

Every program is self-contained and depends only on NumPy, SciPy and Matplotlib.
Stochastic programs fix their random seed, so results are reproducible to the
last digit.

## Reusable building blocks

The same physics is available as a documented, unit-tested library for your own
work:

- [`statistical_thermodynamics.kinetic_theory`](../src/statistical_thermodynamics/kinetic_theory.py)
- [`statistical_thermodynamics.transport`](../src/statistical_thermodynamics/transport.py)
- [`statistical_thermodynamics.potentials`](../src/statistical_thermodynamics/potentials.py)

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
