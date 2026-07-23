# Chapter 5 &mdash; Diatomic and Polyatomic Ideal Gases

> Companion code for Chapter 5 of *Statistical Thermodynamics: Theory,
> Computation, and Molecular Applications* by I. L. Ferreira.

## Overview

Internal molecular structure and its thermodynamic fingerprint: the translation-rotation-vibration heat-capacity staircase, the rotational partition function and its high-temperature expansion, and hindered internal rotation solved by matrix diagonalization.

Following the book's pattern, this chapter contains three programs: an
**analytical** result with a plot, a **direct numerical** computation, and an
**advanced**, research-grade verification study. Every program checks its output
against an exact result, a limiting case, or an independent method, and prints a
verification table to the console.

## Programs

| Program | Role | Computes | Verifies against |
|---|---|---|---|
| [`ex5_1_analytical.py`](ex5_1_analytical.py) | analytical | The heat-capacity staircase of a diatomic gas | the 3/2 -> 5/2 -> 7/2 plateau values against equipartition as each mode unlocks above its characteristic temperature |
| [`ex5_2_rotation.py`](ex5_2_rotation.py) | numerical | The rotational partition function: exact sum vs Euler-Maclaurin expansion | the failure of the classical formula for light molecules at low T and its restoration by the high-temperature correction terms |
| [`ex5_3_advanced.py`](ex5_3_advanced.py) | advanced | Hindered internal rotation by matrix diagonalization in a Fourier basis | the free-rotor limit E_m = B m^2; spectral convergence with basis size; the high-barrier torsional-oscillator spacing n sqrt(B V0) |

Each program also writes a publication-quality figure (`fig5_1.png, fig5_2.png, fig5_3.png`) to its working
directory; `tools/build_all_figures.py` collects these into [`figures/`](figures/).

## Key concepts

- Characteristic temperatures
- Rotational partition function
- Euler-Maclaurin expansion
- Symmetry number
- Hindered internal rotation

## Running

```bash
cd Chapter05_Diatomic_and_Polyatomic_Gases
python ex5_1_analytical.py
```

Every program is self-contained and depends only on NumPy, SciPy and Matplotlib.
Stochastic programs fix their random seed, so results are reproducible to the
last digit.

## Reusable building blocks

The same physics is available as a documented, unit-tested library for your own
work:

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
