# Statistical Thermodynamics: Theory, Computation, and Molecular Applications

### Companion Code Repository

**A Computational Approach with Python** &mdash; by **I. L. Ferreira**

[![License: MIT](https://img.shields.io/badge/License-MIT-2c3e50.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-2980b9.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Built with NumPy · SciPy · Matplotlib](https://img.shields.io/badge/built%20with-NumPy%20·%20SciPy%20·%20Matplotlib-c0392b.svg)](#requirements)
[![Reproducible](https://img.shields.io/badge/results-reproducible-27ae60.svg)](#reproducibility)
[![Tests](https://github.com/ileaof/statistical-thermodynamics-computation-verification/actions/workflows/tests.yml/badge.svg)](https://github.com/ileaof/statistical-thermodynamics-computation-verification/actions/workflows/tests.yml)
[![Repository integrity](https://github.com/ileaof/statistical-thermodynamics-computation-verification/actions/workflows/integrity.yml/badge.svg)](https://github.com/ileaof/statistical-thermodynamics-computation-verification/actions/workflows/integrity.yml)

## Repository

<https://github.com/ileaof/statistical-thermodynamics-computation-verification>

This repository contains the complete, executable Python programs accompanying the
book *Statistical Thermodynamics: Theory, Computation, and Molecular Applications —
A Computational Approach with Python*. Every computational example in the book is
provided here in full, exactly as printed, so that any reader can reproduce its
results and figures to the last digit.

## Table of contents

- [Overview](#overview)
- [About the book](#about-the-book)
- [Guiding principle](#guiding-principle)
- [Repository philosophy](#repository-philosophy)
- [Main features](#main-features)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Creating a virtual environment](#creating-a-virtual-environment)
  - [Installing dependencies](#installing-dependencies)
- [Running the programs](#running-the-programs)
- [Repository structure](#repository-structure)
- [Chapter-by-chapter guide](#chapter-by-chapter-guide)
- [The reusable library](#the-reusable-library)
- [Educational objectives](#educational-objectives)
- [Intended audience](#intended-audience)
- [Computational methodology](#computational-methodology)
- [Verification philosophy](#verification-philosophy)
- [Naming convention](#naming-convention)
- [Reproducibility](#reproducibility)
- [Numerical accuracy](#numerical-accuracy)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Overview

The code is organized one directory per chapter, with three programs each, and a
parallel, unit-tested [`statistical_thermodynamics`](src/statistical_thermodynamics)
library that collects the shared physics for reuse. Documentation, maintenance
tools and continuous-integration workflows keep the whole project maintainable as
the official companion website of the book.

## About the book

*Statistical Thermodynamics: Theory, Computation, and Molecular Applications*
develops the subject from its microscopic foundations to modern computational
practice. Each chapter pairs the theory with runnable Python that computes, and
then **verifies**, the key results — from the multiplicity of a two-state system
to Bose-Einstein condensation, from the Sackur-Tetrode entropy to the finite-size
scaling of the Ising model.

The book comprises **ten chapters** and eight appendices (Probability Theory;
Classical Thermodynamics Review; Quantum Mechanics Review; Mathematical Functions;
Numerical Methods; Python Programming Essentials; Physical Constants; Statistical
Tables). Every chapter closes with a summary, key equations, important concepts,
engineering applications, common mistakes, solved problems, exercises,
computational exercises, programming projects, and further reading.

## Guiding principle

> *A computed number is an opinion until it has been verified.*

Every program checks its results against an exact result, a limiting case, or an
independent method, and each stochastic program fixes its random seed so that its
output is fully reproducible.

## Repository philosophy

- **Fidelity to the book.** The programs are the ones printed in the text; their
  numerical output is authoritative and reproducible.
- **Verification first.** No result is reported without an independent check
  (see [docs/verification.md](docs/verification.md)).
- **Self-contained examples.** Each `ex*.py` runs on its own with only NumPy,
  SciPy and Matplotlib — ideal for reading, teaching and adapting.
- **Reuse without fragmentation.** Shared physics also lives in an installable,
  tested library, so readers can build on the same verified components.
- **Built to last.** Documentation, tests, tooling and CI make this a durable
  companion, not a one-off code dump.

## Main features

| | |
|---|---|
| 📘 **30 worked programs** | Three per chapter: analytical, direct numerical, and advanced verification study |
| ✅ **Self-verifying output** | Each program prints a table comparing its result with an exact or independent reference |
| 📦 **Reusable library** | `statistical_thermodynamics` — 13 documented, `py.typed` modules |
| 🧪 **Unit-tested** | A `pytest` suite covering every library module |
| 📊 **Publication-quality figures** | One consistent, colour-blind-friendly style, saved at 200 dpi |
| 🔁 **Fully reproducible** | Fixed random seeds; identical output on every run and platform |
| 🛠️ **Maintenance tools** | Run all examples, rebuild all figures, format and clean the repo |
| ⚙️ **Continuous integration** | Syntax, style, tests (Python 3.9–3.12), Markdown, integrity |

## Requirements

The programs use only the standard scientific-Python stack:

- Python 3.9 or later
- NumPy
- SciPy
- Matplotlib

Install with:

```bash
pip install numpy scipy matplotlib
```

| Component | Minimum version |
|---|---|
| Python | 3.9 (3.10–3.12 supported) |
| NumPy | 1.21 |
| SciPy | 1.7 |
| Matplotlib | 3.4 |

## Installation

Clone the repository:

```bash
git clone https://github.com/ileaof/statistical-thermodynamics-computation-verification.git
cd statistical-thermodynamics-computation-verification
```

### Creating a virtual environment

Keeping the project's packages isolated avoids version clashes.

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Installing dependencies

```bash
pip install -r requirements.txt        # runtime dependencies only
# or, to also install the reusable library and dev tools:
pip install -e ".[dev]"
```

Full details, including a Conda recipe and troubleshooting, are in
[docs/installation.md](docs/installation.md).

## Running the programs

Each program is self-contained and is run directly:

```bash
cd Chapter04_Monatomic_Ideal_Gas
python ex4_1_analytical.py
```

It prints its verification tables to the console and writes its figure (a `.png`
file) to the working directory.

To run everything at once:

```bash
python tools/run_all_examples.py       # execute every example, report pass/fail
python tools/build_all_figures.py      # regenerate every figure into figures/
```

## Repository structure

The code is organized one directory per chapter. Each chapter contains three
programs following the book's pattern: an **analytical** result with a plot, a
**direct numerical** computation, and an **advanced** research-grade study built as a
verification campaign.

| Chapter | Directory | Programs |
|---|---|---|
| 1. Foundations of Statistical Thermodynamics | [`Chapter01_Foundations/`](Chapter01_Foundations) | two-state multiplicity; Einstein solids; Boltzmann distribution verification |
| 2. Kinetic Theory of Gases and Intermolecular Forces | [`Chapter02_Kinetic_Theory_and_Intermolecular_Forces/`](Chapter02_Kinetic_Theory_and_Intermolecular_Forces) | Maxwell-Boltzmann distribution; mean free path Monte Carlo; Lennard-Jones molecular dynamics |
| 3. Statistical Distributions and Partition Functions | [`Chapter03_Statistical_Distributions/`](Chapter03_Statistical_Distributions) | two-level system; MB/BE/FD statistics; oscillator partition function |
| 4. Monatomic Ideal Gas | [`Chapter04_Monatomic_Ideal_Gas/`](Chapter04_Monatomic_Ideal_Gas) | Sackur-Tetrode entropy; partition-function summation; Gibbs paradox |
| 5. Diatomic and Polyatomic Ideal Gases | [`Chapter05_Diatomic_and_Polyatomic_Gases/`](Chapter05_Diatomic_and_Polyatomic_Gases) | heat-capacity staircase; rotational partition function; hindered rotor |
| 6. Quantum Statistical Thermodynamics | [`Chapter06_Quantum_Statistical_Thermodynamics/`](Chapter06_Quantum_Statistical_Thermodynamics) | photon gas / Planck; Fermi gas; Bose-Einstein condensation |
| 7. Chemical Equilibrium and Imperfect Gases | [`Chapter07_Chemical_Equilibrium_and_Imperfect_Gases/`](Chapter07_Chemical_Equilibrium_and_Imperfect_Gases) | dissociation equilibrium; virial coefficient; cluster-integral Monte Carlo |
| 8. Statistical Thermodynamics of Solids | [`Chapter08_Statistical_Thermodynamics_of_Solids/`](Chapter08_Statistical_Thermodynamics_of_Solids) | Einstein model; Debye model; phonon density of states |
| 9. Phase Transitions and Critical Phenomena | [`Chapter09_Phase_Transitions_and_Critical_Phenomena/`](Chapter09_Phase_Transitions_and_Critical_Phenomena) | mean-field Ising; Ising Monte Carlo; finite-size scaling |
| 10. Computational Statistical Thermodynamics | [`Chapter10_Computational_Statistical_Thermodynamics/`](Chapter10_Computational_Statistical_Thermodynamics) | importance sampling; Metropolis algorithm; verification/validation/reproducibility |

Each chapter directory follows the same template — a `README.md`, three `ex*.py`
programs, and `figures/`, `data/`, `results/` subdirectories. See
[docs/repository_structure.md](docs/repository_structure.md) for the full map.

## Chapter-by-chapter guide

<details>
<summary><strong>Topics covered in each chapter</strong> (click to expand)</summary>

- **1. Foundations of Statistical Thermodynamics** — historical development;
  microscopic and macroscopic descriptions; probability concepts; microstates and
  macrostates; ensembles; the Boltzmann postulate; the statistical interpretation
  of entropy; connection with classical thermodynamics.
- **2. Kinetic Theory of Gases and Intermolecular Forces** — the ideal-gas model;
  pressure from molecular collisions; the molecular speed distribution; mean free
  path; collision frequency; transport properties; Graham's law; intermolecular
  potentials; the Lennard-Jones potential; the hard-sphere model; van der Waals
  interactions; engineering applications.
- **3. Statistical Distributions and Partition Functions** — Maxwell-Boltzmann,
  Bose-Einstein and Fermi-Dirac statistics; the canonical partition function; the
  grand partition function; probability distributions; atmospheric density;
  thermodynamic quantities from partition functions.
- **4. Monatomic Ideal Gas** — the translational partition function; the classical
  limit; internal energy; entropy; Helmholtz energy; heat capacities; chemical
  potential; the Sackur-Tetrode equation.
- **5. Diatomic and Polyatomic Ideal Gases** — the rotational, vibrational and
  electronic partition functions; the equipartition theorem; internal and
  hindered rotations; thermodynamic functions; high- and low-temperature
  approximations.
- **6. Quantum Statistical Thermodynamics** — quantum gases; bosons and fermions;
  the electron gas; the photon gas; blackbody radiation; degenerate gases;
  applications in condensed matter.
- **7. Chemical Equilibrium and Imperfect Gases** — equilibrium constants; the
  partition-function approach; statistical interpretation; the virial equation and
  its coefficients; intermolecular potentials; gas mixtures; applications.
- **8. Statistical Thermodynamics of Solids** — the Einstein and Debye models;
  phonons; the density of states; heat capacity; thermal-conductivity concepts;
  low-temperature behaviour; the high-temperature limit.
- **9. Phase Transitions and Critical Phenomena** — order parameters; first- and
  second-order transitions; the Ising model; mean-field theory; critical
  exponents; Monte Carlo simulation; applications.
- **10. Computational Statistical Thermodynamics** — numerical evaluation of
  partition functions; Monte Carlo methods; the Metropolis algorithm; random
  walks; importance sampling; numerical integration; statistical uncertainty;
  verification, validation and reproducibility; best computational practices.

</details>

Each chapter's own `README.md` describes its three programs and exactly what they
verify.

## The reusable library

Alongside the self-contained examples, the shared physics is available as an
installable, unit-tested package:

```python
import statistical_thermodynamics as st

# Sackur-Tetrode molar entropy of argon at standard conditions
S = st.thermodynamics.sackur_tetrode_molar(39.948 * st.constants.u, 298.15, 1.0e5)

# Debye heat capacity of copper at 100 K
C = st.solids.debye_heat_capacity(100.0, theta_D=343.0)
```

| Module | Contents |
|---|---|
| [`constants`](src/statistical_thermodynamics/constants.py) | CODATA / SI physical constants |
| [`partition_functions`](src/statistical_thermodynamics/partition_functions.py) | two-level, harmonic, rotational, vibrational, translational |
| [`probability`](src/statistical_thermodynamics/probability.py) | multiplicities, combinatorics, canonical probabilities |
| [`thermodynamics`](src/statistical_thermodynamics/thermodynamics.py) | Sackur-Tetrode, chemical potential, fluctuations |
| [`kinetic_theory`](src/statistical_thermodynamics/kinetic_theory.py) | Maxwell-Boltzmann speeds, collisions, mean free path |
| [`transport`](src/statistical_thermodynamics/transport.py) | diffusion, viscosity, thermal conductivity |
| [`potentials`](src/statistical_thermodynamics/potentials.py) | Lennard-Jones and hard-sphere potentials, Mayer function |
| [`quantum_statistics`](src/statistical_thermodynamics/quantum_statistics.py) | occupation numbers, Bose functions, blackbody radiation |
| [`solids`](src/statistical_thermodynamics/solids.py) | Einstein and Debye heat-capacity models |
| [`equilibrium`](src/statistical_thermodynamics/equilibrium.py) | virial coefficients, Carnahan-Starling, dissociation |
| [`numerical_methods`](src/statistical_thermodynamics/numerical_methods.py) | Metropolis, autocorrelation, blocking, bootstrap |
| [`plotting`](src/statistical_thermodynamics/plotting.py) | shared publication-quality Matplotlib styling |
| [`utilities`](src/statistical_thermodynamics/utilities.py) | reproducible RNGs, error metrics, convergence fitting |

## Educational objectives

After working through the code, a reader should be able to:

- connect microscopic counting (multiplicities, partition functions) to
  macroscopic thermodynamics;
- evaluate partition functions and thermodynamic functions both analytically and
  numerically, and know when each is appropriate;
- implement and reason about Monte Carlo and molecular-dynamics simulations;
- quantify statistical uncertainty honestly, correcting for autocorrelation;
- and, above all, treat **verification** as an inseparable part of computation.

## Intended audience

- **Graduate and advanced-undergraduate students** in physics, chemistry,
  materials science and chemical engineering.
- **Instructors** seeking runnable, verified examples for a course in statistical
  thermodynamics or computational physical chemistry.
- **Researchers and practitioners** who want clean, reusable reference
  implementations with trustworthy error analysis.

A working knowledge of thermodynamics, elementary quantum mechanics, and basic
Python is assumed; the appendices of the book review the prerequisites.

## Computational methodology

The programs draw on a compact, recurring toolkit: exact enumeration and
summation with log-gamma arithmetic; numerical integration and root finding
(`scipy.integrate`, `scipy.optimize`); matrix diagonalization for quantum spectra;
molecular dynamics with the symplectic velocity-Verlet integrator; Monte Carlo
methods (uniform and importance sampling, and the Metropolis algorithm); and
statistical error analysis (autocorrelation time, blocking, bootstrap). See
[docs/theory.md](docs/theory.md) and [docs/coding_style.md](docs/coding_style.md).

## Verification philosophy

Every program checks its results against an exact result, a limiting case, or an
independent method, and reports the agreement explicitly. Convergence is measured
rather than assumed: Monte Carlo errors are confirmed to fall as *N*<sup>−1/2</sup>,
the symplectic integrator's energy drift as (Δt)², and the continuum partition
function's error as √α. The full rationale is in
[docs/verification.md](docs/verification.md).

## Naming convention

Programs are named `exN_M_role.py`, where `N` is the chapter, `M` is the example
number, and `role` is one of `analytical` (closed-form result with a plot), a
descriptive numerical-method name, or `advanced` (the research-grade verification
study).

## Reproducibility

All stochastic programs use a fixed random seed (`numpy.random.default_rng`), so
repeated runs give identical results. Verification tables printed to the console
report the agreement between each computed quantity and its exact or limiting value,
and all Monte Carlo estimates carry autocorrelation-corrected statistical error bars
(see Chapter 10).

## Numerical accuracy

Numerically delicate quantities are computed with stable formulations —
log-gamma factorials, `expm1`/`log1p` near zero, maximum-subtraction before
exponentiating Boltzmann weights, and the logistic form of the Fermi function — so
that the printed values are accurate across the full range of temperatures and
system sizes explored in the book.

## Citation

If you use this code in teaching or research, please cite both the book and the
software. A machine-readable citation is provided in [`CITATION.cff`](CITATION.cff).

**BibTeX**

```bibtex
@book{ferreira2026statthermo,
  author    = {Ferreira, I. L.},
  title     = {Statistical Thermodynamics: Theory, Computation, and Molecular
               Applications --- A Computational Approach with Python},
  year      = {2026},
  note      = {In preparation}
}

@software{ferreira2026statthermo_code,
  author  = {Ferreira, I. L.},
  title   = {Statistical Thermodynamics: Computation and Verification
             (companion code)},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/ileaof/statistical-thermodynamics-computation-verification}
}
```

## License

The code is released under the [MIT License](LICENSE) and may be used, modified
and redistributed — including for teaching and derivative works — with attribution.

## Contributing

Contributions — corrections, clarifications, new verified examples, and better
documentation — are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md) first. Bug reports and feature requests use
the templates under [`.github/ISSUE_TEMPLATE`](.github/ISSUE_TEMPLATE).

## Contact

- **Author:** I. L. Ferreira
- **Email:** [ileao@ufpa.br](mailto:ileao@ufpa.br)
- **GitHub:** [@ileaof](https://github.com/ileaof)
- **Issues:** [report a bug or ask a question](https://github.com/ileaof/statistical-thermodynamics-computation-verification/issues)

## Acknowledgments

- Built on the open-source scientific-Python ecosystem — [NumPy](https://numpy.org/),
  [SciPy](https://scipy.org/) and [Matplotlib](https://matplotlib.org/).
- With gratitude to the students and colleagues whose questions shaped both the
  book and this companion code.
