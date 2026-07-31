# Repository structure

The repository is organised so that a reader can find any program from the book
in seconds, reuse its physics through a clean library, and maintain the whole
project over the long term.

## Top-level layout

```
statistical-thermodynamics-computation-verification/
├── README.md                     Project overview and documentation hub
├── LICENSE                       MIT licence
├── CITATION.cff                  How to cite the software
├── requirements.txt              Runtime dependencies
├── pyproject.toml                Packaging and tool configuration
├── CONTRIBUTING.md               Contribution guidelines
├── CODE_OF_CONDUCT.md            Community standards
├── SECURITY.md                   How to report issues
├── CHANGELOG.md                  Release history
├── AUTHORS.md                    Author and contributors
│
├── Chapter01_Foundations/            ┐
├── Chapter02_Kinetic_Theory.../      │  one directory per book chapter,
├── ...                               │  each with three example programs
├── Chapter10_Computational.../       ┘
│
├── src/
│   └── statistical_thermodynamics/   the reusable, tested library
│
├── tests/                        unit tests for the library
├── tools/                        maintenance and build scripts
├── docs/                         extended documentation (this folder)
├── notebooks/                    optional interactive notebooks
├── ThermoLab/                    companion applied-thermodynamics project
├── STATTHERMOPY/                 companion first-principles property simulator
└── .github/                      CI workflows and issue templates
```

## Chapters

Each chapter directory follows an identical template:

```
Chapter0N_Name/
├── README.md                 chapter overview and program table
├── ex N_1_analytical.py       closed-form result with a plot
├── ex N_2_<method>.py         direct numerical computation
├── ex N_3_advanced.py         research-grade verification study
├── figures/                  generated figures (built on demand)
├── data/                     input or reference data, if any
└── results/                  saved numerical output, if any
```

### Naming convention

Programs are named `exN_M_role.py`, where **N** is the chapter, **M** is the
example number, and **role** is one of:

- `analytical` &mdash; a closed-form result accompanied by a plot;
- a descriptive method name (`meanfreepath`, `metropolis`, `debye`, ...) for the
  direct numerical example;
- `advanced` &mdash; the research-grade verification study.

## The `statistical_thermodynamics` library

The example programs are **self-contained**: each runs on its own and reproduces
the results printed in the book. In parallel, the physics and numerical tools
that recur throughout the book are collected into an installable, unit-tested
library so that readers can reuse the same verified building blocks.

| Module | Contents |
|---|---|
| `constants` | CODATA / SI physical constants |
| `partition_functions` | two-level, harmonic, rotational, vibrational, translational |
| `probability` | multiplicities, combinatorics, canonical probabilities |
| `thermodynamics` | Sackur-Tetrode, chemical potential, fluctuations |
| `kinetic_theory` | Maxwell-Boltzmann speeds, collisions, mean free path |
| `transport` | diffusion, viscosity, thermal conductivity |
| `potentials` | Lennard-Jones and hard-sphere potentials, Mayer function |
| `quantum_statistics` | occupation numbers, Bose functions, blackbody radiation |
| `solids` | Einstein and Debye heat-capacity models |
| `equilibrium` | virial coefficients, Carnahan-Starling, dissociation |
| `numerical_methods` | Metropolis, autocorrelation, blocking, bootstrap |
| `plotting` | shared publication-quality Matplotlib styling |
| `utilities` | reproducible RNGs, error metrics, convergence fitting |

## Supporting directories

| Directory | Purpose |
|---|---|
| [`tests/`](../tests/) | `pytest` unit tests for every library module |
| [`tools/`](../tools/) | `run_all_examples.py`, `build_all_figures.py`, `format_repository.py`, `clean_repository.py` |
| [`docs/`](.) | installation, verification philosophy, theory notes, coding style, FAQ |
| [`notebooks/`](../notebooks/) | optional Jupyter notebooks for interactive exploration |
| [`ThermoLab/`](../ThermoLab/) | companion applied-thermodynamics project (properties and cycle analysis) |
| [`STATTHERMOPY/`](../STATTHERMOPY/) | companion first-principles property simulator (partition-function engine) |
| [`.github/`](../.github/) | continuous-integration workflows and issue templates |

### ThermoLab

`ThermoLab/` is a **self-contained project** with its own `pyproject.toml`,
`README.md`, `examples/`, `tests/` and `docs/`. It is installed independently
(`cd ThermoLab && pip install -e .`) and requires Python ≥ 3.11 plus
`thermopack`, neither of which the chapter examples need. Because it ships its
own packaging and test configuration, it is deliberately kept outside `src/` and
the root test suite:

```
ThermoLab/
├── README.md · docs/USAGE.md · handoff.md
├── pyproject.toml                 installable package metadata
├── thermolab/                     the Python package (core API)
│   ├── fluid.py  mixture.py  state.py  flash.py  properties.py
│   ├── transport.py  units.py  tables.py  cfd.py  optimization.py
│   ├── backends/                  BaseBackend ABC + ThermoPack backend
│   └── cycles/                    rankine, brayton, refrigeration, otto, diesel
├── examples/                      20 runnable scripts
└── tests/                         pytest suite (70 tests)
```

### STATTHERMOPY

`STATTHERMOPY/` is a **self-contained simulator** that computes ideal-gas
properties exclusively from the molecular partition function
(`Q = Q_t·Q_r·Q_v·Q_e`), with no empirical correlations in the calculation path.
Like ThermoLab it has its own `pyproject.toml`, tests and docs, installs
independently (`cd STATTHERMOPY && pip install -e ".[dev]"`), and requires
Python ≥ 3.11 (optional extras: `[gui]` for the PySide6 GUI, `[accel]` for the
Numba/OpenMP/CUDA backends):

```
STATTHERMOPY/
├── README.md · handoff.md · LICENSE · pyproject.toml
├── src/statthermopy/
│   ├── partition.py  thermodynamics.py  mixture.py  constants.py  units.py
│   ├── core/          Molecule, Geometry, State, Contribution
│   ├── modes/         translational · rotational · vibrational · electronic
│   ├── database/      registry + data/*.yaml  (22 species)
│   ├── validation/    reference + data/*.yaml  (NIST/JANAF Cp°, S°)
│   ├── backend/       executor + numpy / numba / openmp / cuda
│   ├── io/            exporters (CSV / JSON / YAML / Excel / LaTeX)
│   ├── plots/  cli/  gui/            plotting · REPL+run · PySide6 GUI
│   └── equilibrium/   architecture placeholder (future phase)
├── examples/          runnable scripts + demo notebook + output figures
├── tests/             pytest suite (268 tests, ~96% coverage)
└── docs/              THEORY.md + Sphinx sources
```

## Book chapters &harr; directories

| Book chapter | Directory |
|---|---|
| 1. Foundations of Statistical Thermodynamics | `Chapter01_Foundations/` |
| 2. Kinetic Theory of Gases and Intermolecular Forces | `Chapter02_Kinetic_Theory_and_Intermolecular_Forces/` |
| 3. Statistical Distributions and Partition Functions | `Chapter03_Statistical_Distributions/` |
| 4. Monatomic Ideal Gas | `Chapter04_Monatomic_Ideal_Gas/` |
| 5. Diatomic and Polyatomic Ideal Gases | `Chapter05_Diatomic_and_Polyatomic_Gases/` |
| 6. Quantum Statistical Thermodynamics | `Chapter06_Quantum_Statistical_Thermodynamics/` |
| 7. Chemical Equilibrium and Imperfect Gases | `Chapter07_Chemical_Equilibrium_and_Imperfect_Gases/` |
| 8. Statistical Thermodynamics of Solids | `Chapter08_Statistical_Thermodynamics_of_Solids/` |
| 9. Phase Transitions and Critical Phenomena | `Chapter09_Phase_Transitions_and_Critical_Phenomena/` |
| 10. Computational Statistical Thermodynamics | `Chapter10_Computational_Statistical_Thermodynamics/` |
