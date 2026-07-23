# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] &mdash; 2026-07-23

First public release: the complete companion code for the book, organised as a
professional, maintainable open-source project.

### Added

- **All 30 example programs** for the ten chapters, each an analytical, a direct
  numerical, or an advanced verification study, with self-checking output and a
  publication-quality figure.
- **Reusable library** `statistical_thermodynamics` (13 modules) collecting the
  shared physics and numerical tools, with a `py.typed` marker.
- **Test suite** (`pytest`) covering every library module.
- **Per-chapter READMEs** describing the physics, the programs, and what each
  verifies.
- **Documentation** in `docs/`: installation, repository structure, verification
  philosophy, theory notes, coding style, and an FAQ.
- **Maintenance tools** in `tools/`: `run_all_examples.py`,
  `build_all_figures.py`, `format_repository.py`, `clean_repository.py`.
- **Continuous integration** (GitHub Actions): Python syntax and style, automated
  tests across Python 3.9&ndash;3.12, Markdown validation, and a repository
  integrity check.
- **Project metadata**: MIT `LICENSE`, `CITATION.cff`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, `AUTHORS.md`, `pyproject.toml`,
  `requirements.txt`.

### Changed

- Reorganised the chapters into descriptively named directories
  (`Chapter01_Foundations`, ...), each with `figures/`, `data/` and `results/`
  subdirectories.

[1.0.0]: https://github.com/ileaof/statistical-thermodynamics-computation-verification/releases/tag/v1.0.0
