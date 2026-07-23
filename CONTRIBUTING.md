# Contributing

Thank you for your interest in improving this companion repository. Corrections,
clarifications, new verified examples, and better documentation are all welcome.

## Ways to contribute

- **Report a problem** &mdash; a program that fails, a wrong number, or a typo.
  Open an issue using the bug-report template.
- **Suggest an improvement** &mdash; a new example, a clearer explanation, an
  additional verification. Open an issue using the feature-request template.
- **Submit a pull request** &mdash; fixes and improvements, following the
  guidelines below.

## Ground rules

This repository accompanies a textbook, so two principles come before everything
else:

1. **Correctness is verified, not asserted.** Any new or changed physics must be
   checked against an exact result, a limiting case, or an independent method,
   and the check must be printed to the console. See
   [docs/verification.md](docs/verification.md).
2. **The scientific content of an existing example is not changed casually.** The
   printed results are the ones in the book. Improvements to *filenames,
   formatting, comments, documentation and organisation* are welcome; changes
   that alter a program's numerical output need a clear justification.

## Development setup

```bash
git clone https://github.com/ileaof/statistical-thermodynamics-computation-verification.git
cd statistical-thermodynamics-computation-verification
python -m venv .venv && source .venv/bin/activate   # or the Windows equivalent
pip install -e ".[dev]"
```

## Before you open a pull request

Run the full local check:

```bash
pytest                                   # library tests must pass
python tools/format_repository.py        # black + flake8, no issues
python .github/scripts/check_integrity.py  # structure is intact
```

If you touched or added an example, also run it and confirm its verification
table still reports agreement:

```bash
python tools/run_all_examples.py --chapter N
```

## Style

- Python 3.9+, using only NumPy, SciPy and Matplotlib.
- PEP 8, Black-compatible formatting, NumPy-style docstrings.
- Fixed random seeds for anything stochastic.

The full conventions are in [docs/coding_style.md](docs/coding_style.md).

## Commit messages

Write clear, imperative commit messages (e.g. *"Fix Sommerfeld bracket in the
Fermi-gas root finder"*). Group related changes into a single commit where it
makes review easier.

## Code of conduct

Participation in this project is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md). By contributing, you agree to uphold it.

## Licence

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE) that covers the project.
