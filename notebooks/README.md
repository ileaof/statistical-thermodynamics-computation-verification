# Notebooks

This folder is for **optional** Jupyter notebooks that explore the book's
examples interactively &mdash; sweeping parameters, animating a Monte Carlo
chain, or comparing models side by side.

The authoritative, verified programs are the `ex*.py` files in each chapter
directory; notebooks here are companions for experimentation, not replacements.

## Using notebooks

Install Jupyter alongside the project dependencies:

```bash
pip install -e ".[dev]" notebook
jupyter notebook
```

A notebook can import the reusable library directly:

```python
import statistical_thermodynamics as st

T = 300.0
v_p, v_avg, v_rms = st.kinetic_theory.characteristic_speeds(28.0134 * st.constants.u, T)
print(v_p, v_avg, v_rms)
```

## Contributing a notebook

Notebooks are welcome. Please **clear all outputs before committing**
(`Kernel → Restart & Clear Output`) so the diffs stay small and the repository
does not carry large embedded images. See [CONTRIBUTING.md](../CONTRIBUTING.md).
