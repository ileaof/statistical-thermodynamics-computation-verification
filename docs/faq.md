# Frequently asked questions

### What is this repository?

It is the official companion code for the textbook *Statistical Thermodynamics:
Theory, Computation, and Molecular Applications &mdash; A Computational Approach
with Python* by I. L. Ferreira. Every computational example in the book is here
in full, ready to run and reproduce.

### Do I need the book to use the code?

No. Each program's docstring states the physics, the equations, and what it
verifies, so the code stands on its own as a study resource. The book provides
the full derivations and context.

### Which Python version and packages do I need?

Python 3.9 or later with NumPy, SciPy and Matplotlib &mdash; nothing else. See
[installation.md](installation.md).

### How do I run an example?

```bash
cd Chapter03_Statistical_Distributions
python ex3_1_analytical.py
```

It prints a verification table and writes a figure (`fig3_1.png`) to the
directory.

### Where are the figures? Nothing pops up.

The programs *save* figures as PNG files instead of opening a window, so they run
identically on servers and in notebooks. Look for `figN_M.png` in the working
directory, or run `python tools/build_all_figures.py` to collect them all into
each chapter's `figures/` folder.

### Will I get exactly the numbers printed in the book?

Yes. Every stochastic program fixes its random seed, so results are reproducible
to the last digit on any platform.

### Do I have to install the `statistical_thermodynamics` library?

Only if you want to reuse its functions in your own code. The chapter examples
are self-contained and do not import it. To install it:

```bash
pip install -e .
```

### An advanced example is slow. Is that normal?

The `ex*_3_advanced.py` programs are research-grade verification campaigns (long
Monte Carlo chains, molecular dynamics, Ising sweeps across many sizes). A minute
or two is expected. Run a single chapter with
`python tools/run_all_examples.py --chapter 9` to focus.

### How do I run the tests?

```bash
pytest
```

This exercises the library modules, not the example programs (the examples verify
themselves at runtime).

### I found a bug or a typo. How do I report it?

Open an issue on GitHub. See [CONTRIBUTING.md](../CONTRIBUTING.md) for the
guidelines and the issue templates.

### How should I cite the code?

See the [Citation](../README.md#citation) section of the main README and the
machine-readable [`CITATION.cff`](../CITATION.cff).

### What licence applies?

The MIT licence &mdash; see [LICENSE](../LICENSE). You are free to use, modify and
redistribute the code, including for teaching, with attribution.

### Can I use these programs in my own course?

Yes, that is exactly what they are for. Attribution to the book is appreciated;
the MIT licence permits classroom and derivative use.
