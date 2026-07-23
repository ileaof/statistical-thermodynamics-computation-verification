# Installation

This guide covers everything needed to run the example programs and to use the
reusable `statistical_thermodynamics` library.

## Requirements

| Component | Minimum version | Notes |
|---|---|---|
| Python | 3.9 | 3.10&ndash;3.12 also supported |
| NumPy | 1.21 | array computing |
| SciPy | 1.7 | integration, optimization, special functions |
| Matplotlib | 3.4 | figures |

The examples use **only** the standard scientific-Python stack &mdash; there are
no exotic dependencies.

## 1. Get the code

```bash
git clone https://github.com/ileaof/statistical-thermodynamics-computation-verification.git
cd statistical-thermodynamics-computation-verification
```

## 2. Create a virtual environment (recommended)

Keeping the project's packages isolated avoids version clashes with the rest of
your system.

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

**Conda (any platform)**

```bash
conda create -n statthermo python=3.11 numpy scipy matplotlib
conda activate statthermo
```

## 3. Install dependencies

The lightest option installs just the runtime dependencies:

```bash
pip install -r requirements.txt
```

To also install the reusable library as an editable package (so `import
statistical_thermodynamics` works from anywhere), use:

```bash
pip install -e .
```

And to include the development tools (pytest, black, flake8):

```bash
pip install -e ".[dev]"
```

## 4. Verify the installation

```bash
# run the library test suite
pytest

# run one example
cd Chapter01_Foundations
python ex1_1_analytical.py
```

You should see a verification table printed to the console and a `fig1_1.png`
written to the directory.

## Running everything at once

Two convenience tools drive all 30 programs:

```bash
python tools/run_all_examples.py      # execute every example, report pass/fail
python tools/build_all_figures.py     # regenerate every figure into figures/
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: statistical_thermodynamics` | Run `pip install -e .`, or run examples from within their chapter directory (they are self-contained and do not require the library). |
| `AttributeError: module 'numpy' has no attribute 'trapz'` | NumPy 2.0 renamed `trapz` to `trapezoid`; the code already handles both, so upgrade to the latest checkout. |
| Figures do not appear | The programs *save* PNG files rather than opening a window; look for `figN_M.png` in the working directory. |
| A Monte Carlo example is slow | The advanced (`ex*_3`) programs are research-grade; pass a single chapter to the tools, e.g. `python tools/run_all_examples.py --chapter 1`. |

See the [FAQ](faq.md) for more.
