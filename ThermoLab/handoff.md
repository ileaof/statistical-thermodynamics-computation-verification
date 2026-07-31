# ThermoLab — handoff for commit to `ileaof/statistical-thermodynamics-computation-verification`

This document records the state of the ThermoLab project and how to add it to
the main tree of the existing repository
[`ileaof/statistical-thermodynamics-computation-verification`](https://github.com/ileaof/statistical-thermodynamics-computation-verification).

## What this is

ThermoLab is a unified, object-oriented Python framework for **thermodynamic
properties** and **thermodynamic-cycle analysis**, powered by
[ThermoPack](https://github.com/thermotools/thermopack) behind a single,
backend-agnostic API. It is being added to the verification repository as a
self-contained subfolder.

## Current state (2026-07-27)

| Item | Status |
| --- | --- |
| Package version | `0.1.0` |
| Tests | **70 / 70 passing** (`pytest -q`) |
| Examples | **18 / 18 run** (text-only print; plotting render/save with `--save`) |
| Python | ≥ 3.11 |
| Runtime deps | `numpy>=1.20`, `scipy>=1.7`, `pandas>=1.3`, `matplotlib>=3.3`, `thermopack>=2.0` |
| Dev deps | `pytest>=7`, `pytest-cov` |

Smoke-tested on Windows 11, Python 3.11, with the installed ThermoPack build.

## Folder layout

The ThermoLab **project root** (this directory) is the unit to commit. It is an
installable Python package and should land in the repo as a subfolder, e.g.
`thermolab/`:

```
<project root>/          # →  repo:  thermolab/
  README.md              # package README (self-contained)
  handoff.md             # this file
  pyproject.toml         # installable package metadata
  .gitignore
  thermolab/             # the Python package (core API)
    __init__.py  fluid.py  mixture.py  state.py  flash.py  properties.py
    transport.py  units.py  tables.py  cfd.py  optimization.py  plotting.py
    exceptions.py  _fluid_db.py
    backends/             # BaseBackend ABC + ThermoPack implementation
    cycles/               # rankine, brayton, refrigeration, otto, diesel, base
  examples/              # 18 runnable, self-contained scripts (01–18)
  tests/                 # pytest suite (70 tests)
  docs/USAGE.md          # extended API guide
```

Note the nested `thermolab/thermolab/` layout (project root / Python package) —
standard for installable packages.

## Integration steps

From a working clone of the verification repository:

```bash
git clone https://github.com/ileaof/statistical-thermodynamics-computation-verification.git
cd statistical-thermodynamics-computation-verification

# Place the ThermoLab project as a subfolder named `thermolab/` at the repo root.
# Copy the *contents* of the ThermoLab project folder into ./thermolab/
# (so that ./thermolab/pyproject.toml, ./thermolab/thermolab/, ./thermolab/examples/ ... exist).

git add thermolab
git status                       # sanity-check the staged tree
git commit -m "Add ThermoLab: thermodynamic properties & cycle-analysis framework

Co-Authored-By: Claude <noreply@anthropic.com>"
git push
```

Recommended: keep ThermoLab as a self-contained subfolder with its own
`pyproject.toml`, `README.md`, `examples/`, `tests/`, `docs/`. It installs
independently with `pip install -e thermolab`. Do **not** flatten the package
into the repo root — the nested layout preserves the installable-package
contract and the `thermolab` import name.

## Verification after commit

```bash
cd thermolab
pip install -e .
pytest -q                       # expect: 70 passed
python examples/18_database.py # lists supported gases / liquids / solids / hydrates
python examples/01_air_properties.py
python examples/13_combustion.py --save combustion.png
```

All examples are standalone; the text-only examples
(`01, 02, 03, 08, 11, 18`) print to the terminal, the rest plot and accept
`--save <path>` (headless `Agg` backend, no display required).

## Known limitations & caveats (read before using)

1. **ThermoPack dependency** — requires `thermopack>=2.0`; the Fortran backend
   is used on Windows. ThermoLab is backend-agnostic (`BaseBackend` ABC), but
   only the ThermoPack backend ships today.
2. **Uncatchable Fortran STOP** — a bad ThermoPack call (e.g. saturation at
   `T >= Tc`, or a density solver that cannot converge at extreme T/P) prints a
   Windows access-violation dump and aborts the *whole process*; `try/except`
   cannot catch it. ThermoLab guards the public API against this; the two
   examples that reach into the raw ThermoPack engine (`16_solids`,
   `18_database`) isolate the risky `init_solid` call in a subprocess so a
   failure does not kill the example.
3. **Phase-override subtlety** — `phase="liquid"/"vapor"` is silently ignored
   when the requested phase is unstable at (T, P), and is ambiguous *exactly*
   on the saturation boundary (it can return the other root). The liquid /
   vapour / combustion examples take saturated liquid/vapour at a pressure
   nudged off `Psat` (`1.001·Psat` / `0.999·Psat`) and the combustion example
   reads the 298 K water reference at 1 kPa (below `Psat`) for a consistent
   gas-phase reference. See `examples/13_combustion.py` and `14_liquids.py`.
4. **`saturation_pressure` fails near Tc** — backed by `bubble_pressure`, it
   raises a *catchable* `Exception` within ~2–4 K of the critical point (water:
   works to 642 K, fails at 644 K; `Tc = 647.1 K`). Dome / vapour-pressure
   sampling is capped at 640 K (~0.99·Tc).
5. **Solids are not wrapped by the high-level API** — examples `16_solids` and
   `18_database` reach into ThermoPack's `thermo` engine directly via
   `init_solid`. Only **CO₂ and H₂O** carry solid correlations in this build
   (dry ice and ice). Solid enthalpies use a different zero from ThermoLab's
   fluid enthalpies, so fusion enthalpy is computed from the *same* engine
   (densities are reference-independent and compare freely).
6. **No hydrate model in this ThermoPack Python build** — zero `hydrate`
   references in the installed Python source; the van der Waals-Platteeuw
   model exists in Fortran but has no Python binding. `examples/17_hydrates.py`
   uses engineering correlations instead (Clausius-Clapeyron P-T fit +
   Hammerschmidt inhibitor equation).
7. **Transport properties are gas-phase correlations** — ThermoPack is
   thermodynamic-only, so `mu`, `k`, `α`, `Pr` come from Sutherland viscosity +
   Wilke mixing and modified-Eucken / Wassiljeva conductivity (Chung fallback).
   Accurate for the gas phase; approximate for liquids and flagged with a
   warning.
8. **Curated fluid database** — `list_fluids()` / `KNOWN_SUPPORTED` is the
   supported surface. ThermoPack's full Fortran fluid catalog is not enumerable
   from Python; run `examples/18_database.py` to see what this build supports.

## Non-obvious ThermoPack API quirks (reference)

- Components are a **comma-joined string** (`'N2,O2,CO2,H2O'`), not a list.
- Internal units are **molar** (J/mol, m³/mol); ThermoLab converts to mass-based.
- Component indices are **1-based Fortran**; index 0 → access violation.
- Phase flags: `VAPPH=2`, `LIQPH=1`, `TWOPH=0`, `SINGLEPH=4`.
- `(T, h)` is ill-conditioned for nearly-ideal gases (h depends weakly on P) —
  use `(P, h)`, `(rho, h)`, or `(T, s)`.
- Mixture `critical(z)` is flaky for some compositions — wrap loosely and fall
  back to a dew/bubble bracket for two-phase detection.

These are recorded in the project memory file `thermopack-api-quirks`.

## Pointers

- **Start here:** `examples/18_database.py` — what fluids are supported and what
  phase each can do.
- **Quickstart + supported fluids:** `README.md`
- **Extended API guide:** `docs/USAGE.md`
- **Cycles, plotting, tables, CFD, optimization:** `thermolab/cycles/`,
  `thermolab/plotting.py`, `thermolab/tables.py`, `thermolab/cfd.py`,
  `thermolab/optimization.py`.

## License

MIT — same as the package (`pyproject.toml`).
