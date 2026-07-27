# ThermoLab

A unified, object-oriented Python framework for **thermodynamic properties** and
**thermodynamic-cycle analysis**, powered by [ThermoPack](https://github.com/thermotools/thermopack)
behind a single modern, backend-agnostic API.

ThermoLab gives you, from one consistent interface:

- Pure fluids and gas mixtures (GERG2008 / multiparameter EOS, with cubic
  SRK/PR fallback).
- A thermodynamic state from **any** independent variable pair among
  `T, P, rho, v, h, s, u`.
- ~21 thermodynamic and transport properties, all **SI mass-based**
  (J/kg, J/(kg·K), kg/m³, W/(m·K), Pa·s).
- A lightweight **CFD interface** (per-cell scalar snapshots + vectorized grid
  evaluation) and **property tables** with fast interpolation.
- **Diagrams**: T-s, P-h, P-v, Mollier (h-s), isotherms, isobars, isochores,
  saturation curves.
- **Six cycles**: Rankine, Brayton (Joule), vapor-compression refrigeration,
  Otto, Diesel — with efficiencies, back-work ratio, net work, and plotting.
- A small **optimization** utility for sweeping/optimizing cycle parameters.

All backends sit behind a clean `BaseBackend` ABC, so CoolProp / Cantera /
REFPROP / pycalphad can be added later **without changing the public API**.

## Install

```bash
pip install -e .
```

Requirements: Python ≥ 3.11, `numpy`, `scipy`, `pandas`, `matplotlib`,
`thermopack`.

## Quickstart

```python
from thermolab import Gas

air = Gas("Air", backend="thermopack")
st = air.state(T=800.0, P=5e5)
for attr in ("rho", "cp", "cv", "h", "s", "mu", "k", "gamma", "sound_speed"):
    print(attr, getattr(st, attr))
```

A mixture:

```python
from thermolab import Mixture

flue = Mixture(["N2", "O2", "CO2", "H2O"], [0.78, 0.21, 0.005, 0.005])
st = flue.state(T=1200.0, P=3e5)
print(st.rho, st.cp, st.gamma)
```

Any variable pair works for the flash:

```python
st = air.state(P=5e5, h=6.0e5)      # solve T from (P, h)
st = air.state(rho=1.5, s=800.0)    # solve (T, P) from (rho, s)
```

A cycle:

```python
from thermolab import cycles as C

res = C.brayton(pressure_ratio=12, T3=1400)
print(res)              # efficiency, net work, back-work ratio
res.plot(diagram="ts")  # T-s diagram
```

## Supported fluids (this ThermoPack build)

`H2O, CO2, N2, O2, Ar, H2, He, NH3, R134a, R32, R143a, R1234yf, R1234ze,
R12, R14, R23, R116, Benzene, CO, N2O, SO2, H2S, Kr, Xe, Ne, DME, R125`,
plus the pseudo-fluid **`Air`** (resolved to a dry-air composition: N2 0.7884 /
O2 0.2095 / Ar 0.0093 / CO2 0.0004 on GERG2008).

`Gas("Methane")` raises `UnsupportedFluidError` with a clear message in this
build — add another backend to extend coverage. `list_fluids()` lists what is
available.

## Properties

`State` exposes (mass-based SI): `T, P, rho, v, u, h, s, g, a_helmholtz, cp,
cv, gamma, Z, sound_speed, mu, k, thermal_diffusivity, prandtl, joule_thomson,
beta_thermal_expansion, kappa_t`, plus `phase`, `two_phase`, `quality`.

> **Transport note.** ThermoPack does not provide transport properties, so
> ThermoLab computes gas-phase `mu`, `k`, `α`, `Pr` from engineering
> correlations (Sutherland viscosity with Wilke mixing; modified-Eucken /
> Wassiljeva conductivity). These are accurate gas-phase estimates, not
> multiparameter values, and a warning is issued when used for the liquid phase.

## Examples

The `examples/` directory contains eighteen standalone scripts, each runnable
directly with Python after installing ThermoLab.

| Script | Demonstrates |
| --- | --- |
| `examples/01_air_properties.py` | Pure-fluid properties, round-trip flash |
| `examples/02_mixture.py` | Multicomponent mixture, composition update |
| `examples/03_cfd_interface.py` | CFD scalar snapshot + vectorized grid |
| `examples/04_ts_diagram.py` | T-s diagram with isobars/isochores |
| `examples/05_mollier.py` | Mollier h-s diagram for water |
| `examples/06_brayton_cycle.py` | Brayton cycle + pressure-ratio sweep |
| `examples/07_rankine_cycle.py` | Superheated Rankine cycle on T-s |
| `examples/08_property_tables.py` | Property / saturation tables, interpolated lookup |
| `examples/09_otto_diesel.py` | Otto vs Diesel cycles, compression-ratio sweep |
| `examples/10_refrigeration.py` | Vapor-compression refrigeration, COP, P-h diagram |
| `examples/11_cycle_optimization.py` | Local + global cycle optimization (eta / net work) |
| `examples/12_transport.py` | Transport & acoustic properties, Air vs CO2 sweep |
| `examples/13_combustion.py` | Hydrogen-air combustion, adiabatic flame temperature |
| `examples/14_liquids.py` | Compressed & saturated liquid, compressibility, dome |
| `examples/15_vapour.py` | Vapour-pressure curve, Clausius-Clapeyron, latent heat |
| `examples/16_solids.py` | Solid CO2 (dry ice): melting curve, fusion, density |
| `examples/17_hydrates.py` | Gas-hydrate equilibrium curve + Hammerschmidt inhibition |
| `examples/18_database.py` | Lists supported gases, liquids, solids, hydrate formers |

Start with `examples/18_database.py` to see which fluids ThermoLab supports and
what phase each can do — it tells you which name to pass to `Gas(...)` /
`Mixture(...)` when adapting the other examples to your own fluids.

### Running an example

From the project root (the directory containing `pyproject.toml`):

```bash
python examples/01_air_properties.py
```

This prints the computed property table to the terminal. The text-only
examples (`01`, `02`, `03`, `08`, `11`, `18`) need no display.

The plotting examples (`04`–`07`, `09`, `10`, `12`–`17`) open the figure in a
window by default. To save the figure to a file instead — handy on headless
machines or in a notebook/CI — pass `--save <path>`:

```bash
python examples/04_ts_diagram.py --save ts_air.png
python examples/07_rankine_cycle.py --save rankine.png
python examples/10_refrigeration.py --save refrigeration.png
python examples/13_combustion.py --save combustion.png
python examples/16_solids.py --save solids.png
```

The `--save` flag switches matplotlib to the headless `Agg` backend
automatically, so no display is required.

Run them all at once, saving every figure:

```bash
for ex in examples/*.py; do
  python "$ex" --save "figures/$(basename "$ex" .py).png"
done
```

## Tests

```bash
pytest -q
```

70 tests covering state/flash, properties, mixtures, cycles, transport,
plotting, tables, CFD, optimization, and validation.

## Project layout

```
thermolab/              # the Python package (core API)
  __init__.py  fluid.py  mixture.py  state.py  flash.py  properties.py
  transport.py  units.py  tables.py  cfd.py  optimization.py  plotting.py
  exceptions.py  _fluid_db.py
  backends/        # BaseBackend ABC + ThermoPack implementation
  cycles/          # rankine, brayton, refrigeration, otto, diesel
examples/             # 18 runnable, self-contained scripts
tests/                 # pytest suite
docs/USAGE.md          # extended API guide
```

## Extending with a new backend

Subclass `thermolab.BaseBackend` and implement the abstract molar interface
(`molar_masses`, `guess_phase`, `molar_properties`, `specific_volume`,
saturation/critical helpers). Register it with the backend registry; the
flash solver, `State`, cycles, plotting, and tables all work unchanged.

See [`docs/USAGE.md`](docs/USAGE.md) for the extended guide, and
[`handoff.md`](handoff.md) for integration notes and known limitations.

## License

MIT.