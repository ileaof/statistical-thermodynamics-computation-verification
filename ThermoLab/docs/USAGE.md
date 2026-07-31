# ThermoLab — Extended Usage Guide

This document complements the [`README`](../README.md) with the deeper API
surface: flash behaviour & two-phase handling, cycles, plotting, tables/CFD,
and how to add a new backend.

## 1. Fluids and mixtures

```python
from thermolab import Gas, Mixture, list_fluids

list_fluids()                       # fluids supported by the default backend
air = Gas("Air")                    # pseudo-fluid -> dry-air GERG2008 mix
water = Gas("H2O")
r134a = Gas("R134a")

mix = Mixture(["N2", "O2", "CO2", "H2O"], [0.78, 0.21, 0.005, 0.005])
mix.set_fractions([0.70, 0.10, 0.10, 0.10])   # update composition in place
```

EOS selection is automatic: pure fluid → multiparameter MEOS, with SRK cubic
fallback when MEOS parameters are unavailable; mixture → GERG2008 when every
component is in the GERG core, else SRK. Pass `eos="SRK"` (or `"PR"`) to force
a cubic.

A component absent from this build (for example `Gas("Propane")`) raises
`UnsupportedFluidError` — install another backend to extend coverage.

## 2. State and the flash solver

`fluid.state(**pair)` accepts **any two** of `T, P, rho, v, h, s, u`:

```python
st = air.state(T=800.0, P=5e5)      # direct
st = air.state(P=5e5, h=6.0e5)      # solve T from (P, h)
st = air.state(rho=1.5, s=800.0)    # solve (T, P) from (rho, s)
st = air.state(T=600.0, s=750.0)    # solve P from (T, s)
```

The returned `State` resolves `(T, P, phase)` once; every property is a lazy,
cached attribute (so repeated access — e.g. in a CFD loop — is cheap).

### Two-phase awareness (pure fluids)

For pure fluids the flash is saturation-aware: if an energy/entropy spec lies
between the saturated-liquid and saturated-vapor values, a two-phase state is
returned directly with the vapour quality, **without** asking the backend for a
single-phase root inside the dome (where its density solver can fail):

```python
st = water.state(P=5e4, h=2e6)
st.two_phase      # True
st.quality        # vapour mass fraction in [0, 1]
st.T              # == saturation_temperature(P)
```

Single-phase derivatives (`cp, cv, gamma, sound_speed`) raise `TwoPhaseError`
on a two-phase state; blended values (`rho, h, s, u, v`) are still available
via quality.

### Note on degenerate pairs

For a nearly-ideal gas, enthalpy depends only weakly on pressure, so a `(T, h)`
pair is ill-conditioned (P is essentially undetermined). Prefer `(P, h)`,
`(rho, h)`, or `(T, s)` — all of which are well-conditioned. When a spec is
physically unreachable, `flash` raises `ConvergenceError` rather than returning
a nonsense state.

## 3. Properties

`State` attributes (SI, mass-based):

| Group | Attributes |
| --- | --- |
| Basic | `T, P, rho, v, phase, two_phase, quality` |
| Energy | `u, h, s, g, a_helmholtz` |
| Heat capacities | `cp, cv, gamma` |
| EOS / acoustic | `Z, sound_speed` |
| Transport | `mu, k, thermal_diffusivity, prandtl` |
| Derivatives | `joule_thomson, beta_thermal_expansion, kappa_t` |

`st.bundle()` returns a `PropertyBundle` dataclass; `st.to_dict()` /
`st.to_series()` give dict / pandas Series views; `repr(st)` prints a table.

## 4. CFD interface

```python
from thermolab.cfd import CFDScalars, evaluate_grid, bulk_properties

st = air.state(T=800.0, P=5e5)
scalars = CFDScalars.from_state(st)     # per-cell snapshot
scalars.to_dict()                        # rho, cp, cv, gamma, mu, k, ...

import numpy as np
T = np.linspace(300, 1500, 50)
P = np.full_like(T, 5e5)
df = evaluate_grid(air, T, P)            # DataFrame, one row per node

bulk = bulk_properties([air.state(T=float(t), P=2e5) for t in T])
```

`CFDScalars` is a frozen dataclass — copy it into your solver's cell arrays with
zero per-call overhead after the state is resolved.

## 5. Tables and interpolation

```python
from thermolab import Gas
from thermolab.tables import PropertyTable, SaturationTable

water = Gas("H2O")
tab = PropertyTable(water, T_range=(300, 700, 25), P_range=(1e4, 1e7, 30))
tab.df.head()                            # long-format: T, P, rho, h, s, ...
f = tab.interpolate()                    # callable f(T, P) -> dict of props
props = f(450.0, 1e5)                    # interpolated rho, h, cp, ...

sat = SaturationTable(water, T_range=(300, 640, 40))
```

`T_range` / `P_range` accept either an explicit list or a `(min, max, n)`
triple. Interpolation uses `scipy.RegularGridInterpolator` and is suitable for
fast CFD lookup tables.

## 6. Plotting

All plotters are in `thermolab.plotting`; they return a matplotlib `Axes` and
accept an existing `ax=` to compose diagrams.

```python
from thermolab import Gas
from thermolab import plotting as P

air, water = Gas("Air"), Gas("H2O")
P.plot_ts(air, isobars=[1e5, 5e6], T_range=(250, 1500))
P.plot_ph(water, isotherms=[400, 500], T_range=(300, 600))
P.plot_pv(water, isotherms=[400, 500], T_range=(300, 600))
P.plot_mollier(water, isobars=[1e4, 1e6], T_range=(300, 700))
P.plot_isotherms(air, [300, 500, 800], P_range=(1e4, 1e6))
P.plot_isobars(air, [1e5, 1e6], T_range=(300, 1000))
P.plot_isochores(air, [0.5, 1.0, 2.0], T_range=(300, 800))
P.plot_saturation(water, T_range=(300, 640), diagram="ts")
```

## 7. Cycles

Each cycle builder returns a `CycleResult` with `eta` (or `cop` for
refrigeration), `net_work`, `back_work_ratio`, `points`, and a `plot()` method.

| Cycle | Key parameters |
| --- | --- |
| `rankine` | `P_boiler, P_condenser, T_superheat, eta_turbine, eta_pump` |
| `brayton` (= `joule`) | `P1, pressure_ratio, T1, T3, eta_compressor, eta_turbine` |
| `refrigeration` | `T_evap, T_cond, superheat, eta_compressor` |
| `otto` | `compression_ratio, T1, P1, T3` |
| `diesel` | `compression_ratio, cutoff_ratio, T1, P1` |

```python
from thermolab import cycles as C

C.brayton(pressure_ratio=12, T3=1400)                 # ideal
C.brayton(pressure_ratio=12, T3=1400,
          eta_compressor=0.85, eta_turbine=0.9)       # non-ideal
C.rankine(P_boiler=8e6, P_condenser=1e4, T_superheat=773.0)
C.refrigeration(T_evap=263.15, T_cond=313.15)
C.otto(compression_ratio=8, T3=2500)
C.diesel(compression_ratio=18, cutoff_ratio=2.0)
```

Air-standard cycles (Otto/Diesel/Brayton) use ideal-gas relations cross-checked
against ThermoLab states; Rankine and refrigeration build real `State`s through
the working fluid (water / refrigerant).

## 8. Optimization

```python
from thermolab import cycles as C
from thermolab import optimization as opt

res = opt.optimize_cycle(
    lambda rp: C.brayton(pressure_ratio=rp[0], T3=1400),
    bounds=[(4, 30)], objective="eta", x0=[10.0],
)
print(res.x, res.fun)

df = opt.sweep(lambda pressure_ratio: C.brayton(pressure_ratio=pressure_ratio, T3=1400),
               "pressure_ratio", [4, 8, 12, 16, 20])
```

`optimize_cycle` wraps `scipy.optimize.minimize` (L-BFGS-B by default; pass
`method="differential_evolution"` for global search). `objective` selects
`"eta"` (maximize efficiency) or `"net_work"`.

## 9. Adding a new backend

ThermoLab is backend-agnostic: the flash solver, `State`, cycles, plotting, and
tables only call `BaseBackend`. To add CoolProp / Cantera / REFPROP /
pycalphad:

1. Subclass `thermolab.BaseBackend`.
2. Implement the abstract molar interface:
   - `molar_masses() -> np.ndarray` (kg/mol)
   - `guess_phase(T, P, z) -> Phase`
   - `molar_properties(T, P, z, phase) -> MolarProperties`
     (molar `v, h, s, u, a, g, cp, cv, Z, w, beta, kappa_t, jt`)
   - `specific_volume(T, P, z, phase)`
   - saturation / critical helpers (`saturation_pressure`,
     `saturation_temperature`, `saturation_state`, `critical_temperature`,
     `critical_pressure`, `is_two_phase`) where available — raise
     `NotImplementedError` otherwise.
3. Register it via the backend registry and pass `backend="yourname"` to
   `Gas` / `Mixture`.

No public-API change is required; users keep calling `fluid.state(...)` and
reading the same mass-based `State` attributes.

## 10. Exception hierarchy

`ThermoLabError` (base) → `UnsupportedFluidError`, `BackendError`,
`BackendNotAvailableError`, `ConvergenceError`, `TwoPhaseError`,
`FlashSpecificationError`, `UnsupportedPropertyError`, `FluidAliasError`.
