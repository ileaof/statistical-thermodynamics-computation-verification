# StatThermoPy

**StatThermoPy** is a Python package for computing thermodynamic properties of gases
*exclusively from Statistical Mechanics* — via the molecular partition function

$$Q = Q_t \, Q_r \, Q_v \, Q_e$$

— without any empirical property correlations (NASA polynomials, JANAF, Shomate, CoolProp,
REFPROP). Empirical reference data is used only for *optional* validation, never for the
calculation itself.

## Coverage

- Monoatomic, diatomic, linear- and nonlinear-polyatomic ideal gases.
- Translational, rotational, vibrational (quantum harmonic oscillator) and electronic
  contributions, plus **hindered internal rotation** (1-D Mathieu-eigenvalue rotor) for
  single-bond torsions such as the methyl tops of ethane and propane.
- Molar and massic bases for U, H, S, A, G, Cv, Cp, γ, μ, plus total partition function.
- Ideal-gas mixtures (mole or mass fractions).
- Molecular database of 22 species (easily extensible via YAML).
- CLI scientific terminal, property-vs-T plots, and export to CSV/JSON/YAML/Excel/LaTeX.
- **Statistical transport properties** — first-principles transport & thermophysical coefficients
  of a pure gas from the Chapman–Enskog first-order solution of the Boltzmann equation with the
  Lennard–Jones pair potential: dynamic/kinematic viscosity, thermal conductivity, thermal
  diffusivity, binary & self diffusion, Prandtl/Schmidt/Lewis numbers, compressibility factor,
  speed of sound, expansion coefficient, isothermal compressibility, Joule–Thomson coefficient.
  Heat capacities and γ come from the partition-function engine; the only molecular inputs are
  the LJ σ/ε (no REFPROP/CoolProp). Curves vs T / vs P, 2-D T×P maps, CSV/Excel/Tecplot/PNG/PDF.
- **Qt GUI** (optional, PySide6) with Properties / Plot / Transport / Validate tabs and light/dark theming.
- **Automatic validation** against embedded NIST/JANAF reference data (Cp° and S° for all 22
  species in the database; 20 from NIST WebBook Shomate, C2H6/C3H8 from NASA Glenn polynomials).
- **Performance backends** — pluggable NumPy / Numba / OpenMP / CUDA execution with the same
  physics and API; Numba `@njit` kernels accelerate the quantum rotational J-sum and the
  temperature-batched property grid (CUDA auto-falls back to CPU without an NVIDIA GPU).
- Unit tests targeting ≥95 % coverage.

## Install (development)

```bash
python -m pip install -e ".[dev]"          # core + CLI
python -m pip install -e ".[dev,gui]"       # ... plus the Qt GUI (PySide6)
python -m pip install -e ".[dev,accel]"     # ... plus Numba/OpenMP CPU acceleration
```

## Quick start

```python
from statthermopy import Thermodynamics, State
from statthermopy.database import get

n2 = get("N2")
st = State(T=298.15, P=101325.0)
th = Thermodynamics(n2, st)
print(th.properties())   # molar and massic report
```

CLI:

```
$ statthermopy
> gas N2
> T = 298.15
> P = 101325
> properties
```

## Validation

Cross-check the engine against the embedded NIST/JANAF reference tables (only reference
*values* ship — no empirical correlation coefficients, so the calculation core stays pure
statistical mechanics):

```python
from statthermopy.validation import validate, list_references

print(list_references())          # ['AR', 'C2H2', 'C2H4', 'C2H6', 'C3H8', 'CH4', 'CL2', 'CO', ...]
                                  # 22 species: the full molecular database
r = validate("N2", "Cp")
print(r.mean_abs_error_percent)   # ~0.38 % (rigid-rotor / harmonic-oscillator)
```

Or run the bundled example:

```bash
python examples/validate.py
```

## Performance backends

The engine routes its array work and hot loops through a pluggable `Backend` ABC. The default
`numpy` backend is always available. Three accelerated backends are functional and selectable at
runtime — **same physics, same API, only the numerical execution changes** (no empirical
correlation is introduced):

```bash
python -m pip install -e ".[accel]"     # numba (CPU) → enables numba / openmp / cuda-fallback
```

```python
from statthermopy.backend import set_backend, list_backends, available_backends

print(list_backends())        # ['numpy', 'numba', 'openmp', 'cuda'] — declared architecture
print(available_backends())   # those importable here (cuda only if an NVIDIA GPU is detected)

set_backend("numba")          # @njit CPU kernels (quantum J-sum + T-batched property grid)
set_backend("openmp")         # @njit(parallel=True) + prange over the temperature grid
set_backend("cuda")           # numba.cuda GPU; auto-falls back to numba CPU (+ warning) without a GPU
```

Results are identical to the NumPy path to machine precision. The accelerated backends import
lazily, so `import statthermopy` never pulls in numba. Benchmark with:

```bash
python examples/benchmarks.py
```

## GUI (optional)

```bash
python -m pip install -e ".[gui]"
statthermopy-gui        # or: python -m statthermopy.gui.app
```

The window offers four tabs: **Properties** (pure-gas or mixture editor + state inputs +
results and per-mode breakdown tables, in a two-column layout), **Plot** (any property vs
T, embedded matplotlib canvas), **Transport** (point-evaluation table of all transport
properties at a chosen (T, P), plus curves vs T / vs P and 2-D T×P maps with multi-property
selection, exported to CSV/Excel/Tecplot/PNG/PDF), and **Validate** (engine vs NIST/JANAF
with a coloured PASS/FAIL verdict badge and a comparison plot). The interface follows a
harmonious light/dark design system (semantic palette, accent primary buttons,
hover/pressed/disabled states, rounded borders, vector icons); **View → Theme** toggles
Light / Dark / System, and the matplotlib canvases recolour with the theme. It adds no
physics — it wraps the same public API as the CLI.

## Statistical transport properties

Transport and thermophysical coefficients of a pure gas are derived from the **Chapman–Enskog**
first-order solution of the Boltzmann equation with the **Lennard–Jones 12-6** pair potential,
keeping the engine first-principles: the heat capacities and γ come from the partition-function
engine, the only molecular inputs are the LJ parameters σ (collision diameter) and ε (well
depth) stored per species as `LennardJones`. No external property database (REFPROP/CoolProp)
is used.

```python
from statthermopy.transport import TransportCalculator, binary_diffusion
from statthermopy import State
from statthermopy.database import get

res = TransportCalculator(get("N2"), State(T=300, P=101325)).compute()
res.mu      # 1.77e-5 Pa·s  (dynamic viscosity)
res.k       # 0.0260 W/m·K  (thermal conductivity, Eucken)
res.D_self  # 1.8e-5 m²/s   (self-diffusion)
res.Pr      # 0.737         (Prandtl, Eucken closed form)
res.a       # 353 m/s       (speed of sound)

D = binary_diffusion(get("N2"), get("O2"), 300, 101325)   # 2.06e-5 m²/s, symmetric
```

CLI one-shot (prints all properties and saves a viscosity-vs-T plot):

```
statthermopy transport --gas N2 --T 300 --P 101325 --png n2_visc.png
```

Interactive terminal:

```
> gas N2
> T = 300
> transport              # print all 13 transport properties at (T, P)
> transport mu 300 1500 100 n2_mu.png     # plot μ vs T, save PNG
> transport binary N2 O2                  # binary diffusion D_ij at (T, P)
```

**Scope.** This is the dilute/ideal-gas (Chapman–Enskog first-approximation) regime, valid from
0 K up to the highest supported temperature at any low-to-moderate pressure; μ and k are
T-only, while ν, α, D scale as 1/P. Polar species (H₂O, NH₃, H₂S, SO₂) use the LJ
approximation — larger uncertainty, noted per species. The architecture is open to dense-gas
(Enskog / corresponding-states), mixture diffusion, plasma transport and combustion/CFD
coupling as future extensions behind the same `TransportCalculator` interface.

## License

MIT. See `LICENSE`.
