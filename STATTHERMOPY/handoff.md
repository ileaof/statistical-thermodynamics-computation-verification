# StatThermoPy — Handoff (Phase 6 complete)

> Snapshot before autocompact. Read this first when resuming.
>
> **Phase 6 (hindered internal rotation) is complete and verified** — see the section at the end.
> A separate GUI fix (Plot tab ignored Mixture mode → plotted argon as a flat line) is also done:
> `plots.plot_mixture_property` + `_on_plot`/`_sync_plot_props` in `gui/mainwindow.py`.

## Where things stand

Phase 4 is **complete and verified** (on top of the unchanged Phase-1/2/3 core): embedded
NIST/JANAF validation reference data was extended from the 9-species core set to **all 22 species**
in the molecular database. Phase 5 (GUI overhaul) is also **complete and verified**: the PySide6
window now has a modern light/dark design system, accent primary buttons, vector icons,
reorganised two-column Properties layout, and a live **View → Theme** toggle. See the Phase 4
and Phase 5 notes below. Phase 3 (Numba / OpenMP / CUDA performance
backends) remains **complete and verified**, with **no change to the existing public API**
(additions + internal rewire only):

1. **Backend seam completed** — `Vibrational` and `Electronic` now route their array work
   (`exp/expm1/log/log1p/sum/asarray`) through `get_backend()`; `Rotational`'s quantum linear-rotor
   J-sum calls a backend kernel; `Thermodynamics.property_vs_T` takes a T-batched fast path. With
   the default `NumpyBackend` every kernel returns `None` and the original Python path runs
   unchanged (zero behaviour change, zero physics duplication).
2. **`NumbaBackend` (`"numba"`)** — `@njit(cache=True)` CPU kernels for the exact quantum J-sum
   (`linear_quantum_moments`) and the T-batched molar-property grid (`molar_property_grid`).
3. **`OpenMPBackend` (`"openmp"`)** — `@njit(parallel=True)` + `numba.prange` over the temperature
   grid, reusing the Numba per-temperature device function (no physics duplication).
4. **`CudaBackend` (`"cuda"`)** — `numba.cuda` GPU kernel with **automatic CPU fallback**: when
   `numba.cuda.is_available()` is false, construction warns and delegates to the Numba CPU
   backend, so `set_backend("cuda")` never raises on a GPU-less machine.

The calculation core remains **pure statistical mechanics** — the acceleration changes only the
numerical execution, never the physics or the data. No empirical correlation coefficients
(NASA/Shomate/JANAF/CoolProp/REFPROP) are introduced.

## Verification (all passing)

- `python -m pytest --cov=statthermopy` → **96 % coverage**, all tests pass.
  Every backend module at 100 %; `executor.py` at 98 % (line 131 is the cached `numpy`
  early-return, unreachable after first build).
- `python examples/benchmarks.py` (2000-point grid): **numba 270–440× / openmp 148–303×** speedup
  vs numpy; max relative error **0.00e+00** (machine precision) for N2/CO2/H2.
- CLI regression: `statthermopy run --gas N2 --T 298.15` → Cp_m=29.112901 (unchanged from Phase 1).
- Accelerated validation: `set_backend("numba"); validate("N2","Cp")` → 0.381 % MAE (< 5 %).
- `import statthermopy` stays light — `numba` is **not** imported (subprocess assertion); GUI also
  not imported.
- `set_backend("cuda")` on a GPU-less machine → `RuntimeWarning` + Numba CPU fallback; results
  identical; `available_backends()` correctly omits `cuda`.

## Files added in Phase 3

**Backends:**
- `src/statthermopy/backend/numba_backend.py` — `NumbaBackend`; `_extract_spec(mol)` (geometry /
  symmetry / mass / theta_rot / theta_v / deg_v / theta_e / g_e); `_make_kernels()` →
  `_q_rot_jit`, `_props_at_T` (shared per-T device function), `_molar_props_jit` (T-batched). All
  `@njit` kernel bodies carry `# pragma: no cover` (coverage.py cannot trace compiled machine
  code). Physics uses `math.expm1`/`math.log1p` (NOT `ex-1.0`) to match `np.expm1`/`np.log1p`.
- `src/statthermopy/backend/openmp_backend.py` — `OpenMPBackend(NumbaBackend)`; `_parallel_kernel()`
  builds `@njit(cache=True, parallel=True)` with `prange` over T, reusing `_props_at_T`.
- `src/statthermopy/backend/cuda_backend.py` — `CudaBackend`; `__init__` checks
  `_cuda_available()`, else warns + `self._cpu = NumbaBackend()`. GPU kernel in
  `_build_cuda_kernel()` (`@cuda.jit(device=True)` helpers + `@cuda.jit` one-thread-per-T), all
  `# pragma: no cover` (no NVIDIA GPU in the dev environment). **`self._qsum`** is a *separate*
  delegate from `self._cpu` so the scalar J-sum never flips the GPU flag (a real bug I fixed).
- `tests/test_backend.py` — ~32 tests. Autouse fixture resets the global backend to numpy before
  and after each test. Always-on (numpy + registry, subprocess lazy-import). Numba tests gated on
  `pytest.importorskip("numba")`: 8-species × 10-prop grid match, single-T compute, quantum
  rotation (H2/NO/N2), `linear_quantum_moments` vs Python loop, openmp match, cuda match + fallback
  warning + compute match, numba-accelerated `validate`, 2000-point smoke.
- `examples/benchmarks.py` — times `property_vs_T('Cp_m', 2000 pts)` for numpy/numba/openmp/cuda
  across N2/CO2/H2; prints speedups and confirms identical results.

## Files modified in Phase 3

- `src/statthermopy/backend/executor.py` — string-keyed registry (`get_backend`/`set_backend`/
  `list_backends`/`available_backends`), lazy `_build_backend` (ImportError → hint
  `pip install statthermopy[accel]`), and two **concrete default-`None`** kernels on the ABC
  (`linear_quantum_moments`, `molar_property_grid`) so `NumpyBackend` inherits unchanged behaviour.
- `src/statthermopy/backend/__init__.py` — re-exports the new symbols; updated docstring.
- `src/statthermopy/modes/vibrational.py` & `modes/electronic.py` — `np.*` → `get_backend().*`.
- `src/statthermopy/modes/rotational.py` — `_q_linear_quantum_moments` calls
  `get_backend().linear_quantum_moments(...)` first, falls back to the Python loop if `None`.
  Classical branches (`math.log/math.pi`) unchanged. `Translational` unchanged (closed-form).
- `src/statthermopy/thermodynamics.py` — `_GRID_DERIVABLE` frozenset + `_derive_property_array`;
  `property_vs_T` calls `get_backend().molar_property_grid(...)` and derives H/G/Cp/γ/massic/Q from
  the eight core arrays, else falls back to the per-T Python loop (single source of truth).
- `pyproject.toml` — `accel = ["numba>=0.60"]`, `cuda = ["numba>=0.60"]`, numba added to `dev`.
- `docs/api.rst` — added `numba_backend`/`openmp_backend`/`cuda_backend` automodules.
- `docs/THEORY.md` §9 + `docs/theory.rst` — backend section rewritten: now *functional*
  (Numba/OpenMP/CUDA with fallback), same physics, no empirical correlations, lazy import.
- `README.md` — "Performance backends" section + coverage bullet + `[accel]` install line.

## Key design notes for resuming

- **Default-`None` kernels on the ABC** (not abstract): `NumpyBackend` and any user `Backend`
  subclass inherit them and keep the original Python path — zero behaviour change, zero risk of
  import cycle. Accelerated backends *override* them.
- **Machine-precision match**: the `@njit` kernels mirror the mode modules' closed forms and use
  `math.expm1`/`math.log1p`; verified 0.00e+00 vs numpy for classical **and** quantum rotation.
- **`cache=True`** is on every `@njit` kernel — first run compiles, later runs load from disk.
- **Device-function sharing**: a plain `@njit` function called from another `@njit` is
  auto-treated as a device function. Do **not** pass `device=True` to CPU `@njit` — numba 0.66
  rejects it (it is CUDA-only) with `KeyError: Unrecognized options: {'device'}`.
- **CudaBackend flag discipline**: `self._cpu` is *only* the fallback flag (set when no GPU);
  `self._qsum` is a separate Numba delegate for the scalar J-sum. Delegating the J-sum through
  `self._cpu` would flip the GPU flag and break the GPU `molar_property_grid` path.
- **Coverage & `@njit`**: coverage.py cannot trace compiled function bodies — all kernel `def`
  lines and the GPU-only branches carry `# pragma: no cover`. Traceable dispatch code is covered
  by real tests (type errors, cuda compute/fallback).
- **Environment**: numba 0.66.0 + llvmlite 0.48.0 on numpy 2.4.6 / Python 3.11.4 (Windows 11); no
  NVIDIA GPU. scipy is too old for numpy 2.4 and emits a non-fatal
  "NumPy version >=1.21.6 and <1.28.0 required" warning on numba's BLAS check — kernels still
  compile and run correctly; the noise is hidden by pytest capture on passing tests.
- **Test ordering bug (fixed)**: in `test_*_matches_numpy`, compute all numpy references **first**
  (while the backend is numpy), then switch to the accelerated backend and compare — never flip
  the global backend inside the comparison loop.

## Phase 2 context (still valid)

GUI Qt (PySide6) and automatic NIST/JANAF validation with embedded reference data remain as
delivered in Phase 2. See the Phase-2 notes below for the data-curation method, entropy pressure
default, H-not-shipped limitation, matplotlib backend ordering, and the GUI-test `QMessageBox`
hang — all still apply.

- **Data curation**: embedded Cp°/S° were produced by evaluating the NIST Chemistry WebBook
  **Shomate equations** at the T grid (coefficients fetched via WebFetch, **not** shipped). CO
  and NO WebFetch results were initially swapped (diagnosed via ΔfH°: NO=+90.29, CO=−110.53
  kJ/mol) and corrected. Re-fetch from
  `https://webbook.nist.gov/cgi/cbook.cgi?ID=C<CAS>&Mask=1` and re-evaluate with
  `t=T/1000; Cp=A+B*t+C*t²+D*t³+E/t²; S=A*ln(t)+B*t+C*t²/2+D*t³/3−E/(2t²)+G`.
- **Entropy pressure**: `validate(species,"S")` defaults to the reference's declared pressure
  (1 bar = 100000 Pa), NOT `ValidationRunner`'s 1-atm default — entropy depends on P via the
  translational term.
- **H not shipped**: engine `H_m = U_m + RT` is absolute; NIST gives `H° − H°(298.15)`. A
  reference-state offset would be needed — deferred.
- **Matplotlib backend**: `gui/app.py` must call `matplotlib.use("QtAgg")` before any
  `statthermopy.plots` import (`plotting._get_pyplot()` hard-forces Agg).
- **GUI test hangs**: never exercise `QMessageBox.warning/critical` in tests (modal → offscreen
  hang). Those branches carry `# pragma: no cover`.

## Still deferred (out of Phase 3 scope)

- Technical book in Word (`.docx`) — dedicated future phase.
- H° validation with reference-state offset.
- Backend via cupy puro (in addition to numba.cuda); parallelising the quantum J-sum reduction
  with `prange` if profiling shows it as a bottleneck.

## Phase 4 — validation data extended to all 22 species (complete)

Added embedded NIST/JANAF reference YAMLs (`validation/data/*.yaml`) for the 13 remaining
species: **He, Ne, Kr, Xe, Cl2, NH3, SO2, H2S, N2O, C2H2, C2H4, C2H6, C3H8**. `list_references()`
now returns 22 species (== every species in the molecular database).

**Provenance:**
- 11 species via the NIST WebBook gas-phase **Shomate** coefficients (Chase 1998), evaluated at the
  T grid `[298.15, 400, 600, 800, 1000, 1500, 2000]` (same grid as the Phase-2 core set). Identity
  verified by the listed S°(298.15) cross-check (all match to ~0.01 J/mol/K — no CO/NO-style
  swap).
- **C2H6, C3H8** — NIST WebBook publishes no Shomate fit for these (only discrete Cp; no S°).
  Per user decision, their reference Cp & S come from the **NASA Glenn 7-coefficient polynomials**
  (McBride, Zehe & Gordon, NASA/TP-2002-211556, via Cantera `nasa_gas.yaml`). Only the *values*
  ship — no Shomate/NASA coefficients — so the calculation core stays pure statistical mechanics.
  NASA Cp(298.15) cross-checks NIST's discrete Cp (C2H6 52.50 vs 52.49; C3H8 73.59 vs 73.60).

**Constant refinement (per user decision "refine constants to pass"):**
- **C3H8** initially failed the 5% gate (Cp MAE 7.06%, max 9.64%) because its database YAML used
  approximate grouped fundamentals. Replaced with the 27 experimental gas-phase fundamentals from
  the **Shimanouchi (1972)** compilation (via NIST WebBook vibrational levels, C2v — all
  non-degenerate), including the two low torsions (216, 268 cm⁻¹) the grouped set had
  mis-represented. After refinement: **Cp MAE 2.28%, max 3.69%; S MAE 0.55%**.
- All other species passed <5% with their existing constants (no database changes needed).

**Final per-species MAE (Cp / S, mean abs %):** He 0.001/0.001 · Ne 0.001/0.001 · Kr 0.001/0.000
· Xe 0.001/0.001 · Cl2 1.219/0.210 · NH3 1.358/0.266 · SO2 0.890/0.092 · H2S 0.625/0.060 ·
N2O 0.045/0.007 · C2H2 1.072/0.158 · C2H4 0.083/0.010 · C2H6 2.294/0.909 · C3H8 2.277/0.553.
Worst case across all 22 species: Cp 3.93% (H2 at high T), S 1.44%.

**Files added:** `validation/data/{He,Ne,Kr,Xe,Cl2,NH3,SO2,H2S,N2O,C2H2,C2H4,C2H6,C3H8}.yaml`.
**Files modified:** `database/data/C3H8.yaml` (fundamentals refined); `tests/test_validation_reference.py`
(added `_EXTENDED` set + `test_list_references_contains_extended_set` +
`test_validate_extended_within_tolerance` (×26) + `test_validate_noble_gases_near_exact` (×4);
repointed `test_validate_missing_species_raises` from `C3H8` → `C10H22`); `README.md`;
`docs/THEORY.md` §8b; `docs/theory.rst`; `examples/validate.py` (prints actual worst-case MAE).

**Verification:** `pytest --cov=statthermopy` → 261 passed, **96% coverage**. `python
examples/validate.py` → all 22 species <5%. `import statthermopy` still keeps numba lazy.
Hatchling ships the new YAMLs as package data (same dir as the existing 9).

## Phase 5 — GUI overhaul (complete)

Replaced the unstyled default-Qt window with a modern, professional light/dark design. The
GUI still adds no physics — it wraps the same public API; only presentation changed.

**New module `gui/theme.py`** — a self-contained design system (PySide6-only, imported lazily
by `mainwindow.py`, never by the core):
- `Palette` dataclass of semantic tokens + `LIGHT` / `DARK` instances (bg/surface/surface_alt,
  border/border_strong, text/text_muted/text_disabled, accent(+hover/pressed/text/soft),
  success/danger (+soft), radius/radius_card).
- `qss(p)` — Qt Style Sheet builder: cards (`QFrame#Card`, `QGroupBox`), `QPushButton`
  with hover/pressed/disabled and a dynamic `[primary="true"]` accent rule, inputs (combo/
  spinbox/lineedit) with focus-accent borders, checkbox/radio indicators, tables (alternating
  rows, themed header, accent-soft selection), tabs (accent top bar on the active tab),
  menu/menu-bar, status bar, scroll bars, tooltips.
- `apply_qt_palette(app, p)` — a matching `QPalette` for the native bits QSS doesn't fully
  reach (spinbox arrows, tooltips, the matplotlib toolbar).
- `make_icons(p)` — `QPainter`-drawn vector glyphs (`plus`, `minus`, `play`, `check`,
  `chart`, `refresh`), regenerated per theme so colours track light/dark.
- `detect_dark()` via `QApplication.styleHints().colorScheme()` (PySide6 6.6+; `Unknown`
  → light); `default_font()` → `QFont("Segoe UI", 10)`.

**`gui/mainwindow.py`** (refactored, all pinned widget attributes/methods/tab-labels/
table-shapes preserved — `tests/test_gui.py` unchanged):
- `__init__` calls `_apply_theme(_detect_theme())` after build; stores `_theme_mode`,
  `_theme_palette`, `_theme_choice`, `_theme_actions`.
- `_apply_theme(mode)` sets the app stylesheet + palette, regenerates/sets button icons,
  re-polishes primary buttons, and recolours both matplotlib canvases.
- `_PlotCanvas.apply_theme(palette)` recolours figure/axes/ticks/spines/grid on theme change
  (`ax`/`refresh` unchanged).
- **Matplotlib backend import deferred**: `from matplotlib.backends.backend_qtagg import ...`
  moved out of module top into `_PlotCanvas.__init__`. Reason: the console script
  `statthermopy-gui` does `from statthermopy.gui.app import main`, which imports the
  `statthermopy.gui` package (→ `__init__` → `mainwindow`) *before* `app.main()` can call
  `matplotlib.use("QtAgg")`. Importing `backend_qtagg` with no backend pinned makes matplotlib
  probe for Qt bindings and load a foreign Qt DLL — on this machine a **Qt5 DLL from
  `C:\Program Files\Tecplot\...\bin` (on PATH)** — which then clashes with PySide6's Qt6
  (`ImportError: DLL load failed ... "specified procedure not found"`). Deferring the import
  to canvas construction (which happens after `_ensure_qt_backend()` pinned QtAgg) fixes it.
  `matplotlib.figure` and `..plots` imports are safe at top level (verified); only
  `backend_qtagg` probes. Tests dodge this because `test_gui.py` calls `matplotlib.use("QtAgg")`
  first; the bug only bites a direct `statthermopy-gui` launch. Symptom/error is environment-
  specific (needs a conflicting Qt DLL on PATH) — re-introducing a top-level `backend_qtagg`
  import will silently re-break the launcher on such machines.
- `_build_menu` keeps Export (with a save icon) and adds **View → Theme** (`System`/
  `Light`/`Dark`, exclusive `QActionGroup`) wired to `_on_theme_chosen`.
- Primary accent buttons: `compute_btn` (check icon), `plot_btn` and `val_btn` (play icon);
  `add_row_btn`/`del_row_btn` get plus/minus icons.
- Layout: Properties tab → horizontal `QSplitter` (left = Selection + State + primary
  Compute, right = Results + Per-mode cards); Plot tab → `QFrame#Card` controls + canvas;
  Validate tab → controls card + `val_status` verdict badge (`QLabel[verdict="idle|pass|fail"]`,
  text still contains PASS/FAIL) + horizontal `QSplitter` (table | plot). Tables use
  alternating rows. Window resized to 1100×760.
- `_set_verdict()` re-polishes the badge label.

**`gui/app.py`** — `main()` sets `app.setFont(theme.default_font())` before constructing the
window (window also self-applies the theme for the test path). Backend ordering unchanged.

**Files added:** `src/statthermopy/gui/theme.py`.
**Files modified:** `src/statthermopy/gui/mainwindow.py` (theme + icons + layout + View menu);
`src/statthermopy/gui/app.py` (default font); `src/statthermopy/gui/__init__.py` (trailing
newline); `tests/test_gui.py` (4 new theme tests: applied-on-construct, dark-restyle,
toggle-via-menu, helpers; plus ruff fixes); `README.md` (GUI section).

**Verification:** `pytest --cov=statthermopy` → **265 passed, 96% coverage**
(`theme.py` 98%, `mainwindow.py` 95%, `app.py` 50% — `main()` event loop `# pragma: no cover`).
`import statthermopy` still keeps the GUI out of the core import
(`statthermopy.gui` not in `sys.modules`). Visual check (display required): buttons clearly
contrast, accent primaries, hover/pressed states, rounded corners, View → Theme toggles
light↔dark live with canvas facecolors following. No new heavy deps; PySide6 stays optional.

## Phase 6 — hindered internal rotation (complete)

Added a physically rigorous 1-D **hindered internal-rotor** treatment for single-bond torsions,
replacing the harmonic-oscillator approximation of the methyl torsions in **ethane** and
**propane**. Still pure statistical mechanics: only two spectroscopic constants per rotor
(internal-rotation constant F and barrier V_n) — no empirical property correlation.

**Physics (`modes/hindered_rotor.py`, new):** solves the Mathieu Hamiltonian
`Ĥ = -F d²/dφ² + (V_n/2)(1-cos nφ)` by diagonalising it in the free-rotor basis |m⟩ (2·100+1
states), diagonal `F m² + V_n/2`, off-diagonal (m,m±n) `-V_n/4`, via `np.linalg.eigvalsh`.
Levels are referenced to the ground level (matches the harmonic v=0 zero). Partition function
`q = (1/σ_int) Σ exp(-ε_i/kT)`; thermodynamics from the level-distribution moments
(`U=RT⟨x⟩`, `Cv=R(⟨x²⟩-⟨x⟩²)`, `S=R(ln q+⟨x⟩)`, `A=-RT ln q`), summed over rotors (× degeneracy).
`HinderedRotor(())` is a null mode → molecules without rotors are byte-for-byte unaffected.
Verified limits: free-rotor q matches `√(8π³ I_r kT)/(σh)` to 1e-4 and Cv→R/2; high-barrier→HO;
ethane ladder reproduces the observed ~289 cm⁻¹ torsional fundamental (level index 3 — the 3-fold
symmetry makes each torsional level a near-degenerate triplet with ~0.006 cm⁻¹ tunnelling split).

**Data model:** new frozen `InternalRotor` dataclass (`core/molecule.py`) —
`rotation_constant_cm1` (F), `barrier_cm1` (V_n), `symmetry` (σ_int, default 3), `n_minima`
(default 3), `degeneracy`. `Molecule.internal_rotors` tuple field + `n_internal_rotors`. **DOF
validation changed**: nonlinear/linear now require `n_osc + n_rot == 3N-6 / 3N-5` (not just
`n_osc`); monatomic rejects rotors. Exported at top level and from `core`. YAML loader parses an
`internal_rotors:` list.

**Wiring:** `PartitionFunction` builds `self.internal_rotation = HinderedRotor(mol.internal_rotors)`
and **folds it into Q_v** — `evaluate()` adds its `ln_q` to `lnQv`; `contributions()["vibrational"]`
is harmonic+rotor via `_vibrational_contribution` (name kept "vibrational"). So the reported
4-factor Q (Qt/Qr/Qv/Qe) and the *default* 4-key contributions dict are **unchanged** —
exporters and `Thermodynamics` untouched. `modes` property exposes `internal_rotation` as a 5th
entry (test_coverage updated).

**GUI per-mode exposure:** `contributions(state, split_internal_rotation=True)` (opt-in kwarg,
default folded so all other callers are unaffected) returns a 5th `"internal_rotation"` entry
**only when the molecule has rotors**, with `"vibrational"` then harmonic-only; the two views
give identical totals. `gui/mainwindow._populate_modes` passes `split_internal_rotation=True`,
so C2H6/C3H8 show a dedicated "internal rotation" row (6 rows incl. totals) while the other 20
species are unchanged (5 rows). Row labels underscore→space. Tests:
`test_split_internal_rotation_reports_separate_mode` (partition) +
`test_modes_table_shows_internal_rotation_row_for_ethane` (GUI).

**Backend:** accelerated `molar_property_grid` kernels model only harmonic vibration, so
`_has_internal_rotors(mol)` (in `numba_backend.py`, imported by openmp/cuda) makes all three
return `None` → exact per-T Python fallback for C2H6/C3H8. Zero physics duplication; numba grid
matches numpy to 1e-10 (tested).

**Database:** `C2H6.yaml` — dropped the 289 cm⁻¹ a_u torsion (17 oscillators) + 1 rotor
(F=10.7, V3=1024 cm⁻¹, σ=3). `C3H8.yaml` — dropped the 216 & 268 cm⁻¹ torsions (25 oscillators)
+ 1 rotor entry degeneracy 2 (F=5.3, V3=1190 cm⁻¹, σ=3, two independent identical methyl tops;
top-top coupling neglected). Both still sum to 3N-6.

**Validation impact (vs old harmonic MAE):** C3H8 Cp **2.28 % → 0.29 %** (near-exact 298–2000 K);
C3H8 S 0.55 % → 1.26 % (≈constant +1 % entropy offset); C2H6 Cp 2.29 % → 2.54 % (systematic
−1…−4 % under-prediction, worst at 600–800 K — residual anharmonicity, not the torsion); C2H6 S
0.91 % → 0.75 %. All four well within the 5 % gate. Constants are literature spectroscopic values
(reproduce the observed torsional fundamentals); **not** fitted to Cp/S — keeping the core
first-principles.

**Tests:** `tests/test_hindered_rotor.py` (new, 16 tests, module at 100 %): free-rotor Cv/q
limits, high-barrier/low-T freeze-out, torsional fundamental, `Cv==dU/dT` finite-diff,
`U==RT² dlnq/dT`, degeneracy==independent-rotors, null mode, high-T Cv below HO, DOF split for
C2H6/C3H8, ethane compute sanity, fold-into-vibrational, monatomic-rejects-rotor, param
validation, numba fallback+match. Updated `test_database.py::test_vibrational_oscillator_counts`
(17/25 + rotor counts) and `test_coverage.py::test_partition_modes_dict` (5 keys).

**Verification:** `pytest --cov=statthermopy` → all pass, **96 %** (hindered_rotor 100 %).
`import statthermopy` still keeps numba/GUI lazy. New files ruff-clean.

## Thermal fields T_v, T_p (complete)

Two new derived curves in the visualisation module: the constant-volume thermal field
**T_v = U_m/Cv_m** and the constant-pressure thermal field **T_p = H_m/Cp_m** (both in K; for a
monatomic gas both equal T exactly — a useful diagnostic). Added as first-class properties:

- `ThermoProperties` and `MixtureProperties` gained `T_v`/`T_p` (computed in each `compute()`).
- `property_vs_T` fast path: added to `_GRID_DERIVABLE` + derived in `_derive_property_array`
  (from the grid's U_m/H_m/Cv_m/Cp_m — grid matches per-T path, tested).
- `plots`: added to `MOLAR_PROPS` (⇒ `MIXTURE_PROPS`, GUI combo, `plot_all_properties`). New
  `PROP_UNITS` map + `_ylabel()` put units on every axis; `plot_property`/`plot_mixture_property`
  gained a `color` kwarg and now always draw a legend. New `plot_thermal_fields` /
  `plot_mixture_thermal_fields` overlay both fields with distinct colour-blind-safe colours
  (T_v `#0072B2`, T_p `#D55E00`), descriptive legends and a "Thermal field [K]" y-label.
- GUI: T_v/T_p are individually selectable (via MOLAR_PROPS) and also as a combined
  `_THERMAL_FIELDS_ITEM = "T_v & T_p (thermal fields)"` combo entry (added in `_build_plot_tab`
  and `_sync_plot_props`, handled in `_on_plot`). Results table shows `T_v`/`T_p` rows (unit K,
  no massic counterpart — like γ). Export is automatic (`asdict`).
- Tests: `test_plot_all_properties` 13→**15**; new `test_thermal_fields_definitions_and_units`,
  `test_plot_thermal_fields_two_distinct_curves`, `test_thermal_fields_property_vs_T_grid_matches`,
  `test_mixture_thermal_fields`, `test_thermal_fields_available_in_gui`. Suite: **291 passed, 96%**.

## Predefined fluids — atmospheric Air (complete)

New **`statthermopy/fluids.py`** module: named gas compositions built on the ideal-gas mixture
engine, exposed via the CLI and GUI. Flagship fluid **Air** = standard dry-air mole fractions
(`STANDARD_DRY_AIR` = N2 0.78084, O2 0.20946, AR 0.00934, CO2 0.00040 → M̄ 28.96 g/mol, R 287
J/kg/K) with **optional water vapour** as a *mole-fraction* input (`air(water_mole_fraction=…)`;
dry constituents scale to 1−w, H2O added at w — no saturation-pressure correlation, stays
first-principles). API: `air()`, `PredefinedFluid` (name/description/composition/source/
`humidifiable`; `.build(water_mole_fraction=…)`, `.dry_composition()`), open registry
`register_fluid`/`available_fluids`/`get_fluid` (case-insensitive). Extensible & decoupled from
*evaluation* so a future non-ideal mixture model slots behind the same factory. **Planetary
atmospheres were explicitly cut** (user: "corte atmosferas de outros planetas") — none registered.
All exported from the top-level package.

**Mixture breakdown (`mixture.py`):** new `ComponentContribution` dataclass (per species: x,
molar_mass, its pure-component `*_m` props, and the weighted `*_contrib` = x_i·value that sum to
the mixture molar total). `MixtureProperties` gained `S_mixing` (−R Σ x_i ln x_i, reported
separately though already embedded in S_m via the partial-pressure entropies) and
`components: dict[str, ComponentContribution]`. `compute()` builds both. `asdict`/export handle
the nested dataclasses (JSON-serialisable, verified). `math` import is now actually used.

**CLI (`cli/app.py`):** `fluids` (list) and `fluid Air [h2o=0.01]` (select preset as the active
mixture); `_print_mixture` now prints R_specific, S_mixing and the per-component contribution
table; one-shot `run --fluid Air [--humidity 0.01]`.

**GUI (`gui/mainwindow.py`):** "Preset fluid" combo + "Load preset" button in Selection → switches
to Mixture mode and fills the **editable** mix table with the fluid's composition
(`_on_load_preset`/`_set_mixture_composition`, `Qt.MatchFixedString` for case-insensitive species
match). Right column now has a second `self.components_box`/`components_table` shown for mixtures
(hidden for pure gases, which keep the modes table — pure 5-row test unchanged); `_on_compute`
toggles `modes_box`/`components_box`. `_populate_results` appends M_avg / R_specific / S_mixing
rows for mixtures (detected via `hasattr(res,"S_mixing")`). `_populate_components` fills the table
+ a "Σ total" row. Constants `_COMPONENT_COLS`/`_COMPONENT_ATTRS`. NOTE: the GUI already had a 4th
**Transport** tab (added between my sessions) — untouched.

**Tests:** `tests/test_fluids.py` (10: composition/M̄/R, humidity scaling + bounds, registry
extensibility, contributions-sum-to-totals, S_mixing, as_dict/JSON). `test_auxiliary.py`:
`test_cli_fluid_air`, `test_cli_run_fluid_air`. `test_gui.py`:
`test_air_preset_loads_and_shows_components`. All green; my new files ruff-clean.

## Statistical Humid Air (complete)

New **`statthermopy/humidair/`** subpackage: maximum water-vapour solubility of air vs T and/or P
from vapour–liquid equilibrium `μ_v = μ_l`, with the **vapour phase pure statistical mechanics**
and a **pluggable liquid reference**. No empirical vapour-pressure correlation (Antoine/Magnus/
Tetens). **Planetary atmospheres from the earlier Air task remain out of scope** (user cut).

**Physics / reference reconciliation (the crux).** The vapour Gibbs `g_v(T,P)` comes from
`Thermodynamics(get("H2O"))` (its absolute Sackur–Tetrode+rot+vib+elec entropy = 188.7 vs 188.8
J/mol/K lit). The liquid uses its own reference scale; it is reconciled to the vapour scale with
**two physical anchors at the triple point** (273.16 K, 611.657 Pa): coexistence there, and
`Δh_vap(T_t)=45.054 kJ/mol` — fixing the liquid enthalpy/entropy offsets (Δh₀, Δs₀). Then solve
`g_v(T,P)=g_l(T,P)` for P (explicit `P_sat=P_ref·exp[(g_l−g_v)/RT]` + Poynting fixed-point).
**Validated:** P_sat err −0.09 % (20 °C), −0.15 % (25 °C), −1.5 % (100 °C); Δh_vap(298)=44.0 vs
43.99 kJ/mol; w_sat(25 °C)=20.06 g/kg; wet-bulb(25 °C/50 %RH)=17.9 °C — all match psychrometric
tables.

**Files.** `liquid.py`: `LiquidWaterModel` ABC + `ConstantCpLiquid` (transparent, dependency-free,
default fallback) + `IAPWSLiquid` (IAPWS-95 via optional `iapws`, **auto-selected when importable**
— `default_liquid_model()`; `_sat_liquid` clamps T to [T_triple, T_crit) so edge probes never
raise). `saturation.py`: `SaturationCalculator` (anchor, `saturation_pressure`, `saturation_
temperature`/`dew_point` bounded to liquid-vapour range, `enthalpy_of_vaporisation`). `state.py`:
`HumidAirState` (all ~24 outputs + `components` + `vapor_mode_contributions`, `as_dict`).
`humidair.py`: `HumidAir` — builds the moist mixture (dry air scaled + H2O via `IdealGasMixture`),
psychrometrics, **adiabatic-saturation wet-bulb** (energy balance root-find), per-partition-factor
vapour breakdown (PV=RT assigned to translational so per-mode G sums to total). `plots.py`: P_sat/
solubility/w/RH/dew curves + **3-D solubility(T,P) surface** (mpl_toolkits). Dry background is a
swappable `IdealGasMixture` (extensible to trace gases / other mixtures / future real gas).

**Integration.** Top-level exports `HumidAir`, `HumidAirState`, `SaturationCalculator` (import stays
lazy — numba/gui/**iapws** all deferred; iapws only on first `SaturationCalculator()`). CLI: REPL
`humidair [rh=…|w=…|x=…]` + `humidair <psat|solubility|w|rh> Tmin Tmax` plotting; one-shot
`humidair --T --P [--rh]`. `pyproject` extra `humidair=["iapws>=1.5"]` (+ dev).

**Critical analysis** in **`docs/HUMID_AIR.md`**: why direct stat-mech fails for the liquid
(non-factorising Z_N, H-bond many-body coupling), and a comparison table of SAFT / perturbation
theory / integral equations (OZ+PY/HNC/RISM) / lattice / simulation / IAPWS with
advantages·limits·validity, plus the upgrade path (drop a SAFT `LiquidWaterModel` behind the same
`g_v=g_l` framework).

**Tests:** `tests/test_humidair.py` (19: P_sat vs reference for both liquids, triple-point anchor,
dew-point inversion, Δh_vap, absolute vapour entropy, saturated headline numbers, humidity
definitions/round-trips, degree-of-saturation, wet-bulb bracketing, vapour modes sum to G,
mixture bulk, custom dry background, as_dict/JSON, plots incl. 3-D). `test_auxiliary.py`:
`test_cli_humidair`, `test_cli_run_humidair`.

**GUI: implemented** — a 5th **"Humid Air"** tab (`_build_humidair_tab` + `_on_humidair_compute`/
`_on_humidair_plot`/`_on_humid_mode_changed`/`_populate_humidair`, lazy `self._humidair`). Left:
T/P + humidity mode (Saturated / Relative humidity / Humidity ratio / Mole fraction) + Compute, and
a grouped results table (saturation limit · actual state · bulk · molar thermodynamics). Right: the
water-vapour partition-function contribution table + a plot (P_sat / max mole fraction / max
humidity ratio / RH vs T) on a themed `_PlotCanvas`. Tabs are now Properties/Plot/Transport/**Humid
Air**/Validate (Validate moved to index 4). `_apply_theme` recolours `humid_canvas` and re-polishes
the humid buttons; window construction stays iapws-free (HumidAir built lazily on first Compute).
`test_gui.py`: tab-count test updated to 5 + `test_humidair_tab_computes_and_plots`.

**Plot perf fix:** `plot_humidity_ratio_vs_T` / `plot_relative_humidity_vs_T` previously called the
full `HumidAir.state()` per point (redundant dew-point bisection + mixture build) → ~27 s for 80
points, which froze the GUI ("failed"). Rewritten to compute directly from `P_sat(T)`
(`w_s = ε P_sat/(P−P_sat)`, `RH = P_v/P_sat`) → ~1 s, values bit-identical; above boiling
(`P_sat ≥ P`) → NaN (blank). Tests: `test_humidity_ratio_plot_matches_definition_and_handles_
boiling`, `test_relative_humidity_plot_is_direct`.

## Humid Air — comparative graphical analysis (complete)

Extended the humid-air module with two comparative analyses + GUI section + interactive/export.

**`humidair/analysis.py` (new):** `PsychrometricAnalysis(model)` + `ComparisonTable` dataclass
(x, x_K, columns dict, meta; `to_dataframe`/`to_csv`/`to_excel`). `COMPARISON_PROPERTIES` maps 9
display names → (MixtureProperties field, unit): H,U,S,G,A,Cp,Cv,T_v,T_p.
- `water_vapor_content(T_range,P,*,relative_humidity|humidity_ratio|mole_fraction,temperature_unit,
  T_ref)` → actual w & saturation w_sat (g/kg) + RH; actual = min(w_fixed, w_sat) so it tracks
  saturation below the dew point (condensation) and plateaus above; meta has dew_point.
- `property_comparison(prop_field,T_range,P,*,isobaric,isochoric,temperature_unit,humidity,T_ref)`
  → up to 4 columns (dry/humid × const-P/const-V). **Physics:** isochoric holds molar volume at
  `v_ref=RT_ref/P` so `P(T)=P·T/T_ref`. U/H/Cp/Cv/T_v/T_p are T-only → const-P≡const-V (curves
  coincide, `meta["pressure_independent"]=True`); only S/G/A differ. Composition held fixed (no
  condensation in this comparison — that's the water-vapour-content graph's job).

**`humidair/plots.py`:** `plot_water_vapor_content_vs_T` (actual solid + saturation dashed, dew-
point axvline + "onset of condensation" annotation + shaded condensation region) and
`plot_property_comparison` (4 curves: dry/humid by hue #0072B2/#D55E00, P/V by linestyle; note
appended for T-only props). Both return `(ComparisonTable, ax)`. Interactivity helpers
`make_pickable_legend(ax)` (click legend → toggle curve) and `add_hover_tooltip(ax)` (nearest-point
annotation) — `interactive=True` wires them; both `# pragma: no cover` (need a live canvas).

**GUI — dedicated "Thermodynamic Comparisons" tab** (`_build_comparisons_tab`), inserted **after
Humid Air** so the graph gets a full-width, full-height canvas (tabs are now Properties/Plot/
Transport/Humid Air/**Thermodynamic Comparisons**/Validate → **6 tabs**, Validate at index 5).
Compact controls card on top: Analysis combo (Water vapor content / Dry-vs-Humid property),
Property combo (9), Constant P / Constant V checkboxes, P/Tmin/Tmax/N, X-unit (K/°C), its **own**
humidity mode+value, and **Plot / Export Graph / Export Data**. `_on_comparison_plot` draws on
`cmp_canvas` (`interactive=True`) and caches `self._cmp_table`; `_on_comparison_export_graph` →
QFileDialog PNG(dpi=300)/SVG/PDF; `_on_comparison_export_data` → CSV/Excel;
`_on_comparison_analysis_changed`/`_on_comparison_mode_changed` gate the property/value controls.
Widgets are `cmp_*`; `cmp_plot_btn` in theme re-polish, `cmp_canvas` recoloured in `_apply_theme`.
(The comparison was first added as a section inside the Humid Air tab, then **moved to its own
tab** for space — the Humid Air tab keeps only the psychrometric single-curve plots.)

**Thermal Fields Comparison (independence verified).** Third analysis option on the Comparisons
tab: `PsychrometricAnalysis.thermal_fields_comparison` + `plots.plot_thermal_fields_comparison`
→ 4 curves: dry/humid × {T_v = U_m/Cv_m (const V), T_p = H_m/Cp_m (const P)}. **Each field is
computed from its OWN mixture's `mixture.compute()`** (independent `U_m/H_m/Cv_m/Cp_m`) — no
cross-use of dry properties for humid or vice versa. Verified numerically: dry ≠ humid at every T
(dry T_v 318.75 vs humid 318.63 K; dry T_p 319.11 vs humid 319.02 K at 320 K/50 %RH) and each
equals its own U/Cv, H/Cp. Differences are small (~0.1–0.4 K over a wide T range, honest physics —
air's fields are ≈T), so the plot annotates the max |humid−dry| spread (`meta.max_diff_Tv/Tp`) to
make the distinction evident without zooming. GUI: `cmp_analysis` index 2 (property/const-P/V
controls disabled), dispatched in `_on_comparison_plot`.

**Tests:** `test_humidair.py` +6 (actual-capped-by-saturation with dew point, 4-curve S vs coincident
Cp, ComparisonTable CSV/Excel export, plots return table+axes, °C unit, **thermal-fields
independence + no-reuse**). `test_gui.py` `test_thermodynamic_comparisons_tab` (dedicated tab,
both analyses, 4 curves, CSV export). All new code ruff-clean.

## Plan file

`C:\Users\ileao\.claude\plans\synthetic-wobbling-glacier.md` — the approved Phase-3 plan
(was overwritten from the Phase-2 plan).

## Folder rename (pending)

The project root folder was renamed `SimThermoStat` → `STATTHERMOPY` (cosmetic; done by the
user). The Python package name is `statthermopy` (lowercase) and is independent of the folder
name — **no source/doc changes are required** for the rename. The editable install was
re-pointed to the new folder (`pip install -e .` from `STATTHERMOPY`), updating
`_editable_impl_statthermopy.pth` and `statthermopy-0.1.0.dist-info/direct_url.json`. If the
Claude Code memory dir should carry over, copy
`C:\Users\ileao\.claude\projects\C--Users-ileao-OneDrive-Documentos-SimThermoStat` to
`...-STATTHERMOPY` before reopening in the new location.