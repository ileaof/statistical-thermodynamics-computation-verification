# GUI Improvement Report — StatThermoPy

**StatThermoPy — Statistical Thermodynamics in Python**
Prof. Ivaldo Leão Ferreira · FEM - ITEC - UFPA

Scope of this work: presentation, ergonomics, navigation and maintainability of the Qt
interface. **No physics, thermodynamic model or numerical result was changed.** The one
numerical difference a user will notice is described under *Transport* — it is a display
correction, not a model change, and it is verified against the engine.

---

## ORIGINAL GUI

PySide6 (Qt6) with an embedded matplotlib `QtAgg` canvas. A single file,
`src/statthermopy/gui/mainwindow.py`, 1999 lines, one monolithic `StatThermoPyWindow`
whose seven tabs were built by `_build_*_tab()` methods of 40–300 lines each.

The strongest property of the original, preserved throughout: **the GUI reimplements no
physics.** The entire file contained one arithmetic expression (`R*T/P`, a molar volume).
Everything else came from the public API.

Limitations found:

| Area | Finding |
|---|---|
| Window sizing | The widest tab's layout minimum propagated to the window: **2509 × 663 px**. The window refused to shrink below that, so on any laptop panel the right-hand pane was off-screen and unreachable. The binding constraint was horizontal, not vertical. |
| Scrollbars | None. After adding them, the handle was drawn in `surface_alt` on `bg` — **#eef2f7 on #f4f6f9, a contrast ratio of 1.04:1** — effectively invisible. |
| Menus | Only `Export` and `View`. No File, Tools or Help; no About; no way to reach the manual. |
| Status bar | Present (Qt default) but **never written to**. No progress, no confirmation, no visible state. |
| Tables | Ten tables, **none with an `objectName`**. Headers inconsistent (units in the header, in a separate column, or absent). Numbers left-aligned like prose. No copy. |
| Errors | Eight sites doing `QMessageBox.critical(..., f"Computation failed:\n{exc}")` — **the raw exception text shown to the user**. No field validation, no highlight, no Details. |
| Mixture transport | **Unreachable.** `Transport` accepted one species; `Air Transport` built air itself. The core's `MixtureTransportCalculator` has always been generic in *N* components and the CLI exposed it, but the GUI had no route to it. |
| Per-species transport | `Sc` and `Le` printed without naming the species they belong to; the contributions table omitted them entirely. |
| Species information | None. Thirty names in a combo with nothing to tell them apart. |
| Number display | Every state field padded to six decimals: `101325,000000`, `298,150000`. |
| Composition input | The fraction spin box held four decimals, so a ppm component was **rounded to zero on entry** and never reached the mixture. |
| Responsiveness | All computation synchronous in the callback, with no indication. |

---

## NEW DESIGN

The tab architecture was kept — it fits the workflow and is covered by tests — and
reorganised rather than replaced, as the brief permits. What changed is the application
shell around it: identity, navigation, feedback and presentation.

```
File   Tools   View   Help                        ← full menu bar
┌──────────────────────────────────────────────┐
│ ⬛ Properties  ⬛ Plot  ⬛ Transport  ⬛ Humid   │ ← tab icons, all scrollable
│ ⬛ Comparisons  ⬛ Air Transport     ✓ Validate │
├──────────────────────────────────────────────┤
│  Selection / State / Calculation  │  Results  │
├──────────────────────────────────────────────┤
│ Calculation completed   T = 300 K | P = … | 5 species | FEM - ITEC - UFPA
└──────────────────────────────────────────────┘
```

Two new modules keep `mainwindow.py` from absorbing everything:

- `gui/about.py` — identity constants, version lookup and the About dialog.
- `gui/resources.py` — the bundled artwork, with a per-tab mapping.

---

## MAIN WINDOW

- **Fits the screen.** Every tab now lives in a `QScrollArea` (`widgetResizable`, so it still
  stretches to fill a large window). The window minimum fell from **2509 × 663 to 127 × 118**,
  and the opening size is clamped to the available screen area instead of being requested
  blindly.
- **Scrollbars that can be seen.** The handle moved to `text_muted` at 14 px over a delineated
  track: **1.04:1 → 4.23:1** contrast in light mode, **1.28:1 → 5.71:1** in dark.
- **Menu bar.** `File` (Export ▸ HTML/CSV/JSON/YAML/Excel/LaTeX, Exit), `Tools` (jump to
  Species database / Transport / Validation), `View` (Theme, Reset layout), `Help`
  (Documentation, About). Export moved out of the top level, where nobody looks for it.
- **Status bar.** Transient messages on the left (`Ready`, `Calculating…`, `Calculation
  completed`, `Composition normalized`, `Export completed`, `Copied N cells`), and a permanent
  right-hand summary of the state the next calculation will use, plus `FEM - ITEC - UFPA`.
- **Window and application icon** from the bundled set, so the taskbar and alt-tab show it.
- **Number display.** `_NumberSpin` drops trailing zeros for display only: `101325,000000`
  becomes `101325`, `298,150000` becomes `298,15`. Precision is untouched — a ppm value still
  round-trips exactly, and `100` is not mangled into `1`.

---

## PURE SPECIES

- **Species information panel** under the selector, built only from fields the record actually
  declares: formula, geometry, atom count, symmetry number, molar mass, vibrational mode count,
  hindered rotors, anharmonicity, Lennard-Jones vs Stockmayer, and the measured accuracy bands.
  A species with no transport parameters now says so here rather than failing later.
- **Three reporting bases.** The results table went from `Molar | Massic` to
  `Molar | Massic | Volumetric`, with `V_m` and `ρ` rows so the columns reconcile by hand:
  volumetric = massic × ρ = molar / V_m. Ratios that are not extensive (γ, T_v, T_p) stay blank.
- **State inputs are no longer over-determined.** P/V and n/m became exclusive radio pairs and
  the derived field is greyed out. Previously, ticking `V` sent both P and V, which
  `State.resolve` accepts without checking: PV/nRT reached 0.981 silently, with the entropy
  0.16 J/mol/K off (7.4 J/mol/K at V = 0.010 m³). PV = nRT now holds exactly in all four
  combinations.

---

## MIXTURES

- **Normalize** and **Clear** joined Add/Remove; the live `Σ` indicator was already present.
- **ppm components survive entry.** The fraction spin box went from four to six decimals: at
  four, `H2S: 0.000045` was rounded to zero on entry and the component never reached the
  mixture at all.
- **ppm components survive display.** `format_mole_fraction` was extracted to `mixture.py` and
  is now used by the CLI and the plots alike. Plot titles used two decimals, so a
  five-component biogas was labelled `CH4 0.56, CO2 0.44, O2 0.00, H2S 0.00, CO 0.00` — three
  components apparently absent, printed over a curve computed with all five. It now reads
  `CH4 0.5570, CO2 0.4390, O2 0.0040, H2S 4.50e-05, CO 9.00e-06`. Compositions that were
  already legible are untouched: dry air still shows `CO2 0.0004`.

---

## HUMID AIR

Unchanged in substance; it inherits the shell improvements — scrolling, validation, status
messages, the error dialog with *Details*, and the cleaner number display.

The **Air Transport** tab likewise stays dedicated to air: dry, humid and the dry-vs-humid
comparison. Its per-species table did gain `M`, `Sc_i` and `Le_i`, and its scalar line now
names the tracer (`Sc(H2O)`), but it computes nothing but air.

---

## TRANSPORT

- **Arbitrary mixtures are reachable.** A mixture now heads the **Transport** tab's fluid
  list, above the thirty pure species, so it inherits that tab's property multi-select, sweep
  controls and canvas rather than needing a parallel set of its own. It calls the same
  `MixtureTransportCalculator` the air path uses, and the output matches the CLI digit for
  digit: for the five-component biogas at 300 K and 1 atm, μ = 1.33454 × 10⁻⁵ Pa·s,
  k = 0.0228345 W/m·K, Pr = 0.74758, ρ = 1.15302 kg/m³.

  This was first built as a fluid selector on the *Air Transport* tab, which was the wrong
  place twice over: that tab's plotting is air-specific (`which` ∈ dry / humid / comparison),
  so the curves stayed air while the point calculation followed the mixture; and the tab is
  meant for air. **Air Transport is therefore dedicated to air alone**, and the mixture lives
  where the generic machinery already is.
- **Properties a mixture cannot report are disabled**, not silently wrong: `D_self` is the
  self-diffusion of a gas in itself and `mu_JT` has no mixture counterpart, so both grey out
  when the mixture is selected and `D_eff` takes over for diffusion.
- **The mixture sweep runs over temperature only.** `plot_mixture_property` walks T;
  offering a pressure sweep or a 2-D map would mean implementing physics in the GUI, which
  this layer does not do, so those modes say so and defer to a pure species.
- **Sc and Le name their species.** The per-species table gained `M`, `Sc_i` and `Le_i`
  (7 → 10 columns), and the scalar line reads `Sc(H2O)=0.711, Le(H2O)=0.994` instead of bare
  `Sc=`, `Le=`. A mixture with no natural tracer omits the scalars rather than inventing one.
- **One numerical correction.** `Sc_i` was first wired to the per-component field, which is the
  **pure-species** Schmidt number (ν_i / D_self,i). Beside a `D_i,mix` column it must be the
  **mixture-referenced** one, ν_mix / D_i,mix — which is also what the CLI prints. For CH₄ in
  the biogas these are 0.7616 and 0.7047: a 7.5% difference. The table now shows the latter,
  pinned by a test that recomputes ν_mix / D_i,mix independently.

---

## PLOTS

Plot titles no longer erase small components (above). Otherwise the existing plotting
infrastructure — titles, axis labels with units, grid, legend, the matplotlib navigation
toolbar with zoom/pan and *Save figure* — was already sound and was left alone.

---

## TABLES

A single `_install_table_behaviour()` pass replaces ten construction sites:

- every table gets an `objectName` taken from the attribute that holds it, so a test or a
  stylesheet can find it by the name the code uses;
- result tables are read-only; the composition editor stays editable;
- **numbers right-align automatically** — an `itemChanged` hook aligns any cell whose text
  parses as a float, so magnitudes line up by decimal place without touching the ten populate
  sites;
- **Ctrl+C copies the selection as TSV**, which pastes straight into a spreadsheet;
- tables with more than four columns size columns to their content and scroll horizontally.
  Stretch divided the viewport evenly and truncated every header once a table was wide
  (`Species` became `Specie:`).

---

## ABOUT DIALOG

`Help ▸ About StatThermoPy`. The bundled application mark leads, with the name and tagline
beside it, then the version, the author block, the description, the runtime environment
(Python, PySide6, matplotlib, platform — selectable, for bug reports) and the copyright.

The version is read from installed package metadata (`0.1.0` here) and **omitted entirely if
absent** — never invented. If the artwork is missing the layout falls back to typography alone.

---

## ICONS

The six supplied icons were copied into `src/statthermopy/gui/icons/` so they travel with the
installed wheel the way the YAML species database does, and **downscaled from 1254 px to
256 px: 8.1 MB → 471 KB**, with no visible loss at the sizes actually used (18 px on a tab,
88 px in About, up to 256 px in the taskbar). They were renamed for what they depict:

| File | Depicts | Used for |
|---|---|---|
| `app.png` | overlaid Maxwell–Boltzmann curves, ST monogram | application, window, About, Comparisons |
| `properties.png` | molecule over an energy-level curve | Properties |
| `plot.png` | f(v) distributions vs v with T | Plot |
| `transport.png` | hot/cold molecules crossing a barrier | Transport |
| `humid-air.png` | cloud, humidity, droplet | Humid Air |
| `mixture-transport.png` | two species interdiffusing | Mixture Transport |

*Validate* has no honest match in the set and takes the theme's own vector check glyph, which
recolours with the palette. The originals in `STATTHERMOPY/Icons/` were left in place.

---

## VALIDATION AND ERROR HANDLING

- **Before the core is called**, inputs are checked and reported as a field and a rule:
  *Temperature must be greater than 0 K.*, *Mole fractions must satisfy Σx_i > 0: give at least
  one species a fraction.*, *Negative mole fractions are not allowed.*
- **When the core does raise**, `_fail()` shows *The calculation could not be completed.* with
  the engine's own sentence as the reason and **the traceback behind *Show Details*** — where a
  developer will look for it and an ordinary user will not trip over it. All eight raw-exception
  dialogs were converted; `QMessageBox.critical` and `QMessageBox.warning` no longer appear in
  the file.
- **During a calculation** a busy cursor and a `Calculating…` status message are shown, with the
  window repainted first. The core runs synchronously in the callback; moving it to a worker
  thread is a larger change than this task allows and Qt widgets may only be touched from the
  GUI thread, so the wait is made legible rather than faked asynchronous.

---

## RESPONSIVENESS

| Window | Behaviour |
|---|---|
| 1920 × 1080 | every tab fits; no scrollbars |
| 1400 × 880 | wide tabs scroll horizontally |
| 1000 × 700 | Transport, Plot, Humid Air and Mixture Transport scroll; Properties and Validate fit |
| 800 × 600 | all tabs scroll; nothing is clipped or unreachable |
| 640 × 480 | still usable; the window accepts the size |

Scrollbars appear exactly when the content's minimum exceeds the viewport and not otherwise —
pinned by a test that compares `minimumSizeHint().width()` against `viewport().width()` for
every tab.

---

## TESTING

| Check | Result |
|---|---|
| GUI smoke: 7 tabs, 15 buttons, 30 species, 3 themes, 4 resizes | no exception from any path |
| GUI numbers vs core API (N2, CO2, H2O, C2H6, AR) | agree to 1 × 10⁻⁵ relative |
| Mixture transport GUI vs `MixtureTransportCalculator` | μ, k, Pr agree to 1 × 10⁻⁵ |
| `Sc_i` vs independently recomputed ν_mix / D_i,mix | agree to 5 × 10⁻⁵ |
| HTML export | self-contained, no script/link/http; numbers are the engine's |
| All 30 species reachable, compute and plot | yes |
| Full suite | see the commit message for the run that accompanied this change |

Numerical results are unchanged: the physics modules were not touched in this work, and the
tests above compare what the GUI displays against what the API returns.

---

## SCREENSHOTS

Regenerated in `docs/images/`:

| File | Shows |
|---|---|
| `gui_properties.png` | pure species, three bases, species panel, status bar |
| `gui_mixture.png` | composition editor, per-component contributions |
| `gui_plot.png` | property vs T |
| `gui_transport.png` | pure-gas transport |
| `gui_air_transport.png` | per-species table with Sc_i and Le_i |
| `gui_validate.png` | validation against references |
| `gui_about.png` | About dialog |
| `gui_dark.png` | dark theme |

---

## REMAINING IMPROVEMENTS

Not done here, in rough order of value:

1. **The missing check inside `State.resolve`.** Supplying P and V together is accepted without
   verifying PV = nRT, although the class docstring promises the check and the analogous m-vs-n
   check exists and raises. The GUI can no longer produce the inconsistent pair, but a caller
   using the API directly still gets a silently wrong answer.
2. **Split `mainwindow.py`.** It is still ~2200 lines. One module per tab, over a shared
   application layer, would make it far easier to work on.
3. **Background calculations.** A `QThread` worker with a progress bar and cancellation, for
   long sweeps and 2-D maps.
4. **Narrower tabs.** `Transport` still wants ~2000 px because its splitter panes sum their
   minimums. Reshaping those panes — not just scrolling them — would remove most horizontal
   scrolling. Capping combo and label minimums was tried and measured: it bought 2–6% on three
   tabs and cost 4% on Transport, so it was not kept.
5. **Save/open a configuration.** `File ▸ New / Open / Save configuration` is not offered
   because no such serialisation exists yet; adding it is straightforward but is new
   functionality, not presentation.
6. **Mass fraction column** (`Y_i`) in the per-species table, once the core exposes it — it was
   deliberately not computed in the GUI.
7. **Unit switching** (bar/atm, °C) — the core is SI throughout; this would need a conversion
   layer, kept out of the GUI.
