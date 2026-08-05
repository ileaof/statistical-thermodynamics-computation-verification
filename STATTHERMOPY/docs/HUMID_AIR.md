# Statistical Humid Air — theory, hybrid architecture and critical analysis

This module computes the **maximum solubility of water vapour in air** as a function of absolute
temperature and total pressure — i.e. the saturation limit before condensation — and the full
psychrometric/thermodynamic property set, with the gas phase treated rigorously by statistical
thermodynamics.

## 1. Physical model

The saturation (dew) line is the locus of vapour–liquid equilibrium of water,

$$\mu_v(T,P) = \mu_l(T,P) \quad\Longleftrightarrow\quad g_v(T,P) = g_l(T,P).$$

**Vapour phase — first principles.** The molar Gibbs energy of the water vapour is obtained
*exclusively* from the molecular partition function

$$q = q_\text{trans}\,q_\text{rot}\,q_\text{vib}\,q_\text{elec},$$

through the same engine used throughout StatThermoPy (translational Sackur–Tetrode, rigid-rotor
rotation, quantum harmonic vibration, electronic Boltzmann sum). Its absolute entropy reproduces
the experimental standard entropy of steam, S°(298.15 K) ≈ 188.8 J·mol⁻¹·K⁻¹, to better than
0.1 %, so `g_v(T,P)` carries the correct temperature and pressure dependence **with no fitted
constant**. The per-factor contribution of each partition-function term to every property is
reported, exposing how the translational and rotational motions dominate the vapour Gibbs energy
(and hence the saturation), while water's high-frequency vibrations are nearly frozen below
~400 K and the electronic factor is inert.

**Liquid phase — reference model + two physical anchors.** The liquid Gibbs energy comes from a
pluggable `LiquidWaterModel`. Because the two phases use different energy-reference conventions,
the liquid scale is reconciled to the vapour scale with **two physical anchors at the triple
point** (273.16 K, 611.657 Pa):

1. coexistence there, `g_v(T_t,P_t) = g_l(T_t,P_t)`;
2. the enthalpy of vaporisation `Δh_vap(T_t) = 45.05 kJ·mol⁻¹` (a single calorimetric constant),

which fix the liquid enthalpy- and entropy-reference offsets (Δh₀, Δs₀). **No empirical
vapour-pressure correlation (Antoine, Magnus, Tetens, …) is used**: the temperature dependence of
`P_sat` is *predicted* by the statistical vapour together with the liquid heat capacity.

For an ideal vapour `g_v(T,P) = g_v(T,P_ref) + RT ln(P/P_ref)` and a nearly incompressible liquid
`g_l(T,P) = g_l(T,P_ref) + v_l (P − P_ref)` (Poynting), giving at each temperature

$$P_\text{sat} = P_\text{ref}\,\exp\!\Big[\frac{g_l(T,P) - g_v(T,P_\text{ref})}{RT}\Big].$$

### Validation (no vapour-pressure correlation used)

| T (K) | P_sat computed (Pa) | P_sat reference (Pa) | error |
|------:|--------------------:|---------------------:|------:|
| 273.16 | 611.66 | 611.66 | anchor |
| 293.15 | 2336.9 | 2339.0 | −0.09 % |
| 298.15 | 3165.3 | 3169.9 | −0.15 % |
| 323.15 | 12301  | 12344  | −0.35 % |
| 373.15 | 99828  | 101325 | −1.5 % |

Δh_vap(298 K) comes out at 44.0 kJ·mol⁻¹ (lit. 43.99); the saturation humidity ratio at
25 °C/1 atm is 20.06 g·kg⁻¹ (lit. ≈20.0) and the wet-bulb temperature for 25 °C/50 % RH is
17.9 °C (lit. ≈17.9 °C).

## 2. Critical analysis — why *direct* statistical mechanics fails for the liquid

The vapour is (near-)ideal: molecules are effectively independent, so the *N*-body partition
function factorises, `Q_N = q^N / N!`, and `q` separates into translational × rotational ×
vibrational × electronic factors. **None of this survives in the liquid.**

- **The configurational integral does not factorise.** With intermolecular potential
  `U(r₁…r_N)`, `Q_N = (q_int^N / N! Λ^{3N}) Z_N`, where `Z_N = ∫ e^{−βU} dr₁…dr_N` couples all
  molecules. `Z_N` is *the* problem: it is a 3*N*-dimensional integral with no closed form.
- **Hydrogen bonding and strong association.** Liquid water is a transient, tetrahedral
  hydrogen-bond network. The pair potential is highly directional and many-body (polarisation,
  cooperativity), so even a good pair potential is only approximate.
- **Rotational/translational coupling.** In the dense phase, "rotation" becomes hindered libration
  and "translation" becomes cage rattling + diffusion; the clean mode separation of the ideal gas
  is gone.
- **Reference-scale mismatch.** Even with a liquid model, its absolute enthalpy/entropy zero
  differs from the spectroscopic vapour zero — hence the triple-point + Δh_vap anchoring above.

Directly building `q_liquid` from single-molecule spectroscopy (as for the gas) is therefore
**not physically valid** for water. A rigorous route must treat the many-body `Z_N`.

## 3. Recommended statistical models for the liquid (advantages, limits, validity)

| Approach | Idea | Advantages | Limitations | Validity for water |
|---|---|---|---|---|
| **SAFT** (SAFT-VR, PC-SAFT, SAFT-γ Mie) | Perturbation on a chain-of-segments reference with an explicit **association** term (Wertheim TPT1) for hydrogen bonds | Analytic EOS; association term is physically matched to H-bonding; excellent VLE and `P_sat` for water and mixtures; extends to humid air + electrolytes | A handful of molecular parameters fitted to data (segment size/energy, association volume/energy); TPT1 ignores ring/cooperative bonding | **Recommended primary upgrade.** Good over the whole liquid range and into supercritical |
| **Thermodynamic perturbation theory** (Barker–Henderson, WCA) | Expand `A` about a hard-sphere (or soft) reference in the attractive perturbation | Rigorous, systematically improvable; foundation under SAFT | Needs an accurate reference + pair correlation `g(r)`; slow/again-approximate for strong, directional H-bonds | Good for simple/LJ fluids; for water only via an association extension (→ SAFT) |
| **Integral equations** (Ornstein–Zernike + PY / HNC / RISM, RISM-KH) | Solve OZ for `g(r)` with a closure; molecular RISM for site–site structure | Delivers structure *and* thermodynamics from the pair potential; no VLE fitting | Closure-dependent accuracy; thermodynamic inconsistency; convergence hard near coexistence and for strong association | Structure good; quantitative `P_sat` sensitive to closure — a research-grade route |
| **Lattice / association models** (Ising-like, two-state, cell/hole theories) | Discretise positions/bond states; count H-bond configurations combinatorially | Very transparent; captures the entropy of the H-bond network; cheap | Coarse; lattice artefacts; semi-quantitative | Good pedagogy and qualitative trends; not reference-accurate |
| **Molecular simulation** (MC/MD with TIP4P/2005, MB-pol, ML potentials) | Sample `Z_N` numerically | Most rigorous given the potential; MB-pol/ML reach near-experimental accuracy | Cost; force-field/potential dependence; free energies need special methods (Gibbs-ensemble, thermodynamic integration) | **Gold standard** reference; too heavy for an interactive property call |
| **IAPWS-95 / IF97** (used here) | Multiparameter empirical Helmholtz EOS fit to all data | The international reference; machine-precision liquid `c_p, s, v` | Empirical (not first-principles); it *is* a correlation for the liquid | The pragmatic hybrid reference — highest accuracy for the liquid leg |

## 4. Hybrid architecture and how to upgrade the liquid

The code is deliberately split so the liquid model is a single pluggable object:

```
HumidAir ── SaturationCalculator ── LiquidWaterModel (ABC)
   │              │                     ├── ConstantCpLiquid   (transparent, dependency-free)
   │              │                     └── IAPWSLiquid        (IAPWS-95 reference, optional)
   │              └── vapour: Thermodynamics(H2O)  ← pure statistical mechanics
   └── dry air: IdealGasMixture (N2/O2/Ar/CO2; extensible to trace gases / other backgrounds)
```

To introduce a fuller statistical liquid, implement `LiquidWaterModel.{enthalpy, entropy,
molar_volume}` for a SAFT/perturbation model and pass it to `SaturationCalculator`. Nothing else
changes: the vapour stays first-principles and the equilibrium is still `g_v = g_l`. A SAFT liquid
would also remove the constant-`c_p` approximation and extend validity toward the critical point.

## 5. Hypotheses, validity and accuracy of the current implementation

- **Ideal-gas vapour & ideal mixing** of dry air + vapour. Valid to ≲ a few bar; the
  air–water *enhancement factor* (≈ +0.5 % on `P_sat` at 1 atm from air's presence and vapour
  non-ideality) is neglected. Impact: sub-percent on humidity ratio near ambient.
- **Liquid reference.** `ConstantCpLiquid` assumes constant `c_p,l`, `v_l` (incompressible):
  ≈0.1 % on `P_sat` near 25 °C, ≈1.5 % at 100 °C. `IAPWSLiquid` removes this within the liquid's
  reference accuracy.
- **Two calorimetric/defining anchors** (triple point + `Δh_vap(T_t)`). These are fundamental
  constants, not a fitted vapour-pressure curve.
- **Range.** Liquid–vapour only, `T ∈ [T_triple, T_critical)`. Below the triple point the
  equilibrium is with ice (frost point) — not covered; the dew-point search clamps to the triple
  point. Harmonic-oscillator vibrations slightly overestimate the vapour `c_p` above ~1500 K.

### Path to a fully first-principles humid-air model

1. Replace `IAPWSLiquid` with a **SAFT-γ Mie / PC-SAFT** liquid (molecular parameters), keeping the
   `g_v = g_l` framework — removes the empirical liquid leg.
2. Add the **enhancement factor** from a real-gas mixture model (virial/SAFT vapour) for `P > 1`
   bar.
3. For research accuracy, compute the liquid free energy by **molecular simulation** (MB-pol / ML
   potentials with thermodynamic integration) and tabulate it as a `LiquidWaterModel`.
4. Generalise the dry background to trace gases, other gas mixtures, and real-gas equations of
   state for high-pressure / planetary applications.

## 6. Comparative graphical analysis

Two comparative analyses (module `humidair.analysis.PsychrometricAnalysis`, plotted by
`humidair.plots`, exposed in the GUI's **Thermodynamic Comparisons** section):

**Water-vapour content vs temperature.** The *actual* humidity ratio (fixed water content set by
the humidity spec) and the *saturation* humidity ratio `w_s = ε P_sat/(P − P_sat)` (g/kg dry air),
on one graph. The actual content follows the saturation curve below the dew point (condensation
removes water) and plateaus above it; the dew point marks the onset of condensation.

**Dry air vs humid air, isobaric vs isochoric.** A thermodynamic property (H, U, S, G, A, Cp, Cv,
T_v, T_p) of dry and humid air versus temperature under two constraints — **isobaric** (constant P)
and **isochoric** (constant molar volume `v_ref = R T_ref/P`, so `P(T) = P·T/T_ref`) — giving up to
four curves. Because an ideal gas has `U, H, Cp, Cv, T_v, T_p` depending on temperature *only*,
their isobaric and isochoric curves coincide; only the pressure-dependent `S, G, A` differ between
the two constraints (the code flags the temperature-only case). The composition is held fixed in
this comparison (condensation is shown separately by the water-vapour-content graph).

Both analyses return a `ComparisonTable` (numerical data → CSV/Excel) and a matplotlib figure with
an interactive click-to-toggle legend, hover tooltips, zoom/pan (toolbar) and high-resolution
PNG/SVG/PDF export.
