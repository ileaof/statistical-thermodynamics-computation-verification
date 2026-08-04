# StatThermoPy — Theoretical Foundations

This document derives, from first principles of Statistical Mechanics, every thermodynamic
property computed by StatThermoPy: from the molecular partition function to U, H, S, A, G, Cv,
Cp, γ and μ, for monoatomic, diatomic, linear- and nonlinear-polyatomic ideal gases and their
mixtures. For each result the **physical origin**, **hypotheses**, **limitations** and
**validity range** are stated, with references.

The companion `.docx` technical book (deferred to a dedicated phase) will expand this material
with worked numerical examples and validation tables.

---

## 1. The canonical ensemble and the molecular partition function

### 1.1 Origin

For a system of N identical, non-interacting molecules in contact with a heat bath at
temperature T and enclosed in volume V (a closed system: fixed N, V, T), the canonical
partition function is

$$Z_N = \sum_\alpha e^{-\beta E_\alpha}, \qquad \beta = \frac{1}{k_B T},$$

where the sum runs over all quantum states α of the *entire* N-particle system and E_α are the
corresponding total energies. For an **ideal gas** the molecules are independent, so the total
energy separates: E = Σ_i ε_i. Because the molecules are **indistinguishable** (quantum
identical particles), the sum over states over-counts by N!, giving

$$Z_N = \frac{Q^N}{N!}, \qquad Q = \sum_j g_j\, e^{-\beta \varepsilon_j}.$$

Q is the **molecular partition function**: a sum over the energy levels ε_j of a *single*
molecule, weighted by their degeneracies g_j. The factorisation Q = Q_t Q_r Q_v Q_e follows
because the single-molecule Hamiltonian separates (to an excellent approximation) into
translation, rotation, vibration and electronic contributions whose energies add and whose
partition functions therefore multiply.

### 1.2 Thermodynamic connection (bridge equations)

All thermodynamic quantities follow from ln Z_N:

$$U = k_B T^2 \left(\frac{\partial \ln Z_N}{\partial T}\right)_{V,N}, \qquad
  S = k_B\!\left[\ln Z_N + T\!\left(\frac{\partial \ln Z_N}{\partial T}\right)_{V,N}\right],$$

$$A = -k_B T \ln Z_N, \qquad P = k_B T\!\left(\frac{\partial \ln Z_N}{\partial V}\right)_{T,N},$$

and the derived potentials H = U + PV, G = A + PV, μ = (∂G/∂N)_T,P. Because ln N! is
temperature-independent, the N! correction contributes to S, A, G, μ but **not** to U or Cv.

### 1.3 Molar form

Setting N = n N_A (n moles) and using k_B N_A = R, Stirling's approximation ln N! ≈ N ln N − N
turns the bridge equations into the **molar** relations used throughout the code:

| Quantity | Molar formula |
|---|---|
| U_m  | R T² (∂ ln Q / ∂T)_V |
| S_m  | R [ ln Q − ln N_A + 1 + T (∂ ln Q / ∂T)_V ] |
| A_m  | −R T [ ln Q − ln N_A + 1 ] |
| H_m  | U_m + R T |
| G_m  | A_m + R T = −R T [ ln Q − ln N_A ] |
| μ_m  | G_m |
| Cv_m | (∂U_m/∂T)_V |
| Cp_m | Cv_m + R |
| γ    | Cp_m / Cv_m |

The translation mode carries the volume dependence (ln V) and therefore hosts the
indistinguishability correction (−ln N_A + 1) and the P V work term (R T per mole); the internal
modes (rotation, vibration, electronic) are volume-independent.

### 1.4 Hypotheses, limitations, validity

- **Ideal gas**: no intermolecular interactions. Valid at low pressures / high temperatures
  away from the saturation line and critical point.
- **Separability** of translation/rotation/vibration/electronic (Born–Oppenheimer + rigid
  rotor + harmonic oscillator). Breaks down for floppy molecules, near dissociation, or when
  rovibrational coupling (centrifugal distortion, anharmonicity, Coriolis) is significant.
- **Distinguishability correction** uses Stirling's approximation, accurate for N_A ~ 10²³.
- **Maxwell–Boltzmann statistics**: non-degenerate limit. Valid when Q/N = e^{ln Q − ln N} ≫ 1,
  i.e. the thermal de Broglie wavelength is much smaller than the intermolecular spacing — the
  classical (non-Bose/Fermi) regime. Fails for He at very low T or very high density.

---

## 2. Translational contribution

### 2.1 Derivation

A free particle of mass m in a 3-D box of volume V has, in the continuum (high-quantum-number)
limit, the partition function obtained by integrating the Boltzmann factor over momentum:

$$Q_t = \frac{1}{h^3}\int\!\!\int e^{-\beta p^2/2m}\,d^3p\,d^3r
      = \left(\frac{2\pi m k_B T}{h^2}\right)^{3/2}\! V .$$

Hence ln Q_t = (3/2) ln(2π m k_B T / h²) + ln V, and (∂ ln Q_t / ∂T)_V = 3/(2T). The molar
quantities (with the N! correction absorbed here):

- U_m,t = (3/2) R T,
- Cv_m,t = (3/2) R,
- S_m,t = R [ ln Q_t − ln N_A + 1 + 3/2 ] = R [ (3/2) ln(2π m k_B T / h²) + ln(k_B T/P) + 5/2 ],

the **Sackur–Tetrode equation** (the molar volume V_m = RT/P gives V_m/N_A = k_B T/P).
- A_m,t = −R T [ ln Q_t − ln N_A + 1 ],
- G_m,t = −R T [ ln Q_t − ln N_A ].

### 2.2 Notes

This is the only contribution that depends on P (through V_m) and on density; it is the locus
of the entropy of mixing and the chemical potential's pressure dependence (μ = μ° + RT ln P).
Validity: classical, non-relativistic, ideal-gas regime. The Sackur–Tetrode formula fails when
Q/N is not large (quantum degeneracy).

---

## 3. Rotational contribution

### 3.1 Monoatomic

A point atom has no rotational degrees of freedom: Q_r = 1, all rotational contributions zero.

### 3.2 Linear rigid rotor (diatomics and linear polyatomics)

Energy levels ε_J = B J(J+1) with degeneracy (2J+1), where B = ℏ²/(2I) is the rotational
constant (I = moment of inertia about axes perpendicular to the bond). The exact sum is

$$Q_r = \sum_{J=0}^{\infty} (2J+1)\, e^{-J(J+1)\theta_r/T}, \qquad
  \theta_r = \frac{h^2}{8\pi^2 I k_B}.$$

In the **high-temperature limit** T ≫ θ_r the sum is replaced by an integral (Euler–Maclaurin),
giving the classical closed form

$$Q_r = \frac{T}{\sigma\,\theta_r},$$

where σ is the **symmetry number** (the number of indistinguishable orientations produced by
rotation: 2 for homonuclear diatomics, 1 for heteronuclear, higher for symmetric polyatomics).
From it: U_m,r = R T, Cv_m,r = R, S_m,r = R [ ln Q_r + 1 ], A_m,r = −R T ln Q_r.

StatThermoPy implements **both**: the classical closed form (default) and the exact quantum sum
(`use_quantum_rotation=True`), the latter required at low T (T ≲ θ_r) where rotation "freezes"
and Cv_m,r drops below R — observable for H2 (θ_r = 85.3 K).

### 3.3 Nonlinear rigid rotor (asymmetric top)

A general molecule has three principal moments I_A ≤ I_B ≤ I_C. In the high-T limit the
partition function (derivable by integrating over the three Euler angles and their conjugate
momenta) is

$$Q_r = \frac{\sqrt{\pi}}{\sigma}\sqrt{\frac{T^3}{\theta_A \theta_B \theta_C}},
      \qquad \theta_i = \frac{h^2}{8\pi^2 I_i k_B}.$$

Then ln Q_r = (3/2) ln T + const − ln σ, (∂ ln Q_r / ∂T) = 3/(2T), giving U_m,r = (3/2) R T,
Cv_m,r = (3/2) R, S_m,r = R [ ln Q_r + 3/2 ], A_m,r = −R T ln Q_r.

### 3.4 Limitations

Rigid-rotor approximation: neglects centrifugal distortion. High-T classical formula requires
T ≫ θ_r for every axis; for light rotors at low T use the quantum sum (linear) — the nonlinear
quantum sum (asymmetric-top eigenvalues) is not implemented in Phase 1.

---

## 4. Vibrational contribution (quantum harmonic oscillator)

Each normal mode i is an independent harmonic oscillator of frequency ν_i (wavenumber ṽ_i);
with the energy zero at v = 0,

$$Q_{v,i} = \frac{1}{1 - e^{-\theta_{v,i}/T}}, \qquad \theta_{v,i} = \frac{h c \tilde\nu_i}{k_B}.$$

For a mode of degeneracy g_i the contribution is multiplied by g_i, and ln Q_v = − Σ_i g_i
ln(1 − e^{−θ_{v,i}/T}). The closed forms (each a textbook result obtained by differentiating ln
Q_v):

- U_m,v  = Σ_i g_i R θ_{v,i} / (e^{θ_{v,i}/T} − 1),
- Cv_m,v = Σ_i g_i R (θ_{v,i}/T)² e^{θ_{v,i}/T} / (e^{θ_{v,i}/T} − 1)²,
- S_m,v  = Σ_i g_i R [ (θ_{v,i}/T)/(e^{θ_{v,i}/T} − 1) − ln(1 − e^{−θ_{v,i}/T}) ],
- A_m,v  = Σ_i g_i R T ln(1 − e^{−θ_{v,i}/T}).

Limits: T ≪ θ_v → mode frozen (U → 0, Cv → 0); T ≫ θ_v → equipartition (U → R T, Cv → R per
oscillator). Degenerate modes (e.g. CO2 bend g=2, CH4 ν3 g=3) are counted through g_i.

**Limitations**: harmonic approximation — neglects anharmonicity, mode coupling, and
zero-point-energy shifts. Adequate up to a few thousand K for most species; above that the
heat capacity is overestimated slightly. Low-frequency torsions about single bonds are better
described as *hindered internal rotors* (§4b) than as harmonic oscillators.

---

## 4b. Hindered internal rotation

A torsion about a single bond (e.g. a methyl top in ethane or propane) is not well approximated
by a harmonic oscillator: as temperature rises the motion crosses over from libration in a well
to nearly free rotation over the barrier, and its heat capacity rolls **down** from R (harmonic)
toward R/2 (free rotor) rather than saturating at R. StatThermoPy treats each such mode with a
one-dimensional hindered-rotor model. For an n-fold symmetric potential

$$V(\varphi) = \tfrac{1}{2} V_n\,[1 - \cos(n\varphi)],$$

the torsional Hamiltonian is the Mathieu operator

$$\hat H = -F\,\frac{d^2}{d\varphi^2} + \tfrac{1}{2}V_n[1-\cos(n\varphi)], \qquad
F = \frac{\hbar^2}{2 I_r},$$

with reduced moment of inertia I_r and internal-rotation constant F. Its eigenvalues are obtained
**exactly** (to basis truncation) by diagonalising Ĥ in the free-rotor basis |m⟩ = e^{imφ}/√(2π),
where ⟨m|Ĥ|m⟩ = F m² + V_n/2 and ⟨m|Ĥ|m±n⟩ = −V_n/4. From the resulting levels ε_i (referenced to
the ground level, matching the v = 0 vibrational convention),

$$q_\text{ir} = \frac{1}{\sigma_\text{int}}\sum_i e^{-\varepsilon_i / k_B T},$$

with σ_int the internal symmetry number (3 for a methyl top). The thermodynamic functions follow
from the level-distribution moments (x_i = ε_i/k_B T):

- U_m = R T ⟨x⟩,
- Cv_m = R (⟨x²⟩ − ⟨x⟩²),
- S_m = R (ln q_ir + ⟨x⟩),
- A_m = −R T ln q_ir,

summed over every rotor. The model reproduces both limits automatically: as V_n → 0 it becomes
the free internal rotor q = (8π³ I_r k_B T)^{1/2}/(σ_int h) with Cv → R/2, and as V_n → ∞ it
becomes a harmonic oscillator of the small-oscillation frequency ṽ = n√(F V_n) with Cv → R. Only
the two spectroscopic constants F and V_n enter — no empirical property correlation — so the
engine remains pure statistical mechanics. Each rotor replaces one 3N−6 (or 3N−5) harmonic
oscillator; in the reported four-factor partition function it is folded into Q_v (the internal
nuclear-motion factor).

Applied to the C–C torsion of **ethane** (V₃ = 1024 cm⁻¹, F = 10.7 cm⁻¹, σ_int = 3) the Mathieu
ladder reproduces the observed ~289 cm⁻¹ torsional fundamental; the two methyl torsions of
**propane** (V₃ = 1190 cm⁻¹, F = 5.3 cm⁻¹, σ_int = 3, treated as two independent identical rotors)
are handled the same way. Relative to the harmonic-torsion treatment this leaves propane's C_p
essentially exact across 298–2000 K (mean error ≈ 0.3 %) and keeps ethane within a few percent.

**Limitations**: one-dimensional and uncoupled — top–top and top–frame coupling, and the change
of I_r with the overall rotation, are neglected. The potential is a single cosine term (only the
leading n-fold barrier V_n); higher harmonics are not represented.

---

## 5. Electronic contribution

The electronic partition function sums over the electronic terms of the molecule, with the
ground state as the energy zero:

$$Q_e = \sum_j g_j\, e^{-\theta_{e,j}/T}, \qquad \theta_{e,j} = \frac{h c \tilde T_j}{k_B},$$

where T̃_j (cm⁻¹) is the term energy relative to the ground state and g_j the electronic
degeneracy. Defining the Boltzmann population p_j = g_j e^{−θ_{e,j}/T} / Q_e and the thermal
averages ⟨θ_e⟩ = Σ p_j θ_{e,j}, ⟨θ_e²⟩ = Σ p_j θ_{e,j}²:

- U_m,e  = R ⟨θ_e⟩,
- Cv_m,e = R (⟨θ_e²⟩ − ⟨θ_e⟩²) / T²   (energy fluctuation / heat capacity theorem),
- S_m,e  = R [ ln Q_e + ⟨θ_e⟩ / T ],
- A_m,e  = −R T ln Q_e.

For most closed-shell species (N2, CO2, CH4, noble gases) only the ground state is populated and
Q_e = g_0, contributing R ln g_0 to S and nothing to U or Cv. Open-shell species matter:
**O2** has a ³Σ_g⁻ ground state (g_0 = 3) and low-lying excited terms; **NO** has a ²Π ground
doublet with the ²Π_{3/2} component at 121 cm⁻¹, giving a non-negligible electronic heat
capacity at room temperature.

**Limitations**: only the explicitly listed terms are included; very high-lying terms are
negligible below a few thousand K. Ionisation and dissociation are outside scope.

---

## 6. Assembly of properties (Thermodynamics)

The four `Contribution` objects are summed; the P V term and the N! correction are applied once
(translation hosts them). The aggregator computes, in molar and massic (per kg) bases:

- U_m = Σ U_m,mode ;  H_m = U_m + R T ;
- S_m = Σ S_m,mode ;  A_m = Σ A_m,mode ;  G_m = A_m + R T ;  μ_m = G_m ;
- Cv_m = Σ Cv_m,mode ; Cp_m = Cv_m + R ;  γ = Cp_m/Cv_m ;
- massic = molar / M ;  R_specific = R / M.

The extensive totals use the amount of substance n from the resolved state: X = n X_m.

---

## 7. Ideal-gas mixtures

Each component occupies the full volume at its partial pressure P_i = x_i P. There is no
enthalpy of mixing; the entropy of mixing is the classical

$$\Delta S_{\text{mix}} = -R \sum_i x_i \ln x_i .$$

With mole fractions x_i (converted from mass fractions via x_i ∝ w_i/M_i):

- M̅ = Σ x_i M_i,  R_specific = R/M̅ ;
- U_m, H_m, Cv_m, Cp_m = Σ x_i (·)_i(T)  (pressure-independent for ideal gases) ;
- S_m = Σ x_i S_m,i(T, P_i) = Σ x_i [S_m,i(T,P) − R ln x_i] ;
- G_m = Σ x_i μ_i(T, P_i) = Σ x_i [G_m,i(T,P) + R T ln x_i] ;
- A_m = U_m − T S_m = G_m − R T ;  γ = Cp_m/Cv_m.

---

## 8. Chemical equilibrium (architecture only — Phase 1 placeholder)

Because μ_i(T,P) is already available from the partition function, future phases can implement:
(a) **Gibbs-energy minimisation** subject to element conservation (Lagrange multipliers / RAND
algorithm); (b) **reaction equilibrium** with ΔG°(T) = Σ ν_i G_m,i and K(T) = exp(−ΔG°/RT);
(c) **equilibrium composition** of reacting mixtures — all without empirical correlations. The
`statthermopy.equilibrium` module exposes the intended interfaces (presently raising
`EquilibriumNotImplemented`).

## 8b. Validation against embedded NIST/JANAF reference data

StatThermoPy ships **curated reference tables** of molar Cp° and absolute molar S° (J/mol/K,
standard state 1 bar) for all 22 species in the molecular database — the monoatomic and diatomic
gases (Ar, He, Ne, Kr, Xe, H2, N2, O2, Cl2, NO, CO), the triatomics (H2O, CO2, N2O, SO2, H2S),
and the larger polyatomics (NH3, CH4, C2H2, C2H4, C2H6, C3H8) — used solely by the optional
validation layer (`statthermopy.validation.validate`) for cross-checking the first-principles
engine.

**What is and is not shipped.** Only the *reference values* (numbers at a T grid) are embedded.
No empirical correlation coefficients (NASA/Shomate/JANAF polynomials) live in the package, so
the calculation core (§1–7) remains pure statistical mechanics. The tabulated values were
produced by evaluating the NIST Chemistry WebBook Shomate equations for each species at the grid
temperatures (the Shomate coefficients themselves are not shipped). The two species for which
NIST WebBook publishes no Shomate fit (C2H6 ethane, C3H8 propane) use the NASA Glenn
7-coefficient polynomials (McBride, Zehe & Gordon, NASA/TP-2002-211556) instead, evaluated at the
same grid; again only the values ship, not the coefficients. Each YAML file under
`statthermopy.validation.data` cites its source and is refinable.

**Why Cp and S compare directly.** Cp is pressure-independent for an ideal gas, and S° is an
absolute (third-law) quantity, so both match the engine's molar output without a reference-state
offset. (H is *not* shipped in Phase 2 because the engine's H_m = U_m + RT is an absolute
enthalpy while NIST tabulates H° − H°(298.15); a reference-state convention would be required.)
Entropy is validated at the reference's declared standard-state pressure (1 bar), since the
translational entropy carries the pressure dependence (ln(kT/P)).

**Tolerance and limits.** The rigid-rotor / harmonic-oscillator model departs from experiment by
up to a few percent at high temperature — neglected anharmonicity makes the engine
*underestimate* Cp at 2000 K (≈4% for H2, the worst case in the set). The validation tolerance is
5% mean absolute error. Open-shell species (O2 ³Σ_g⁻, NO ²Π) include their low-lying electronic
terms, which the engine reproduces. Phase 2 covers the core subset above; extending to all 22
species is deferred (some, e.g. C2H6/C3H8/SO2/H2S, carry approximate, refinable spectroscopic
constants).

---

## 9. Backend abstraction (performance)

All array work (exponentials, logs, sums over levels) is routed through a thin `Backend` ABC.
The reference `NumpyBackend` is always available. Three accelerated backends are **functional** and
can be selected at runtime with `set_backend(...)` **without altering the public API**:

* `"numba"` — Numba `@njit(cache=True)` CPU kernels. The two hot loops are compiled: the exact
  quantum linear-rotor J-sum (`linear_quantum_moments`) and a temperature-batched molar-property
  kernel (`molar_property_grid`) that returns the eight core molar arrays
  `(U_m, S_m, A_m, Cv_m, ln_Qt, ln_Qr, ln_Qv, ln_Qe)` over a temperature grid.
* `"openmp"` — `OpenMPBackend` extends `NumbaBackend` with `@njit(parallel=True)` + `numba.prange`
  over the temperature grid (Numba's OpenMP-style threaded loops). It reuses the same per-temperature
  device function, so there is no physics duplication.
* `"cuda"` — `numba.cuda` GPU kernel (one thread per temperature), with **automatic CPU fallback**:
  when `numba.cuda.is_available()` is false, construction emits a `RuntimeWarning` and transparently
  delegates to the Numba CPU backend, so `set_backend("cuda")` never raises on a GPU-less machine.

The per-temperature physics inside the kernels **mirrors `modes/translational`,
`modes/rotational`, `modes/vibrational`, `modes/electronic` exactly** — the acceleration changes only
the numerical execution, never the physics or the data. No empirical property correlation
(NASA/Shomate/JANAF/CoolProp/REFPROP) is introduced; the calculation core remains pure Statistical
Mechanics. The accelerated backends are imported lazily, so `import statthermopy` never pulls in
numba. `list_backends()` returns the declared architecture (`["numpy","numba","openmp","cuda"]`);
`available_backends()` returns those whose optional dependencies are importable in the current
environment (`pip install statthermopy[accel]`).

---

## 9.5 Statistical transport properties (Chapman–Enskog + Lennard–Jones)

Everything above is **pure Statistical Mechanics of the ideal gas**. The transport coefficients —
viscosity, thermal conductivity, diffusion — are *kinetic* properties: they depend on the
collision dynamics, not on the equilibrium partition function alone. They are derived here from
the **Chapman–Enskog** first-order solution of the Boltzmann equation for a dilute gas of
molecules interacting through the **Lennard–Jones 12-6** pair potential, keeping the engine
first-principles: the only molecular inputs are the LJ parameters `σ` (collision diameter) and
`ε` (well depth), stored per species as `LennardJones` on the same footing as the spectroscopic
constants, and the heat capacities/`γ` taken **directly** from the partition-function engine.

Primary transport coefficients (SI):

$$\mu = \frac{5}{16}\frac{\sqrt{m k_B T / \pi}}{\sigma^2\,\Omega^{(2,2)*}(T^*)}, \qquad
k = \mu\,c_v\,\frac{9\gamma - 5}{4}\quad\text{(Eucken)}, \qquad
D_{ij} = \frac{3}{16}\frac{k_B T}{P}\,\frac{1}{\sigma_{ij}^2\,\Omega^{(1,1)*}(T^*_{ij})}
\sqrt{\frac{2 k_B T}{\pi m_{ij}}}$$

with `T* = k_B T/ε`, the Lorentz–Berthelot combining rules
`σ_ij = (σ_i+σ_j)/2`, `ε_ij = √(ε_i ε_j)` and reduced mass `m_ij = m_i m_j/(m_i+m_j)`. The
self-diffusion `D_ii` is the `i = j` case (`m_ii = m/2`). The collision integrals `Ω^(l,s)*` use
the **Neufeld et al. (1972)** fits, valid for `0.3 ≤ T* ≤ 100` with a graceful low-`T*` branch so
curves stay continuous down to 0 K.

The Eucken factor `(9γ−5)/4` equals the Chapman–Enskog monatomic multiplier `5/2` on `c_v` for
`γ = 5/3`, recovering the exact CE result `k = (5/2) c_v μ = (15/4)(k_B/m) μ`. Because `Cv_m`,
`Cp_m` and `γ` come from `Thermodynamics`, the transport coefficients inherit the full
statistical-mechanics temperature dependence — the quantum vibrational/electronic excitation and,
when enabled, the quantum rotor propagate into viscosity and conductivity with thermodynamic
consistency.

Derived thermophysical coefficients follow from the ideal-gas EOS (exact for this engine):

$$\rho = \frac{PM}{RT},\quad \nu = \frac{\mu}{\rho},\quad \alpha = \frac{k}{\rho c_p},\quad
Pr = \frac{4\gamma}{9\gamma - 5},\quad Sc = \frac{5}{6}\frac{\Omega^{(1,1)*}}{\Omega^{(2,2)*}},\quad
Le = \frac{Sc}{Pr}$$

$$Z = 1,\quad a = \sqrt{\gamma R_\text{sp} T},\quad \beta = \frac{1}{T},\quad
\kappa_T = \frac{1}{P},\quad \mu_{JT} = 0$$

Pressure enters through density (`ν`, `α`, `D ∝ 1/P`) and `κ_T`; `μ` and `k` are
pressure-independent in the dilute limit — the physically correct ideal-gas behaviour. The
dimensionless groups use closed forms so they stay finite at `T = 0` (where `μ, k, D → 0`).

**Scope.** This is the dilute/ideal-gas (Chapman–Enskog first-approximation) regime, valid from
0 K up to the highest supported temperature at any (low-to-moderate) pressure. Polar species
(H₂O, NH₃, H₂S, SO₂) use the LJ approximation — larger uncertainty, noted per species. The
architecture is open to dense-gas (Enskog / corresponding-states), mixture diffusion, plasma
transport and combustion/CFD coupling as future extensions behind the same
`TransportCalculator` / `TransportProperties` interface.

---

## 10. References

1. McQuarrie, D. A. *Statistical Mechanics* (University Science Books).
2. Hill, T. L. *An Introduction to Statistical Thermodynamics* (Dover).
3. Herzberg, G. *Molecular Spectra and Molecular Structure*, vols. I–III (Van Nostrand).
4. Chase, M. W. et al. *NIST-JANAF Thermochemical Tables* (used only for optional validation).
5. CODATA 2018 recommended values of the fundamental physical constants.
6. NIST Computational Chemistry Comparison and Benchmark Database (CCCDB) — spectroscopic
   constants used to populate the molecular database.
7. Hirschfelder, J. O., Curtiss, C. F. & Bird, R. B. *Molecular Theory of Gases and Liquids*
   (Wiley) — Chapman–Enskog theory and the Lennard–Jones transport framework.
8. Neufeld, P. D., Janzen, A. R. & Aziz, R. A. (1972) *J. Chem. Phys.* **57**, 1100–1102 — the
   collision-integral correlations `Ω^(l,s)*` used by the transport module.
9. Poling, B. E., Prausnitz, J. M. & O'Connell, J. P. *The Properties of Gases and Liquids*
   (McGraw-Hill) — source of the Lennard–Jones `σ`/`ε` parameters in the molecular database.