# Theory notes

A concise reference to the central results implemented in the code. This is a
map, not a substitute for the textbook; each entry points to the chapter and the
programs where the result is computed and verified.

## Foundations (Chapter 1)

- **Boltzmann entropy:** *S* = *k*<sub>B</sub> ln Ω, with the multiplicity Ω
  the number of microstates of a macrostate.
- **Two-state multiplicity:** Ω(*N*, *n*) = *C*(*N*, *n*); in the large-*N* limit
  a Gaussian of relative width ∝ *N*<sup>&minus;1/2</sup>.
- **Einstein solid:** Ω(*N*, *q*) = *C*(*q* + *N* &minus; 1, *q*).
- **Boltzmann distribution:** for a small system in contact with a reservoir,
  *P*(*i*) ∝ e<sup>&minus;*E*<sub>i</sub>/*k*<sub>B</sub>*T*</sup>.

## Kinetic theory (Chapter 2)

- **Maxwell-Boltzmann speeds:** *f*(*v*) = 4π (*m*/2π*k*<sub>B</sub>*T*)<sup>3/2</sup>
  *v*<sup>2</sup> e<sup>&minus;*mv*<sup>2</sup>/2*k*<sub>B</sub>*T*</sup>.
- **Characteristic speeds:** *v*<sub>p</sub> = √(2*k*<sub>B</sub>*T*/*m*),
  ⟨*v*⟩ = √(8*k*<sub>B</sub>*T*/π*m*), *v*<sub>rms</sub> = √(3*k*<sub>B</sub>*T*/*m*).
- **Mean free path:** λ = 1 / (√2 *n* σ), σ = π*d*<sup>2</sup>.
- **Lennard-Jones potential:** *V*(*r*) = 4ε[(σ/*r*)<sup>12</sup> &minus; (σ/*r*)<sup>6</sup>].

## Partition functions (Chapter 3)

- **Canonical bridge:** *U* = &minus;∂ ln *Z*/∂β, *F* = &minus;*k*<sub>B</sub>*T* ln *Z*,
  *S* = (*U* &minus; *F*)/*T*, *C* = (⟨*E*<sup>2</sup>⟩ &minus; ⟨*E*⟩<sup>2</sup>)/*k*<sub>B</sub>*T*<sup>2</sup>.
- **Two-level system:** *Z* = 1 + e<sup>&minus;ε/*k*<sub>B</sub>*T*</sup>; the heat
  capacity shows the Schottky anomaly.
- **Quantum statistics:** ⟨*n*⟩ = 1/(e<sup>(ε&minus;μ)/*k*<sub>B</sub>*T*</sup> ± 1),
  with +1 for fermions, &minus;1 for bosons, and the classical limit for large argument.

## Monatomic ideal gas (Chapter 4)

- **Thermal wavelength:** Λ = *h* / √(2π*mk*<sub>B</sub>*T*).
- **Sackur-Tetrode entropy:** *S* = *Nk*<sub>B</sub>[ln(*V*/*N*Λ<sup>3</sup>) + 5/2].
- **Gibbs correction:** *Z* = *z*<sup>*N*</sup>/*N*! makes entropy extensive and
  resolves the Gibbs paradox.

## Diatomic and polyatomic gases (Chapter 5)

- **Rotational partition function:** *z*<sub>rot</sub> = (1/σ) Σ (2*J* + 1)
  e<sup>&minus;θ<sub>rot</sub>*J*(*J*+1)/*T*</sup>, with the high-*T* limit *T*/σθ<sub>rot</sub>.
- **Heat-capacity staircase:** *C*<sub>V</sub> climbs 3/2 → 5/2 → 7/2 *R* as
  rotation and vibration unlock.

## Quantum gases (Chapter 6)

- **Planck spectrum:** *u*(ω, *T*) = ℏω<sup>3</sup>/(π<sup>2</sup>*c*<sup>3</sup>) ·
  1/(e<sup>ℏω/*k*<sub>B</sub>*T*</sup> &minus; 1).
- **Stefan-Boltzmann:** *u* = *aT*<sup>4</sup>, *a* = π<sup>2</sup>*k*<sup>4</sup>/15ℏ<sup>3</sup>*c*<sup>3</sup>.
- **Fermi gas:** μ(*T*) from the number constraint; low-*T* *C* ∝ *T* (Sommerfeld).
- **Bose-Einstein condensation:** below *T*<sub>c</sub>, *N*<sub>0</sub>/*N* = 1 &minus; (*T*/*T*<sub>c</sub>)<sup>3/2</sup>.

## Chemical equilibrium and imperfect gases (Chapter 7)

- **Equilibrium constant:** *K*<sub>p</sub> from ratios of molecular partition
  functions and the dissociation energy.
- **Second virial coefficient:** *B*(*T*) = &minus;2π*N*<sub>A</sub> ∫ (e<sup>&minus;*u*/*k*<sub>B</sub>*T*</sup> &minus; 1) *r*<sup>2</sup> d*r*.
- **Hard-sphere coefficients:** *B*<sub>2</sub> = (2/3)πσ<sup>3</sup>, *B*<sub>3</sub> = (5/18)π<sup>2</sup>σ<sup>6</sup>.

## Solids (Chapter 8)

- **Einstein model:** *C*<sub>V</sub> = 3*R* *x*<sup>2</sup>e<sup>*x*</sup>/(e<sup>*x*</sup>&minus;1)<sup>2</sup>, *x* = θ<sub>E</sub>/*T*.
- **Debye model:** *C*<sub>V</sub> = 9*R*(*T*/θ<sub>D</sub>)<sup>3</sup> ∫<sub>0</sub><sup>θ<sub>D</sub>/*T*</sup> *x*<sup>4</sup>e<sup>*x*</sup>/(e<sup>*x*</sup>&minus;1)<sup>2</sup> d*x*, with the low-*T* law *C*<sub>V</sub> ∝ *T*<sup>3</sup>.

## Phase transitions (Chapter 9)

- **Mean-field Ising:** *m* = tanh(*m*/*t*), with β = 1/2 and γ = 1.
- **Onsager (2-D):** *k*<sub>B</sub>*T*<sub>c</sub>/*J* = 2/ln(1+√2) = 2.26919, with
  *m* = (1 &minus; sinh<sup>&minus;4</sup>(2/*T*))<sup>1/8</sup>.
- **Finite-size scaling:** χ<sub>max</sub> ∝ *L*<sup>γ/ν</sup>, *m*(*T*<sub>c</sub>) ∝ *L*<sup>&minus;β/ν</sup>,
  with the exact ratios γ/ν = 7/4, β/ν = 1/8.

## Computational methods (Chapter 10)

- **Importance sampling:** ⟨*O*⟩ = Σ *O*(*x*<sub>i</sub>)*w*<sub>i</sub> / Σ *w*<sub>i</sub>, *w* = target/proposal.
- **Metropolis:** accept *x* → *x*′ with probability min(1, e<sup>&minus;β[*V*(*x*′)&minus;*V*(*x*)]</sup>).
- **Error analysis:** integrated autocorrelation time τ, blocking, bootstrap, and
  the pull distribution.

---

For the derivations behind these results, see the corresponding chapter of the
book; for the code that computes and verifies them, follow the links in each
chapter's README.
