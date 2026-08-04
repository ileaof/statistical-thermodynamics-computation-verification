Theoretical foundations
======================

StatThermoPy computes thermodynamic properties of ideal gases **exclusively from Statistical
Mechanics**, via the molecular partition function :math:`Q = Q_t\,Q_r\,Q_v\,Q_e`, with no
empirical property correlations. This page states the derivation, physical origin, hypotheses,
limitations and validity range of every result. A fuller, prose-form treatment is kept in
``THEORY.md`` alongside this file.

The canonical ensemble
----------------------

For :math:`N` identical, non-interacting molecules at :math:`(T,V,N)`,

.. math::

   Z_N = \sum_\alpha e^{-\beta E_\alpha},\qquad
   Z_N = \frac{Q^N}{N!},\qquad Q = \sum_j g_j\, e^{-\beta\varepsilon_j},\qquad
   \beta=\frac{1}{k_B T}.

The factorisation :math:`Q=Q_tQ_rQ_vQ_e` follows from separation of the single-molecule
Hamiltonian. The bridge equations give every thermodynamic quantity from :math:`\ln Z_N`:

.. math::

   U = k_B T^2\!\left(\tfrac{\partial\ln Z_N}{\partial T}\right)_{\!V,N},\quad
   S = k_B\!\left[\ln Z_N + T\!\left(\tfrac{\partial\ln Z_N}{\partial T}\right)_{\!V,N}\right],\quad
   A = -k_B T\ln Z_N.

Molar form (with Stirling's :math:`\ln N!\approx N\ln N-N`, :math:`N=nN_A`, :math:`R=N_A k_B`):

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Quantity
     - Molar formula
   * - :math:`U_m`
     - :math:`R T^2(\partial\ln Q/\partial T)_V`
   * - :math:`S_m`
     - :math:`R[\ln Q - \ln N_A + 1 + T(\partial\ln Q/\partial T)_V]`
   * - :math:`A_m`
     - :math:`-RT[\ln Q - \ln N_A + 1]`
   * - :math:`H_m`
     - :math:`U_m + RT`
   * - :math:`G_m`
     - :math:`A_m + RT = -RT[\ln Q - \ln N_A]`
   * - :math:`\mu_m`
     - :math:`G_m`
   * - :math:`C_{v,m}`
     - :math:`(\partial U_m/\partial T)_V`
   * - :math:`C_{p,m}`
     - :math:`C_{v,m} + R`
   * - :math:`\gamma`
     - :math:`C_{p,m}/C_{v,m}`

Translation (Sackur–Tetrode)
---------------------------

.. math::

   Q_t = \left(\frac{2\pi m k_B T}{h^2}\right)^{3/2}\!V,\qquad
   U_{m,t}=\tfrac32 RT,\quad C_{v,m,t}=\tfrac32 R,

.. math::

   S_{m,t}=R\!\left[\tfrac32\ln\!\left(\frac{2\pi m k_B T}{h^2}\right)+\ln\!\left(\frac{k_B T}{P}\right)+\tfrac52\right].

This is the only volume- (pressure-) dependent mode and hosts the indistinguishability
correction :math:`-\ln N_A + 1`. Valid in the classical, non-degenerate regime.

Rotation
--------

* **Monoatomic**: :math:`Q_r=1`, no contribution.
* **Linear rigid rotor**, :math:`\theta_r=h^2/(8\pi^2 I k_B)`:

  .. math::

     Q_r=\sum_{J=0}^{\infty}(2J+1)e^{-J(J+1)\theta_r/T}
        \;\xrightarrow{T\gg\theta_r}\;\frac{T}{\sigma\,\theta_r},

  with :math:`U_{m,r}=RT`, :math:`C_{v,m,r}=R`, :math:`S_{m,r}=R[\ln Q_r+1]`. The package offers
  the exact quantum sum (``use_quantum_rotation=True``), needed at :math:`T\lesssim\theta_r`.
* **Nonlinear** (asymmetric top), :math:`\theta_i=h^2/(8\pi^2 I_i k_B)`:

  .. math::

     Q_r=\frac{\sqrt\pi}{\sigma}\sqrt{\frac{T^3}{\theta_A\theta_B\theta_C}},\qquad
     U_{m,r}=\tfrac32 RT,\quad C_{v,m,r}=\tfrac32 R,\quad S_{m,r}=R[\ln Q_r+\tfrac32].

Vibration (quantum harmonic oscillator)
---------------------------------------

For each normal mode :math:`i` of degeneracy :math:`g_i`, :math:`\theta_{v,i}=hc\tilde\nu_i/k_B`:

.. math::

   Q_{v,i}=\frac{1}{1-e^{-\theta_{v,i}/T}},\quad
   U_{m,v}=\sum_i g_i\frac{R\theta_{v,i}}{e^{\theta_{v,i}/T}-1},\quad
   C_{v,m,v}=\sum_i g_i R\!\left(\frac{\theta_{v,i}}{T}\right)^{\!2}\frac{e^{\theta_{v,i}/T}}{(e^{\theta_{v,i}/T}-1)^2},

.. math::

   S_{m,v}=\sum_i g_i R\!\left[\frac{\theta_{v,i}/T}{e^{\theta_{v,i}/T}-1}-\ln\!\left(1-e^{-\theta_{v,i}/T}\right)\right].

Limits: :math:`T\ll\theta_v` mode frozen; :math:`T\gg\theta_v` equipartition. Harmonic
approximation neglects anharmonicity/coupling.

Electronic
---------

.. math::

   Q_e=\sum_j g_j\, e^{-\theta_{e,j}/T},\qquad
   U_{m,e}=R\langle\theta_e\rangle,\quad
   C_{v,m,e}=\frac{R}{T^2}\!\left(\langle\theta_e^2\rangle-\langle\theta_e\rangle^2\right),

with Boltzmann populations :math:`p_j=g_j e^{-\theta_{e,j}/T}/Q_e`. For closed-shell species
:math:`Q_e=g_0`; open-shell species (O₂, NO) carry low-lying terms that raise :math:`C_{v,m,e}`.

Ideal-gas mixtures
------------------

Components at partial pressure :math:`P_i=x_iP`; no enthalpy of mixing; classical mixing entropy

.. math::

   \Delta S_{\text{mix}}=-R\sum_i x_i\ln x_i,\qquad
   S_m=\sum_i x_i\!\left[S_{m,i}(T,P)-R\ln x_i\right],\qquad
   G_m=\sum_i x_i\!\left[G_{m,i}(T,P)+RT\ln x_i\right].

Chemical equilibrium
--------------------

:math:`\mu_i(T,P)` is already available from the partition function, so Gibbs-energy
minimisation and reaction equilibrium :math:`K(T)=e^{-\Delta G^\circ/RT}` can be built on top
without empirical correlations. Phase 1 exposes the interfaces only.

Validation against embedded NIST/JANAF data
--------------------------------------------

StatThermoPy ships curated reference tables of molar :math:`C_p^\circ` and absolute
:math:`S^\circ` (J/mol/K, standard state 1 bar) for all 22 species in the molecular database
(monoatomic, diatomic, triatomic, and larger polyatomic gases), used only for optional
cross-checking via :func:`~statthermopy.validation.validate`. **Only the reference values are
embedded** — no empirical correlation coefficients (NASA/Shomate/JANAF polynomials) live in the
package, so the calculation core stays pure statistical mechanics. The values were produced by
evaluating the NIST Chemistry WebBook Shomate equations at a temperature grid (the coefficients
themselves are not shipped); for C2H6 and C3H8, for which NIST WebBook publishes no Shomate fit,
the NASA Glenn 7-coefficient polynomials (McBride, Zehe & Gordon, NASA/TP-2002-211556) are used
instead, again with only the values shipped. Each YAML under
``statthermopy.validation.data`` cites its source.

Because :math:`C_p` is pressure-independent for an ideal gas and :math:`S^\circ` is an absolute
third-law quantity, both compare directly to the engine's molar output — no reference-state
offset is needed. Entropy is validated at the reference's standard-state pressure (1 bar), since
the translational term carries the pressure dependence. The rigid-rotor / harmonic-oscillator
model departs from experiment by up to a few percent at high temperature (neglected
anharmonicity makes the engine *underestimate* :math:`C_p` at 2000 K, e.g. ~4% for H2); the
validation tolerance is 5% mean absolute error. Open-shell species (O2, NO) include low-lying
electronic terms, which the engine reproduces.

Hypotheses, limitations, validity
----------------------------------

* Ideal gas — no intermolecular interactions; valid away from condensation/liquefaction (the
  transport layer adds the dilute-gas Chapman–Enskog coefficients on top of this engine).
* Born–Oppenheimer + rigid rotor + harmonic oscillator separability.
* Maxwell–Boltzmann (non-degenerate) statistics; fails for He at very low :math:`T`/high density.
* Spectroscopic constants from McQuarrie, Herzberg, NIST CCCDB.

Backend abstraction (performance)
--------------------------------

All array work (exponentials, logs, sums over levels) is routed through a thin ``Backend`` ABC.
The reference ``NumpyBackend`` is always available. Three accelerated backends are *functional* and
selected at runtime with :func:`~statthermopy.backend.set_backend` **without altering the public
API**:

* ``"numba"`` — Numba ``@njit(cache=True)`` CPU kernels compiling the exact quantum linear-rotor
  J-sum and a temperature-batched molar-property kernel;
* ``"openmp"`` — ``@njit(parallel=True)`` with ``numba.prange`` over the temperature grid (Numba's
  OpenMP-style threaded loops), reusing the same per-temperature device function (no physics
  duplication);
* ``"cuda"`` — a ``numba.cuda`` GPU kernel with **automatic CPU fallback**: when no NVIDIA GPU is
  present, construction warns and delegates to the Numba CPU backend, so
  ``set_backend("cuda")`` never raises on a GPU-less machine.

The per-temperature physics inside the kernels mirrors the mode modules exactly — the acceleration
changes only the numerical execution, never the physics or the data. No empirical property
correlation is introduced; the calculation core stays pure Statistical Mechanics. The accelerated
backends import lazily, so ``import statthermopy`` never pulls in numba. See
:func:`~statthermopy.backend.list_backends` / :func:`~statthermopy.backend.available_backends`.

References
----------

1. McQuarrie, D. A. *Statistical Mechanics* (University Science Books).
2. Hill, T. L. *An Introduction to Statistical Thermodynamics* (Dover).
3. Herzberg, G. *Molecular Spectra and Molecular Structure*, vols. I–III.
4. Chase, M. W. et al. *NIST-JANAF Thermochemical Tables* (validation only).
5. CODATA 2018 recommended values of the fundamental physical constants.
6. NIST Computational Chemistry Comparison and Benchmark Database (CCCDB).