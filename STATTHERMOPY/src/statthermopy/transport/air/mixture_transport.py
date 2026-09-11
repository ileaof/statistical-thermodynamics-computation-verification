"""Mixture transport properties from kinetic-theory mixing rules.

This module computes the transport properties of a **dilute ideal-gas mixture** by combining the
pure-species values produced by :class:`~statthermopy.transport.transport.TransportCalculator`
(viscosity, Eucken thermal conductivity, self-diffusion) and the binary-pair values produced by
:func:`~statthermopy.transport.transport.binary_diffusion` through the standard, well-established
dilute-gas mixing rules:

* **Viscosity — Wilke (1950):**

      μ_mix = Σ_i x_i μ_i / (Σ_j x_j φ_ij),
      φ_ij = [1 + (μ_i/μ_j)^{1/2} (M_j/M_i)^{1/4}]^2 / √(8 (1 + M_i/M_j)).

* **Thermal conductivity — Mason–Saxena (1958):** the same structural form as Wilke applied to the
  pure-species Eucken conductivity ``k_i`` (the conductivity-ratio form of Mason–Saxena is
  well approximated by the viscosity-ratio weighting for the gases considered here):

      k_mix = Σ_i x_i k_i / (Σ_j x_j φ_ij^(k)),
      φ_ij^(k) = [1 + (k_i/k_j)^{1/2} (M_j/M_i)^{1/4}]^2 / √(8 (1 + M_i/M_j)).

* **Mass diffusivity of a trace species in the mixture — Blanc's law** (dilute multicomponent):

      D_t,m = (1 − x_t) / Σ_{j≠t} (x_j / D_tj),

  with ``D_tj`` from :func:`~statthermopy.transport.transport.binary_diffusion` (Lorentz–Berthelot +
  ``Ω^(1,1)*``). For "water-vapour diffusivity in air" the trace is H₂O: when H₂O is a component of
  the (humid) mixture it diffuses through the remaining dry species; when it is not (dry air) it
  is treated as an external trace diffusing into the dry background.

The derived mixture quantities (``ρ``, ``ν``, ``α``, ``Pr``, ``Sc``, ``Le``, speed of sound,
thermal-expansion coefficient, isothermal compressibility) follow from the ideal-gas EOS of the
mixture, with the heat capacities ``Cp_m``, ``Cv_m`` and ``γ`` taken from
:meth:`~statthermopy.mixture.IdealGasMixture.compute` so the full quantum-mode temperature
dependence propagates into the transport coefficients. The architecture is open to dense-gas
(Enskog / corresponding-states) corrections as a future extension behind the same interface.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from ...constants import R
from ...core.state import ResolvedState, State
from ...mixture import IdealGasMixture
from ..transport import TransportCalculator, binary_diffusion

__all__ = [
    "MixtureTransportProperties",
    "SpeciesTransportContribution",
    "MixtureTransportCalculator",
    "wilke_viscosity",
    "mason_saxena_conductivity",
    "blanc_diffusion",
    "AIR_TRANSPORT_PROPS",
    "AIR_TRANSPORT_UNITS",
    "AIR_TRANSPORT_LABELS",
]

#: Headline transport properties reported for the air-transport panel (the eight vs-T plots).
AIR_TRANSPORT_PROPS: list[str] = ["mu", "nu", "k", "alpha", "D_eff", "Pr", "Sc", "Le"]

#: Full property set available on :class:`MixtureTransportProperties`.
AIR_TRANSPORT_ALL_PROPS: list[str] = [
    "mu", "nu", "k", "alpha", "D_eff", "Pr", "Sc", "Le",
    "rho", "R_specific", "a", "beta", "kappa_T", "gamma", "cp_s", "cv_s", "Z",
]

#: SI units of each property (axis labels / legends / export headers).
AIR_TRANSPORT_UNITS: dict[str, str] = {
    "mu": "Pa·s", "nu": "m^2/s", "k": "W/m/K", "alpha": "m^2/s", "D_eff": "m^2/s",
    "Pr": "-", "Sc": "-", "Le": "-",
    "rho": "kg/m^3", "R_specific": "J/kg/K", "a": "m/s",
    "beta": "1/K", "kappa_T": "1/Pa", "gamma": "-", "cp_s": "J/kg/K", "cv_s": "J/kg/K",
    "Z": "-",
}

#: Human-readable labels for plots / tables.
AIR_TRANSPORT_LABELS: dict[str, str] = {
    "mu": "Dynamic viscosity",
    "nu": "Kinematic viscosity",
    "k": "Thermal conductivity",
    "alpha": "Thermal diffusivity",
    "D_eff": "Water-vapour diffusivity in air",
    "Pr": "Prandtl number",
    "Sc": "Schmidt number",
    "Le": "Lewis number",
    "rho": "Density",
    "R_specific": "Specific gas constant",
    "a": "Speed of sound",
    "beta": "Thermal expansion coefficient",
    "kappa_T": "Isothermal compressibility",
    "gamma": "Heat-capacity ratio",
    "cp_s": "Specific heat Cp",
    "cv_s": "Specific heat Cv",
    "Z": "Compressibility factor",
}


# -- mixing-rule helpers ------------------------------------------------------


def _wilke_phi(value_i: float, value_j: float, m_i: float, m_j: float) -> float:
    """Wilke/Mason–Saxena interaction factor ``φ_ij`` from a pair of like-weight quantities.

    ``value`` is ``μ`` for viscosity or ``k`` for conductivity; ``m`` is the molar mass. The form
    ``[1 + (v_i/v_j)^½ (M_j/M_i)^¼]^2 / √(8(1 + M_i/M_j))`` is shared by both rules.
    """
    ratio_v = math.sqrt(value_i / value_j) if value_j > 0.0 else 0.0
    ratio_m = (m_j / m_i) ** 0.25 if m_i > 0.0 else 0.0
    return (1.0 + ratio_v * ratio_m) ** 2 / math.sqrt(8.0 * (1.0 + m_i / m_j))


def wilke_viscosity(
    species: list[tuple[float, float, float]],
) -> tuple[float, list[float]]:
    """Wilke mixture viscosity from per-species ``(x_i, μ_i, M_i)`` triples.

    Returns ``(μ_mix, contributions)`` where ``contributions[i] = x_i μ_i / Σ_j x_j φ_ij`` (they
    sum to ``μ_mix``).
    """
    n = len(species)
    denom = [0.0] * n
    for i in range(n):
        xi, mu_i, mi = species[i]
        s = 0.0
        for j in range(n):
            xj, mu_j, mj = species[j]
            s += xj * _wilke_phi(mu_i, mu_j, mi, mj)
        denom[i] = s
    mu_mix = 0.0
    contribs = [0.0] * n
    for i in range(n):
        xi, mu_i, _ = species[i]
        contribs[i] = xi * mu_i / denom[i] if denom[i] > 0.0 else 0.0
        mu_mix += contribs[i]
    return mu_mix, contribs


def mason_saxena_conductivity(
    species: list[tuple[float, float, float]],
) -> tuple[float, list[float]]:
    """Mason–Saxena mixture conductivity from per-species ``(x_i, k_i, M_i)`` triples.

    Structurally identical to :func:`wilke_viscosity` with ``k_i`` in place of ``μ_i``. Returns
    ``(k_mix, contributions)``.
    """
    return wilke_viscosity(species)  # identical algebra; named separately for clarity


def blanc_diffusion(
    trace_index: int,
    x: list[float],
    D_pairs: list[list[float]],
) -> float:
    """Blanc's-law effective diffusivity of the trace species into the mixture.

    ``D_t,m = (1 − x_t) / Σ_{j≠t} (x_j / D_tj)``. ``D_pairs[i][j]`` must hold ``D_ij`` for the trace
    row ``i = trace_index``. ``D_pairs`` is square over all species; the trace row is used.
    """
    n = len(x)
    x_t = x[trace_index]
    s = 0.0
    for j in range(n):
        if j == trace_index:
            continue
        D = D_pairs[trace_index][j]
        if D > 0.0 and x[j] > 0.0:
            s += x[j] / D
    if s <= 0.0:
        return 0.0
    return (1.0 - x_t) / s


# -- result containers --------------------------------------------------------


@dataclass
class SpeciesTransportContribution:
    """One species' pure-species transport values and its share of the mixture totals.

    The ``*_contrib`` fields are the mole-fraction-weighted amounts this species adds to the
    corresponding mixture total under the Wilke / Mason–Saxena rules; they sum (over species) to
    the mixture viscosity / conductivity.
    """

    name: str
    x: float                 # mole fraction in the mixture
    molar_mass: float        # kg/mol
    # pure-species transport at (T, P) — from TransportCalculator
    mu_i: float              # Pa·s
    k_i: float               # W/m/K
    D_self_i: float          # m^2/s
    Pr_i: float
    Sc_i: float
    Le_i: float
    # diffusion of this species into the rest of the mixture (Blanc)
    D_im: float              # m^2/s
    # weighted contributions to the mixture Wilke/Mason–Saxena totals (sum to mu_mix/k_mix)
    mu_contrib: float
    k_contrib: float


@dataclass
class MixtureTransportProperties:
    """Full transport & thermophysical report for a gas mixture at one state.

    All quantities are SI. ``mu`` is the Wilke dynamic viscosity, ``k`` the Mason–Saxena thermal
    conductivity, ``a`` the speed of sound, ``beta`` the thermal-expansion coefficient,
    ``kappa_T`` the isothermal compressibility. ``components`` carries the per-species
    :class:`SpeciesTransportContribution` breakdown.

    **Mixture scalars vs per-species quantities.** ``mu``, ``k``, ``rho``, ``alpha``, ``gamma``
    and ``Pr = μ c_p / k`` are properties *of the mixture* — single numbers. Diffusion is not:
    ``D_i,m``, and therefore the Schmidt and Lewis numbers built on it, are defined **per
    species**. They are reported in the ``D_im``, ``Sc_i`` and ``Le_i`` dictionaries, keyed by
    species name.

    ``D_eff``, ``Sc`` and ``Le`` are the single-species view for the one species named by
    ``trace_species``. They are ``None`` unless a trace was requested — a generic mixture has no
    natural trace, and silently defaulting to H₂O would report water diffusing through a mixture
    that may contain none. :class:`~statthermopy.transport.air.AirTransport` asks for ``"H2O"``
    explicitly, where it *is* the physically natural choice.
    """

    # conditions
    T: float
    P: float
    label: str                       # "Dry air", "Humid air", …
    basis: str
    x: dict                          # name -> mole fraction
    M_avg: float                     # kg/mol
    humidity_ratio: float | None     # kg/kg dry air (None for dry air)
    # transport (mixing-rule outputs)
    mu: float                        # Pa·s
    nu: float                         # m^2/s
    k: float                         # W/m/K
    alpha: float                     # m^2/s
    # per-species diffusion (name -> value); diffusion is never a single mixture scalar
    D_im: dict                        # m^2/s, species i into the rest of the mixture (Blanc)
    Sc_i: dict                        # nu / D_i,m
    Le_i: dict                        # alpha / D_i,m
    # single-species view, only when a trace species was requested (else None)
    trace_species: str | None
    D_eff: float | None               # m^2/s — D_im[trace]
    Sc: float | None                  # Sc_i[trace]
    Le: float | None                  # Le_i[trace]
    # dimensionless group of the mixture itself
    Pr: float
    # thermophysical (ideal-gas EOS of the mixture)
    Z: float
    a: float                          # m/s
    beta: float                       # 1/K
    kappa_T: float                    # 1/Pa
    # supporting
    rho: float                        # kg/m^3
    gamma: float
    cv_s: float                       # J/kg/K
    cp_s: float                       # J/kg/K
    R_specific: float                 # J/kg/K
    #: Mole-fraction-weighted |error| bands of the mixture's coefficients (percent). A species
    #: without a validated band contributes nothing and is listed in ``accuracy_unvalidated``.
    accuracy: dict = field(default_factory=dict)
    #: Components carrying no validated accuracy band.
    accuracy_unvalidated: tuple = ()
    # provenance
    mixing_rules: dict = field(default_factory=dict)
    # per-species breakdown
    components: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Flat dictionary view suitable for export (components serialised as nested dicts)."""
        d = asdict(self)
        return d

    # -- per-species accessors -------------------------------------------------

    def effective_diffusivity(self, species: str) -> float:
        """Blanc diffusivity ``D_i,m`` of one species into the rest of the mixture (m²/s)."""
        return self._per_species(self.D_im, species, "effective diffusivity")

    def schmidt_number(self, species: str) -> float:
        """Schmidt number ``Sc_i = ν / D_i,m`` of one species."""
        return self._per_species(self.Sc_i, species, "Schmidt number")

    def lewis_number(self, species: str) -> float:
        """Lewis number ``Le_i = α / D_i,m`` of one species."""
        return self._per_species(self.Le_i, species, "Lewis number")

    def effective_diffusivities(self) -> dict:
        """``{species: D_i,m}`` for every component."""
        return dict(self.D_im)

    def schmidt_numbers(self) -> dict:
        """``{species: Sc_i}`` for every component."""
        return dict(self.Sc_i)

    def lewis_numbers(self) -> dict:
        """``{species: Le_i}`` for every component."""
        return dict(self.Le_i)

    def _per_species(self, table: dict, species: str, what: str) -> float:
        key = {k.upper(): k for k in table}.get(str(species).upper())
        if key is None:
            raise KeyError(
                f"{what} requested for {species!r}, which is not a component of this mixture "
                f"({', '.join(table) or 'none'}). Pass a trace species to the calculator to get "
                f"the diffusivity of a species that is not itself present."
            )
        return table[key]


# -- calculator ---------------------------------------------------------------


class MixtureTransportCalculator:
    """Compute all transport & thermophysical properties of a gas mixture at a state.

    Parameters
    ----------
    mixture : IdealGasMixture
        The gas mixture (dry air, humid air, or any custom ideal-gas composition).
    trace : str or None, default ``None``
        Species for which the single-value :attr:`.D_eff`, :attr:`.Sc` and :attr:`.Le` are
        reported. ``None`` — the generic default — leaves those three ``None`` and reports
        diffusion only per species, in :attr:`.D_im` / :attr:`.Sc_i` / :attr:`.Le_i`. A generic
        mixture has no natural trace, so none is assumed. When a trace *is* named and is absent
        from the mixture, it is treated as an external species diffusing into the background.

    Notes
    -----
    The pure-species ``μ_i``, ``k_i`` come from
    :class:`~statthermopy.transport.transport.TransportCalculator` (Chapman–Enskog + Eucken), the
    binary-pair ``D_ij`` from
    :func:`~statthermopy.transport.transport.binary_diffusion`, and the heat capacities / ``γ``
    from :meth:`~statthermopy.mixture.IdealGasMixture.compute`. No empirical property correlation
    is used in the calculation path.
    """

    def __init__(self, mixture: IdealGasMixture, *, trace: str | None = None) -> None:
        self.mixture = mixture
        self.trace = trace

    def _resolved(self, state: State) -> ResolvedState:
        return state.resolve(self.mixture.M_avg)

    def compute(self, state: State, *, label: str = "") -> MixtureTransportProperties:
        """Evaluate every mixture property at ``state`` and return a report."""
        rs = self._resolved(state)
        T, P = rs.T, rs.P
        mix = self.mixture
        x_map = mix.x                                  # dict[Molecule, mole fraction]
        items = list(x_map.items())                     # [(mol, x_i), ...]
        names = [mol.name for mol, _ in items]
        xs = [float(xi) for _, xi in items]
        molar = [mol.molar_mass for mol, _ in items]
        n = len(items)

        # mixture thermodynamics (γ, Cp_m, Cv_m) — one evaluation
        pm = mix.compute(state)
        M_avg = mix.M_avg
        gamma = pm.gamma
        cp_s = pm.Cp_m / M_avg
        cv_s = pm.Cv_m / M_avg
        R_specific = R / M_avg

        # per-species pure transport
        mu_i = [0.0] * n
        k_i = [0.0] * n
        D_self_i = [0.0] * n
        Pr_i = [0.0] * n
        Sc_i = [0.0] * n
        Le_i = [0.0] * n
        for idx, (mol, _) in enumerate(items):
            pure = TransportCalculator(mol, State(T=T, P=P)).compute()
            mu_i[idx] = pure.mu
            k_i[idx] = pure.k
            D_self_i[idx] = pure.D_self
            Pr_i[idx] = pure.Pr
            Sc_i[idx] = pure.Sc
            Le_i[idx] = pure.Le

        # Wilke viscosity + Mason–Saxena conductivity
        mu_mix, mu_contribs = wilke_viscosity(list(zip(xs, mu_i, molar, strict=False)))
        k_mix, k_contribs = mason_saxena_conductivity(list(zip(xs, k_i, molar, strict=False)))

        # binary-pair diffusion matrix D_ij for all species pairs (used by Blanc)
        D_pairs = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    D_pairs[i][j] = D_self_i[i]
                else:
                    D_pairs[i][j] = binary_diffusion(items[i][0], items[j][0], T, P)

        # per-species Blanc diffusivity into the rest of the mixture
        D_im = [0.0] * n
        for i in range(n):
            D_im[i] = blanc_diffusion(i, xs, D_pairs)

        # trace diffusivity, only when a trace species was explicitly requested
        D_eff = (
            self._trace_diffusivity(items, xs, D_pairs, T, P)
            if self.trace is not None
            else None
        )

        # density and per-density derivatives (ideal-gas EOS of the mixture)
        if T > 0.0 and P > 0.0:
            rho = P * M_avg / (R * T)
            nu = mu_mix / rho if rho > 0.0 else 0.0
            alpha = k_mix / (rho * cp_s) if (rho > 0.0 and cp_s > 0.0) else 0.0
        else:
            rho = 0.0
            nu = 0.0
            alpha = 0.0

        # Pr is a mixture scalar; Sc and Le are per-species because D is.
        Pr = (mu_mix * cp_s / k_mix) if k_mix > 0.0 else 0.0
        D_im_map = {name: D_im[i] for i, name in enumerate(names)}
        Sc_i_map = {n_: (nu / d if d > 0.0 else 0.0) for n_, d in D_im_map.items()}
        Le_i_map = {n_: (alpha / d if d > 0.0 else 0.0) for n_, d in D_im_map.items()}
        if D_eff is not None and D_eff > 0.0:
            Sc = nu / D_eff
            Le = (Sc / Pr) if Pr != 0.0 else 0.0
        else:
            Sc = Le = None if D_eff is None else 0.0

        # thermophysical (ideal-gas EOS — exact for this engine)
        Z = 1.0
        a = math.sqrt(gamma * R_specific * T) if T > 0.0 else 0.0
        beta = (1.0 / T) if T > 0.0 else 0.0
        kappa_T = (1.0 / P) if P > 0.0 else 0.0

        # Mole-fraction-weighted accuracy bands. A mixture is only as trustworthy as its
        # components, and a little water vapour drags the conductivity band up sharply.
        acc: dict[str, float] = {}
        unvalidated: list[str] = []
        for key in ("viscosity_percent", "conductivity_percent", "diffusion_percent"):
            total = 0.0
            weight = 0.0
            for (mol, xi) in items:
                ta = mol.transport_accuracy
                value = getattr(ta, key, None) if ta is not None else None
                if value is None:
                    continue
                total += xi * float(value)
                weight += xi
            if weight > 0.0:
                acc[key] = total / weight
        for (mol, _xi) in items:
            if mol.transport_accuracy is None:
                unvalidated.append(mol.name)
        unvalidated = tuple(unvalidated)

        # per-species contribution breakdown
        components: dict[str, SpeciesTransportContribution] = {}
        for idx, (mol, xi) in enumerate(items):
            components[mol.name] = SpeciesTransportContribution(
                name=mol.name, x=xi, molar_mass=mol.molar_mass,
                mu_i=mu_i[idx], k_i=k_i[idx], D_self_i=D_self_i[idx],
                Pr_i=Pr_i[idx], Sc_i=Sc_i[idx], Le_i=Le_i[idx],
                D_im=D_im[idx],
                mu_contrib=mu_contribs[idx], k_contrib=k_contribs[idx],
            )

        return MixtureTransportProperties(
            T=T, P=P, label=label, basis=mix.basis,
            x={name: xi for name, xi in zip(names, xs, strict=False)},
            M_avg=M_avg, humidity_ratio=None,
            mu=mu_mix, nu=nu, k=k_mix, alpha=alpha,
            D_im=D_im_map, Sc_i=Sc_i_map, Le_i=Le_i_map,
            trace_species=self.trace, D_eff=D_eff, Sc=Sc, Le=Le,
            Pr=Pr,
            Z=Z, a=a, beta=beta, kappa_T=kappa_T,
            rho=rho, gamma=gamma, cv_s=cv_s, cp_s=cp_s, R_specific=R_specific,
            accuracy=acc, accuracy_unvalidated=unvalidated,
            mixing_rules={"mu": "Wilke", "k": "Mason-Saxena", "D_im": "Blanc"},
            components=components,
        )

    properties = compute

    # -- internal: trace diffusivity (water vapour in air) ---------------------

    def _trace_diffusivity(
        self,
        items: list[tuple],
        xs: list[float],
        D_pairs: list[list[float]],
        T: float,
        P: float,
    ) -> float:
        """Water-vapour-in-air diffusivity via Blanc's law.

        If H₂O is a component of the mixture, ``D_eff = (1 − x_H2O) / Σ_{j≠H2O} (x_j / D_H2O,j)``.
        If it is absent (dry air), H₂O is an external trace diffusing into the background and
        ``D_eff = 1 / Σ_j (x_j / D_H2O,j)`` (``1 − x_H2O = 1``). The H₂O :class:`Molecule` is loaded
        from the database; ``D_H2O,j`` uses :func:`binary_diffusion`.
        """
        from ...database import get

        names = [mol.name for mol, _ in items]
        # normalise the trace lookup (database keys are uppercase)
        trace_key = self.trace.upper()
        index = {name.upper(): i for i, name in enumerate(names)}
        h2o = get(self.trace)

        if trace_key in index:
            ti = index[trace_key]
            return blanc_diffusion(ti, xs, D_pairs)

        # trace not in mixture: H2O diffusing into the dry background
        if T <= 0.0 or P <= 0.0:
            return 0.0
        s = 0.0
        for i, (mol, xi) in enumerate(items):
            if xi <= 0.0:
                continue
            D = binary_diffusion(h2o, mol, T, P)
            if D > 0.0:
                s += xi / D
        return (1.0 / s) if s > 0.0 else 0.0

    # -- vectorised helpers ----------------------------------------------------

    def property_vs_T(self, prop: str, T_range, P: float) -> tuple:
        """Evaluate a mixture property over a temperature range at constant pressure.

        Returns ``(Ts, values)``. ``prop`` is any attribute of :class:`MixtureTransportProperties`.
        """
        Ts = list(T_range)
        out = []
        for T in Ts:
            st = State(T=float(T), P=float(P))
            out.append(getattr(self.compute(st), prop))
        return Ts, out

    def property_vs_P(self, prop: str, P_range, T: float) -> tuple:
        """Evaluate a mixture property over a pressure range at constant temperature."""
        Ps = list(P_range)
        out = []
        for P in Ps:
            st = State(T=float(T), P=float(P))
            out.append(getattr(self.compute(st), prop))
        return Ps, out