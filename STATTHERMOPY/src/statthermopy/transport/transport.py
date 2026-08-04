"""Transport and thermophysical properties from Statistical Mechanics + Kinetic Theory.

This module computes, for a pure gas at a state ``(T, P)``, the transport coefficients of the
dilute gas from the **Chapman–Enskog** first-order solution of the Boltzmann equation with the
**Lennard–Jones** pair potential, and the thermophysical coefficients from the **ideal-gas
equation of state** (the exact statistical-mechanics ideal gas this engine models). Every primary
coefficient is derived from the molecular partition function (which supplies ``Cv``, ``Cp``,
``γ``, the molar mass) and the LJ potential parameters (``σ``, ``ε``, stored per species as
:class:`~statthermopy.core.molecule.LennardJones`); no external property database is used.

Primary coefficients (Chapman–Enskog, dilute/ideal-gas limit)
------------------------------------------------------------
Dynamic viscosity (SI, Pa·s)::

    μ = (5/16) · √(m k_B T / π) / (σ² · Ω^(2,2)*(T*)),   T* = k_B T / ε = T / (ε/k_B)

Thermal conductivity (Eucken correlation, W/m·K)::

    k = μ · c_v · (9γ − 5) / 4,   c_v = Cv_m / M,   γ = Cp_m / Cv_m

The Eucken factor ``(9γ − 5)/4`` equals the Chapman–Enskog monatomic multiplier ``5/2`` on
``c_v`` for ``γ = 5/3``, recovering the exact CE result ``k = (5/2) c_v μ = (15/4)(k_B/m) μ``
(per-mass ``c_v = (3/2) R/M`` for a monatomic gas). ``Cv_m``, ``Cp_m`` and ``γ`` are taken
**directly** from
:class:`~statthermopy.thermodynamics.Thermodynamics`, so the quantum-mode heat capacities (and
their Third-Law behaviour) propagate into the transport coefficients with full thermodynamic
consistency.

Binary (and self-) diffusion (SI, m²/s)::

    D_ij = (3/16) · (k_B T / P) · (σ_ij² Ω^(1,1)*(T*_ij))⁻¹ · √(2 k_B T / (π m_ij))

with the Lorentz–Berthelot combining rules ``σ_ij = (σ_i+σ_j)/2``,
``ε_ij = √(ε_i ε_j)`` and the reduced mass ``m_ij = m_i m_j/(m_i+m_j)``. The pure-gas
self-diffusion coefficient ``D_self = D_ii`` is the ``i = j`` case (``m_ii = m_i/2``).

Derived coefficients (ideal-gas EOS — exact for this engine)
-----------------------------------------------------------
::

    ρ = P M / (R T);   ν = μ/ρ;   α = k/(ρ c_p);
    Pr = μ c_p / k = 4γ/(9γ − 5);          (Eucken — independent of μ, finite at all T)
    Sc = ν / D_self = (5/6) Ω^(1,1)*/Ω^(2,2)*;   Le = α / D_self = Sc / Pr;
    Z = 1;   a = √(γ R_specific T);   β = 1/T;   κ_T = 1/P;   μ_JT = 0.

Pressure dependence enters through the density (``ν``, ``α``, ``D ∝ 1/P``) and ``κ_T``; ``μ``
and ``k`` are pressure-independent in the dilute limit — the physically correct ideal-gas
behaviour. The architecture is open to dense-gas (Enskog / corresponding-states) corrections as
a future extension.

Continuity at T = 0
-------------------
The ``√T`` prefactor drives ``μ, k → 0`` and ``D → 0`` as ``T → 0``, while ``Pr``, ``Sc``, ``Le``
(the dimensionless ratios) stay finite via their closed-form expressions; ``a → 0``. The singular
ideal-gas coefficients (``β = 1/T``, ``ρ``) are clamped to ``0`` at ``T = 0`` so every reported
curve is finite and continuous down to 0 K (documented as the ``T = 0`` ideal-gas limit).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from ..constants import R, k_B
from ..core.molecule import Molecule
from ..core.state import ResolvedState, State
from ..thermodynamics import Thermodynamics
from .collision import omega_11, omega_22, t_star
from .lennard_jones import pair_epsilon_over_k, pair_sigma_m, reduced_mass

__all__ = [
    "TransportProperties",
    "TransportCalculator",
    "binary_diffusion",
    "self_diffusion",
    "TRANSPORT_PROPS",
    "TRANSPORT_UNITS",
]


#: All transport/thermophysical properties reported by :class:`TransportCalculator`.
TRANSPORT_PROPS: list[str] = [
    "mu", "nu", "k", "alpha", "D_self",
    "Pr", "Sc", "Le",
    "Z", "a", "beta", "kappa_T", "mu_JT",
]

#: SI units of each property (for axis labels / legends / export headers).
TRANSPORT_UNITS: dict[str, str] = {
    "mu": "Pa·s", "nu": "m^2/s", "k": "W/m/K", "alpha": "m^2/s", "D_self": "m^2/s",
    "Pr": "-", "Sc": "-", "Le": "-",
    "Z": "-", "a": "m/s", "beta": "1/K", "kappa_T": "1/Pa", "mu_JT": "K/Pa",
    "rho": "kg/m^3", "gamma": "-",
}


@dataclass
class TransportProperties:
    """Full transport & thermophysical report for one molecule at one state.

    All quantities are SI. ``mu`` is the dynamic viscosity, ``nu`` the kinematic viscosity, ``k``
    the thermal conductivity, ``alpha`` the thermal diffusivity, ``D_self`` the self-diffusion
    coefficient, ``Pr``/``Sc``/``Le`` the Prandtl/Schmidt/Lewis numbers, ``Z`` the compressibility
    factor, ``a`` the speed of sound, ``beta`` the thermal-expansion coefficient, ``kappa_T`` the
    isothermal compressibility and ``mu_JT`` the Joule–Thomson coefficient. ``rho``, ``gamma``,
    ``cv_s`` and ``cp_s`` are carried for completeness and downstream reuse.
    """

    # conditions
    T: float
    P: float
    molar_mass: float
    # transport
    mu: float          # dynamic viscosity      [Pa·s]
    nu: float          # kinematic viscosity     [m^2/s]
    k: float           # thermal conductivity   [W/m/K]
    alpha: float       # thermal diffusivity     [m^2/s]
    D_self: float      # self-diffusion          [m^2/s]
    # dimensionless groups
    Pr: float
    Sc: float
    Le: float
    # thermophysical
    Z: float
    a: float           # speed of sound          [m/s]
    beta: float        # thermal expansion       [1/K]
    kappa_T: float     # isothermal compress.    [1/Pa]
    mu_JT: float       # Joule–Thomson           [K/Pa]
    # supporting (for reuse / export)
    rho: float         # mass density            [kg/m^3]
    gamma: float
    cv_s: float        # massic Cv               [J/kg/K]
    cp_s: float        # massic Cp               [J/kg/K]

    def as_dict(self) -> dict:
        """Flat dictionary view suitable for export."""
        return asdict(self)


class TransportCalculator:
    """Compute all transport & thermophysical properties of a molecule at a state.

    Parameters
    ----------
    molecule : Molecule
        The molecular species. Must carry :class:`~statthermopy.core.molecule.LennardJones`
        parameters (``molecule.lennard_jones``); a species without them cannot run the
        Chapman–Enskog transport layer.
    state : State
        The thermodynamic state (only ``T`` and ``P`` are used by the transport layer).

    Notes
    -----
    The heat capacities ``Cv_m``, ``Cp_m`` and ``γ`` are obtained from
    :class:`~statthermopy.thermodynamics.Thermodynamics` at the same state, so the transport
    coefficients inherit the full statistical-mechanics temperature dependence (including the
    quantum vibrational/electronic excitation and, when enabled, the quantum rotor).
    """

    def __init__(self, molecule: Molecule, state: State) -> None:
        if molecule.lennard_jones is None:
            raise ValueError(
                f"{molecule.name} has no Lennard–Jones parameters; transport properties "
                "require a `lennard_jones` entry in its database record."
            )
        self.molecule = molecule
        self.state = state

    def _resolved(self) -> ResolvedState:
        return self.state.resolve(self.molecule.molar_mass)

    # -- primary Chapman–Enskog coefficients -----------------------------------

    def viscosity(self, T: float) -> float:
        """Dynamic viscosity ``μ(T)`` (Pa·s). Pressure-independent (dilute gas)."""
        if T <= 0.0:
            return 0.0
        lj = self.molecule.lennard_jones
        m = self.molecule.molecular_mass
        sigma = lj.sigma_m
        Ts = t_star(T, lj.epsilon_over_k)
        omega = omega_22(Ts)
        return (5.0 / 16.0) * math.sqrt(m * k_B * T / math.pi) / (sigma * sigma * omega)

    def conductivity(self, T: float, mu: float | None = None) -> float:
        """Thermal conductivity ``k(T)`` (W/m·K) via the Eucken correlation."""
        if T <= 0.0:
            return 0.0
        if mu is None:
            mu = self.viscosity(T)
        th = Thermodynamics(self.molecule, State(T=float(T), P=float(self._resolved().P))).compute()
        cv_s = th.Cv_m / th.molar_mass
        gamma = th.gamma
        return mu * cv_s * (9.0 * gamma - 5.0) / 4.0

    def self_diffusion_coeff(self, T: float, P: float) -> float:
        """Self-diffusion coefficient ``D_self = D_ii`` (m²/s)."""
        if T <= 0.0 or P <= 0.0:
            return 0.0
        lj = self.molecule.lennard_jones
        m = self.molecule.molecular_mass
        sigma = lj.sigma_m
        Ts = t_star(T, lj.epsilon_over_k)
        omega = omega_11(Ts)
        m_ii = 0.5 * m  # reduced mass of an identical pair
        return (3.0 / 16.0) * (k_B * T / P) * (1.0 / (sigma * sigma * omega)) \
            * math.sqrt(2.0 * k_B * T / (math.pi * m_ii))

    # -- full report ----------------------------------------------------------

    def compute(self) -> TransportProperties:
        """Evaluate every property and return a :class:`TransportProperties`."""
        rs = self._resolved()
        T, P = rs.T, rs.P
        M = self.molecule.molar_mass
        lj = self.molecule.lennard_jones
        Ts = t_star(T, lj.epsilon_over_k) if T > 0.0 else 0.0

        # heat capacities / gamma from the statistical-mechanics engine (one evaluation)
        th = Thermodynamics(self.molecule, self.state).compute()
        Cv_m, Cp_m, gamma = th.Cv_m, th.Cp_m, th.gamma
        cv_s = Cv_m / M
        cp_s = Cp_m / M
        R_specific = R / M

        # primary transport coefficients
        mu = self.viscosity(T)
        k_th = self.conductivity(T, mu=mu) if T > 0.0 else 0.0
        D_self = self.self_diffusion_coeff(T, P)

        # density and the transport "per-density" derivatives
        if T > 0.0 and P > 0.0:
            rho = P * M / (R * T)
            nu = mu / rho if rho > 0.0 else 0.0
            alpha = k_th / (rho * cp_s) if (rho > 0.0 and cp_s > 0.0) else 0.0
        else:
            rho = 0.0
            nu = 0.0
            alpha = 0.0

        # dimensionless groups — closed forms, finite at every T (including T = 0)
        Pr = 4.0 * gamma / (9.0 * gamma - 5.0)
        # Sc = ν/D_self = (5/6) Ω11/Ω22 for self-diffusion (exact; avoids 0/0 at T = 0)
        o11 = omega_11(Ts) if T > 0.0 else omega_11(0.0)
        o22 = omega_22(Ts) if T > 0.0 else omega_22(0.0)
        Sc = (5.0 / 6.0) * (o11 / o22)
        Le = Sc / Pr if Pr != 0.0 else 0.0

        # thermophysical (ideal-gas EOS — exact for this engine)
        Z = 1.0
        a = math.sqrt(gamma * R_specific * T) if T > 0.0 else 0.0
        beta = (1.0 / T) if T > 0.0 else 0.0          # ideal gas; clamped at T = 0
        kappa_T = (1.0 / P) if P > 0.0 else 0.0        # ideal gas
        mu_JT = 0.0                                     # ideal gas (isenthalpic)

        return TransportProperties(
            T=T, P=P, molar_mass=M,
            mu=mu, nu=nu, k=k_th, alpha=alpha, D_self=D_self,
            Pr=Pr, Sc=Sc, Le=Le,
            Z=Z, a=a, beta=beta, kappa_T=kappa_T, mu_JT=mu_JT,
            rho=rho, gamma=gamma, cv_s=cv_s, cp_s=cp_s,
        )

    properties = compute  # convenience alias, mirroring Thermodynamics

    # -- vectorised helpers ----------------------------------------------------

    def property_vs_T(self, prop: str, T_range, P: float | None = None) -> tuple:
        """Evaluate a property over a temperature range at constant pressure.

        Returns ``(Ts, values)``. ``prop`` is any attribute of :class:`TransportProperties`.
        """
        P = P if P is not None else self._resolved().P
        Ts = list(T_range)
        out = []
        for T in Ts:
            st = State(T=float(T), P=float(P))
            out.append(getattr(TransportCalculator(self.molecule, st).compute(), prop))
        return Ts, out

    def property_vs_P(self, prop: str, P_range, T: float) -> tuple:
        """Evaluate a property over a pressure range at constant temperature.

        Returns ``(Ps, values)``. ``prop`` is any attribute of :class:`TransportProperties`.
        """
        Ps = list(P_range)
        out = []
        for P in Ps:
            st = State(T=float(T), P=float(P))
            out.append(getattr(TransportCalculator(self.molecule, st).compute(), prop))
        return Ps, out


# -- standalone binary diffusion ----------------------------------------------


def binary_diffusion(mol_i: Molecule, mol_j: Molecule, T: float, P: float) -> float:
    """Binary diffusion coefficient ``D_ij(T, P)`` (m²/s) for a pair of species.

    Uses the Lorentz–Berthelot combining rules (see :mod:`statthermopy.transport.lennard_jones`)
    and the ``Ω^(1,1)*`` collision integral. ``D_ij == D_ji`` by construction.

    Raises
    ------
    ValueError
        If either species lacks Lennard–Jones parameters.
    """
    if mol_i.lennard_jones is None or mol_j.lennard_jones is None:
        raise ValueError(
            f"binary diffusion requires LJ parameters for both {mol_i.name} and {mol_j.name}."
        )
    if T <= 0.0 or P <= 0.0:
        return 0.0
    sigma_ij = pair_sigma_m(mol_i.lennard_jones, mol_j.lennard_jones)
    eps_k_ij = pair_epsilon_over_k(mol_i.lennard_jones, mol_j.lennard_jones)
    m_ij = reduced_mass(mol_i, mol_j)
    Ts = t_star(T, eps_k_ij)
    omega = omega_11(Ts)
    return (3.0 / 16.0) * (k_B * T / P) * (1.0 / (sigma_ij * sigma_ij * omega)) \
        * math.sqrt(2.0 * k_B * T / (math.pi * m_ij))


def self_diffusion(molecule: Molecule, T: float, P: float) -> float:
    """Self-diffusion coefficient ``D_ii(T, P)`` (m²/s) for a single species (convenience)."""
    return TransportCalculator(molecule, State(T=float(T), P=float(P))).self_diffusion_coeff(T, P)
