"""Ideal-gas mixtures.

An ideal-gas mixture is treated as independent components each occupying the full volume at its
partial pressure ``P_i = x_i P``. There is no enthalpy of mixing; the entropy of mixing is the
classical ``ΔS_mix = -R Σ x_i ln x_i``.

Given mole fractions ``x_i`` (converted from mass fractions when needed):

    M̅     = Σ x_i M_i
    U_m    = Σ x_i U_m,i(T)
    H_m    = Σ x_i H_m,i(T)
    Cv_m   = Σ x_i Cv_m,i(T)
    Cp_m   = Σ x_i Cp_m,i(T)
    S_m    = Σ x_i S_m,i(T, P_i)         (P_i = x_i P)
    G_m    = Σ x_i μ_i(T, P_i) = Σ x_i [G_m,i(T, P) + R T ln x_i]
    A_m    = U_m - T S_m  =  G_m - R T
    γ      = Cp_m / Cv_m
    R_spec = R / M̅
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .constants import R
from .core.molecule import Molecule
from .core.state import State
from .thermodynamics import Thermodynamics

__all__ = ["IdealGasMixture", "MixtureProperties", "ComponentContribution"]


@dataclass
class ComponentContribution:
    """One species' share of an ideal-gas mixture's properties.

    The ``*_m`` fields are the pure component's own molar properties — energetic ones
    (``U_m``/``H_m``/``Cv_m``/``Cp_m``) at the mixture pressure ``P`` (pressure-independent for an
    ideal gas), the pressure-dependent ones (``S_m``/``G_m``/``mu_m``) at the component's partial
    pressure ``P_i = x_i P``. The ``*_contrib`` fields are the mole-fraction-weighted amounts
    ``x_i · value`` that this component adds to the mixture total (they sum, over components, to
    the mixture's molar property).
    """

    name: str
    x: float             # mole fraction
    molar_mass: float    # kg/mol
    # pure-component molar properties
    U_m: float
    H_m: float
    S_m: float
    A_m: float
    G_m: float
    Cv_m: float
    Cp_m: float
    mu_m: float
    # weighted contributions to the mixture molar totals (sum to the mixture value)
    U_contrib: float
    H_contrib: float
    S_contrib: float
    A_contrib: float
    G_contrib: float
    Cv_contrib: float
    Cp_contrib: float


@dataclass
class MixtureProperties:
    """Thermodynamic report for an ideal-gas mixture (molar + massic)."""

    T: float
    P: float
    basis: str
    x: dict          # mole fractions
    M_avg: float     # kg/mol
    # molar
    U_m: float
    H_m: float
    S_m: float
    A_m: float
    G_m: float
    Cv_m: float
    Cp_m: float
    gamma: float
    T_v: float       # constant-volume thermal field, U_m / Cv_m (K)
    T_p: float       # constant-pressure thermal field, H_m / Cp_m (K)
    mu_m: float
    R_molar: float
    S_mixing: float  # entropy of mixing, -R Σ x_i ln x_i (J/mol/K)
    # massic
    U_s: float
    H_s: float
    S_s: float
    A_s: float
    G_s: float
    Cv_s: float
    Cp_s: float
    R_specific: float
    # per-component breakdown (species name -> ComponentContribution)
    components: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


class IdealGasMixture:
    """Ideal-gas mixture of pure species.

    Parameters
    ----------
    components : dict[Molecule, float]
        Mapping of molecule -> fraction (mole or mass, summing to 1).
    basis : str, default "mole"
        ``"mole"`` or ``"mass"``.
    """

    def __init__(self, components: dict[Molecule, float], *, basis: str = "mole") -> None:
        if basis not in ("mole", "mass"):
            raise ValueError("basis must be 'mole' or 'mass'.")
        if not components:
            raise ValueError("At least one component is required.")
        self.basis = basis
        # normalize and store
        total = sum(components.values())
        if total <= 0:
            raise ValueError("Fractions must sum to a positive number.")
        self._input = {mol: v / total for mol, v in components.items()}
        self.x = self._mole_fractions()

    # -- construction helpers --------------------------------------------------

    @classmethod
    def from_names(cls, fractions: dict[str, float], *, basis: str = "mole") -> "IdealGasMixture":
        """Build a mixture from molecule names resolved through the database."""
        from .database import get
        comps = {get(name): float(f) for name, f in fractions.items()}
        return cls(comps, basis=basis)

    def _mole_fractions(self) -> dict[Molecule, float]:
        if self.basis == "mole":
            total = sum(self._input.values())
            return {mol: v / total for mol, v in self._input.items()}
        # mass -> mole
        w_over_M = {mol: (w / mol.molar_mass) for mol, w in self._input.items()}
        norm = sum(w_over_M.values())
        return {mol: v / norm for mol, v in w_over_M.items()}

    # -- properties -----------------------------------------------------------

    @property
    def M_avg(self) -> float:
        """Average molar mass (kg/mol)."""
        return sum(x * mol.molar_mass for mol, x in self.x.items())

    def compute(self, state: State) -> MixtureProperties:
        """Evaluate mixture properties at ``state``."""
        rs = state.resolve(self.M_avg)
        T = rs.T
        P = rs.P
        x = self.x

        U_m = H_m = Cv_m = Cp_m = S_m = G_m = 0.0
        S_mixing = 0.0
        components: dict[str, ComponentContribution] = {}
        for mol, xi in x.items():
            # component at its partial pressure P_i = x_i P
            Pi = xi * P
            p = Thermodynamics(mol, State(T=T, P=Pi)).compute()
            # U, H, Cv, Cp are pressure-independent for ideal gases -> evaluate at total P.
            pp = Thermodynamics(mol, State(T=T, P=P)).compute()
            U_m += xi * pp.U_m
            H_m += xi * pp.H_m
            Cv_m += xi * pp.Cv_m
            Cp_m += xi * pp.Cp_m
            # entropy / Gibbs at partial pressure already include the mixing shift
            S_m += xi * p.S_m
            G_m += xi * p.G_m
            if xi > 0.0:
                S_mixing += -R * xi * math.log(xi)
            # per-component Helmholtz mirrors the mixture definition A = U(T,P) - T S(T,P_i)
            A_i = pp.U_m - T * p.S_m
            components[mol.name] = ComponentContribution(
                name=mol.name, x=xi, molar_mass=mol.molar_mass,
                U_m=pp.U_m, H_m=pp.H_m, S_m=p.S_m, A_m=A_i, G_m=p.G_m,
                Cv_m=pp.Cv_m, Cp_m=pp.Cp_m, mu_m=p.mu_m,
                U_contrib=xi * pp.U_m, H_contrib=xi * pp.H_m, S_contrib=xi * p.S_m,
                A_contrib=xi * A_i, G_contrib=xi * p.G_m,
                Cv_contrib=xi * pp.Cv_m, Cp_contrib=xi * pp.Cp_m,
            )

        A_m = U_m - T * S_m
        gamma = Cp_m / Cv_m
        # Thermal fields: characteristic temperatures U_m/Cv_m and H_m/Cp_m (K).
        T_v = U_m / Cv_m
        T_p = H_m / Cp_m
        mu_m = G_m
        M_avg = self.M_avg

        return MixtureProperties(
            T=T, P=P, basis=self.basis, x={mol.name: xi for mol, xi in x.items()},
            M_avg=M_avg,
            U_m=U_m, H_m=H_m, S_m=S_m, A_m=A_m, G_m=G_m,
            Cv_m=Cv_m, Cp_m=Cp_m, gamma=gamma, T_v=T_v, T_p=T_p, mu_m=mu_m, R_molar=R,
            S_mixing=S_mixing,
            U_s=U_m / M_avg, H_s=H_m / M_avg, S_s=S_m / M_avg, A_s=A_m / M_avg,
            G_s=G_m / M_avg, Cv_s=Cv_m / M_avg, Cp_s=Cp_m / M_avg, R_specific=R / M_avg,
            components=components,
        )

    properties = compute

    def __repr__(self) -> str:
        comp = ", ".join(f"{mol.name}={x:.4f}" for mol, x in self.x.items())
        return f"IdealGasMixture({comp})"