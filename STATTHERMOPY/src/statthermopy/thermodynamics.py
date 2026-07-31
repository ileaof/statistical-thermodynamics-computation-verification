"""Thermodynamic properties from the molecular partition function.

:class:`Thermodynamics` is the aggregator: it sums the per-mode :class:`Contribution` objects
and adds the two ideal-gas terms that belong to the whole gas rather than to any single mode —

* the ``P V`` work term, ``R T`` per mole, giving ``H = U + R T`` and ``G = A + R T``;
* the ``N!`` indistinguishability correction, already folded into the translational
  contribution.

It reports every requested property on a *molar* and a *massic* basis, plus the extensive
(total) amounts when the state specifies the amount of substance.

Molar relations (per mole, ``n = 1``):

    U_m  = Σ U_m,mode
    S_m  = Σ S_m,mode
    A_m  = Σ A_m,mode
    H_m  = U_m + R T
    G_m  = A_m + R T          (= μ_m for a pure ideal gas)
    Cv_m = Σ Cv_m,mode
    Cp_m = Cv_m + R
    γ    = Cp_m / Cv_m

Massic (per kg) quantities are the molar ones divided by the molar mass ``M`` (kg/mol); the
specific gas constant is ``R_specific = R / M``. ``γ`` is intensive and identical on both bases.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .constants import R
from .core.molecule import Molecule
from .core.state import ResolvedState, State
from .partition import PartitionFunction

__all__ = ["Thermodynamics", "ThermoProperties"]


@dataclass
class ThermoProperties:
    """Full thermodynamic report for one molecule at one state.

    The ``_m`` fields are molar (per mol), the ``_s`` fields are massic (per kg), and the bare
    fields are extensive totals (require a non-trivial amount of substance).
    """

    # --- conditions ---
    T: float
    P: float
    V: float
    n: float
    m: float
    molar_mass: float  # kg/mol

    # --- molar properties (per mol) ---
    U_m: float
    H_m: float
    S_m: float
    A_m: float
    G_m: float
    Cv_m: float
    Cp_m: float
    gamma: float
    mu_m: float
    R_molar: float

    # --- massic properties (per kg) ---
    U_s: float
    H_s: float
    S_s: float
    A_s: float
    G_s: float
    Cv_s: float
    Cp_s: float
    R_specific: float

    # --- extensive (total) properties ---
    U: float
    H: float
    S: float
    A: float
    G: float
    Cv: float
    Cp: float

    # --- partition function ---
    Qt: float
    Qr: float
    Qv: float
    Qe: float
    Qtotal: float
    ln_Qtotal: float

    # --- per-mode breakdown (molar) ---
    contributions: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Flat dictionary view (contributions nested) suitable for export."""
        d = asdict(self)
        return d


class Thermodynamics:
    """Compute all thermodynamic properties of a molecule at a state.

    Parameters
    ----------
    molecule : Molecule
        The molecular species.
    state : State
        The thermodynamic state.
    use_quantum_rotation : bool, default False
        Use the exact quantum rigid-rotor sum for linear molecules.
    """

    def __init__(
        self,
        molecule: Molecule,
        state: State,
        *,
        use_quantum_rotation: bool = False,
    ) -> None:
        self.molecule = molecule
        self.state = state
        self.partition = PartitionFunction(molecule, use_quantum_rotation=use_quantum_rotation)

    def _resolved(self) -> ResolvedState:
        return self.state.resolve(self.molecule.molar_mass)

    def compute(self) -> ThermoProperties:
        """Evaluate all properties and return a :class:`ThermoProperties`."""
        rs = self._resolved()
        contribs = self.partition.contributions(self.state)

        U_m = sum(c.U_m for c in contribs.values())
        S_m = sum(c.S_m for c in contribs.values())
        A_m = sum(c.A_m for c in contribs.values())
        Cv_m = sum(c.Cv_m for c in contribs.values())

        T = rs.T
        H_m = U_m + R * T
        G_m = A_m + R * T
        Cp_m = Cv_m + R
        gamma = Cp_m / Cv_m
        mu_m = G_m

        M = self.molecule.molar_mass
        # massic (per kg) = molar / M
        U_s = U_m / M
        H_s = H_m / M
        S_s = S_m / M
        A_s = A_m / M
        G_s = G_m / M
        Cv_s = Cv_m / M
        Cp_s = Cp_m / M
        R_specific = R / M

        # extensive (total)
        n = rs.n
        U = n * U_m
        H = n * H_m
        S = n * S_m
        A = n * A_m
        G = n * G_m
        Cv = n * Cv_m
        Cp = n * Cp_m

        pv = self.partition.evaluate(self.state)

        contrib_dict = {k: {
            "ln_q": c.ln_q, "d_ln_q_dT": c.d_ln_q_dT,
            "U_m": c.U_m, "S_m": c.S_m, "A_m": c.A_m, "Cv_m": c.Cv_m,
        } for k, c in contribs.items()}

        return ThermoProperties(
            T=T, P=rs.P, V=rs.V, n=n, m=rs.m, molar_mass=M,
            U_m=U_m, H_m=H_m, S_m=S_m, A_m=A_m, G_m=G_m,
            Cv_m=Cv_m, Cp_m=Cp_m, gamma=gamma, mu_m=mu_m, R_molar=R,
            U_s=U_s, H_s=H_s, S_s=S_s, A_s=A_s, G_s=G_s,
            Cv_s=Cv_s, Cp_s=Cp_s, R_specific=R_specific,
            U=U, H=H, S=S, A=A, G=G, Cv=Cv, Cp=Cp,
            Qt=pv.Qt, Qr=pv.Qr, Qv=pv.Qv, Qe=pv.Qe,
            Qtotal=pv.Qtotal, ln_Qtotal=pv.ln_Qtotal,
            contributions=contrib_dict,
        )

    # convenience alias
    properties = compute

    # -- vectorised helper for property-vs-T curves ----------------------------

    #: Attributes derivable from the backend's molar-property grid. Anything not in this set
    #: (notably ``"contributions"``, a nested dict) falls back to the per-T Python path.
    _GRID_DERIVABLE: frozenset[str] = frozenset({
        "T", "P", "V", "n", "m", "molar_mass",
        "U_m", "H_m", "S_m", "A_m", "G_m", "Cv_m", "Cp_m", "gamma", "mu_m", "R_molar",
        "U_s", "H_s", "S_s", "A_s", "G_s", "Cv_s", "Cp_s", "R_specific",
        "U", "H", "S", "A", "G", "Cv", "Cp",
        "Qt", "Qr", "Qv", "Qe", "Qtotal",
        "ln_Qt", "ln_Qr", "ln_Qv", "ln_Qe", "ln_Qtotal",
    })

    def _derive_property_array(self, prop, res, Ts, P):
        """Map a :class:`ThermoProperties` attribute to its array over ``Ts`` from the grid.

        The backend grid supplies the eight core molar arrays (``U_m``, ``S_m``, ``A_m``,
        ``Cv_m`` and the four ``ln_Q*``); everything else is derived here, identically to
        :meth:`compute`, so the fast path matches the per-T Python path exactly.
        """
        import numpy as np

        U_m = np.asarray(res["U_m"], dtype=float)
        S_m = np.asarray(res["S_m"], dtype=float)
        A_m = np.asarray(res["A_m"], dtype=float)
        Cv_m = np.asarray(res["Cv_m"], dtype=float)
        lnQt = np.asarray(res["ln_Qt"], dtype=float)
        lnQr = np.asarray(res["ln_Qr"], dtype=float)
        lnQv = np.asarray(res["ln_Qv"], dtype=float)
        lnQe = np.asarray(res["ln_Qe"], dtype=float)
        T = np.asarray(Ts, dtype=float)
        M = self.molecule.molar_mass

        H_m = U_m + R * T
        G_m = A_m + R * T
        Cp_m = Cv_m + R
        gamma = Cp_m / Cv_m
        lnQtotal = lnQt + lnQr + lnQv + lnQe
        V = R * T / P
        table = {
            "T": T, "P": np.full_like(T, P), "V": V, "n": np.ones_like(T),
            "m": np.full_like(T, M), "molar_mass": np.full_like(T, M),
            "U_m": U_m, "H_m": H_m, "S_m": S_m, "A_m": A_m, "G_m": G_m,
            "Cv_m": Cv_m, "Cp_m": Cp_m, "gamma": gamma, "mu_m": G_m, "R_molar": np.full_like(T, R),
            "U_s": U_m / M, "H_s": H_m / M, "S_s": S_m / M, "A_s": A_m / M, "G_s": G_m / M,
            "Cv_s": Cv_m / M, "Cp_s": Cp_m / M, "R_specific": np.full_like(T, R / M),
            # extensive totals with n = 1 mol (the default for property_vs_T): equal to molar
            "U": U_m, "H": H_m, "S": S_m, "A": A_m, "G": G_m, "Cv": Cv_m, "Cp": Cp_m,
            "Qt": np.exp(lnQt), "Qr": np.exp(lnQr), "Qv": np.exp(lnQv), "Qe": np.exp(lnQe),
            "Qtotal": np.exp(lnQtotal),
            "ln_Qt": lnQt, "ln_Qr": lnQr, "ln_Qv": lnQv, "ln_Qe": lnQe, "ln_Qtotal": lnQtotal,
        }
        return table[prop]

    def property_vs_T(self, prop: str, T_range, P: float | None = None) -> tuple:
        """Evaluate a property over a temperature range at constant pressure.

        Parameters
        ----------
        prop : str
            Attribute name of :class:`ThermoProperties` (e.g. ``"Cp_m"``, ``"Qtotal"``).
        T_range : array-like
            Temperatures (K).
        P : float, optional
            Pressure (Pa). Defaults to the state's pressure.

        Notes
        -----
        When the active backend provides a :meth:`~statthermopy.backend.Backend.
        molar_property_grid` kernel (Numba/OpenMP/CUDA), the grid is computed in one vectorised
        pass; otherwise each temperature is evaluated with a fresh :class:`Thermodynamics`, as in
        Phase 1. Both paths return identical values.
        """
        from .backend import get_backend

        P = P if P is not None else self._resolved().P
        Ts = list(T_range)
        res = get_backend().molar_property_grid(
            self.molecule, Ts, P, self.partition.rotational.use_quantum,
            self.partition.rotational.quantum_cutoff,
        )
        if res is not None and prop in self._GRID_DERIVABLE:
            return Ts, list(self._derive_property_array(prop, res, Ts, P))
        # fallback: per-T Python path (single source of truth for the NumPy backend)
        out = []
        for T in Ts:
            st = State(T=T, P=P)
            th = Thermodynamics(self.molecule, st,
                                use_quantum_rotation=self.partition.rotational.use_quantum)
            out.append(getattr(th.compute(), prop))
        return Ts, out