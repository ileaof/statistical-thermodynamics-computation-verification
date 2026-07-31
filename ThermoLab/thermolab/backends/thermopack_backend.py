"""ThermoPack backend.

Wraps the installed ThermoPack library (multiparameter GERG2008 / MEOS and cubic
SRK) behind :class:`BaseBackend`.

Design constraints (verified against ThermoPack 2.2.x, see the project plan):

* ThermoPack returns **molar** quantities; this backend stays in molar units and
  :class:`~thermolab.state.State` converts to mass-based.
* A *failed* ``init`` calls Fortran ``STOP`` and aborts the whole interpreter,
  so the EOS is chosen **upfront from a classification table** — we never
  attempt an init that could fail. Fluids absent from the database raise
  :class:`UnsupportedFluidError` without touching the engine.
* Component names are case-insensitive; components are passed to ThermoPack as a
  comma-joined string.
"""

from __future__ import annotations

import numpy as np

from .._fluid_db import normalize_name
from ..exceptions import UnsupportedFluidError
from ..units import R_GAS
from .base import BaseBackend, MolarProperties, Phase, SaturationState

# ---------------------------------------------------------------------------
# Component-name translation.
# ---------------------------------------------------------------------------
# ThermoPack identifies light alkanes by carbon number (methane is "C1", not
# "CH4"/"METHANE"), which differs from the formula/name a caller is likely to
# type. Map the common aliases onto ThermoPack's own identifier so that
# ``Gas("CH4")`` and ``Gas("methane")`` both resolve correctly.
_TP_COMPONENT_ALIASES: dict[str, str] = {
    "CH4": "C1",
    "METHANE": "C1",
}


def _to_tp_id(name: str) -> str:
    """Return the ThermoPack component identifier for a user-facing name."""
    n = normalize_name(name)
    return _TP_COMPONENT_ALIASES.get(n, n)

# ---------------------------------------------------------------------------
# Fluid classification (avoids any failing init -> process abort).
# ---------------------------------------------------------------------------

# Fluids with fitted multiparameter (MEOS) parameters in this ThermoPack build.
_MEOS_OK: frozenset[str] = frozenset({
    "H2O", "CO2", "N2", "O2", "AR", "H2", "HE", "NH3",
    "R134A", "R32", "R143A", "R1234YF", "R1234ZE", "R12", "R14", "R23", "R116",
    "BENZENE", "CO", "N2O", "SO2", "H2S", "KR", "XE", "NE",
})

# Fluids supported only via cubic EOS in this build (no MEOS parameters).
_CUBIC_ONLY: frozenset[str] = frozenset({"R125", "DME"})

# Fluids supported only via the GERG2008 multiparameter model. The default
# NIST_MEOS reference has no parameters for these (a MEOS init aborts the
# interpreter), but GERG2008 does — e.g. methane as "C1".
_GERG_ONLY: frozenset[str] = frozenset({"C1"})

# Everything the database knows about (MEOS, GERG2008, or cubic).
_SUPPORTED: frozenset[str] = _MEOS_OK | _CUBIC_ONLY | _GERG_ONLY

# Components eligible for the GERG2008 mixture model in this build.
_GERG_CORE: frozenset[str] = frozenset({
    "N2", "O2", "AR", "CO2", "H2O", "CO", "H2", "H2S", "HE", "C1",
})

# Phase label -> ThermoPack integer flag mapping (filled at runtime).
_VAPOR = 2
_LIQUID = 1


def _phase_flag(phase: Phase | str) -> int:
    if isinstance(phase, Phase):
        phase = phase.value
    p = str(phase).lower()
    if p in ("vapor", "vap", "gas", "v"):
        return _VAPOR
    if p in ("liquid", "liq", "l"):
        return _LIQUID
    raise ValueError(f"Unsupported single-phase label for property evaluation: {phase!r}")


class ThermoPackBackend(BaseBackend):
    """ThermoPack-backed thermodynamic engine."""

    name = "thermopack"

    def __init__(
        self,
        components: list[str],
        *,
        reference_state: str = "DEFAULT",
        eos: str | None = None,
    ) -> None:
        super().__init__(components, reference_state=reference_state, eos=eos)

        # Translate user-facing names to ThermoPack component identifiers
        # (e.g. methane "CH4" -> "C1") before any lookup or engine call.
        tp_ids = [_to_tp_id(c) for c in self.components]

        # Validate every component is in the database (reject upfront).
        missing = [c for c in tp_ids if c not in _SUPPORTED]
        if missing:
            raise UnsupportedFluidError(
                ", ".join(missing),
                self.name,
                detail=(
                    "The installed ThermoPack build has no parameters for these "
                    "components. Use a backend that supports them (CoolProp/Cantera) "
                    "or restrict to fluids in thermolab.list_fluids()."
                ),
            )

        # Pick the EOS deterministically.
        if self.nc == 1:
            comp = tp_ids[0]
            if comp in _MEOS_OK:
                chosen = "MEOS"
            elif comp in _GERG_ONLY:
                chosen = "GERG2008"
            else:
                chosen = "SRK"
        else:
            if all(c in _GERG_CORE for c in tp_ids):
                chosen = "GERG2008"
            else:
                chosen = "SRK"

        # Allow explicit override (caller knows what they want).
        if eos is not None:
            chosen = eos.upper()

        self._eos_chosen = chosen
        self._comps_str = ",".join(tp_ids)
        self._engine = self._build_engine(chosen)

        # Phase flags from the live engine (authoritative).
        global _VAPOR, _LIQUID
        _VAPOR = int(self._engine.VAPPH)
        _LIQUID = int(self._engine.LIQPH)

        # Cache molar masses (kg/mol).
        mw_gmol = np.array(
            [self._engine.compmoleweight(i + 1) for i in range(self.nc)],
            dtype=float,
        )
        self._molar_masses = mw_gmol / 1000.0  # g/mol -> kg/mol

    # ------------------------------------------------------------------
    # Engine construction
    # ------------------------------------------------------------------
    def _build_engine(self, eos: str):
        """Build a fresh ThermoPack engine for the chosen EOS."""
        from thermopack import multiparameter as mp
        from thermopack import thermo

        if eos in ("GERG2008", "MEOS", "MBWR32", "MBWR19"):
            eng = mp.multiparam()
            eng.init(self._comps_str, eos, self.reference_state)
            return eng
        if eos in ("SRK", "PR", "PRSV", "CPA"):
            eng = thermo.thermo()
            alpha = "Classic"
            eng.init_thermo(eos, "Classic", alpha, self._comps_str, 2, silent=True)
            return eng
        raise UnsupportedFluidError(
            self._comps_str, self.name, detail=f"Unknown EOS {eos!r}."
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    def molar_masses(self) -> np.ndarray:
        return self._molar_masses.copy()

    def component_names_normalized(self) -> list[str]:
        """Return component names normalized for transport-coefficient lookup."""
        return [normalize_name(c) for c in self.components]

    def component_critical_temperatures(self) -> np.ndarray:
        """Per-component critical temperatures [K] (for transport fallback)."""
        eng = self._engine
        return np.array(
            [float(eng.critical_temperature(i + 1)) for i in range(self.nc)],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Phase handling
    # ------------------------------------------------------------------
    def guess_phase(self, T: float, P: float, z: np.ndarray) -> Phase:
        z = np.asarray(z, dtype=float)
        ph = int(self._engine.guess_phase(T, P, z))
        if ph == _VAPOR:
            return Phase.VAPOR
        if ph == _LIQUID:
            return Phase.LIQUID
        # Fallback: decide from density vs critical (rare for single-phase root).
        return Phase.VAPOR

    # ------------------------------------------------------------------
    # Single-phase properties
    # ------------------------------------------------------------------
    def specific_volume(self, T: float, P: float, z: np.ndarray, phase: Phase | str) -> float:
        z = np.asarray(z, dtype=float)
        return float(self._engine.specific_volume(T, P, z, _phase_flag(phase))[0])

    def pressure_at_volume(self, T: float, v_molar: float, z: np.ndarray) -> float:
        z = np.asarray(z, dtype=float)
        return float(self._engine.pressure_tv(T, v_molar, z)[0])

    def molar_properties(self, T: float, P: float, z: np.ndarray, phase: Phase | str) -> MolarProperties:
        z = np.asarray(z, dtype=float)
        ph = _phase_flag(phase)
        eng = self._engine

        # Molar volume + (dv/dT)|P, (dv/dP)|T  -> beta, kappa_t, JT
        v, dvdt, dvdp = eng.specific_volume(T, P, z, ph, dvdt=True, dvdp=True)

        # Enthalpy + (dh/dT)|P -> cp
        h, cp_mol = eng.enthalpy(T, P, z, ph, dhdt=True)[:2]

        # Entropy (+ ds/dT|P, available but not required here)
        s = eng.entropy(T, P, z, ph)[0]

        # cv via thermodynamic identity: cv = cp + T*(dP/dT|v)^2 / (dP/dv|T)
        # using the (T, v) property interface.
        P_calc, dpdt_v, dpdv_t = eng.pressure_tv(T, v, z, dpdt=True, dpdv=True)
        cv_mol = cp_mol + T * (dpdt_v ** 2) / dpdv_t

        # Speed of sound at constant entropy (m/s)
        w = float(eng.speed_of_sound_tv(T, v, z))

        # Compressibility factor
        Z = P_calc * v / (R_GAS * T)

        # Derived energies (molar)
        u = h - P_calc * v
        a_helm = u - T * s
        g_gibbs = h - T * s

        # Isobaric thermal expansion: beta = -(1/v)(dv/dT)|P
        beta = -dvdt / v
        # Isothermal compressibility: kappa_t = -(1/v)(dv/dP)|T
        kappa_t = -dvdp / v
        # Joule-Thomson: mu_JT = (1/cp)*(T*(dv/dT)|P - v)
        jt = (T * dvdt - v) / cp_mol if cp_mol != 0.0 else float("inf")

        return MolarProperties(
            v=v, h=h, s=s, u=u, a=a_helm, g=g_gibbs,
            cp=cp_mol, cv=cv_mol, Z=Z, w=w,
            beta=beta, kappa_t=kappa_t, jt=jt,
        )

    # ------------------------------------------------------------------
    # Saturation (pure fluids)
    # ------------------------------------------------------------------
    def _pure_z(self) -> np.ndarray:
        return np.ones(self.nc)

    def saturation_pressure(self, T: float, z: np.ndarray | None = None) -> float:
        z = self._pure_z() if z is None else np.asarray(z, dtype=float)
        # For a pure fluid bubble == dew == Psat.
        return float(self._engine.bubble_pressure(T, z)[0])

    def saturation_temperature(self, P: float, z: np.ndarray | None = None) -> float:
        z = self._pure_z() if z is None else np.asarray(z, dtype=float)
        return float(self._engine.bubble_temperature(P, z)[0])

    def saturation_state(self, T: float, z: np.ndarray | None = None) -> SaturationState:
        z = self._pure_z() if z is None else np.asarray(z, dtype=float)
        eng = self._engine
        P = float(eng.bubble_pressure(T, z)[0])
        vf = float(eng.specific_volume(T, P, z, _LIQUID)[0])
        vg = float(eng.specific_volume(T, P, z, _VAPOR)[0])
        hf = float(eng.enthalpy(T, P, z, _LIQUID)[0])
        hg = float(eng.enthalpy(T, P, z, _VAPOR)[0])
        sf = float(eng.entropy(T, P, z, _LIQUID)[0])
        sg = float(eng.entropy(T, P, z, _VAPOR)[0])
        return SaturationState(
            T=T, P=P,
            rho_f=1.0 / vf, rho_g=1.0 / vg,
            h_f=hf, h_g=hg, s_f=sf, s_g=sg,
        )

    def is_two_phase(self, T: float, P: float, z: np.ndarray) -> bool:
        """Return True if (T, P, z) lies inside the two-phase region.

        * **Pure fluid**: an independently specified (T, P) is two-phase only on
          the saturation line (``P == Psat(T)``); the dome interior is reached
          via an energy/entropy spec, handled by the flash solver. So this
          returns True only when ``P`` is essentially equal to ``Psat(T)``.
        * **Mixture**: the two-phase region is ``dew_P(T,z) < P < bubble_P(T,z)``.
        """
        z = np.asarray(z, dtype=float)
        # The critical-temperature guard is only an early-exit optimization; the
        # mixture critical-point solver (critical(z)) is flaky for some
        # compositions (e.g. H2O/CO2-rich), so a failure here must not crash the
        # flash — fall through to the dew/bubble bracket instead.
        try:
            Tc = self.critical_temperature(z)
        except Exception:
            Tc = np.inf
        if T >= Tc:
            return False

        if self.nc == 1:
            try:
                Psat = self.saturation_pressure(T, z)
            except Exception:
                return False
            if Psat <= 0:
                return False
            return abs(P - Psat) / Psat < 1e-4

        # Mixture: dew/bubble bracket.
        try:
            Pbub = float(self._engine.bubble_pressure(T, z)[0])
            Pdew = float(self._engine.dew_pressure(T, z)[0])
        except Exception:
            return False
        lo, hi = sorted((Pbub, Pdew))
        return lo < P < hi

    # ------------------------------------------------------------------
    # Critical point
    # ------------------------------------------------------------------
    def critical_temperature(self, z: np.ndarray | None = None) -> float:
        if self.nc == 1:
            return float(self._engine.critical_temperature(1))
        z = self._pure_z() if z is None else np.asarray(z, dtype=float)
        return float(self._engine.critical(z)[0])

    def critical_pressure(self, z: np.ndarray | None = None) -> float:
        if self.nc == 1:
            return float(self._engine.critical_pressure(1))
        z = self._pure_z() if z is None else np.asarray(z, dtype=float)
        return float(self._engine.critical(z)[2])