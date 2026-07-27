"""Cycle analysis primitives: points, results, and shared helpers.

A thermodynamic cycle is represented as an ordered list of
:class:`CyclePoint` (each wrapping a :class:`~thermolab.state.State`) plus
aggregate quantities (heat, work, efficiency). Each cycle module builds the
states with real-fluid flashes wherever the working fluid is supported by the
backend, so efficiencies reflect real gas behavior rather than the cold
air-standard assumption (unless the fluid is unavailable, in which case an
ideal-gas fallback is used).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..exceptions import ThermoLabError


@dataclass
class CyclePoint:
    """A labelled state on a cycle."""

    label: str
    state: Optional[object] = None  # thermolab.state.State
    note: str = ""

    @property
    def h(self) -> float:
        return self.state.h if self.state is not None else float("nan")

    @property
    def s(self) -> float:
        return self.state.s if self.state is not None else float("nan")

    @property
    def P(self) -> float:
        return self.state.P if self.state is not None else float("nan")

    @property
    def T(self) -> float:
        return self.state.T if self.state is not None else float("nan")


@dataclass
class CycleResult:
    """Result of a thermodynamic cycle calculation."""

    name: str
    points: list[CyclePoint] = field(default_factory=list)
    # per-process heat and work [J/kg], keyed by "1->2" etc.
    q: dict[str, float] = field(default_factory=dict)
    w: dict[str, float] = field(default_factory=dict)
    eta: float = float("nan")
    cop: float = float("nan")
    net_work: float = float("nan")
    back_work_ratio: float = float("nan")
    notes: str = ""

    def __repr__(self) -> str:
        lines = [f"Cycle: {self.name}"]
        for pt in self.points:
            if pt.state is not None:
                lines.append(
                    f"  {pt.label}: T={pt.T:10.2f} K  P={pt.P:11.3g} Pa  "
                    f"h={pt.h:12.3g} J/kg  s={pt.s:10.3g} J/(kg.K)"
                )
            else:
                lines.append(f"  {pt.label}: (no state)")
        if self.q:
            lines.append("  Heat [J/kg]: " + ", ".join(f"{k}={v:.3g}" for k, v in self.q.items()))
        if self.w:
            lines.append("  Work [J/kg]: " + ", ".join(f"{k}={v:.3g}" for k, v in self.w.items()))
        if not (self.eta != self.eta):  # not NaN
            lines.append(f"  Thermal efficiency: {self.eta:.4%}")
        if not (self.cop != self.cop):
            lines.append(f"  COP: {self.cop:.4f}")
        if not (self.net_work != self.net_work):
            lines.append(f"  Net work: {self.net_work:.3g} J/kg")
        if not (self.back_work_ratio != self.back_work_ratio):
            lines.append(f"  Back work ratio: {self.back_work_ratio:.4f}")
        return "\n".join(lines)

    def plot(self, *, diagram="ts", ax=None):
        from .. import plotting
        return plotting.plot_cycle(self, diagram=diagram, ax=ax)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def isentropic_state(fluid, *, P, s):
    """State at pressure ``P`` with the same entropy as the given value ``s``."""
    return fluid.state(P=P, s=s)


def sat_liquid(fluid, P):
    """Saturated-liquid state at pressure ``P`` (pure fluid)."""
    sat = fluid.backend.saturation_state(fluid.backend.saturation_temperature(P), fluid.fractions)
    hf = sat.h_f / fluid.molar_mass
    return fluid.state(P=P, h=hf)


def sat_vapor(fluid, P):
    """Saturated-vapor state at pressure ``P`` (pure fluid)."""
    sat = fluid.backend.saturation_state(fluid.backend.saturation_temperature(P), fluid.fractions)
    hg = sat.h_g / fluid.molar_mass
    return fluid.state(P=P, h=hg)


def _dh(s_lo, s_hi):
    return s_hi.h - s_lo.h


class CycleError(ThermoLabError):
    """Raised when a cycle cannot be constructed (e.g. bad parameters)."""