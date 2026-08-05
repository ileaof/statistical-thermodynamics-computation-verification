"""Statistical Humid Air — maximum water-vapour solubility from statistical thermodynamics.

The gas phase (dry air + water vapour) is described rigorously by the molecular partition function
(translational + rotational + vibrational + electronic) through the ideal-mixture relations; the
maximum water content the air can hold before condensing is set by the vapour–liquid equilibrium
``μ_v(T,P) = μ_l(T,P)``, with the vapour Gibbs energy taken purely from statistical mechanics and
the liquid from a pluggable reference model (IAPWS-95 when available, else a transparent
constant-``c_p`` reference), reconciled through the triple-point anchor. No empirical
vapour-pressure correlation (Antoine, Magnus, Tetens, …) is used.

Public API
----------
- :class:`HumidAir` — the model: ``.state(T, P, ...)`` → :class:`HumidAirState` with the full
  psychrometric and thermodynamic property set and the per-partition-function breakdown.
- :class:`SaturationCalculator` — the vapour–liquid saturation solver.
- :class:`LiquidWaterModel`, :class:`ConstantCpLiquid`, :class:`IAPWSLiquid` — liquid references.
- plotting helpers in :mod:`statthermopy.humidair.plots`.
"""

from __future__ import annotations

from .analysis import COMPARISON_PROPERTIES, ComparisonTable, PsychrometricAnalysis
from .humidair import HumidAir
from .liquid import (
    DH_VAP_TRIPLE,
    P_TRIPLE,
    T_TRIPLE,
    ConstantCpLiquid,
    IAPWSLiquid,
    LiquidWaterModel,
    default_liquid_model,
)
from .saturation import SaturationCalculator
from .state import HumidAirState

__all__ = [
    "HumidAir",
    "HumidAirState",
    "SaturationCalculator",
    "PsychrometricAnalysis",
    "ComparisonTable",
    "COMPARISON_PROPERTIES",
    "LiquidWaterModel",
    "ConstantCpLiquid",
    "IAPWSLiquid",
    "default_liquid_model",
    "T_TRIPLE",
    "P_TRIPLE",
    "DH_VAP_TRIPLE",
]
