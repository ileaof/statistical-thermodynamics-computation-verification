"""The :class:`HumidAirState` result container for a moist-air condition."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

__all__ = ["HumidAirState"]


@dataclass
class HumidAirState:
    """A complete moist-air state: saturation limit, psychrometrics and mixture thermodynamics.

    ``*_max`` fields are the **saturation** (maximum water-holding) values — the headline result:
    the most water vapour the air can carry at ``(T, P)`` before condensation. The remaining
    fields describe the actual state at the requested humidity.
    """

    # --- conditions ---
    T: float                       # K
    P: float                       # Pa (total)
    saturated: bool                # whether the actual state is at the saturation limit
    liquid_model: str              # liquid reference used for the saturation calculation

    # --- saturation limit (maximum water content) ---
    P_sat: float                   # saturation vapour pressure of water (Pa)
    x_h2o_max: float               # maximum H2O mole fraction (-)
    mass_fraction_h2o_max: float   # maximum H2O mass fraction (-)
    humidity_ratio_max: float      # maximum humidity ratio w_s (kg vapour / kg dry air)
    absolute_humidity_max: float   # maximum vapour mass density (kg/m^3)
    vapor_concentration_max: float # maximum vapour molar concentration (mol/m^3)

    # --- actual moist-air state ---
    P_vapor: float                 # actual water partial pressure (Pa)
    x_h2o: float                   # actual H2O mole fraction (-)
    mass_fraction_h2o: float       # actual H2O mass fraction (-)
    humidity_ratio: float          # actual humidity ratio w (kg vapour / kg dry air)
    absolute_humidity: float       # actual vapour mass density (kg/m^3)
    vapor_concentration: float     # actual vapour molar concentration (mol/m^3)
    relative_humidity: float       # P_vapor / P_sat (-)
    degree_of_saturation: float    # w / w_s (-)
    dew_point: float               # K
    wet_bulb: float                # K (adiabatic-saturation temperature)

    # --- mixture bulk properties ---
    density: float                 # moist-air mass density (kg/m^3)
    M_avg: float                   # average molar mass (kg/mol)
    R_specific: float              # specific gas constant of the mixture (J/kg/K)

    # --- molar thermodynamic properties (per mol of mixture) ---
    U_m: float
    H_m: float
    S_m: float
    A_m: float
    G_m: float
    Cv_m: float
    Cp_m: float
    gamma: float
    mu_m: float
    S_mixing: float                # entropy of mixing (J/mol/K)

    # --- massic thermodynamic properties (per kg of mixture) ---
    U_s: float
    H_s: float
    S_s: float
    A_s: float
    G_s: float
    Cv_s: float
    Cp_s: float

    # --- breakdowns ---
    components: dict = field(default_factory=dict)             # per-species contributions (mixture)
    vapor_mode_contributions: dict = field(default_factory=dict)  # per partition-function factor

    def as_dict(self) -> dict:
        return asdict(self)
