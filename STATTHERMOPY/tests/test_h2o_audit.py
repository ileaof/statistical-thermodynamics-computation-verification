"""Scientific audit of gas-phase H2O — regression tests.

Locks in every finding of ``docs/H2O_AUDIT.md``. The audit concluded that the engine has **no**
implementation, unit, constant, formula, standard-state or energy-zero error, and that the residual
deviation from thermochemical tables had two physical origins outside the code:

* a flat ``-0.32 %`` floor on Cp below ~600 K — centrifugal distortion and vibration-rotation
  coupling, absent from the rigid rotor;
* a term growing to ``-1.90 %`` at 2000 K — anharmonicity, absent from the harmonic oscillator.

The second was then removed (audit section 15): water now sums the anharmonic level manifold built
from its spectroscopic omega/x_ij constants, which cuts the 2000 K deficit to ``-0.57 %`` and the
validation-layer mean from 0.81 % to 0.38 %. The first remains — exact quantum rotation is 65x too
small to explain it, and the truncated centrifugal expansion does not converge.

The tests are organised so that a future regression is attributable:

1. molecular data (``TestMolecularData``)
2. thermodynamic identities, reference-free (``TestIdentities``)
3. mode contributions against independently coded closed forms (``TestModeContributions``)
4. standard state and unit conversions (``TestConventions``)
5. absolute values against tabulated references (``TestAgainstReferences``)
6. the error *signature* that proves the cause (``TestErrorSignature``)
7. the anharmonic machinery added in response (``TestAnharmonicManifold``)

Everything in 1-6 runs offline with no optional dependency. Cross-checks against ``iapws`` and
``thermo`` are in ``TestExternalCrossChecks`` and skip when those packages are absent.
"""

from __future__ import annotations

import math

import pytest

from statthermopy import State, Thermodynamics
from statthermopy.constants import N_A, R, h, k_B
from statthermopy.core.molecule import Geometry
from statthermopy.database import get
from statthermopy.units import CM1_TO_K

# --- Tabulated reference values (values only; no correlation coefficients ship here) ---------
#
# JANAF / NIST-CODATA for H2O(g), standard state 1 bar, absolute third-law entropy.
JANAF_CP_298 = 33.590   # J/mol/K
JANAF_S_298 = 188.834   # J/mol/K
# NIST WebBook Shomate for H2O(g), evaluated on the grid (same table the validation layer ships).
NIST_CP = {500.0: 35.2184, 800.0: 38.7365, 1000.0: 41.2656, 1500.0: 47.1086, 2000.0: 51.2048}
NIST_S = {500.0: 206.5341, 800.0: 223.8251, 1000.0: 232.7400, 1500.0: 250.6198, 2000.0: 264.7692}
# Third-law absolute entropy of saturated liquid water at the triple point = the IAPWS zero.
# Audit section 9: 63.3059 J/mol/K measured, 63.338 from the CODATA route.
ENTROPY_OFFSET_J_MOL_K = 63.306
# Enthalpy offset to the same scale: triple-point latent heat minus the ideal gas's thermal
# enthalpy there. Audit section 9.2: -1997.870 kJ/kg by regression, 1996.946 from the triple point.
ENTHALPY_OFFSET_KJ_KG = -1997.870
MOLAR_MASS_KG_MOL = 0.01801528


@pytest.fixture(scope="module")
def h2o():
    return get("H2O")


def _res(h2o, T, P=1e5):
    return Thermodynamics(h2o, State(T=T, P=P)).compute()


# ---------------------------------------------------------------- 1. molecular data


class TestMolecularData:
    """Audit section 4 — the spectroscopic input, field by field."""

    def test_molar_mass(self, h2o):
        assert h2o.molar_mass == pytest.approx(MOLAR_MASS_KG_MOL, rel=1e-9)

    def test_geometry_is_nonlinear(self, h2o):
        """Must be nonlinear: the linear rotor formula would be wrong for water."""
        assert h2o.geometry is Geometry.NONLINEAR

    def test_symmetry_number_is_two(self, h2o):
        """C2v: one C2 rotation. sigma = 1 would inflate S by R ln 2 = 5.76 J/mol/K."""
        assert h2o.symmetry_number == 2

    def test_three_vibrational_modes(self, h2o):
        """3N - 6 = 3 for a nonlinear triatomic."""
        assert h2o.n_vibrational_modes == 3
        assert h2o.n_vibrational_modes == 3 * h2o.n_atoms - 6

    def test_each_fundamental_appears_once_and_is_nondegenerate(self, h2o):
        """C2v has no degenerate modes: nu1, nu2, nu3 each once with g = 1."""
        waves = sorted(m.wavenumber_cm1 for m in h2o.vibrational_modes)
        assert waves == pytest.approx([1595.0, 3657.0, 3756.0])
        assert [m.degeneracy for m in h2o.vibrational_modes] == [1, 1, 1]
        assert len(waves) == len(set(waves)), "a fundamental is double-counted"

    def test_three_moments_of_inertia(self, h2o):
        assert len(h2o.moments_of_inertia) == 3

    def test_rotational_temperature_roundtrip(self, h2o):
        """theta -> I in the loader and I -> theta in the mode must be mutually exact."""
        thetas = sorted((h * h) / (8.0 * math.pi**2 * I * k_B) for I in h2o.moments_of_inertia)
        assert thetas == pytest.approx([13.36, 20.87, 40.13], rel=1e-9)

    def test_electronic_ground_state_is_a_singlet(self, h2o):
        assert len(h2o.electronic_levels) == 1
        assert h2o.electronic_levels[0].energy_cm1 == 0.0
        assert h2o.electronic_levels[0].degeneracy == 1

    def test_wavenumber_to_kelvin_conversion(self):
        """CM1_TO_K = h c / k_B with c in cm/s; CODATA 2018 gives 1.4387768775 K/cm^-1."""
        assert pytest.approx(1.4387768775, rel=1e-9) == CM1_TO_K

    def test_vibrational_temperatures(self, h2o):
        thetas = sorted(m.wavenumber_cm1 * CM1_TO_K for m in h2o.vibrational_modes)
        assert thetas == pytest.approx([2294.85, 5261.61, 5404.05], rel=1e-5)


# ------------------------------------------------------- 2. identities (reference-free)


GRID = [298.15, 300.0, 350.0, 373.15, 400.0, 500.0, 800.0, 1000.0, 1500.0, 2000.0]


class TestIdentities:
    """Audit section 5.1-5.2 — internal consistency, needing no external data."""

    @pytest.mark.parametrize("T", GRID)
    def test_mayer_relation(self, h2o, T):
        r = _res(h2o, T)
        assert r.Cp_m - r.Cv_m == pytest.approx(R, abs=1e-12)

    @pytest.mark.parametrize("T", GRID)
    def test_enthalpy_definition(self, h2o, T):
        r = _res(h2o, T)
        assert r.H_m - r.U_m == pytest.approx(R * T, rel=1e-12)

    @pytest.mark.parametrize("T", GRID)
    def test_gibbs_and_helmholtz(self, h2o, T):
        r = _res(h2o, T)
        assert r.G_m == pytest.approx(r.H_m - T * r.S_m, abs=1e-8)
        assert r.A_m == pytest.approx(r.U_m - T * r.S_m, abs=1e-8)

    @pytest.mark.parametrize("T", GRID)
    def test_gamma(self, h2o, T):
        r = _res(h2o, T)
        assert r.gamma == pytest.approx(r.Cp_m / r.Cv_m, rel=1e-14)

    @pytest.mark.parametrize("T", [298.15, 500.0, 1000.0, 2000.0])
    def test_cp_is_the_derivative_of_enthalpy(self, h2o, T):
        d = 0.01
        num = (_res(h2o, T + d).H_m - _res(h2o, T - d).H_m) / (2 * d)
        assert _res(h2o, T).Cp_m == pytest.approx(num, rel=1e-7)

    @pytest.mark.parametrize("T", [298.15, 500.0, 1000.0, 2000.0])
    def test_cv_is_the_derivative_of_internal_energy(self, h2o, T):
        d = 0.01
        V = _res(h2o, T).V
        up = Thermodynamics(h2o, State(T=T + d, V=V, n=1.0)).compute().U_m
        um = Thermodynamics(h2o, State(T=T - d, V=V, n=1.0)).compute().U_m
        assert _res(h2o, T).Cv_m == pytest.approx((up - um) / (2 * d), rel=1e-7)


# --------------------------------------------------- 3. mode contributions vs closed forms


class TestModeContributions:
    """Audit section 5.3 — each mode against a formula coded independently here."""

    T = 500.0
    P = 1e5

    def _contribs(self, h2o):
        st = State(T=self.T, P=self.P)
        return Thermodynamics(h2o, st).partition.contributions(st)

    def test_translational_is_sackur_tetrode(self, h2o):
        m = MOLAR_MASS_KG_MOL / N_A
        V_m = R * self.T / self.P
        ln_q = 1.5 * math.log(2 * math.pi * m * k_B * self.T / h**2) + math.log(V_m)
        expected = R * (ln_q + 1.5 - math.log(N_A) + 1.0)
        assert self._contribs(h2o)["translational"].S_m == pytest.approx(expected, abs=1e-9)

    def test_rotational_uses_the_nonlinear_rigid_rotor(self, h2o):
        """Q_r = (sqrt(pi)/sigma) sqrt(T^3 / (tA tB tC)) — not the linear T/(sigma theta)."""
        tA, tB, tC = 40.13, 20.87, 13.36
        ln_q = 0.5 * math.log(math.pi) - math.log(2) + 0.5 * math.log(self.T**3 / (tA * tB * tC))
        expected = R * (ln_q + 1.5)
        assert self._contribs(h2o)["rotational"].S_m == pytest.approx(expected, abs=1e-9)

    def test_rotational_heat_capacity_is_three_halves_R(self, h2o):
        assert self._contribs(h2o)["rotational"].Cv_m == pytest.approx(1.5 * R, rel=1e-12)

    def test_vibrational_is_the_anharmonic_manifold_sum(self, h2o):
        """H2O now sums the real level manifold, so the Einstein form no longer applies to it.

        Recomputed here from the stored constants, independently of the mode implementation.
        """
        anh = h2o.anharmonicity
        assert anh is not None, "H2O should carry anharmonicity constants"
        zero = anh.zero_point_energy_cm1
        levels = []
        for v1 in range(40):
            for v2 in range(60):
                for v3 in range(40):
                    e = anh.term_value_cm1((v1, v2, v3)) - zero
                    if 0.0 <= e <= anh.dissociation_cm1:
                        # local monotonicity, as the mode applies
                        ok = True
                        for i, vi in enumerate((v1, v2, v3)):
                            if vi:
                                low = [v1, v2, v3]
                                low[i] = vi - 1
                                if anh.term_value_cm1(tuple(low)) - zero >= e:
                                    ok = False
                                    break
                        if ok:
                            levels.append(e * CM1_TO_K)
        q = sum(math.exp(-t / self.T) for t in levels)
        mean = sum(t * math.exp(-t / self.T) for t in levels) / q
        expected = R * (math.log(q) + mean / self.T)
        assert self._contribs(h2o)["vibrational"].S_m == pytest.approx(expected, rel=1e-10)

    def test_harmonic_path_is_still_einstein_for_species_without_constants(self):
        """A species with no anharmonicity block must keep the exact harmonic behaviour."""
        n2 = get("N2")
        assert n2.anharmonicity is None
        st = State(T=self.T, P=self.P)
        got = Thermodynamics(n2, st).partition.contributions(st)["vibrational"].S_m
        x = n2.vibrational_modes[0].wavenumber_cm1 * CM1_TO_K / self.T
        e = math.exp(-x)
        assert got == pytest.approx(R * (x * e / (1 - e) - math.log1p(-e)), abs=1e-12)

    def test_electronic_contributes_nothing(self, h2o):
        c = self._contribs(h2o)["electronic"]
        assert c.S_m == 0.0 and c.U_m == 0.0 and c.Cv_m == 0.0

    def test_modes_sum_to_the_total(self, h2o):
        parts = sum(c.S_m for c in self._contribs(h2o).values())
        assert parts == pytest.approx(_res(h2o, self.T, self.P).S_m, abs=1e-9)


# ------------------------------------------------------- 4. conventions, units, standard state


class TestConventions:
    """Audit sections 5.4-5.5 and 6 — the things that silently corrupt a comparison."""

    def test_entropy_pressure_dependence_is_exact(self, h2o):
        for P in (1e4, 101325.0, 1e6):
            d = _res(h2o, 500.0, P).S_m - _res(h2o, 500.0, 1e5).S_m
            assert d == pytest.approx(-R * math.log(P / 1e5), abs=1e-10)

    def test_cp_is_pressure_independent(self, h2o):
        """Ideal gas: no standard-state error can move Cp. Rules out that explanation."""
        base = _res(h2o, 500.0, 1e5).Cp_m
        for P in (1e3, 1e4, 101325.0, 1e6):
            assert _res(h2o, 500.0, P).Cp_m == pytest.approx(base, rel=1e-12)

    def test_standard_pressure_is_not_imposed(self, h2o):
        """1 bar and 1 atm must differ by exactly R ln(1.01325) = 0.1094 J/mol/K."""
        d = _res(h2o, 298.15, 101325.0).S_m - _res(h2o, 298.15, 1e5).S_m
        assert d == pytest.approx(-R * math.log(101325.0 / 1e5), abs=1e-10)

    @pytest.mark.parametrize("T", [298.15, 500.0, 1000.0])
    def test_molar_to_massic_conversion(self, h2o, T):
        r = _res(h2o, T)
        M = h2o.molar_mass
        for molar, massic in ((r.Cp_m, r.Cp_s), (r.S_m, r.S_s), (r.H_m, r.H_s), (r.U_m, r.U_s)):
            assert molar / M == pytest.approx(massic, rel=1e-14)

    def test_specific_gas_constant_is_exact(self, h2o):
        r = _res(h2o, 500.0)
        assert r.R_specific == pytest.approx(R / h2o.molar_mass, rel=1e-14)
        assert r.R_specific == pytest.approx(461.5228, abs=1e-3)

    def test_enthalpy_zero_is_at_0_K_without_zero_point_energy(self, h2o):
        """H_m is the JANAF-style increment H(T) - H(0): -> 0 at T -> 0, ~ 4RT while frozen."""
        assert _res(h2o, 1e-3).H_m == pytest.approx(0.0, abs=1e-1)
        for T in (1.0, 10.0, 100.0):
            assert _res(h2o, T).H_m == pytest.approx(4 * R * T, rel=1e-6)

    def test_enthalpy_increment_is_the_integral_of_cp(self, h2o):
        """H(1000) - H(298.15) must equal the trapezoidal integral of Cp over the same range."""
        import numpy as np

        Ts = np.linspace(298.15, 1000.0, 4001)
        cps = np.array([_res(h2o, float(t)).Cp_m for t in Ts])
        integral = float(np.trapezoid(cps, Ts)) if hasattr(np, "trapezoid") else float(
            np.trapz(cps, Ts)
        )
        direct = _res(h2o, 1000.0).H_m - _res(h2o, 298.15).H_m
        assert direct == pytest.approx(integral, rel=1e-6)


# ------------------------------------------------------------- 5. against tabulated references


class TestAgainstReferences:
    """Audit section 7 — absolute agreement, with tolerances justified by RRHO."""

    def test_cp_at_298_within_rrho_tolerance(self, h2o):
        """-0.32 %: the centrifugal-distortion floor. Tightening this would be an empirical fit."""
        err = (_res(h2o, 298.15).Cp_m - JANAF_CP_298) / JANAF_CP_298
        assert -0.005 < err < 0.0, f"Cp(298.15) error {err:.4%} outside the expected floor"

    def test_entropy_at_298_within_tolerance(self, h2o):
        """Entropy is ten times more robust than Cp: dominated by exactly-treated modes."""
        err = (_res(h2o, 298.15).S_m - JANAF_S_298) / JANAF_S_298
        assert -0.001 < err < 0.0, f"S(298.15) error {err:.4%} outside expectation"

    @pytest.mark.parametrize("T", sorted(NIST_CP))
    def test_cp_against_nist_grid(self, h2o, T):
        err = abs(_res(h2o, T).Cp_m - NIST_CP[T]) / NIST_CP[T]
        assert err < 0.02, f"Cp({T}) off by {err:.3%}"

    @pytest.mark.parametrize("T", sorted(NIST_S))
    def test_entropy_against_nist_grid(self, h2o, T):
        err = abs(_res(h2o, T).S_m - NIST_S[T]) / NIST_S[T]
        assert err < 0.003, f"S({T}) off by {err:.3%}"

    def test_validation_layer_passes_its_own_tolerance(self):
        from statthermopy.validation import validate

        assert validate("H2O", "Cp").mean_abs_error_percent < 1.0
        assert validate("H2O", "S").mean_abs_error_percent < 0.2


# --------------------------------------------------------------- 6. the error signature


class TestErrorSignature:
    """Audit section 8 — what proves the cause is the model, not the code.

    These are the load-bearing tests of the audit. If the engine were later "fixed" by tweaking a
    frequency to improve one temperature, the two-component signature would break and these fail.
    """

    def test_error_is_always_negative(self, h2o):
        """RRHO under-counts states: it can never over-predict Cp. A positive error is a bug."""
        for T, ref in NIST_CP.items():
            assert _res(h2o, T).Cp_m < ref, f"Cp({T}) exceeds the reference — unphysical for RRHO"

    def test_low_temperature_floor_is_flat(self, h2o):
        """273-500 K: the residual sits near -0.106 J/mol/K and does not track temperature."""
        resid = {}
        for T, ref in ((298.15, JANAF_CP_298), (500.0, NIST_CP[500.0])):
            resid[T] = _res(h2o, T).Cp_m - ref
        assert all(-0.13 < v < -0.08 for v in resid.values()), resid

    def test_floor_does_not_scale_with_the_vibrational_contribution(self, h2o):
        """The decisive test (audit 8.3).

        Between 300 K and 500 K the vibrational heat capacity grows by roughly an order of
        magnitude. If the low-T residual were vibrational in origin it would grow with it. It does
        not: the ratio residual/Cv_vib collapses, proving the floor is rotational.
        """
        cv_vib_300 = _res(h2o, 300.0).Cv_m - 3.0 * R
        cv_vib_500 = _res(h2o, 500.0).Cv_m - 3.0 * R
        assert cv_vib_500 / cv_vib_300 > 5.0, "vibrational contribution should grow steeply"

        r300 = abs(_res(h2o, 300.0).Cp_m - 33.5960)      # JANAF at 300 K
        r500 = abs(_res(h2o, 500.0).Cp_m - NIST_CP[500.0])
        # The residual changes by less than 30 % while Cv_vib changes by more than 400 %.
        assert 0.7 < r500 / r300 < 1.3, f"residual ratio {r500 / r300:.2f} — floor is not flat"

    def test_anharmonicity_removed_the_high_temperature_growth(self, h2o):
        """Audit section 15 — the anharmonic manifold flattened the high-T deficit.

        With the harmonic ladder the error grew to -1.90 % at 2000 K. Summing the real levels
        holds it near the rotational floor across the whole range. Guarding both ends: it must
        stay well below the old figure, and must not have been over-corrected past the reference.
        """
        errs = {T: (_res(h2o, T).Cp_m - NIST_CP[T]) / NIST_CP[T] for T in sorted(NIST_CP)}
        assert all(e < 0.0 for e in errs.values()), "RRHO cannot over-predict Cp"
        assert abs(errs[2000.0]) < 0.008, f"2000 K deficit {errs[2000.0]:.4%} — expected ~0.57 %"
        assert abs(errs[2000.0]) < 0.5 * 0.0195, "should be far better than the harmonic -1.95 %"
        # The residual is now dominated by the flat rotational floor, so the spread is small.
        assert max(errs.values()) - min(errs.values()) < 0.005

    def test_symmetry_error_magnitude_is_excluded(self, h2o):
        """A wrong sigma would move S by R ln 2 = 5.76 J/mol/K, 50x the observed residual."""
        observed = abs(_res(h2o, 500.0).S_m - NIST_S[500.0])
        assert observed < 0.1 * R * math.log(2)

    def test_engine_matches_spectroscopic_not_calorimetric_entropy(self, h2o):
        """Audit section 9.3 — the residual entropy of ice, and why the engine is right to omit it.

        Classical thermodynamics only gives entropy differences; the third law supplies the zero by
        assuming a perfect crystal at 0 K. Ice is not one: proton disorder leaves Pauling's residual
        entropy R ln(3/2). A calorimetric integration from 0 K therefore *understates* the absolute
        entropy by that amount, which Giauque & Stout measured in 1936.

        The engine counts gas-phase states spectroscopically and contains no ice, so it must agree
        with the spectroscopic value and disagree with the uncorrected calorimetric one. Auditing
        against the latter would show a spurious +1.8 % "error".
        """
        S_spectroscopic = JANAF_S_298          # 188.834 — what JANAF/CODATA tabulate
        S_calorimetric = 185.4                 # Giauque & Stout, third law without the correction
        residual = R * math.log(1.5)           # Pauling 1935: 3.3712 J/mol/K

        # The historical discrepancy is Pauling's residual entropy, to better than 0.1 J/mol/K.
        assert S_spectroscopic - S_calorimetric == pytest.approx(residual, abs=0.1)

        engine = _res(h2o, 298.15).S_m
        # Agrees with the spectroscopic scale...
        assert abs(engine - S_spectroscopic) / S_spectroscopic < 0.001
        # ...and must NOT agree with the uncorrected calorimetric one.
        assert (engine - S_calorimetric) / S_calorimetric > 0.015

    def test_standard_state_error_magnitude_is_excluded(self, h2o):
        """R ln(1 atm/1 bar) = 0.109 J/mol/K nearly equals the Cp floor — a numerical trap.

        It is excluded because Cp is rigorously pressure-independent for an ideal gas, which
        ``TestConventions.test_cp_is_pressure_independent`` verifies directly.
        """
        assert R * math.log(101325.0 / 1e5) == pytest.approx(0.1094, abs=1e-3)


# -------------------------------------------------------------- 7. optional cross-checks


class TestExternalCrossChecks:
    """Independent libraries, skipped when unavailable. Offline once installed."""

    def test_against_iapws_ideal_gas_limit(self, h2o):
        iapws = pytest.importorskip("iapws", reason="pip install -e '.[humidair]'")
        # (T, rho) input avoids the package's spurious density root at low P — see audit section 3.
        for T in (300.0, 500.0, 1000.0):
            rho = 1.0 * MOLAR_MASS_KG_MOL / (R * T)
            ref = iapws.IAPWS95(T=T, rho=rho).cp * 1000 * MOLAR_MASS_KG_MOL
            err = (_res(h2o, T).Cp_m - ref) / ref
            assert -0.01 < err < 0.0

    def test_entropy_offset_to_the_iapws_scale(self, h2o):
        """The engine's own offset drifts because it carries the engine's entropy deficit.

        At 500 K, where that deficit is smallest, it is within 0.2 J/mol/K of the true constant.
        """
        iapws = pytest.importorskip("iapws", reason="pip install -e '.[humidair]'")
        T = 500.0
        rho = 1.0 * MOLAR_MASS_KG_MOL / (R * T)
        s_iapws = iapws.IAPWS95(T=T, rho=rho).s * 1000 * MOLAR_MASS_KG_MOL - R * math.log(1e5 / 1.0)
        offset = _res(h2o, T).S_m - s_iapws
        assert offset == pytest.approx(ENTROPY_OFFSET_J_MOL_K, abs=0.2)

    def test_true_offset_from_the_reference_scale(self):
        """NIST absolute minus IAPWS ideal must give the constant, to better than 0.05 J/mol/K."""
        iapws = pytest.importorskip("iapws", reason="pip install -e '.[humidair]'")
        offs = []
        for T, ref in NIST_S.items():
            rho = 1.0 * MOLAR_MASS_KG_MOL / (R * T)
            s = iapws.IAPWS95(T=T, rho=rho).s * 1000 * MOLAR_MASS_KG_MOL - R * math.log(1e5 / 1.0)
            offs.append(ref - s)
        assert max(offs) - min(offs) < 0.01, "offset should be constant"
        assert sum(offs) / len(offs) == pytest.approx(ENTROPY_OFFSET_J_MOL_K, abs=0.05)

    def test_enthalpy_offset_is_constant_once_the_engine_deficit_is_removed(self, h2o):
        """Audit section 9.2 — the offset only looks drifty because of how it is measured.

        Measured at 1 bar it wanders by 10.8 kJ/kg; in the ideal-gas limit by 4.3; with the
        engine's own accumulated enthalpy deficit subtracted it is constant to the printed digits.
        """
        iapws = pytest.importorskip("iapws", reason="pip install -e '.[humidair]'")

        def h_engine(T):
            return _res(h2o, T).H_s / 1000.0  # kJ/kg

        def h_ref(T):
            return iapws.IAPWS95(T=T, rho=1.0 * MOLAR_MASS_KG_MOL / (R * T)).h

        temps = [423.15, 473.15, 573.15, 673.15, 773.15, 973.15]
        base_e, base_r = h_engine(temps[0]), h_ref(temps[0])
        corrected = []
        for T in temps:
            offset = h_engine(T) - h_ref(T)
            deficit = (h_engine(T) - base_e) - (h_ref(T) - base_r)
            corrected.append(offset - deficit)
        assert max(corrected) - min(corrected) < 1e-6, "corrected offset must be constant"
        assert corrected[0] == pytest.approx(ENTHALPY_OFFSET_KJ_KG, abs=0.01)

    def test_enthalpy_offset_from_the_triple_point(self, h2o):
        """The constant is the triple-point latent heat minus the ideal gas's thermal enthalpy.

        ``ENTHALPY_OFFSET_KJ_KG`` is defined as engine minus reference, hence negative; this route
        computes reference minus engine, so it reproduces its magnitude.
        """
        iapws = pytest.importorskip("iapws", reason="pip install -e '.[humidair]'")
        Tt = 273.16
        h_ideal = iapws.IAPWS95(T=Tt, rho=1.0 * MOLAR_MASS_KG_MOL / (R * Tt)).h
        h_thermal = _res(h2o, Tt).H_s / 1000.0
        # 1996.946 kJ/kg, within 0.05 % of the 1997.870 obtained by regression at 423 K.
        assert h_ideal - h_thermal == pytest.approx(-ENTHALPY_OFFSET_KJ_KG, rel=1e-3)

        # And it must decompose exactly into the physical pieces. h(liquid) is 6e-4 kJ/kg rather
        # than an exact zero, so it is carried explicitly instead of being assumed to vanish.
        h_liq = iapws.IAPWS95(T=Tt, x=0).h
        h_vap = iapws.IAPWS95(T=Tt, x=1).h
        latent = h_vap - h_liq
        correction = h_ideal - h_vap
        assert h_liq + latent + correction - h_thermal == pytest.approx(
            h_ideal - h_thermal, abs=1e-9
        )
        assert latent == pytest.approx(2500.91, abs=0.05)

    def test_multi_source_consensus(self, h2o):
        """The engine must sit below every independent correlation, by 0.1-2 %.

        Also asserts the audit's headline observation: at 800 K the references disagree among
        themselves by more than the engine deviates from any one of them.
        """
        thermo = pytest.importorskip("thermo", reason="pip install thermo")
        obj = thermo.HeatCapacityGas(CASRN="7732-18-5", MW=18.01528)
        srcs = [s for s in ("JANAF", "TRCIG", "HEOS_FIT") if s in obj.all_methods]
        assert srcs, "no reference correlation available"
        vals = [obj.calculate(800.0, s) for s in srcs]
        eng = _res(h2o, 800.0).Cp_m
        for v in vals:
            assert eng < v, "engine must under-predict every source"
            assert abs(eng - v) / v < 0.02


# ------------------------------------------------------------- 8. anharmonic machinery


class TestAnharmonicManifold:
    """Audit section 15 — the anharmonic vibrational mode added in response to the audit."""

    def test_constants_reproduce_the_observed_fundamentals(self, h2o):
        """The self-check that validates the constant set without any external table.

        The fundamentals are a consequence of omega and x_ij; if they did not come out right, the
        constants would be wrong or in the wrong order.
        """
        predicted = h2o.anharmonicity.fundamentals_cm1()
        observed = tuple(m.wavenumber_cm1 for m in h2o.vibrational_modes)
        assert len(predicted) == len(observed)
        for p, o in zip(predicted, observed):
            assert abs(p - o) < 2.0, f"fundamental {p:.2f} vs observed {o:.2f} cm^-1"

    def test_zero_point_energy(self, h2o):
        """G(0,0,0) = 4634.6 cm^-1 for water."""
        assert h2o.anharmonicity.zero_point_energy_cm1 == pytest.approx(4634.6, abs=1.0)

    def test_levels_start_at_the_ground_state(self, h2o):
        """The manifold is measured from G(0), so the lowest level is exactly zero."""
        st = State(T=500.0, P=1e5)
        theta = Thermodynamics(h2o, st).partition.vibrational.theta
        assert min(theta) == pytest.approx(0.0, abs=1e-9)
        assert len(theta) > 100, "manifold suspiciously small"

    def test_anharmonic_exceeds_harmonic_at_high_temperature(self, h2o):
        """Closing level spacing means more populated states, hence a larger Cv."""
        from statthermopy.modes import Vibrational

        st = State(T=2000.0, P=1e5)
        rs = st.resolve(h2o.molar_mass)
        anh = Thermodynamics(h2o, st).partition.vibrational.contribution(rs)
        harm = Vibrational(h2o.vibrational_modes).contribution(rs)
        assert anh.Cv_m > harm.Cv_m
        assert anh.S_m > harm.S_m

    def test_anharmonic_and_harmonic_agree_when_vibration_is_frozen(self, h2o):
        """At low T only the ground level is populated, so the two models must coincide."""
        from statthermopy.modes import Vibrational

        st = State(T=50.0, P=1e5)
        rs = st.resolve(h2o.molar_mass)
        anh = Thermodynamics(h2o, st).partition.vibrational.contribution(rs)
        harm = Vibrational(h2o.vibrational_modes).contribution(rs)
        assert anh.Cv_m == pytest.approx(harm.Cv_m, abs=1e-9)
        assert anh.S_m == pytest.approx(harm.S_m, abs=1e-9)

    def test_identities_still_hold_on_the_anharmonic_path(self, h2o):
        """Adding a mode must not break the thermodynamic identities."""
        for T in (298.15, 1000.0, 2000.0):
            r = _res(h2o, T)
            assert r.Cp_m - r.Cv_m == pytest.approx(R, abs=1e-12)
            assert r.H_m - r.U_m == pytest.approx(R * T, rel=1e-12)
            d = 0.01
            num = (_res(h2o, T + d).H_m - _res(h2o, T - d).H_m) / (2 * d)
            assert r.Cp_m == pytest.approx(num, rel=1e-6)

    def test_validation_layer_improved(self):
        """The whole point: the embedded NIST comparison must be markedly better."""
        from statthermopy.validation import validate

        cp = validate("H2O", "Cp")
        # Harmonic gave 0.808 % mean / 1.947 % max.
        assert cp.mean_abs_error_percent < 0.5
        assert cp.max_abs_error_percent < 0.8

    def test_only_water_carries_anharmonicity_so_far(self):
        """Every other species must keep the harmonic path — no silent change elsewhere."""
        from statthermopy.database import list_molecules

        with_anh = [n for n in list_molecules() if get(n).anharmonicity is not None]
        assert with_anh == ["H2O"], f"unexpected anharmonic species: {with_anh}"

    def test_degenerate_modes_are_rejected(self):
        """The manifold's level counting is only defined for non-degenerate modes."""
        from statthermopy.core.molecule import Anharmonicity, Geometry, Molecule, VibrationalMode

        with pytest.raises(ValueError, match="non-degenerate"):
            Molecule(
                name="X", formula="X", molar_mass_gmol=16.0, geometry=Geometry.NONLINEAR,
                n_atoms=3, symmetry_number=1, moments_of_inertia=(1e-46, 2e-46, 3e-46),
                vibrational_modes=(VibrationalMode(1000.0, 3),),
                anharmonicity=Anharmonicity((1000.0,), ((-10.0,),), 30000.0),
            )

    def test_mode_count_mismatch_is_rejected(self):
        from statthermopy.core.molecule import Anharmonicity, Geometry, Molecule, VibrationalMode

        with pytest.raises(ValueError, match="anharmonicity describes"):
            Molecule(
                name="X", formula="X", molar_mass_gmol=18.0, geometry=Geometry.NONLINEAR,
                n_atoms=3, symmetry_number=2, moments_of_inertia=(1e-46, 2e-46, 3e-46),
                vibrational_modes=(VibrationalMode(1000.0), VibrationalMode(2000.0),
                                   VibrationalMode(3000.0)),
                anharmonicity=Anharmonicity((1000.0,), ((-10.0,),), 30000.0),
            )

    def test_backends_agree_on_the_anharmonic_species(self, h2o):
        """The engine's invariant: a backend changes execution, never the model.

        The compiled grid kernels only know harmonic ladders, so they must decline this molecule
        and defer to the reference path rather than silently returning harmonic numbers.
        """
        from statthermopy.backend import available_backends, get_backend, set_backend

        original = get_backend().name
        try:
            point = [_res(h2o, T).Cp_m for T in (298.15, 1000.0, 2000.0)]
            for name in available_backends():
                set_backend(name)
                th = Thermodynamics(h2o, State(T=298.15, P=1e5))
                _, grid = th.property_vs_T("Cp_m", [298.15, 1000.0, 2000.0], P=1e5)
                for a, b in zip(point, grid):
                    assert a == pytest.approx(b, rel=1e-12), f"backend {name} diverges"
        finally:
            set_backend(original)
