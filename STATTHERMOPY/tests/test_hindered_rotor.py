"""Tests for the hindered internal-rotation mode and its integration.

The physics is checked against the two rigorous analytic limits (free rotor and harmonic
oscillator), the internal thermodynamic consistency (Cv = dU/dT), and the end-to-end molecule
path (ethane / propane), including the accelerated-backend fallback.
"""

from __future__ import annotations

import math

import pytest

from statthermopy import InternalRotor, Thermodynamics
from statthermopy.constants import R, h, hc, k_B
from statthermopy.core.state import State
from statthermopy.database import get
from statthermopy.modes.hindered_rotor import HinderedRotor, torsional_levels_kelvin


def _rs(T: float):
    return State(T=T, P=101325.0).resolve(0.03)


def _rotor_mode(F=10.7, V=1024.0, sym=3, n=3, deg=1):
    return HinderedRotor((InternalRotor(F, V, symmetry=sym, n_minima=n, degeneracy=deg),))


# -- analytic limits ---------------------------------------------------------

def test_free_rotor_heat_capacity_approaches_half_R():
    """A vanishing barrier is a free 1-D rotor: Cv -> R/2 at high T."""
    m = _rotor_mode(V=1e-4)
    assert m.cv_m(_rs(1000.0)) == pytest.approx(0.5 * R, rel=1e-3)


def test_free_rotor_partition_function_matches_closed_form():
    """q_free = sqrt(8 pi^3 I_r k_B T) / (sigma h), with F = hbar^2 / (2 I_r)."""
    F_cm1, sigma = 10.7, 3
    F_J = F_cm1 * hc * 100.0
    hbar = h / (2.0 * math.pi)
    I_r = hbar * hbar / (2.0 * F_J)
    m = _rotor_mode(F=F_cm1, V=1e-6, sym=sigma)
    for T in (300.0, 1200.0):
        q_num = math.exp(m.ln_q(_rs(T)))
        q_ana = math.sqrt(8.0 * math.pi**3 * I_r * k_B * T) / (sigma * h)
        assert q_num == pytest.approx(q_ana, rel=1e-4)


def test_high_barrier_low_T_partition_goes_to_one():
    """Deep wells well below the first torsional level: only the (sigma-fold) ground manifold is
    populated (q -> 1) and the heat capacity freezes out (Cv -> 0). The ethane torsion sits at
    ~287 cm^-1 (~413 K), so this holds at 25 K but not yet at 50 K."""
    m = _rotor_mode(V=1024.0)
    assert math.exp(m.ln_q(_rs(25.0))) == pytest.approx(1.0, abs=1e-3)
    assert m.cv_m(_rs(25.0)) == pytest.approx(0.0, abs=1e-3)


def test_torsional_fundamental_matches_ethane():
    """The Mathieu ladder of the ethane rotor reproduces the observed ~289 cm^-1 torsion.

    The 3-fold symmetry makes each torsional level a near-degenerate triplet; the physical
    fundamental is the spacing to the next manifold (level index 3), not the tunnelling split.
    """
    levels_K = torsional_levels_kelvin(10.7, 1024.0, 3)
    from statthermopy.units import CM1_TO_K
    fundamental_cm1 = levels_K[3] / CM1_TO_K
    assert fundamental_cm1 == pytest.approx(289.0, abs=8.0)


# -- consistency -------------------------------------------------------------

def test_cv_matches_numerical_dU_dT():
    """cv_m must equal the finite-difference derivative of U_m (single source of truth)."""
    m = _rotor_mode(F=5.3, V=1190.0, deg=2)

    def U(T):
        return m.contribution(_rs(T)).U_m

    T, dT = 700.0, 0.05
    numeric = (U(T + dT) - U(T - dT)) / (2.0 * dT)
    assert m.cv_m(_rs(T)) == pytest.approx(numeric, rel=1e-5)


def test_d_ln_q_dT_matches_internal_energy():
    """The canonical relation U_m = R T^2 (d ln q / dT) must hold for the rotor."""
    m = _rotor_mode(F=10.7, V=1024.0)
    st = _rs(750.0)
    c = m.contribution(st)
    assert R * st.T * st.T * m.d_ln_q_dT(st) == pytest.approx(c.U_m, rel=1e-9)


def test_degeneracy_equals_independent_rotors():
    """One rotor of degeneracy 2 == two identical independent rotors."""
    r = InternalRotor(5.3, 1190.0, symmetry=3, n_minima=3)
    one_deg2 = HinderedRotor((InternalRotor(5.3, 1190.0, symmetry=3, n_minima=3, degeneracy=2),))
    two_deg1 = HinderedRotor((r, r))
    st = _rs(600.0)
    assert one_deg2.cv_m(st) == pytest.approx(two_deg1.cv_m(st), rel=1e-12)
    assert one_deg2.ln_q(st) == pytest.approx(two_deg1.ln_q(st), rel=1e-12)


def test_empty_rotor_is_null_mode():
    """No rotors -> zero contribution, so molecules without internal rotation are unaffected."""
    m = HinderedRotor(())
    c = m.contribution(_rs(500.0))
    assert m.ln_q(_rs(500.0)) == 0.0
    assert (c.U_m, c.S_m, c.A_m, c.Cv_m) == (0.0, 0.0, 0.0, 0.0)


def test_rotor_lowers_high_T_cv_vs_harmonic():
    """Physical signature: above the barrier the hindered rotor sheds heat capacity toward the
    free-rotor R/2, i.e. it sits below a harmonic oscillator of the same small-oscillation
    frequency."""
    F, V, n = 10.7, 1024.0, 3
    nu = n * math.sqrt(F * V)  # small-oscillation harmonic wavenumber (cm^-1)
    from statthermopy.units import CM1_TO_K
    theta = nu * CM1_TO_K

    def ho_cv(T):
        x = theta / T
        return R * x * x * math.exp(x) / (math.exp(x) - 1.0) ** 2

    rotor = _rotor_mode(F=F, V=V, sym=3, n=n)
    T = 2000.0  # well above the barrier (1024 cm^-1 ~ 1474 K)
    assert rotor.cv_m(_rs(T)) < ho_cv(T)
    assert rotor.cv_m(_rs(T)) > 0.5 * R  # but still above the free-rotor floor


# -- molecule integration ----------------------------------------------------

@pytest.mark.parametrize("name,n_osc,n_rot", [("C2H6", 17, 1), ("C3H8", 25, 2)])
def test_database_molecule_dof_split(name, n_osc, n_rot):
    mol = get(name)
    assert mol.n_vibrational_modes == n_osc
    assert mol.n_internal_rotors == n_rot
    assert n_osc + n_rot == 3 * mol.n_atoms - 6  # internal-DOF conservation


def test_ethane_compute_runs_and_is_reasonable():
    res = Thermodynamics(get("C2H6"), State(T=298.15, P=101325.0)).compute()
    # Cp of ethane at 298 K is ~52.5 J/mol/K; within a few percent of the reference.
    assert res.Cp_m == pytest.approx(52.5, abs=1.5)
    assert res.Cv_m > 0.0 and res.gamma > 1.0


def test_internal_rotation_folds_into_vibrational_contribution():
    """The rotor is reported inside the 'vibrational' mode, keeping the four-mode breakdown."""
    pf = Thermodynamics(get("C2H6"), State(T=800.0, P=101325.0)).partition
    contribs = pf.contributions(State(T=800.0, P=101325.0))
    assert set(contribs) == {"translational", "rotational", "vibrational", "electronic"}
    # vibrational contribution == harmonic + internal rotation
    rs = _rs(800.0)
    harm = pf.vibrational.contribution(rs)
    rot = pf.internal_rotation.contribution(rs)
    assert contribs["vibrational"].Cv_m == pytest.approx(harm.Cv_m + rot.Cv_m, rel=1e-12)
    assert rot.Cv_m > 0.0  # the rotor actually contributes


def test_split_internal_rotation_reports_separate_mode():
    """split_internal_rotation=True gives a distinct 'internal_rotation' entry for a molecule
    with rotors, with the same total as the folded view; molecules without rotors are unchanged."""
    st = State(T=800.0, P=101325.0)
    pf = Thermodynamics(get("C2H6"), st).partition
    folded = pf.contributions(st)
    split = pf.contributions(st, split_internal_rotation=True)
    assert set(split) == {
        "translational", "rotational", "vibrational", "internal_rotation", "electronic",
    }
    # vibrational (harmonic-only) + internal_rotation == the folded vibrational
    for attr in ("ln_q", "U_m", "S_m", "A_m", "Cv_m"):
        combined = getattr(split["vibrational"], attr) + getattr(split["internal_rotation"], attr)
        assert combined == pytest.approx(getattr(folded["vibrational"], attr), rel=1e-12)
    # a molecule without internal rotors keeps the four-mode view even when split is requested
    pf_n2 = Thermodynamics(get("N2"), st).partition
    assert set(pf_n2.contributions(st, split_internal_rotation=True)) == {
        "translational", "rotational", "vibrational", "electronic",
    }


def test_monatomic_rejects_internal_rotors():
    from statthermopy import Geometry, Molecule
    with pytest.raises(ValueError):
        Molecule(
            name="BadAr", formula="Ar", molar_mass_gmol=39.948,
            geometry=Geometry.MONOATOMIC, n_atoms=1,
            internal_rotors=(InternalRotor(10.0, 100.0),),
        )


def test_internal_rotor_parameter_validation():
    with pytest.raises(ValueError):
        InternalRotor(0.0, 100.0)          # F must be > 0
    with pytest.raises(ValueError):
        InternalRotor(10.0, -1.0)          # barrier must be >= 0
    with pytest.raises(ValueError):
        InternalRotor(10.0, 100.0, symmetry=0)


# -- accelerated-backend fallback --------------------------------------------

def test_numba_grid_falls_back_for_internal_rotors():
    """The compiled grid has no rotor term, so property_vs_T on the numba backend must match the
    NumPy path exactly (via the None-return fallback)."""
    pytest.importorskip("numba")
    from statthermopy.backend import set_backend

    mol = get("C2H6")
    Ts = [300.0, 600.0, 1000.0, 1500.0]
    # NumPy reference first (default backend).
    set_backend("numpy")
    ref = Thermodynamics(mol, State(T=300.0, P=101325.0)).property_vs_T("Cp_m", Ts)[1]
    try:
        set_backend("numba")
        be = __import__("statthermopy.backend", fromlist=["get_backend"]).get_backend()
        assert be.molar_property_grid(mol, Ts, 101325.0, False) is None
        got = Thermodynamics(mol, State(T=300.0, P=101325.0)).property_vs_T("Cp_m", Ts)[1]
    finally:
        set_backend("numpy")
    assert got == pytest.approx(ref, rel=1e-10)


# -- the T = 0 limit ----------------------------------------------------------

class TestZeroKelvinLimit:
    """Every other mode guards ``T == 0``; :class:`HinderedRotor` did not.

    The GUI plots from Tmin = 0 K by default, so ``x = levels / T`` raised ZeroDivisionError
    and the three species carrying an internal rotor -- C2H6, C3H8 and CH3CCL3 -- produced no
    property graph at all, while every other species plotted normally.
    """

    ROTOR_SPECIES = ["C2H6", "C3H8", "CH3CCL3"]

    @pytest.mark.parametrize("name", ROTOR_SPECIES)
    def test_properties_at_absolute_zero_do_not_raise(self, name):
        props = Thermodynamics(get(name), State(T=0.0, P=101325.0)).properties()
        assert math.isfinite(props.Cp_m)
        assert math.isfinite(props.Cv_m)
        assert math.isfinite(props.U_m)

    @pytest.mark.parametrize("name", ROTOR_SPECIES)
    def test_the_rotor_freezes_out(self, name):
        """Third Law: the torsional mode carries no energy and no heat capacity at T = 0."""
        rotor = HinderedRotor(get(name).internal_rotors)
        contribution = rotor.contribution(State(T=0.0, P=101325.0).resolve(get(name).molar_mass))
        assert contribution.U_m == pytest.approx(0.0)
        assert contribution.Cv_m == pytest.approx(0.0)
        assert contribution.A_m == pytest.approx(0.0)
        assert math.isfinite(contribution.ln_q)
        assert math.isfinite(contribution.S_m)

    @staticmethod
    def _tunnelling_splitting(name: str) -> float:
        """Energy (K) of the first level above the torsional ground state.

        This sets the scale on which the T -> 0 limit is actually reached, and it varies by
        five orders of magnitude across these three species because it depends exponentially
        on the barrier: 7.9e-03 K for C2H6 (1024 cm^-1), 7.4e-06 K for C3H8 (1190 cm^-1) and
        5.2e-08 K for CH3CCL3 (1913 cm^-1). A fixed approach temperature cannot serve all
        three -- ethane's ln q is still -0.10 at 0.05 K.
        """
        rotor = get(name).internal_rotors[0]
        levels = torsional_levels_kelvin(
            rotor.rotation_constant_cm1, rotor.barrier_cm1, rotor.n_minima
        )
        return float(min(v for v in levels if v > 0.0))

    @pytest.mark.parametrize("name", ROTOR_SPECIES)
    def test_the_limit_is_continuous(self, name):
        """Approaching 0 K must converge on the value returned exactly at 0 K."""
        rotor = HinderedRotor(get(name).internal_rotors)
        M = get(name).molar_mass
        T_low = self._tunnelling_splitting(name) / 100.0
        at_zero = rotor.contribution(State(T=0.0, P=101325.0).resolve(M))
        near_zero = rotor.contribution(State(T=T_low, P=101325.0).resolve(M))
        assert near_zero.Cv_m == pytest.approx(at_zero.Cv_m, abs=1e-6)
        assert near_zero.U_m == pytest.approx(at_zero.U_m, abs=1e-6)
        assert near_zero.ln_q == pytest.approx(at_zero.ln_q, abs=1e-6)

    def test_the_limit_is_the_ground_state_degeneracy(self):
        """ln q -> ln(g0 / sigma_int): ethane's ground level is a singlet under sigma = 3."""
        rotor = HinderedRotor(get("C2H6").internal_rotors)
        at_zero = rotor.contribution(State(T=0.0, P=101325.0).resolve(get("C2H6").molar_mass))
        assert at_zero.ln_q == pytest.approx(-math.log(3.0))

    @pytest.mark.parametrize("name", ROTOR_SPECIES)
    def test_a_curve_from_zero_kelvin_is_plottable(self, name):
        """The reported symptom: Cp vs T over the GUI's default range."""
        Ts = [1500.0 * i / 99 for i in range(100)]   # the GUI's default range, starting at 0 K
        _, values = Thermodynamics(
            get(name), State(T=300.0, P=101325.0)
        ).property_vs_T("Cp_m", Ts)
        assert len(values) == len(Ts)
        assert all(math.isfinite(v) for v in values)

    def test_rotor_species_now_behave_like_the_rest(self):
        """C2H6 was the outlier; at 0 K it must now be as finite as argon."""
        for name in ["AR", "N2", *self.ROTOR_SPECIES]:
            props = Thermodynamics(get(name), State(T=0.0, P=101325.0)).properties()
            assert math.isfinite(props.Cp_m), name
