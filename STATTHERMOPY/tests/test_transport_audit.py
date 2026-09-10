"""Regression tests for the transport-module audit corrections.

Locks in the findings and fixes of ``docs/TRANSPORT_MIXTURE_AUDIT.md`` and
``docs/TRANSPORT_CORRECTION_REPORT.md``:

* **D-1** — species names must not be YAML booleans (covered in ``test_database.py``).
* **D-2** — a zero mole fraction means the species is absent, not infinitely dilute.
* **D-3** — a generic mixture has no implicit trace species; diffusion is per species.

Plus the properties the audit *confirmed* and that must not regress: Wilke really active,
all 30 species independent, every binary pair computable and symmetric, and the mixing rules
generic in the number of components.
"""

from __future__ import annotations

import itertools

import pytest

from statthermopy import State, Thermodynamics
from statthermopy.database import get, list_molecules
from statthermopy.mixture import IdealGasMixture
from statthermopy.transport import TransportCalculator, binary_diffusion
from statthermopy.transport.air import MixtureTransportCalculator

T0, P0 = 300.0, 101325.0


def _mix(fractions, **kw):
    return MixtureTransportCalculator(IdealGasMixture.from_names(fractions), **kw)


# ----------------------------------------------------------- D-2: zero fractions


class TestZeroMoleFractions:
    """A species at x = 0 is absent; the mixture must behave as if it were never listed."""

    def test_zero_component_is_exactly_equivalent_to_omitting_it(self):
        a = _mix({"N2": 1.0, "H2O": 0.0}).compute(State(T=T0, P=P0))
        b = _mix({"N2": 1.0}).compute(State(T=T0, P=P0))
        for prop in ("mu", "k", "rho", "Pr", "M_avg", "nu", "alpha", "cp_s", "gamma"):
            assert getattr(a, prop) == getattr(b, prop), f"{prop} differs"
        assert list(a.x) == list(b.x)

    def test_inactive_species_are_recorded(self):
        m = IdealGasMixture.from_names({"N2": 1.0, "H2O": 0.0, "CO2": 0.0})
        assert set(m.inactive) == {"H2O", "CO2"}
        assert {n.name for n in m.x} == {"N2"}

    def test_sparse_composition_needs_no_manual_pruning(self):
        r = _mix({"N2": 0.78, "O2": 0.21, "Ar": 0.01, "H2O": 0.0, "CO2": 0.0}).compute(
            State(T=T0, P=P0)
        )
        assert len(r.x) == 3
        assert r.mu > 0 and r.k > 0

    @pytest.mark.parametrize("x_h2o", [0.0, 1e-15, 1e-12, 1e-9, 1e-6, 0.01, 0.5, 1.0])
    def test_sweep_through_zero_is_finite(self, x_h2o):
        f = {
            "N2": 0.7808 * (1 - x_h2o), "O2": 0.2095 * (1 - x_h2o),
            "Ar": 0.0093 * (1 - x_h2o), "CO2": 0.0004 * (1 - x_h2o), "H2O": x_h2o,
        }
        r = _mix(f).compute(State(T=T0, P=P0))
        import math

        for prop in ("mu", "k", "rho", "Pr", "alpha", "nu"):
            v = getattr(r, prop)
            assert math.isfinite(v) and v > 0.0, f"{prop} = {v} at x_H2O = {x_h2o}"

    def test_sweep_is_continuous_at_zero(self):
        """Crossing x = 0 must not jump: the limit x -> 0+ equals the x = 0 value."""
        def mu(x):
            f = {
                "N2": 0.7808 * (1 - x), "O2": 0.2095 * (1 - x),
                "Ar": 0.0093 * (1 - x), "CO2": 0.0004 * (1 - x), "H2O": x,
            }
            return _mix(f).compute(State(T=T0, P=P0)).mu

        assert mu(1e-15) == pytest.approx(mu(0.0), rel=1e-12)
        assert mu(1e-12) == pytest.approx(mu(0.0), rel=1e-9)

    def test_negative_fraction_is_rejected(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            IdealGasMixture.from_names({"N2": 1.0, "H2O": -0.1})

    def test_all_zero_is_rejected(self):
        with pytest.raises(ValueError):
            IdealGasMixture.from_names({"N2": 0.0, "H2O": 0.0})

    def test_zero_fraction_on_mass_basis(self):
        a = IdealGasMixture.from_names({"N2": 1.0, "H2O": 0.0}, basis="mass")
        b = IdealGasMixture.from_names({"N2": 1.0}, basis="mass")
        assert a.M_avg == b.M_avg


# ------------------------------------------------------------- D-3: trace species


class TestTraceSpeciesSemantics:
    """Sc and Le are per species; a generic mixture assumes no trace."""

    HEAVY = {"CCl4": 0.4, "I2": 0.3, "C6H6": 0.2, "SO2": 0.1}

    def test_generic_mixture_has_no_implicit_trace(self):
        r = _mix(self.HEAVY).compute(State(T=T0, P=P0))
        assert r.trace_species is None
        assert r.D_eff is None and r.Sc is None and r.Le is None

    def test_per_species_diffusion_is_always_reported(self):
        r = _mix(self.HEAVY).compute(State(T=T0, P=P0))
        assert set(r.effective_diffusivities()) == set(r.x)
        assert set(r.schmidt_numbers()) == set(r.x)
        assert set(r.lewis_numbers()) == set(r.x)
        for name, D in r.effective_diffusivities().items():
            assert D > 0.0, name

    def test_per_species_values_differ(self):
        """Distinct species must not share a diffusivity."""
        r = _mix(self.HEAVY).compute(State(T=T0, P=P0))
        vals = list(r.effective_diffusivities().values())
        assert len(set(vals)) == len(vals)

    def test_schmidt_and_lewis_consistency(self):
        r = _mix(self.HEAVY).compute(State(T=T0, P=P0))
        for name, D in r.effective_diffusivities().items():
            assert r.schmidt_number(name) == pytest.approx(r.nu / D, rel=1e-12)
            assert r.lewis_number(name) == pytest.approx(r.alpha / D, rel=1e-12)

    def test_asking_for_an_absent_species_raises_clearly(self):
        r = _mix(self.HEAVY).compute(State(T=T0, P=P0))
        with pytest.raises(KeyError, match="not a component of this mixture"):
            r.schmidt_number("H2O")

    def test_explicit_trace_still_works_for_an_absent_species(self):
        r = _mix(self.HEAVY, trace="H2O").compute(State(T=T0, P=P0))
        assert r.trace_species == "H2O"
        assert r.D_eff > 0.0 and r.Sc > 0.0 and r.Le > 0.0

    def test_air_transport_keeps_water_as_its_trace(self):
        from statthermopy.transport.air import AirTransport

        r = AirTransport().humid(T0, P0, relative_humidity=0.5)
        assert r.trace_species == "H2O"
        assert r.D_eff > 0.0
        assert r.Sc == pytest.approx(r.nu / r.D_eff, rel=1e-12)

    def test_prandtl_stays_a_mixture_scalar(self):
        """Pr is a property of the mixture and must be present with or without a trace."""
        for kw in ({}, {"trace": "H2O"}):
            r = _mix(self.HEAVY, **kw).compute(State(T=T0, P=P0))
            assert r.Pr == pytest.approx(r.mu * r.cp_s / r.k, rel=1e-12)


# --------------------------------------------------- confirmed physics: no regression


class TestWilkeStaysActive:
    """Wilke must not be replaced by a weighted mean — the audit's headline evidence."""

    def test_he_xe_exceeds_both_pure_viscosities(self):
        """The non-monotonic maximum for disparate masses is Wilke's signature."""
        he = TransportCalculator(get("He"), State(T=T0, P=P0)).compute().mu
        xe = TransportCalculator(get("Xe"), State(T=T0, P=P0)).compute().mu
        mix = _mix({"He": 0.5, "Xe": 0.5}).compute(State(T=T0, P=P0)).mu
        assert mix > max(he, xe), (
            f"mu_mix = {mix:.4e} should exceed both pure values "
            f"(He {he:.4e}, Xe {xe:.4e}) — Wilke may have been replaced"
        )

    def test_he_xe_is_not_a_mole_weighted_mean(self):
        he = TransportCalculator(get("He"), State(T=T0, P=P0)).compute().mu
        xe = TransportCalculator(get("Xe"), State(T=T0, P=P0)).compute().mu
        mix = _mix({"He": 0.5, "Xe": 0.5}).compute(State(T=T0, P=P0)).mu
        assert abs(mix - 0.5 * (he + xe)) / mix > 0.10

    def test_conductivity_mixing_is_not_a_plain_mean(self):
        he = TransportCalculator(get("He"), State(T=T0, P=P0)).compute().k
        xe = TransportCalculator(get("Xe"), State(T=T0, P=P0)).compute().k
        mix = _mix({"He": 0.5, "Xe": 0.5}).compute(State(T=T0, P=P0)).k
        assert abs(mix - 0.5 * (he + xe)) / mix > 0.10


class TestAllSpeciesIndependent:
    """Every species carries its own transport data — no silent sharing."""

    def test_every_species_has_lennard_jones(self):
        missing = [n for n in list_molecules() if get(n).lennard_jones is None]
        assert missing == []

    def test_lennard_jones_parameters_are_all_distinct(self):
        pairs = {(get(n).lennard_jones.sigma_angstrom, get(n).lennard_jones.epsilon_over_k)
                 for n in list_molecules()}
        assert len(pairs) == len(list_molecules()), "two species share LJ parameters"

    @pytest.mark.parametrize("prop", ["mu", "k", "D_self", "rho"])
    def test_no_species_matches_nitrogen(self, prop):
        """The audit's identity test: nothing may be numerically identical to N2."""
        n2 = getattr(TransportCalculator(get("N2"), State(T=T0, P=P0)).compute(), prop)
        for name in list_molecules():
            if name == "N2":
                continue
            v = getattr(TransportCalculator(get(name), State(T=T0, P=P0)).compute(), prop)
            assert v != n2, f"{name}.{prop} is identical to N2 — check for a fallback"

    def test_all_thirty_species_report_finite_positive_transport(self):
        for name in list_molecules():
            r = TransportCalculator(get(name), State(T=T0, P=P0)).compute()
            th = Thermodynamics(get(name), State(T=T0, P=P0)).compute()
            for prop, value in (("mu", r.mu), ("k", r.k), ("rho", r.rho), ("alpha", r.alpha),
                                ("Pr", r.Pr), ("Sc", r.Sc), ("Cp", th.Cp_m), ("Cv", th.Cv_m)):
                import math

                assert math.isfinite(value), f"{name}.{prop} = {value}"
                assert value > 0.0, f"{name}.{prop} = {value}"

    def test_pure_species_do_not_depend_on_load_order(self):
        forward = {n: TransportCalculator(get(n), State(T=T0, P=P0)).compute().mu
                   for n in list_molecules()}
        backward = {n: TransportCalculator(get(n), State(T=T0, P=P0)).compute().mu
                    for n in reversed(list_molecules())}
        assert forward == backward


class TestBinaryDiffusionMatrix:
    """All N(N-1)/2 pairs computable from molecular parameters, none stored."""

    def test_every_pair_is_symmetric_and_positive(self):
        names = list_molecules()
        pairs = list(itertools.combinations(names, 2))
        assert len(pairs) == len(names) * (len(names) - 1) // 2
        for a, b in pairs:
            d_ab = binary_diffusion(get(a), get(b), T0, P0)
            d_ba = binary_diffusion(get(b), get(a), T0, P0)
            assert d_ab > 0.0, f"D({a},{b}) = {d_ab}"
            assert d_ab == pytest.approx(d_ba, rel=1e-14), f"D({a},{b}) != D({b},{a})"

    def test_scaling_with_temperature_and_pressure(self):
        base = binary_diffusion(get("N2"), get("O2"), T0, P0)
        assert binary_diffusion(get("N2"), get("O2"), T0, 2 * P0) == pytest.approx(
            base / 2, rel=1e-12
        )
        assert binary_diffusion(get("N2"), get("O2"), 2 * T0, P0) > 2 * base


class TestMulticomponentGenerality:
    """One formulation for any number of components; order must not matter."""

    CASES = [
        ({"N2": 1.0}, 1),
        ({"N2": 0.5, "O2": 0.5}, 2),
        ({"N2": 0.5, "O2": 0.3, "Ar": 0.2}, 3),
        ({"N2": 0.7808, "O2": 0.2095, "Ar": 0.0093, "CO2": 0.0004}, 4),
        ({"N2": 0.76, "O2": 0.20, "Ar": 0.009, "CO2": 0.0004, "H2O": 0.03}, 5),
    ]

    @pytest.mark.parametrize("fractions,n", CASES)
    def test_mixtures_of_every_size(self, fractions, n):
        r = _mix(fractions).compute(State(T=T0, P=P0))
        assert len(r.x) == n
        assert r.mu > 0 and r.k > 0 and r.rho > 0

    def test_fifteen_components(self):
        names = ["H2", "He", "CH4", "N2", "O2", "CO2", "NH3", "Xe",
                 "Ar", "Ne", "CO", "NO", "SO2", "Cl2", "H2S"]
        r = _mix(dict.fromkeys(names, 1.0 / len(names))).compute(State(T=T0, P=P0))
        assert len(r.x) == 15
        assert r.mu > 0 and r.k > 0

    def test_all_thirty_components(self):
        names = list_molecules()
        r = _mix(dict.fromkeys(names, 1.0 / len(names))).compute(State(T=T0, P=P0))
        assert len(r.x) == 30
        assert r.mu > 0 and r.k > 0

    def test_result_is_invariant_to_species_order(self):
        a = _mix({"N2": 0.7, "O2": 0.2, "CO2": 0.1}).compute(State(T=T0, P=P0))
        b = _mix({"CO2": 0.1, "N2": 0.7, "O2": 0.2}).compute(State(T=T0, P=P0))
        for prop in ("mu", "k", "rho", "Pr", "M_avg", "alpha", "nu"):
            assert getattr(a, prop) == pytest.approx(getattr(b, prop), rel=1e-14), prop

    def test_binary_blanc_reduces_to_the_pair_diffusivity(self):
        """For A+B, D_A,m must be exactly D_AB — Blanc's binary limit."""
        r = _mix({"N2": 0.6, "O2": 0.4}).compute(State(T=T0, P=P0))
        d_ab = binary_diffusion(get("N2"), get("O2"), T0, P0)
        assert r.effective_diffusivity("N2") == pytest.approx(d_ab, rel=1e-12)
        assert r.effective_diffusivity("O2") == pytest.approx(d_ab, rel=1e-12)

    def test_mixture_thermodynamic_identities(self):
        from statthermopy.constants import R

        for fractions, _ in self.CASES:
            r = _mix(fractions).compute(State(T=T0, P=P0))
            assert r.R_specific == pytest.approx(R / r.M_avg, rel=1e-14)
            assert r.rho == pytest.approx(P0 * r.M_avg / (R * T0), rel=1e-14)
            assert sum(r.x.values()) == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------- Phase 2: polar species (Stockmayer)


class TestStockmayerPolarPotential:
    """A permanent dipole is a molecular constant, so treating it stays first-principles.

    Water is strongly polar (1.85 D) and the spherically symmetric LJ fit over-predicts its
    viscosity by ~10 % at 300 K. The Stockmayer set plus Brokaw's delta correction cuts that to
    ~2 %. Only species carrying the parameters are affected.
    """

    def test_only_water_carries_stockmayer_parameters(self):
        polar = [n for n in list_molecules() if get(n).stockmayer is not None]
        assert polar == ["H2O"], f"unexpected Stockmayer species: {polar}"

    def test_reduced_dipole_of_water_is_unity(self):
        """delta = mu^2/(2 eps sigma^3) ~ 1.0 is the textbook value for water."""
        assert get("H2O").stockmayer.reduced_dipole == pytest.approx(1.0, abs=0.01)

    def test_delta_zero_reproduces_lennard_jones_exactly(self):
        from statthermopy.transport.collision import omega_11, omega_22

        for ts in (0.5, 1.0, 2.0, 10.0, 50.0):
            assert omega_22(ts, 0.0) == omega_22(ts)
            assert omega_11(ts, 0.0) == omega_11(ts)

    def test_dipole_raises_the_collision_integral(self):
        """A deeper attractive well deflects slow molecules more, so Omega goes up."""
        from statthermopy.transport.collision import omega_22

        assert omega_22(2.0, 1.0) > omega_22(2.0, 0.0)

    def test_every_non_polar_species_is_bit_for_bit_unchanged(self):
        """The refinement must not perturb the other 29 species."""
        import math

        from statthermopy.constants import k_B
        from statthermopy.transport.collision import omega_22

        for name in list_molecules():
            mol = get(name)
            if mol.stockmayer is not None:
                continue
            lj = mol.lennard_jones
            T = 300.0
            manual = (5 / 16) * math.sqrt(mol.molecular_mass * k_B * T / math.pi) / (
                lj.sigma_m**2 * omega_22(T / lj.epsilon_over_k)
            )
            got = TransportCalculator(mol, State(T=T, P=P0)).compute().mu
            assert got == manual, f"{name} changed with the Stockmayer path"

    @pytest.mark.parametrize(
        "T,mu_ref", [(300.0, 9.80e-6), (400.0, 1.32e-5), (500.0, 1.73e-5), (600.0, 2.14e-5)]
    )
    def test_water_viscosity_within_five_percent(self, T, mu_ref):
        mu = TransportCalculator(get("H2O"), State(T=T, P=P0)).compute().mu
        assert abs(mu - mu_ref) / mu_ref < 0.05, f"mu(H2O, {T} K) off by more than 5 %"

    def test_water_viscosity_beats_the_plain_lennard_jones_fit(self):
        """Guard the improvement: the LJ-only value was +9.8 % at 300 K, Stockmayer is +2.3 %."""
        import math

        from statthermopy.constants import k_B
        from statthermopy.transport.collision import omega_22

        mol = get("H2O")
        lj = mol.lennard_jones
        T, ref = 300.0, 9.80e-6
        lj_only = (5 / 16) * math.sqrt(mol.molecular_mass * k_B * T / math.pi) / (
            lj.sigma_m**2 * omega_22(T / lj.epsilon_over_k)
        )
        now = TransportCalculator(mol, State(T=T, P=P0)).compute().mu
        assert abs(now - ref) < abs(lj_only - ref), "Stockmayer should be closer than plain LJ"
        assert abs(now - ref) / ref < 0.04

    def test_conductivity_of_water_improved_but_still_limited(self):
        """k(H2O) improves with mu but keeps a large residual — the Eucken limitation.

        Documented, not hidden: the residual is the harmonic Eucken treatment of internal modes
        in a polar molecule, which needs Mason-Monchick and a rotational collision number.
        """
        k = TransportCalculator(get("H2O"), State(T=300.0, P=P0)).compute().k
        k_ref = 0.018563  # IAPWS dilute-gas limit at 300 K
        err = (k - k_ref) / k_ref
        assert 0.0 < err < 0.40, f"k(H2O) error {err:.1%} outside the documented band"

    def test_water_is_less_conductive_than_air_at_room_temperature(self):
        """A qualitative ordering the plain LJ fit got backwards.

        The IAPWS dilute-gas limit gives k(H2O) = 0.0186 W/m/K at 300 K against 0.0263 for air:
        water vapour is the *less* conductive of the two. With the LJ-only parameters the engine
        returned 0.0262 and put water above air. The Stockmayer refinement restores the ordering.
        """
        from statthermopy.transport.air import AirTransport

        k_water = TransportCalculator(get("H2O"), State(T=T0, P=P0)).compute().k
        k_air = AirTransport().dry(T0, P0).k
        assert k_water < k_air

    def test_humid_air_still_consistent(self):
        """The water refinement must flow through the mixture path without breaking it."""
        from statthermopy.transport.air import AirTransport

        dry = AirTransport().dry(T0, P0)
        humid = AirTransport().humid(T0, P0, relative_humidity=1.0)
        assert humid.mu < dry.mu       # water vapour is the least viscous component
        assert humid.k < dry.k         # and, at 300 K, the less conductive one
        assert humid.rho < dry.rho     # and the lightest


# ------------------------------------- Phase 2b: rotational relaxation (Mason-Monchick)


class TestMasonMonchickConductivity:
    """Rotational relaxation refines k where its central assumption holds.

    Mason & Monchick (1962) scale the rotational contribution by the *mass*-diffusion rate
    rho*D/mu. That is right for a non-polar molecule and wrong for a strongly polar one, where
    resonant dipole-dipole exchange moves rotational quanta without moving molecules. Measured
    over the four polar species, applying it makes their error worse by an amount that grows
    with the reduced dipole (r = 0.90). Polar species therefore keep Eucken.

    Z_rot comes from ultrasonic relaxation, independent of any conductivity data.
    """

    MM_SPECIES = ["H2", "N2", "O2", "CO", "NO", "Cl2", "CO2", "N2O", "CH4", "C2H4", "C2H6"]
    POLAR = ["H2O", "NH3", "SO2", "H2S"]

    def test_only_the_intended_species_declare_a_collision_number(self):
        declared = sorted(
            n for n in list_molecules() if get(n).rotational_relaxation is not None
        )
        assert declared == sorted(get(s).name.upper() for s in self.MM_SPECIES)

    def test_polar_species_keep_eucken(self):
        """Their k would get worse under Mason-Monchick, for a stated physical reason."""
        for name in self.POLAR:
            assert get(name).rotational_relaxation is None, name

    def test_monatomics_keep_eucken(self):
        for name in ("He", "Ne", "Ar", "Kr", "Xe"):
            assert get(name).rotational_relaxation is None, name

    def test_parker_scaling_increases_z_rot_with_temperature(self):
        rr = get("N2").rotational_relaxation
        eps = get("N2").lennard_jones.epsilon_over_k
        assert rr.z_rot(298.15, eps) == pytest.approx(rr.z_rot_298, rel=1e-12)
        assert rr.z_rot(1000.0, eps) > rr.z_rot(300.0, eps)

    @pytest.mark.parametrize(
        "name,k_ref,tol",
        [("N2", 0.0260, 0.03), ("O2", 0.0266, 0.03), ("H2", 0.187, 0.03),
         ("Cl2", 0.00895, 0.04), ("CH4", 0.0343, 0.06)],
    )
    def test_conductivity_of_the_refined_species(self, name, k_ref, tol):
        k = TransportCalculator(get(name), State(T=T0, P=P0)).compute().k
        assert abs(k - k_ref) / k_ref < tol, f"k({name}) off by more than {tol:.0%}"

    def test_nitrogen_conductivity_beats_the_eucken_value(self):
        """Eucken gave -4.0 %; Mason-Monchick gives -0.7 %."""
        k = TransportCalculator(get("N2"), State(T=T0, P=P0)).compute().k
        assert abs(k - 0.0260) / 0.0260 < 0.02

    def test_prandtl_of_air_matches_literature(self):
        """The headline consequence: Pr(dry air) was +4.3 % off, now under 1 %."""
        from statthermopy.transport.air import AirTransport

        Pr = AirTransport().dry(T0, P0).Pr
        assert abs(Pr - 0.707) / 0.707 < 0.02

    def test_air_conductivity_improved(self):
        from statthermopy.transport.air import AirTransport

        k = AirTransport().dry(T0, P0).k
        assert abs(k - 0.0263) / 0.0263 < 0.03

    def test_monatomic_limit_collapses_to_chapman_enskog(self):
        """With no internal modes the Mason-Monchick expression is the exact CE result."""
        import math

        from statthermopy.constants import R

        mol = get("Ar")
        r = TransportCalculator(mol, State(T=T0, P=P0)).compute()
        exact = (15.0 / 4.0) * (R / mol.molar_mass) * r.mu
        assert r.k == pytest.approx(exact, rel=1e-12)
        assert math.isfinite(r.k)

    def test_species_without_a_record_are_bit_for_bit_unchanged(self):
        """Eucken species must not shift: k = mu*cv*(9g-5)/4 exactly."""
        from statthermopy import Thermodynamics

        for name in list_molecules():
            mol = get(name)
            if mol.rotational_relaxation is not None:
                continue
            r = TransportCalculator(mol, State(T=T0, P=P0)).compute()
            th = Thermodynamics(mol, State(T=T0, P=P0)).compute()
            expected = r.mu * (th.Cv_m / mol.molar_mass) * (9.0 * th.gamma - 5.0) / 4.0
            assert r.k == expected, name

    def test_conductivity_stays_positive_and_monotonic_in_temperature(self):
        for name in self.MM_SPECIES:
            ks = [
                TransportCalculator(get(name), State(T=T, P=P0)).compute().k
                for T in (250.0, 300.0, 500.0, 1000.0)
            ]
            assert all(k > 0 for k in ks), name
            assert ks == sorted(ks), f"k({name}) not increasing with T"
