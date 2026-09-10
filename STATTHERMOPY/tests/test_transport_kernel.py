"""The vectorised transport kernel must reproduce the reference path exactly enough.

:class:`~statthermopy.transport.kernel.TransportKernel` reorganises the data for array
evaluation; it must not change the physics. These tests pin it against
:class:`~statthermopy.transport.air.MixtureTransportCalculator`, which stays the authority.

The only admitted difference is the ``C_v,i(T)`` table interpolation, which the tolerance below
quantifies (a few parts in 10⁻⁶ with the default grid). Viscosity, density and diffusion carry no
tabulation and must match to machine precision.
"""

from __future__ import annotations

import numpy as np
import pytest

from statthermopy import State
from statthermopy.mixture import IdealGasMixture
from statthermopy.transport import TransportKernel
from statthermopy.transport.air import MixtureTransportCalculator

CASES = [
    (["N2"], [1.0]),
    (["N2", "O2"], [0.5, 0.5]),
    (["He", "Xe"], [0.5, 0.5]),
    (["N2", "O2", "Ar"], [0.7, 0.2, 0.1]),
    (["N2", "O2", "Ar", "CO2"], [0.7808, 0.2095, 0.0093, 0.0004]),
    (["N2", "O2", "Ar", "CO2", "H2O"], [0.76, 0.20, 0.009, 0.0004, 0.03]),
    (["CCl4", "I2", "C6H6", "SO2"], [0.4, 0.3, 0.2, 0.1]),
    (["H2", "He", "CH4", "NH3"], [0.4, 0.3, 0.2, 0.1]),
]

#: Tabulation tolerance for quantities that depend on the interpolated C_v.
TABLE_RTOL = 1e-5
#: Everything else must be exact to floating-point round-off.
EXACT_RTOL = 1e-12


def _reference(species, x, T, P):
    mix = IdealGasMixture.from_names(dict(zip(species, x, strict=False)))
    return MixtureTransportCalculator(mix).compute(State(T=T, P=P))


class TestKernelMatchesReference:
    @pytest.mark.parametrize("species,x", CASES)
    def test_untabulated_quantities_are_exact(self, species, x):
        """mu, rho, M and the diffusion matrix involve no interpolation."""
        out = TransportKernel(species)(np.array(300.0), np.array(101325.0), np.array(x))
        ref = _reference(species, x, 300.0, 101325.0)
        assert float(out["mu"]) == pytest.approx(ref.mu, rel=EXACT_RTOL)
        assert float(out["rho"]) == pytest.approx(ref.rho, rel=EXACT_RTOL)
        assert float(out["M_mix"]) == pytest.approx(ref.M_avg, rel=EXACT_RTOL)
        for i, name in enumerate(species):
            assert float(out["D_im"][i]) == pytest.approx(
                ref.effective_diffusivity(name), rel=EXACT_RTOL
            ), name

    @pytest.mark.parametrize("species,x", CASES)
    def test_tabulated_quantities_within_interpolation_tolerance(self, species, x):
        out = TransportKernel(species)(np.array(300.0), np.array(101325.0), np.array(x))
        ref = _reference(species, x, 300.0, 101325.0)
        for prop, ref_value in (("k", ref.k), ("Pr", ref.Pr), ("alpha", ref.alpha),
                                ("nu", ref.nu), ("cp_s", ref.cp_s), ("gamma", ref.gamma)):
            assert float(out[prop]) == pytest.approx(ref_value, rel=TABLE_RTOL), prop

    @pytest.mark.parametrize("T", [250.0, 300.0, 500.0, 1000.0, 2000.0])
    def test_agreement_holds_across_temperature(self, T):
        species = ["N2", "O2", "Ar", "CO2", "H2O"]
        x = [0.76, 0.20, 0.009, 0.0004, 0.03]
        out = TransportKernel(species)(np.array(T), np.array(101325.0), np.array(x))
        ref = _reference(species, x, T, 101325.0)
        assert float(out["mu"]) == pytest.approx(ref.mu, rel=EXACT_RTOL)
        assert float(out["k"]) == pytest.approx(ref.k, rel=TABLE_RTOL)

    @pytest.mark.parametrize("P", [1.0e3, 1.0e4, 101325.0, 1.0e6])
    def test_agreement_holds_across_pressure(self, P):
        species = ["N2", "O2"]
        x = [0.79, 0.21]
        out = TransportKernel(species)(np.array(300.0), np.array(P), np.array(x))
        ref = _reference(species, x, 300.0, P)
        assert float(out["rho"]) == pytest.approx(ref.rho, rel=EXACT_RTOL)
        assert float(out["D_im"][0]) == pytest.approx(
            ref.effective_diffusivity("N2"), rel=EXACT_RTOL
        )

    def test_thirty_species_mixture(self):
        from statthermopy.database import list_molecules

        names = list_molecules()
        x = [1.0 / len(names)] * len(names)
        out = TransportKernel(names)(np.array(300.0), np.array(101325.0), np.array(x))
        ref = _reference(names, x, 300.0, 101325.0)
        assert float(out["mu"]) == pytest.approx(ref.mu, rel=EXACT_RTOL)
        assert float(out["k"]) == pytest.approx(ref.k, rel=TABLE_RTOL)


class TestKernelArraySemantics:
    SPECIES = ["N2", "O2", "Ar", "CO2", "H2O"]
    X = [0.76, 0.20, 0.009, 0.0004, 0.03]

    def test_shapes_broadcast(self):
        kern = TransportKernel(self.SPECIES)
        T = np.full((4, 5), 300.0)
        P = np.full((4, 5), 101325.0)
        X = np.broadcast_to(self.X, (4, 5, 5))
        out = kern(T, P, X)
        assert out["mu"].shape == (4, 5)
        assert out["D_im"].shape == (4, 5, 5)
        assert out["Sc_i"].shape == (4, 5, 5)

    def test_every_cell_gets_its_own_answer(self):
        """A temperature gradient must produce a viscosity gradient, cell by cell."""
        kern = TransportKernel(self.SPECIES)
        T = np.array([250.0, 300.0, 400.0, 800.0])
        out = kern(T, np.full(4, 101325.0), np.tile(self.X, (4, 1)))
        assert np.all(np.diff(out["mu"]) > 0.0)
        assert np.all(np.diff(out["k"]) > 0.0)

    def test_array_result_equals_looping_one_cell_at_a_time(self):
        kern = TransportKernel(self.SPECIES)
        T = np.array([250.0, 300.0, 400.0])
        P = np.array([1.0e4, 101325.0, 2.0e5])
        X = np.tile(self.X, (3, 1))
        block = kern(T, P, X)
        for i in range(3):
            single = kern(T[i], P[i], X[i])
            for prop in ("mu", "k", "rho", "Pr"):
                assert block[prop][i] == pytest.approx(float(single[prop]), rel=1e-14)

    def test_zero_fraction_drops_out(self):
        """A zero entry must behave as if the species were not listed."""
        kern = TransportKernel(["N2", "H2O"])
        with_zero = kern(np.array(300.0), np.array(101325.0), np.array([1.0, 0.0]))
        pure = TransportKernel(["N2"])(np.array(300.0), np.array(101325.0), np.array([1.0]))
        assert float(with_zero["mu"]) == pytest.approx(float(pure["mu"]), rel=1e-14)
        assert float(with_zero["rho"]) == pytest.approx(float(pure["rho"]), rel=1e-14)

    def test_rows_are_normalised(self):
        kern = TransportKernel(["N2", "O2"])
        a = kern(np.array(300.0), np.array(101325.0), np.array([0.79, 0.21]))
        b = kern(np.array(300.0), np.array(101325.0), np.array([7.9, 2.1]))
        assert float(a["mu"]) == pytest.approx(float(b["mu"]), rel=1e-14)

    def test_species_order_is_the_axis_order(self):
        a = TransportKernel(["N2", "O2"])(np.array(300.0), np.array(101325.0),
                                           np.array([0.79, 0.21]))
        b = TransportKernel(["O2", "N2"])(np.array(300.0), np.array(101325.0),
                                           np.array([0.21, 0.79]))
        assert float(a["mu"]) == pytest.approx(float(b["mu"]), rel=1e-14)

    def test_no_nan_or_inf(self):
        kern = TransportKernel(self.SPECIES)
        T = np.linspace(250.0, 2000.0, 50)
        out = kern(T, np.full(50, 101325.0), np.tile(self.X, (50, 1)))
        for name, arr in out.items():
            assert np.all(np.isfinite(arr)), name


class TestKernelValidation:
    def test_wrong_species_axis_is_rejected(self):
        kern = TransportKernel(["N2", "O2"])
        with pytest.raises(ValueError, match="expected 2"):
            kern(np.array(300.0), np.array(101325.0), np.array([1.0, 0.0, 0.0]))

    def test_negative_fraction_is_rejected(self):
        kern = TransportKernel(["N2", "O2"])
        with pytest.raises(ValueError, match=">= 0"):
            kern(np.array(300.0), np.array(101325.0), np.array([1.2, -0.2]))

    def test_all_zero_row_is_rejected(self):
        kern = TransportKernel(["N2", "O2"])
        with pytest.raises(ValueError, match="positive fraction"):
            kern(np.array(300.0), np.array(101325.0), np.array([0.0, 0.0]))

    def test_extrapolation_is_refused_not_silent(self):
        """Outside the tabulated range the kernel must fail loudly."""
        kern = TransportKernel(["N2"], T_min=200.0, T_max=1000.0)
        with pytest.raises(ValueError, match="outside the tabulated range"):
            kern(np.array(1500.0), np.array(101325.0), np.array([1.0]))

    def test_empty_species_list_is_rejected(self):
        with pytest.raises(ValueError, match="At least one species"):
            TransportKernel([])
