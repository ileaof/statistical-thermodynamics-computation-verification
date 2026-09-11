"""CLI access to mixture transport — the D-4 gap the audit recorded.

``MixtureTransportCalculator`` was always generic in the number of components, but the only way
in from the command line was ``airtransport``, which builds air itself; ``transport`` required a
single ``--gas``. These tests pin the new route: any composition, one-shot or interactive.
"""

from __future__ import annotations

import io as _io

import pytest

from statthermopy.cli.app import StatThermoPyShell, main


def _run(capsys, argv):
    assert main(argv) == 0
    return capsys.readouterr().out


def _shell():
    sh = StatThermoPyShell(stdin=_io.StringIO(""))
    sh.use_rawinput = False
    return sh


class TestOneShotMixtureTransport:
    def test_mixture_reports_the_full_table(self, capsys):
        out = _run(capsys, ["transport", "--mixture", "CO2:0.5", "CH4:0.5", "--T", "300"])
        assert "Mixture transport" in out
        assert "CO2=0.5000, CH4=0.5000" in out
        assert "Wilke" in out and "Mason-Saxena" in out and "Blanc" in out
        assert "Dynamic viscosity" in out and "Thermal conductivity" in out

    def test_per_species_breakdown_is_printed(self, capsys):
        out = _run(capsys, ["transport", "--mixture", "CO2:0.5", "CH4:0.5", "--T", "300"])
        assert "Per-species contributions" in out
        assert "Sc_i" in out
        assert "CO2" in out and "CH4" in out

    def test_single_property(self, capsys):
        out = _run(
            capsys, ["transport", "--mixture", "CO2:0.5", "CH4:0.5", "--prop", "mu", "--T", "300"]
        )
        assert "mu(mixture)" in out

    def test_generic_mixture_has_no_implicit_trace(self, capsys):
        """Sc must not silently be water diffusing through a mixture that contains none."""
        out = _run(capsys, ["transport", "--mixture", "CO2:0.5", "CH4:0.5", "--T", "300"])
        assert "trace species" not in out

    def test_trace_species_can_be_named(self, capsys):
        out = _run(
            capsys,
            ["transport", "--mixture", "CO2:0.5", "CH4:0.5", "--trace", "H2O", "--T", "300"],
        )
        assert "Schmidt number" in out
        assert "trace species H2O" in out

    def test_asking_for_sc_without_a_trace_explains_how(self, capsys):
        out = _run(
            capsys, ["transport", "--mixture", "CO2:0.5", "CH4:0.5", "--prop", "Sc", "--T", "300"]
        )
        assert "--trace" in out

    def test_predefined_fluid(self, capsys):
        out = _run(capsys, ["transport", "--fluid", "Air", "--T", "300"])
        assert "Mixture transport" in out and "Air" in out

    def test_mass_basis_converts_to_mole_fractions(self, capsys):
        out = _run(
            capsys,
            ["transport", "--mixture", "CO2:0.5", "CH4:0.5", "--basis", "mass", "--T", "300"],
        )
        # CH4 is far lighter, so equal masses give it the larger mole fraction.
        assert "CO2=0.2671" in out and "CH4=0.7329" in out

    def test_accuracy_band_is_reported(self, capsys):
        out = _run(capsys, ["transport", "--mixture", "CO2:0.5", "CH4:0.5", "--T", "300"])
        assert "validated accuracy" in out

    def test_plot_is_written(self, capsys, tmp_path):
        png = tmp_path / "mix.png"
        out = _run(
            capsys,
            ["transport", "--mixture", "CO2:0.5", "CH4:0.5", "--prop", "mu",
             "--Tmin", "300", "--Tmax", "600", "--N", "10", "--png", str(png)],
        )
        assert "saved plot" in out
        assert png.exists()

    def test_pure_gas_path_is_untouched(self, capsys):
        out = _run(capsys, ["transport", "--gas", "N2", "--T", "300"])
        assert "Transport properties" in out
        assert "Mixture transport" not in out

    def test_binary_diffusion_still_works(self, capsys):
        out = _run(capsys, ["transport", "--gas", "N2", "--binary", "N2", "O2", "--T", "300"])
        assert "D(N2,O2)" in out


class TestOneShotErrors:
    def test_no_species_at_all(self, capsys):
        out = _run(capsys, ["transport", "--T", "300"])
        assert "--gas, --mixture or --fluid" in out

    def test_malformed_spec(self, capsys):
        out = _run(capsys, ["transport", "--mixture", "CO2", "--T", "300"])
        assert "name:fraction" in out

    def test_unparseable_fraction(self, capsys):
        out = _run(capsys, ["transport", "--mixture", "CO2:half", "--T", "300"])
        assert "cannot parse fraction" in out

    def test_unknown_species(self, capsys):
        out = _run(capsys, ["transport", "--mixture", "CO2:0.5", "XX:0.5", "--T", "300"])
        assert "Unknown molecule" in out


class TestInteractiveMixtureTransport:
    def test_transport_uses_the_active_mixture(self, capsys):
        sh = _shell()
        sh.onecmd("mixture CO2:0.5 CH4:0.5")
        sh.onecmd("T = 300")
        sh.onecmd("P = 101325")
        sh.onecmd("transport")
        out = capsys.readouterr().out
        assert "Mixture transport" in out
        assert "CO2" in out and "CH4" in out

    def test_trace_keyword(self, capsys):
        sh = _shell()
        sh.onecmd("mixture CO2:0.5 CH4:0.5")
        sh.onecmd("T = 300")
        sh.onecmd("transport trace=H2O")
        out = capsys.readouterr().out
        assert "trace species H2O" in out

    def test_binary_still_names_its_own_pair(self, capsys):
        """``transport binary`` must keep working while a mixture is selected."""
        sh = _shell()
        sh.onecmd("mixture CO2:0.5 CH4:0.5")
        sh.onecmd("T = 300")
        sh.onecmd("P = 101325")
        sh.onecmd("transport binary CO2 CH4")
        out = capsys.readouterr().out
        assert "D(CO2,CH4)" in out

    def test_plot_from_the_shell(self, capsys, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sh = _shell()
        sh.onecmd("mixture CO2:0.5 CH4:0.5")
        sh.onecmd("T = 300")
        sh.onecmd("transport mu 300 600 10 m.png")
        out = capsys.readouterr().out
        assert "saved plot" in out
        assert (tmp_path / "m.png").exists()

    def test_unknown_property_lists_the_choices(self, capsys):
        sh = _shell()
        sh.onecmd("mixture CO2:0.5 CH4:0.5")
        sh.onecmd("T = 300")
        sh.onecmd("transport bogus 300 600")
        out = capsys.readouterr().out
        assert "unknown property" in out

    def test_nothing_selected_names_both_options(self, capsys):
        sh = _shell()
        sh.onecmd("T = 300")
        sh.onecmd("transport")
        out = capsys.readouterr().out
        assert "gas or a mixture" in out

    def test_pure_gas_in_the_shell_is_untouched(self, capsys):
        sh = _shell()
        sh.onecmd("gas N2")
        sh.onecmd("T = 300")
        sh.onecmd("P = 101325")
        sh.onecmd("transport")
        out = capsys.readouterr().out
        assert "Transport properties" in out
        assert "Mixture transport" not in out

    def test_selecting_a_gas_after_a_mixture_switches_back(self, capsys):
        sh = _shell()
        sh.onecmd("mixture CO2:0.5 CH4:0.5")
        sh.onecmd("gas N2")
        sh.onecmd("T = 300")
        sh.onecmd("P = 101325")
        sh.onecmd("transport")
        out = capsys.readouterr().out
        assert "Transport properties" in out


class TestAgreesWithTheApi:
    def test_cli_numbers_match_the_engine(self, capsys):
        from statthermopy import State
        from statthermopy.mixture import IdealGasMixture
        from statthermopy.transport.air import MixtureTransportCalculator

        out = _run(
            capsys,
            ["transport", "--mixture", "CO2:0.5", "CH4:0.5", "--prop", "mu",
             "--T", "300", "--P", "101325"],
        )
        ref = MixtureTransportCalculator(
            IdealGasMixture.from_names({"CO2": 0.5, "CH4": 0.5})
        ).compute(State(T=300.0, P=101325.0))
        assert f"{ref.mu:.6g}" in out

    @pytest.mark.parametrize(
        "spec",
        [
            ["N2:1.0"],
            ["N2:0.5", "O2:0.5"],
            ["N2:0.7", "O2:0.2", "Ar:0.1"],
            ["N2:0.76", "O2:0.20", "Ar:0.009", "CO2:0.0004", "H2O:0.03"],
        ],
    )
    def test_any_number_of_components(self, capsys, spec):
        out = _run(capsys, ["transport", "--mixture", *spec, "--T", "300"])
        assert "Mixture transport" in out
        assert "Dynamic viscosity" in out


class TestMixtureExportAndClearErrors:
    """Paths a user reaches for that used to fail silently or say the wrong thing."""

    @pytest.mark.parametrize("fmt,ext", [("csv", "csv"), ("json", "json"), ("yaml", "yaml"),
                                          ("latex", "tex"), ("excel", "xlsx")])
    def test_mixture_exports_in_every_format(self, capsys, tmp_path, fmt, ext):
        """It used to write a file only for JSON, and say nothing for the rest."""
        out_file = tmp_path / f"m.{ext}"
        out = _run(capsys, ["run", "--mixture", "CO2:0.5", "CH4:0.5", "--T", "300",
                            "--export", fmt, str(out_file)])
        assert "exported ->" in out
        assert out_file.exists() and out_file.stat().st_size > 0

    def test_fluid_exports_too(self, capsys, tmp_path):
        out_file = tmp_path / "air.csv"
        out = _run(capsys, ["run", "--fluid", "Air", "--T", "300",
                            "--export", "csv", str(out_file)])
        assert "exported ->" in out and out_file.exists()

    def test_unknown_format_is_reported_not_silently_coerced(self, capsys, tmp_path):
        """A pure gas used to fall back to CSV; a mixture wrote nothing at all."""
        for spec in (["--gas", "N2"], ["--mixture", "CO2:0.5", "CH4:0.5"]):
            out = _run(capsys, ["run", *spec, "--T", "300",
                                "--export", "bogus", str(tmp_path / "x.out")])
            assert "unknown format" in out
            assert not (tmp_path / "x.out").exists()

    def test_shell_export_works_right_after_mixture_properties(self, capsys, tmp_path,
                                                               monkeypatch):
        """It used to claim nothing had been computed, immediately after printing it."""
        monkeypatch.chdir(tmp_path)
        sh = _shell()
        sh.onecmd("mixture CO2:0.5 CH4:0.5")
        sh.onecmd("T = 300")
        sh.onecmd("P = 101325")
        sh.onecmd("properties")
        sh.onecmd("export csv m.csv")
        out = capsys.readouterr().out
        assert "exported ->" in out
        assert (tmp_path / "m.csv").exists()

    def test_shell_export_computes_a_mixture_on_the_fly(self, capsys, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sh = _shell()
        sh.onecmd("mixture CO2:0.5 CH4:0.5")
        sh.onecmd("T = 300")
        sh.onecmd("P = 101325")
        sh.onecmd("export json m.json")
        out = capsys.readouterr().out
        assert "exported ->" in out

    def test_plot_on_a_mixture_points_at_the_working_command(self, capsys):
        sh = _shell()
        sh.onecmd("mixture CO2:0.5 CH4:0.5")
        sh.onecmd("T = 300")
        sh.onecmd("plot Cp_m 300 600")
        out = capsys.readouterr().out
        assert "pure-gas properties" in out
        assert "transport" in out and "gas CO2" in out

    def test_modes_on_a_mixture_points_at_the_working_command(self, capsys):
        sh = _shell()
        sh.onecmd("mixture CO2:0.5 CH4:0.5")
        sh.onecmd("T = 300")
        sh.onecmd("modes")
        out = capsys.readouterr().out
        assert "partition function" in out
        assert "properties" in out

    def test_nothing_to_export_message_names_both_options(self, capsys):
        sh = _shell()
        sh.onecmd("export csv x.csv")
        out = capsys.readouterr().out
        assert "gas or mixture" in out
