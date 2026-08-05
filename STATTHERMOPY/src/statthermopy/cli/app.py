"""Command-line interface for StatThermoPy.

Two entry points share one engine:

* an interactive scientific terminal (REPL) launched with ``statthermopy`` (no args), inspired by
  a Python session::

      > gas CH4
      > T = 800
      > P = 5e5
      > properties
      > plot Cp_m 300 1500
      > export csv ch4.csv

* a one-shot ``argparse`` mode for scripting (``statthermopy run --gas N2 --T 298.15 ...``).
"""

from __future__ import annotations

import argparse
import shlex
import sys
from cmd import Cmd

from ..core.state import State
from ..database import get, list_molecules
from ..fluids import available_fluids, get_fluid
from ..humidair import HumidAir
from ..io import Exporter
from ..mixture import IdealGasMixture
from ..plots import MOLAR_PROPS, PARTITION_PROPS, plot_property
from ..thermodynamics import Thermodynamics
from ..transport import TRANSPORT_PROPS, TRANSPORT_UNITS, TransportCalculator

__all__ = ["StatThermoPyShell", "main"]


def _split(arg: str) -> list[str]:
    """Tokenise a command line, preserving backslashes (Windows paths).

    ``shlex.split`` with ``posix=True`` mangles Windows paths (treats ``\\`` as an escape
    character). We use ``posix=False`` and strip surrounding quotes so paths like
    ``C:\\Users\\...\\n2.json`` survive intact.
    """
    try:
        tokens = shlex.split(arg, posix=False)
    except ValueError:
        return arg.split()
    return [t.strip().strip('"').strip("'") for t in tokens]

# Properties shown in the human-readable report and accepted by `plot`/`export table`.
_REPORT_PROPS = ["U_m", "H_m", "S_m", "A_m", "G_m", "Cv_m", "Cp_m", "gamma", "mu_m"]
_REPORT_PROPS_MASSIC = ["U_s", "H_s", "S_s", "A_s", "G_s", "Cv_s", "Cp_s", "R_specific"]
_UNITS = {
    "U_m": "J/mol", "H_m": "J/mol", "S_m": "J/mol/K", "A_m": "J/mol", "G_m": "J/mol",
    "Cv_m": "J/mol/K", "Cp_m": "J/mol/K", "gamma": "-", "mu_m": "J/mol",
    "U_s": "J/kg", "H_s": "J/kg", "S_s": "J/kg/K", "A_s": "J/kg", "G_s": "J/kg",
    "Cv_s": "J/kg/K", "Cp_s": "J/kg/K", "R_specific": "J/kg/K",
}


class StatThermoPyShell(Cmd):
    """Interactive scientific terminal."""

    intro = (
        "StatThermoPy — statistical thermodynamics terminal.\n"
        "Type 'help' for commands, 'list' to see available gases, 'quit' to exit."
    )
    prompt = "> "

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.molecule = None
        self.mixture = None
        self.T: float | None = None
        self.P: float | None = None
        self.V: float | None = None
        self.n: float | None = None
        self.m: float | None = None
        self._last_result = None

    # -- helpers --------------------------------------------------------------

    def _set_var(self, name: str, value: str) -> None:
        try:
            v = float(value)
        except ValueError:
            print(f"  error: cannot parse number {value!r}.")
            return
        setattr(self, name, v)

    def _make_state(self) -> State | None:
        if self.T is None:
            print("  error: set temperature first, e.g.  T = 298.15")
            return None
        kwargs = {"T": self.T}
        if self.P is not None:
            kwargs["P"] = self.P
        if self.V is not None:
            kwargs["V"] = self.V
        if self.n is not None:
            kwargs["n"] = self.n
        if self.m is not None:
            kwargs["m"] = self.m
        return State(**kwargs)

    # -- commands -------------------------------------------------------------

    def do_list(self, _arg: str) -> None:
        """List all available gases."""
        names = list_molecules()
        print(f"  {len(names)} gases: " + ", ".join(names))

    def do_gas(self, arg: str) -> None:
        """Select a pure gas:  gas N2"""
        name = arg.strip()
        if not name:
            print("  usage: gas <name>")
            return
        try:
            self.molecule = get(name)
            self.mixture = None
            print(f"  selected: {self.molecule}")
        except KeyError as exc:
            print(f"  error: {exc}")

    def do_mixture(self, arg: str) -> None:
        """Define an ideal-gas mixture:  mixture Ar:0.7 N2:0.3   (append 'mass' for mass basis)."""
        tokens = arg.split()
        if not tokens:
            print("  usage: mixture Ar:0.7 N2:0.3 [mass]")
            return
        basis = "mole"
        if tokens[-1].lower() in ("mass", "mole"):
            basis = tokens[-1].lower()
            tokens = tokens[:-1]
        fractions: dict[str, float] = {}
        for tok in tokens:
            if ":" not in tok:
                print(f"  error: expected name:frac, got {tok!r}")
                return
            nm, fr = tok.rsplit(":", 1)
            try:
                fractions[nm] = float(fr)
            except ValueError:
                print(f"  error: cannot parse fraction {fr!r}")
                return
        try:
            self.mixture = IdealGasMixture.from_names(fractions, basis=basis)
            self.molecule = None
            print(f"  mixture ({basis}): {self.mixture}")
        except (KeyError, ValueError) as exc:
            print(f"  error: {exc}")

    def do_fluids(self, _arg: str) -> None:
        """List the available predefined fluids (e.g. Air)."""
        print("  predefined fluids: " + ", ".join(available_fluids()))

    def do_fluid(self, arg: str) -> None:
        """Select a predefined fluid:  fluid Air   |   fluid Air h2o=0.01

        ``h2o=<x>`` adds water vapour at mole fraction ``x`` (dry constituents scaled to fill the
        rest). The fluid becomes the active mixture; use ``properties`` to evaluate it.
        """
        parts = arg.split()
        if not parts:
            print("  usage: fluid <name> [h2o=<mole fraction>]   (e.g. fluid Air h2o=0.01)")
            print(f"  available: {', '.join(available_fluids())}")
            return
        name = parts[0]
        water = 0.0
        for tok in parts[1:]:
            if tok.lower().startswith("h2o="):
                try:
                    water = float(tok.split("=", 1)[1])
                except ValueError:
                    print(f"  error: cannot parse {tok!r}")
                    return
        try:
            fluid = get_fluid(name)
            self.mixture = fluid.build(water_mole_fraction=water)
            self.molecule = None
        except (KeyError, ValueError) as exc:
            print(f"  error: {exc}")
            return
        print(f"  fluid: {fluid.name} — {fluid.description}")
        if water:
            print(f"    with H2O at x = {water}")
        print(f"    {self.mixture}")

    def do_T(self, arg: str) -> None:
        """Set temperature (K):  T = 298.15"""
        self._set_var("T", arg.replace("=", " ").strip())

    def do_P(self, arg: str) -> None:
        """Set pressure (Pa):  P = 101325"""
        self._set_var("P", arg.replace("=", " ").strip())

    def do_V(self, arg: str) -> None:
        """Set volume (m^3):  V = 0.024"""
        self._set_var("V", arg.replace("=", " ").strip())

    def do_n(self, arg: str) -> None:
        """Set amount (mol):  n = 1"""
        self._set_var("n", arg.replace("=", " ").strip())

    def do_m(self, arg: str) -> None:
        """Set mass (kg):  m = 0.028"""
        self._set_var("m", arg.replace("=", " ").strip())

    def do_state(self, _arg: str) -> None:
        """Show the currently set state variables."""
        for v in ("T", "P", "V", "n", "m"):
            val = getattr(self, v)
            if val is not None:
                print(f"  {v} = {val}")
        if self.molecule:
            print(f"  gas: {self.molecule}")
        if self.mixture:
            print(f"  mixture: {self.mixture}")

    def do_properties(self, _arg: str) -> None:
        """Compute and print all thermodynamic properties (molar + massic)."""
        if self.mixture is not None:
            st = self._make_state()
            if st is None:
                return
            res = self.mixture.compute(st)
            self._print_mixture(res)
            self._last_result = None
            return
        if self.molecule is None:
            print("  error: select a gas first, e.g.  gas N2")
            return
        st = self._make_state()
        if st is None:
            return
        res = Thermodynamics(self.molecule, st).compute()
        self._last_result = res
        self._print_properties(res)

    do_props = do_properties

    def do_modes(self, _arg: str) -> None:
        """Print the per-mode breakdown of the partition function and contributions."""
        if self.molecule is None or self.T is None:
            print("  error: select a gas and set T first.")
            return
        st = self._make_state()
        pf = Thermodynamics(self.molecule, st).partition
        vals = pf.evaluate(st)
        contribs = pf.contributions(st)
        print(f"  Partition function factors (T={st.resolve(self.molecule.molar_mass).T} K):")
        for f, ln in [("Qt", vals.ln_Qt), ("Qr", vals.ln_Qr),
                      ("Qv", vals.ln_Qv), ("Qe", vals.ln_Qe)]:
            print(f"    ln {f} = {ln:.6f}   {f} = {getattr(vals, f):.6e}")
        print(f"    ln Q  = {vals.ln_Qtotal:.6f}   Q  = {vals.Qtotal:.6e}")
        print("  Per-mode contributions (molar):")
        for name, c in contribs.items():
            print(f"    {name:13s} U={c.U_m:12.4f}  S={c.S_m:12.4f}  "
                  f"A={c.A_m:14.4f}  Cv={c.Cv_m:10.4f}")

    def do_plot(self, arg: str) -> None:
        """Plot a property vs T (saved to PNG):  plot Cp_m 300 1500 [out.png]"""
        if self.molecule is None:
            print("  error: select a gas first.")
            return
        parts = _split(arg)
        if not parts:
            print("  usage: plot <prop> [Tmin] [Tmax] [npoints] [out.png]")
            print(f"  props: {', '.join(MOLAR_PROPS + PARTITION_PROPS)}")
            return
        prop = parts[0]
        Tmin = float(parts[1]) if len(parts) > 1 else 300.0
        Tmax = float(parts[2]) if len(parts) > 2 else 1500.0
        npts = int(float(parts[3])) if len(parts) > 3 else 100
        out = parts[4] if len(parts) > 4 else f"{self.molecule.name}_{prop}.png"
        import numpy as np
        Ts = np.linspace(Tmin, Tmax, npts)
        P = self.P if self.P is not None else 101325.0
        ax = plot_property(self.molecule, prop, Ts, P=P, logy=(prop in PARTITION_PROPS))
        ax.figure.savefig(out, dpi=120, bbox_inches="tight")
        print(f"  saved plot -> {out}")

    def do_export(self, arg: str) -> None:
        """Export last result:  export csv out.csv   (csv|json|yaml|excel|latex)."""
        parts = _split(arg)
        if len(parts) < 2:
            print("  usage: export <csv|json|yaml|excel|latex> <file>")
            return
        fmt, path = parts[0], parts[1]
        if self._last_result is None:
            # compute on the fly if a gas is selected
            if self.molecule is None or self.T is None:
                print("  error: nothing to export; run `properties` first.")
                return
            self._last_result = Thermodynamics(self.molecule, self._make_state()).compute()
        exp = Exporter(self._last_result)
        try:
            {"csv": exp.to_csv, "json": exp.to_json, "yaml": exp.to_yaml,
             "latex": exp.to_latex, "excel": exp.to_excel}[fmt](path)
        except KeyError:
            print(f"  error: unknown format {fmt!r}. Use csv|json|yaml|excel|latex.")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"  error: {exc}")
            return
        print(f"  exported -> {path}")

    def do_humidair(self, arg: str) -> None:
        """Statistical humid air — maximum water-vapour solubility at the current (T, P).

        ``humidair``                -> saturation limit (maximum water the air can hold)
        ``humidair rh=0.5``         -> at 50 % relative humidity
        ``humidair w=0.012``        -> at humidity ratio 0.012 kg/kg dry air
        ``humidair x=0.02``         -> at H2O mole fraction 0.02
        ``humidair <psat|solubility|w|rh> Tmin Tmax [N] [out.png]`` -> plot vs T
        """
        parts = _split(arg)
        T = self.T if self.T is not None else 298.15
        P = self.P if self.P is not None else 101325.0
        ha = HumidAir()
        plot_keys = {"psat", "solubility", "w", "rh"}
        if parts and parts[0] in plot_keys:
            self._humidair_plot(ha, parts, P)
            return
        rh = w = xw = None
        for tok in parts:
            low = tok.lower()
            try:
                if low.startswith("rh="):
                    rh = float(tok[3:])
                elif low.startswith("w="):
                    w = float(tok[2:])
                elif low.startswith("x="):
                    xw = float(tok[2:])
            except ValueError:
                print(f"  error: cannot parse {tok!r}")
                return
        try:
            st = ha.state(T, P, relative_humidity=rh, humidity_ratio=w, mole_fraction=xw)
        except ValueError as exc:
            print(f"  error: {exc}")
            return
        self._print_humidair(st)

    @staticmethod
    def _humidair_plot(ha, parts: list[str], P: float) -> None:
        import numpy as np

        from ..humidair import plots as hp
        key = parts[0]
        Tmin = float(parts[1]) if len(parts) > 1 else 273.16
        Tmax = float(parts[2]) if len(parts) > 2 else 373.15
        npts = int(float(parts[3])) if len(parts) > 3 else 100
        out = parts[4] if len(parts) > 4 else f"humidair_{key}.png"
        Ts = np.linspace(Tmin, Tmax, npts)
        fn = {
            "psat": lambda: hp.plot_saturation_pressure_vs_T(Ts, model=ha),
            "solubility": lambda: hp.plot_max_solubility_vs_T(Ts, P=P, model=ha),
            "w": lambda: hp.plot_humidity_ratio_vs_T(Ts, P=P, model=ha),
            "rh": lambda: hp.plot_relative_humidity_vs_T(Ts, 0.01, P=P, model=ha),
        }[key]
        ax = fn()
        ax.figure.savefig(out, dpi=120, bbox_inches="tight")
        print(f"  saved plot -> {out}")

    def do_transport(self, arg: str) -> None:
        """Transport properties of the selected gas.

        ``transport``                      — print all properties at the current (T, P)
        ``transport <prop> Tmin Tmax [N] [out.png]``
                                          — plot a property vs T (saved to PNG)
        ``transport binary N2 O2``         — binary diffusion D_ij at current (T, P)

        Available props (use any single name): """ + ", ".join(TRANSPORT_PROPS) + """
        """
        if self.molecule is None:
            print("  error: select a gas first, e.g.  gas N2")
            return
        parts = _split(arg)
        if not parts:
            st = self._make_state()
            if st is None:
                return
            self._print_transport(self.molecule, st)
            return
        if parts[0] == "binary":
            self._transport_binary(parts[1:])
            return
        prop = parts[0]
        if prop not in TRANSPORT_PROPS:
            print(f"  error: unknown property {prop!r}. Choose from: {', '.join(TRANSPORT_PROPS)}")
            return
        Tmin = float(parts[1]) if len(parts) > 1 else 300.0
        Tmax = float(parts[2]) if len(parts) > 2 else 1500.0
        npts = int(float(parts[3])) if len(parts) > 3 else 100
        out = parts[4] if len(parts) > 4 else f"{self.molecule.name}_transport_{prop}.png"
        P = self.P if self.P is not None else 101325.0
        import numpy as np

        from ..transport.plots import plot_transport_vs_T
        Ts = np.linspace(Tmin, Tmax, npts)
        ax = plot_transport_vs_T(self.molecule, prop, Ts, P=P)
        ax.figure.savefig(out, dpi=120, bbox_inches="tight")
        print(f"  saved plot -> {out}")

    def _transport_binary(self, tokens: list[str]) -> None:
        """Print the binary diffusion coefficient D_ij of two gases at the current (T, P)."""
        from ..transport import binary_diffusion
        if len(tokens) < 2:
            print("  usage: transport binary <gasA> <gasB>")
            return
        st = self._make_state()
        if st is None:
            return
        T = self.T
        P = self.P if self.P is not None else 101325.0
        try:
            mol_i = get(tokens[0])
            mol_j = get(tokens[1])
        except KeyError as exc:
            print(f"  error: {exc}")
            return
        D = binary_diffusion(mol_i, mol_j, T, P)
        print(f"  D({tokens[0]},{tokens[1]}) @ T={T:.2f} K, P={P:.4g} Pa = {D:.6e} m^2/s")

    @staticmethod
    def _print_transport(mol, st: State) -> None:
        """Pretty-print all transport properties of ``mol`` at state ``st``."""
        res = TransportCalculator(mol, st).compute()
        print(f"  Transport properties — {mol.name} @ T={res.T:.4f} K, P={res.P:.6g} Pa")
        for prop in TRANSPORT_PROPS:
            val = getattr(res, prop)
            unit = TRANSPORT_UNITS.get(prop, "")
            print(f"    {prop:8s} = {val:14.6g}  {unit}")

    @staticmethod
    def _print_humidair(st) -> None:
        tag = "SATURATED (maximum solubility)" if st.saturated else "moist air"
        print(f"  Statistical Humid Air — {tag}")
        print(f"  T = {st.T:.4f} K ({st.T-273.15:.2f} C)   P = {st.P:.6g} Pa   "
              f"liquid model: {st.liquid_model}")
        print("  --- Saturation limit (max water the air can hold) ---")
        print(f"    P_sat                 = {st.P_sat:14.4f}  Pa")
        print(f"    x_H2O,max             = {st.x_h2o_max:14.6f}  mol/mol")
        print(f"    mass_fraction,max     = {st.mass_fraction_h2o_max:14.6f}  kg/kg")
        print(f"    humidity_ratio,max    = {st.humidity_ratio_max*1e3:14.6f}  g/kg dry air")
        print(f"    abs_humidity,max      = {st.absolute_humidity_max*1e3:14.6f}  g/m3")
        print(f"    vapor_conc,max        = {st.vapor_concentration_max:14.6f}  mol/m3")
        print("  --- Actual state ---")
        print(f"    relative_humidity     = {st.relative_humidity:14.6f}")
        print(f"    humidity_ratio        = {st.humidity_ratio*1e3:14.6f}  g/kg dry air")
        print(f"    x_H2O                 = {st.x_h2o:14.6f}  mol/mol")
        print(f"    degree_of_saturation  = {st.degree_of_saturation:14.6f}")
        print(f"    dew_point             = {st.dew_point:14.4f}  K ({st.dew_point-273.15:.2f} C)")
        print(f"    wet_bulb              = {st.wet_bulb:14.4f}  K ({st.wet_bulb-273.15:.2f} C)")
        print("  --- Mixture bulk ---")
        print(f"    density               = {st.density:14.6f}  kg/m3")
        print(f"    M_avg                 = {st.M_avg*1e3:14.6f}  g/mol")
        print(f"    R_specific            = {st.R_specific:14.6f}  J/kg/K")
        print("  --- Thermodynamics (molar) ---")
        for p, u in [("U_m", "J/mol"), ("H_m", "J/mol"), ("S_m", "J/mol/K"), ("A_m", "J/mol"),
                     ("G_m", "J/mol"), ("Cv_m", "J/mol/K"), ("Cp_m", "J/mol/K"),
                     ("gamma", "-"), ("mu_m", "J/mol"), ("S_mixing", "J/mol/K")]:
            print(f"    {p:9s} = {getattr(st, p):16.6f}  {u}")
        print("  --- Water-vapour partition-function contributions (to G_m, J/mol) ---")
        print(f"    {'factor':14s}{'G_m':>14}{'S_m':>12}{'Cv_m':>10}")
        for name, c in st.vapor_mode_contributions.items():
            print(f"    {name:14s}{c['G_m']:14.3f}{c['S_m']:12.4f}{c['Cv_m']:10.4f}")

    def do_quit(self, _arg: str) -> bool:
        """Exit the terminal."""
        print("  goodbye.")
        return True

    do_EOF = do_quit
    do_exit = do_quit

    # -- pretty printing ------------------------------------------------------

    @staticmethod
    def _print_properties(res) -> None:
        print(f"  State: T={res.T:.4f} K  P={res.P:.6g} Pa  "
              f"V={res.V:.6e} m^3  n={res.n:.6g} mol  m={res.m:.6g} kg")
        print(f"  Molar mass: {res.molar_mass*1e3:.4f} g/mol")
        print("  --- Molar (per mol) ---")
        for p in _REPORT_PROPS:
            print(f"    {p:7s} = {getattr(res, p):14.6f}  {_UNITS.get(p,'')}")
        print("  --- Massic (per kg) ---")
        for p in _REPORT_PROPS_MASSIC:
            print(f"    {p:14s} = {getattr(res, p):14.6f}  {_UNITS.get(p,'')}")
        print("  --- Partition function ---")
        for f in ("Qt", "Qr", "Qv", "Qe", "Qtotal"):
            print(f"    {f:7s} = {getattr(res, f):14.6e}")
        print(f"    ln Q   = {res.ln_Qtotal:.6f}")

    @staticmethod
    def _print_mixture(res) -> None:
        comp = ", ".join(f"{k}={v:.4f}" for k, v in res.x.items())
        print(f"  Mixture ({res.basis}): {comp}")
        print(f"  T={res.T:.4f} K  P={res.P:.6g} Pa  M_avg={res.M_avg*1e3:.4f} g/mol  "
              f"R_specific={res.R_specific:.4f} J/kg/K")
        print("  --- Molar (per mol) ---")
        for p in ("U_m", "H_m", "S_m", "A_m", "G_m", "Cv_m", "Cp_m", "gamma", "mu_m"):
            print(f"    {p:8s} = {getattr(res, p):14.6f}  {_UNITS.get(p,'')}")
        print(f"    {'S_mixing':8s} = {res.S_mixing:14.6f}  J/mol/K")
        print("  --- Massic (per kg) ---")
        for p in ("U_s", "H_s", "S_s", "A_s", "G_s", "Cv_s", "Cp_s", "R_specific"):
            print(f"    {p:14s} = {getattr(res, p):14.6f}  {_UNITS.get(p,'')}")
        if res.components:
            print("  --- Per-component contribution to molar totals ---")
            print(f"    {'species':8s}{'x_i':>9}{'U':>12}{'S':>10}{'G':>14}{'Cp':>9}")
            for name, c in res.components.items():
                print(f"    {name:8s}{c.x:9.4f}{c.U_contrib:12.2f}{c.S_contrib:10.3f}"
                      f"{c.G_contrib:14.2f}{c.Cp_contrib:9.3f}")


# --- one-shot argparse mode --------------------------------------------------


def _run_one_shot(args: argparse.Namespace) -> None:
    if args.fluid:
        mix = get_fluid(args.fluid).build(water_mole_fraction=args.humidity)
        st = State(T=args.T, P=args.P)
        res = mix.compute(st)
        StatThermoPyShell._print_mixture(res)
        if args.export:
            fmt, path = args.export
            Exporter(res).to_json(path) if fmt == "json" else None
        return
    if args.mixture:
        fractions: dict[str, float] = {}
        for tok in args.mixture:
            nm, fr = tok.rsplit(":", 1)
            fractions[nm] = float(fr)
        mix = IdealGasMixture.from_names(fractions, basis=args.basis)
        st = State(T=args.T, P=args.P)
        res = mix.compute(st)
        StatThermoPyShell._print_mixture(res)
        if args.export:
            fmt, path = args.export
            Exporter(res).to_json(path) if fmt == "json" else None
        return
    mol = get(args.gas)
    st = State(T=args.T, P=args.P, n=args.n)
    res = Thermodynamics(mol, st).compute()
    StatThermoPyShell._print_properties(res)
    if args.export:
        fmt, path = args.export
        exp = Exporter(res)
        getattr(exp, f"to_{fmt}", exp.to_csv)(path)
        print(f"  exported -> {path}")


def _run_transport(args: argparse.Namespace) -> None:
    """One-shot transport computation: print a property (or all) at (T, P), optionally plot."""
    import numpy as np

    from ..transport import binary_diffusion
    from ..transport.plots import plot_transport_vs_T

    if args.binary:
        mol_i = get(args.binary[0])
        mol_j = get(args.binary[1])
        D = binary_diffusion(mol_i, mol_j, args.T, args.P)
        print(f"  D({args.binary[0]},{args.binary[1]}) @ T={args.T:.2f} K, P={args.P:.4g} Pa "
              f"= {D:.6e} m^2/s")
        return
    mol = get(args.gas)
    st = State(T=args.T, P=args.P)
    res = TransportCalculator(mol, st).compute()
    if args.prop:
        if args.prop not in TRANSPORT_PROPS:
            print(f"  error: unknown property {args.prop!r}. Choose from: {', '.join(TRANSPORT_PROPS)}")
            return
        print(f"  {args.prop}({mol.name}) @ T={res.T:.2f} K, P={res.P:.4g} Pa "
              f"= {getattr(res, args.prop):.6g} {TRANSPORT_UNITS.get(args.prop, '')}")
    else:
        StatThermoPyShell._print_transport(mol, st)
    if args.png:
        prop = args.prop or "mu"
        Ts = np.linspace(args.Tmin, args.Tmax, args.N)
        ax = plot_transport_vs_T(mol, prop, Ts, P=args.P)
        ax.figure.savefig(args.png, dpi=120, bbox_inches="tight")
        print(f"  saved plot -> {args.png}")


def _run_humidair(args: argparse.Namespace) -> None:
    """One-shot humid-air computation: print the saturated (or given-RH) state at (T, P)."""
    st = HumidAir().state(args.T, args.P, relative_humidity=args.rh)
    StatThermoPyShell._print_humidair(st)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="statthermopy", description="StatThermoPy CLI.")
    sub = p.add_subparsers(dest="command")

    # interactive (default)
    sub.add_parser("repl", help="interactive scientific terminal (default)")

    # one-shot
    run = sub.add_parser("run", help="one-shot computation")
    run.add_argument("--gas", help="gas name")
    run.add_argument("--T", type=float, default=298.15, help="temperature (K)")
    run.add_argument("--P", type=float, default=101325.0, help="pressure (Pa)")
    run.add_argument("--n", type=float, default=1.0, help="moles")
    run.add_argument("--mixture", nargs="+", help="mixture spec, e.g. Ar:0.7 N2:0.3")
    run.add_argument("--fluid", help="predefined fluid, e.g. Air (overrides --gas/--mixture)")
    run.add_argument("--humidity", type=float, default=0.0,
                     help="water-vapour mole fraction for --fluid (e.g. 0.01)")
    run.add_argument("--basis", default="mole", choices=["mole", "mass"])
    run.add_argument("--export", nargs=2, metavar=("FMT", "PATH"),
                     help="export result, e.g. csv out.csv")

    # one-shot transport
    tr = sub.add_parser("transport", help="one-shot transport properties")
    tr.add_argument("--gas", required=True, help="gas name")
    tr.add_argument("--T", type=float, default=300.0, help="temperature (K)")
    tr.add_argument("--P", type=float, default=101325.0, help="pressure (Pa)")
    tr.add_argument("--prop", help=f"property to report; default = all. One of: {', '.join(TRANSPORT_PROPS)}")
    tr.add_argument("--binary", nargs=2, metavar=("GAS_A", "GAS_B"),
                    help="binary diffusion D_ij of two gases (overrides --prop)")
    tr.add_argument("--Tmin", type=float, default=300.0, help="plot range start (K)")
    tr.add_argument("--Tmax", type=float, default=1500.0, help="plot range end (K)")
    tr.add_argument("--N", type=int, default=100, help="number of plot points")
    tr.add_argument("--png", help="save a property-vs-T plot to this path")

    # one-shot humid air
    hu = sub.add_parser("humidair", help="statistical humid air / maximum water solubility")
    hu.add_argument("--T", type=float, default=298.15, help="temperature (K)")
    hu.add_argument("--P", type=float, default=101325.0, help="total pressure (Pa)")
    hu.add_argument("--rh", type=float, default=None,
                    help="relative humidity 0-1 (default: saturated = maximum solubility)")

    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``statthermopy`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        _run_one_shot(args)
        return 0
    if args.command == "transport":
        _run_transport(args)
        return 0
    if args.command == "humidair":
        _run_humidair(args)
        return 0
    # default: interactive REPL
    try:
        StatThermoPyShell().cmdloop()
    except KeyboardInterrupt:
        print("\n  goodbye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
