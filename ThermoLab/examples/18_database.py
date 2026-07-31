"""Example 18 — What's in the database: gases, liquids, solids, hydrates.

Introspects ThermoLab's curated fluid database (``list_fluids()``) and
classifies every supported fluid by phase capability, so you can pick the
right fluid when adapting the other examples.

* **Gases**   — every supported fluid has a vapour phase; listed with Tc.
* **Liquids** — condensables, split by critical temperature: *ambient* liquids
  (Tc above room temperature, liquid near ambient conditions) and
  *cryogenic* liquids (Tc <= 300 K).
* **Solids**  — detected by attempting ThermoPack's ``init_solid`` on each
  fluid. That call is an *uncatchable Fortran STOP* when a fluid has no solid
  correlation, so each probe runs in its own subprocess; a fluid is reported as
  solid-capable only if the subprocess survives. (This makes the example take
  ~30-60 s — it is a one-off discovery tool, not a hot loop.)
* **Hydrates** — ThermoPack's Python wrapper exposes no hydrate model in this
  build, so this lists the *hydrate-former* components present in the database
  (water is the lattice host). Use Example 17's correlation approach for actual
  equilibrium calculations.

Run with::

    python examples/18_database.py
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np

from thermolab import Gas, list_fluids
from thermolab._fluid_db import FLUID_ALIASES, KNOWN_SUPPORTED

# Classic gas-hydrate formers (structure I/II). Only those present in the
# database will be listed; CH4 is supported, while heavier alkanes
# (C2H6/C3H8/C4H10) are shown for reference but are not in this ThermoPack build.
HYDRATE_FORMERS = {
    "CO2", "H2S", "N2", "O2", "Ar", "Xe", "Kr", "H2", "N2O",
    "CH4", "C2H6", "C3H8", "C4H10",
}
HYDRATE_HOST = "H2O"

AMBIENT_Tc = 300.0  # K, split between ambient and cryogenic liquids


def _probe_solid(comp: str) -> bool:
    """True if ThermoPack has a solid correlation for ``comp``.

    Runs in an isolated subprocess because ``init_solid`` on an unsupported
    fluid is an uncatchable Fortran STOP that would kill this process.
    """
    snippet = (
        "from thermopack import thermo\n"
        "import sys\n"
        "c = sys.argv[1]\n"
        "t = thermo.thermo()\n"
        "t.init_thermo('PR', 'Classic', 'Classic', c, 1)\n"
        "t.init_solid(c)\n"
        "print('SOLID_OK')\n"
    )
    try:
        r = subprocess.run([sys.executable, "-c", snippet, comp],
                           capture_output=True, text=True, timeout=40)
    except Exception:
        return False
    return "SOLID_OK" in r.stdout


def main() -> None:
    # --- Build the supported pure-fluid set (in-process) -----------------
    supported: list[tuple[str, float]] = []  # (name, Tc [K])
    unsupported: list[str] = []
    for name in KNOWN_SUPPORTED:
        try:
            g = Gas(name)
            Tc = g.critical_temperature()
            supported.append((name, Tc))
        except Exception:
            unsupported.append(name)
    supported.sort(key=lambda x: x[1])  # by Tc

    print("=" * 64)
    print(f"ThermoLab database: {len(KNOWN_SUPPORTED)} curated pure fluids, "
          f"{len(FLUID_ALIASES)} pseudo-fluid aliases")
    print(f"  supported in this build : {len(supported)}")
    if unsupported:
        print(f"  unavailable in this build: {', '.join(unsupported)}")
    print("=" * 64)

    # --- Gases (vapour phase) --------------------------------------------
    print("\nGASES / VAPOUR  (every supported fluid has a vapour phase)")
    print(f"  {'name':>10} {'Tc [K]':>10}")
    for name, Tc in supported:
        print(f"  {name:>10} {Tc:10.1f}")
    print(f"  pseudo-fluids (mixtures): {', '.join(FLUID_ALIASES)}")

    # --- Liquids (condensables, by Tc) -----------------------------------
    ambient = [(n, T) for n, T in supported if T > AMBIENT_Tc]
    cryo = [(n, T) for n, T in supported if T <= AMBIENT_Tc]
    print("\nLIQUIDS  (condensable below Tc)")
    print(f"  ambient  (Tc > {AMBIENT_Tc:g} K, liquid near room temperature):")
    print(f"    {', '.join(n for n, _ in ambient)}")
    print(f"  cryogenic (Tc <= {AMBIENT_Tc:g} K):")
    print(f"    {', '.join(n for n, _ in cryo)}")

    # --- Solids (subprocess-probed) --------------------------------------
    print("\nSOLIDS  (probing each fluid in an isolated subprocess ...)")
    solids = []
    for i, (name, _) in enumerate(supported, 1):
        ok = _probe_solid(name)
        print(f"  [{i:2d}/{len(supported)}] {name:>10}  "
              f"{'has solid correlation' if ok else 'no solid model'}")
        if ok:
            solids.append(name)
    print(f"  -> solid-capable: {', '.join(solids) if solids else '(none)'}")

    # --- Hydrates (hydrate formers present in the database) --------------
    names = {n for n, _ in supported}
    formers_present = sorted(HYDRATE_FORMERS & names)
    print("\nHYDRATES  (no hydrate model in this ThermoPack Python build)")
    print(f"  lattice host   : {HYDRATE_HOST if HYDRATE_HOST in names else '(H2O not available)'}")
    print(f"  formers present : {', '.join(formers_present) if formers_present else '(none)'}")
    print("  (use examples/17_hydrates.py for correlation-based equilibrium)")

    # --- Adapt-the-examples cheat sheet ---------------------------------
    print("\n" + "=" * 64)
    print("ADAPTING THE EXAMPLES")
    print("=" * 64)
    print("  from thermolab import Gas, Mixture, list_fluids")
    print("  list_fluids()                          # all curated names")
    print("  air   = Gas('Air')                      # pseudo-mixture")
    print("  water = Gas('H2O')                      # liquid + vapour")
    print(f"  dryice supported: {', '.join(solids) if solids else '(none in this build)'}")
    print("  # solids reach into ThermoPack directly: see examples/16_solids.py")


if __name__ == "__main__":
    main()