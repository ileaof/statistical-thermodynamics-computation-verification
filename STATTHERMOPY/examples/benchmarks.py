"""Benchmark the accelerated backends against the NumPy reference.

Times ``property_vs_T`` over a large temperature grid for the NumPy, Numba and OpenMP backends
(and CUDA if available) and confirms the accelerated paths return values identical to the NumPy
path. Run with::

    python examples/benchmarks.py
"""

from __future__ import annotations

import time

import numpy as np

from statthermopy import State, Thermodynamics
from statthermopy.backend import available_backends, set_backend
from statthermopy.database import get


def _time(species: str, n: int = 2000) -> dict:
    mol = get(species)
    Ts = np.linspace(300.0, 3000.0, n)
    results: dict = {}
    for be in ("numpy", "numba", "openmp", "cuda"):
        if be not in available_backends():
            continue
        set_backend(be)
        # warm-up (compile the JIT kernels) so we measure steady-state throughput
        Thermodynamics(mol, State(T=298.15, P=101325.0)).property_vs_T("Cp_m", Ts[:10], P=101325.0)
        t0 = time.perf_counter()
        _, vals = Thermodynamics(mol, State(T=298.15, P=101325.0)).property_vs_T(
            "Cp_m", Ts, P=101325.0
        )
        results[be] = (time.perf_counter() - t0, vals)
    set_backend("numpy")
    return results


def main() -> None:
    print("StatThermoPy backend benchmark — property_vs_T('Cp_m', 2000 points)\n")
    ref_time = None
    for species in ("N2", "CO2", "H2"):
        print(f"  {species}:")
        res = _time(species)
        for be, (dt, _vals) in res.items():
            tag = ""
            if be == "numpy":
                ref_time = dt
            elif ref_time is not None:
                tag = f"  ({ref_time / dt:.1f}x vs numpy)" if dt > 0 else ""
            print(f"    {be:8s} {dt * 1000:8.2f} ms{tag}")
        # correctness: every backend matches numpy
        if "numpy" in res:
            np_vals = np.array(res["numpy"][1])
            for be, (_dt, vals) in res.items():
                if be == "numpy":
                    continue
                err = float(np.max(np.abs(np.array(vals) - np_vals) / np.abs(np_vals)))
                print(f"    {be:8s} max rel. error vs numpy = {err:.2e}")
        print()

    print(f"Available backends: {available_backends()}")
    print("CUDA falls back to the Numba CPU backend when no NVIDIA GPU is present.")


if __name__ == "__main__":  # pragma: no cover
    main()