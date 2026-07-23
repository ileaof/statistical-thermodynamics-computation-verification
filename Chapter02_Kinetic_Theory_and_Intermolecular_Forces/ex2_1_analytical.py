#!/usr/bin/env python3
"""
Example 2.1 (analytical) -- The Maxwell-Boltzmann speed distribution.

The number of molecules of an ideal gas with speed in [v, v+dv] is N f(v) dv,
where the Maxwell-Boltzmann speed distribution is

    f(v) = 4 pi (m / (2 pi k_B T))^{3/2} v^2 exp(-m v^2 / (2 k_B T)).

This script derives the three characteristic speeds analytically, verifies the
normalisation and the first two moments by numerical quadrature against their
closed forms, and plots f(v) for several gases at fixed T and for one gas at
several temperatures.  Only numpy, scipy, matplotlib are used.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

k_B = 1.380649e-23          # J/K
u   = 1.660539e-27          # atomic mass unit, kg


def f_speed(v, m, T):
    """Maxwell-Boltzmann speed probability density (per m/s), SI units."""
    a = m / (2.0 * np.pi * k_B * T)
    return 4.0 * np.pi * a ** 1.5 * v ** 2 * np.exp(-m * v ** 2 / (2.0 * k_B * T))


def characteristic_speeds(m, T):
    """Closed-form most-probable, mean and rms speeds (m/s)."""
    v_p   = np.sqrt(2.0 * k_B * T / m)                 # argmax of f(v)
    v_avg = np.sqrt(8.0 * k_B * T / (np.pi * m))       # <v>
    v_rms = np.sqrt(3.0 * k_B * T / m)                 # sqrt(<v^2>)
    return v_p, v_avg, v_rms


def main():
    print("=" * 70)
    print("Example 2.1  Maxwell-Boltzmann speed distribution")
    print("=" * 70)

    # --- Verification: normalisation and moments by quadrature ----------
    gases = {"He": 4.0026, "N2": 28.0134, "O2": 31.998, "CO2": 44.009}
    T = 300.0
    print(f"\nT = {T:.0f} K.  Quadrature checks vs closed forms:")
    print(f"{'gas':>5} {'norm':>10} {'<v> num':>10} {'<v> exact':>10} "
          f"{'vrms num':>10} {'vrms exact':>11}")
    for name, M in gases.items():
        m = M * u
        norm, _  = quad(f_speed, 0, np.inf, args=(m, T))
        v1, _    = quad(lambda v: v * f_speed(v, m, T), 0, np.inf)
        v2, _    = quad(lambda v: v * v * f_speed(v, m, T), 0, np.inf)
        v_p, v_avg, v_rms = characteristic_speeds(m, T)
        print(f"{name:>5} {norm:>10.6f} {v1:>10.2f} {v_avg:>10.2f} "
              f"{np.sqrt(v2):>10.2f} {v_rms:>11.2f}")

    # Ordering v_p < <v> < v_rms with fixed ratios sqrt(2):sqrt(8/pi):sqrt(3)
    m = 28.0134 * u
    v_p, v_avg, v_rms = characteristic_speeds(m, T)
    print(f"\nN2 at {T:.0f} K:  v_p = {v_p:.1f},  <v> = {v_avg:.1f},  "
          f"v_rms = {v_rms:.1f} m/s")
    print(f"  ratios <v>/v_p = {v_avg/v_p:.4f} (exact {np.sqrt(4/np.pi):.4f}), "
          f"v_rms/v_p = {v_rms/v_p:.4f} (exact {np.sqrt(1.5):.4f})")

    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4),
                                   constrained_layout=True)
    v = np.linspace(0, 3000, 600)
    colors = {"He": "#c0392b", "N2": "#2c3e50", "O2": "#2980b9", "CO2": "#27ae60"}
    for name, M in gases.items():
        ax1.plot(v, f_speed(v, M * u, T), color=colors[name], lw=2, label=name)
    ax1.set_xlabel("molecular speed  $v$  (m/s)")
    ax1.set_ylabel(r"$f(v)$   (s/m)")
    ax1.set_title(f"(a)  different gases at $T = {T:.0f}$ K")
    ax1.legend(frameon=False)

    m = 28.0134 * u
    for Temp, c in [(100, "#2980b9"), (300, "#2c3e50"), (600, "#e67e22"),
                    (1200, "#c0392b")]:
        ax2.plot(v, f_speed(v, m, Temp), color=c, lw=2, label=f"{Temp} K")
        vp = np.sqrt(2 * k_B * Temp / m)
        ax2.axvline(vp, color=c, ls=":", lw=1)
    ax2.set_xlabel("molecular speed  $v$  (m/s)")
    ax2.set_ylabel(r"$f(v)$   (s/m)")
    ax2.set_title(r"(b)  N$_2$ at several temperatures")
    ax2.legend(frameon=False, title="dotted: $v_p$")

    fig.suptitle("Maxwell-Boltzmann speed distribution", y=1.06, fontsize=13)
    fig.savefig("fig2_1.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig2_1.png")


if __name__ == "__main__":
    main()
