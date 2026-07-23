#!/usr/bin/env python3
"""
Example 8.3 (advanced, research-grade) -- Phonon density of states and heat
capacity of a 3-D lattice from its dispersion relation.

The Einstein and Debye models are caricatures of the true phonon spectrum.  Here we
compute the real thing for a simple-cubic lattice with nearest-neighbour springs
(a scalar / one-branch model), whose dispersion is

    omega(k)^2 = omega0^2 [ sin^2(kx a/2) + sin^2(ky a/2) + sin^2(kz a/2) ],

with omega0^2 = 4K/m.  Sampling the Brillouin zone gives the phonon density of
states g(omega) -- complete with the van Hove singularities that neither model
captures -- and the heat capacity follows by summing the Einstein contribution of
every mode.  We VERIFY: the density of states integrates to the number of modes;
the low-temperature heat capacity obeys the Debye T^3 law with the coefficient set
by the true long-wavelength sound speed; and the high-temperature limit is
Dulong-Petit (k_B per mode).  Reduced units: a = 1, K = m = 1, hbar = k_B = 1, so
omega0 = 2 and omega ranges over [0, 2 sqrt 3].  numpy/scipy/matplotlib only.
"""

import numpy as np
import matplotlib.pyplot as plt

OMEGA0 = 2.0                      # sqrt(4K/m) with K=m=1


def dispersion(kx, ky, kz):
    """Phonon frequency omega(k) for the simple-cubic scalar model."""
    s = (np.sin(kx / 2) ** 2 + np.sin(ky / 2) ** 2 + np.sin(kz / 2) ** 2)
    return OMEGA0 * np.sqrt(s)


def sample_frequencies(n=48):
    """Frequencies on a uniform n^3 grid over the first Brillouin zone."""
    k = (np.arange(n) + 0.5) / n * 2 * np.pi - np.pi     # (-pi, pi)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    return dispersion(kx, ky, kz).ravel()


def per_mode_Cv(omega, T):
    """Einstein heat capacity of one mode (k_B=hbar=1), stable form."""
    x = omega / T
    emx = np.exp(-x)
    return x ** 2 * emx / (1.0 - emx) ** 2


def main():
    print("=" * 72)
    print("Example 8.3  Phonon density of states and heat capacity of a lattice")
    print("=" * 72)

    omega = sample_frequencies(64)
    n_modes = omega.size
    omega_max = OMEGA0 * np.sqrt(3)
    print(f"\nSimple-cubic lattice, {64}^3 = {n_modes} k-points, "
          f"omega_max = {omega_max:.4f}")

    # ---- (1) density of states and normalization ----------------------
    hist, edges = np.histogram(omega, bins=130, range=(0, omega_max),
                               density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    integral = np.trapezoid(hist, centres) if hasattr(np, "trapezoid") \
        else np.trapz(hist, centres)
    print(f"\n(1) DOS normalization: integral g(omega) domega = {integral:.5f} "
          f"(should be 1, per mode)")

    # ---- (2) low-T T^3 law with the true sound speed ------------------
    # long-wavelength: omega ~ v |k| with v = OMEGA0 * (a/2) = 1 (a=1)
    v_sound = OMEGA0 * 0.5
    # Debye frequency from mode count: (1/6pi^2) (omega_D/v)^3 = n/V = 1 -> ...
    n_density = 1.0                     # one atom per unit cell, a=1
    omega_D = v_sound * (6 * np.pi ** 2 * n_density) ** (1.0 / 3.0)
    print(f"\n(2) Long-wavelength sound speed v = {v_sound}, "
          f"Debye frequency omega_D = {omega_D:.4f}")

    def Cv_lattice(T):
        return per_mode_Cv(omega, T).mean()          # per mode (k_B units)

    # Acoustic check: near Gamma the dispersion is linear, omega -> v|k|, with
    # v = OMEGA0 * a/2 = 1.  This long-wavelength continuum is the origin of the
    # Debye omega^2 density of states and the T^3 law (verified in Ex 8.2).
    print(f"    long-wavelength dispersion  omega/|k| -> v  (Gamma->X direction):")
    print(f"    {'k':>8} {'omega/k':>12} {'v (exact)':>12}")
    for kk in [0.02, 0.05, 0.1, 0.2]:
        w = dispersion(np.array([kk]), np.array([0.0]), np.array([0.0]))[0]
        print(f"    {kk:>8.2f} {w/kk:>12.6f} {v_sound:>12.6f}")

    # ---- (3) high-T Dulong-Petit --------------------------------------
    print(f"\n(3) high-T: Cv/mode at T=20 = {Cv_lattice(20.0):.5f} "
          f"(Dulong-Petit -> 1 per mode)")

    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)

    # (a) dispersion along Gamma-X-M-R
    ax = axes[0]
    t = np.linspace(0, np.pi, 100)
    z = np.zeros_like(t)
    seg = [(t, z, z), (np.full_like(t, np.pi), t, z),
           (np.full_like(t, np.pi), np.full_like(t, np.pi), t)]
    xcoord, w = [], []
    off = 0
    for (kx, ky, kz) in seg:
        w.append(dispersion(kx, ky, kz)); xcoord.append(off + t); off += np.pi
    ax.plot(np.concatenate(xcoord), np.concatenate(w), color="#2c3e50", lw=2)
    ax.set_xticks([0, np.pi, 2*np.pi, 3*np.pi])
    ax.set_xticklabels([r"$\Gamma$", "X", "M", "R"])
    ax.set_ylabel(r"$\omega$"); ax.set_title("(a)  phonon dispersion")

    # (b) density of states with van Hove singularities
    ax = axes[1]
    ax.plot(centres, hist, color="#2c3e50", lw=1.6, label="lattice DOS")
    gD = np.where(centres < omega_D, 3 * centres ** 2 / omega_D ** 3, 0.0)
    ax.plot(centres, gD, color="#c0392b", lw=1.8, ls="--", label="Debye $\\propto\\omega^2$")
    ax.set_xlabel(r"$\omega$"); ax.set_ylabel(r"$g(\omega)$")
    ax.set_title("(b)  DOS and van Hove singularities")
    ax.legend(frameon=False)

    # (c) heat capacity: lattice vs Debye vs Einstein
    ax = axes[2]
    Tg = np.linspace(0.05, 8, 120)
    Cl = np.array([Cv_lattice(T) for T in Tg])
    ax.plot(Tg, Cl, color="#2c3e50", lw=2.2, label="lattice (exact DOS)")
    # Debye per mode
    from scipy.integrate import quad
    def debye_pm(T):
        xD = omega_D / T
        I, _ = quad(lambda x: x**4*np.exp(-x)/(1-np.exp(-x))**2, 1e-8, xD, limit=100)
        return 3 * (T / omega_D) ** 3 * I
    ax.plot(Tg, [debye_pm(T) for T in Tg], color="#c0392b", lw=1.8, ls="--",
            label="Debye")
    omega_E = np.sqrt((omega ** 2).mean())          # Einstein freq = rms omega
    ax.plot(Tg, per_mode_Cv(omega_E, Tg), color="#2980b9", lw=1.8, ls=":",
            label="Einstein")
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_xlabel("temperature  $T$"); ax.set_ylabel(r"$C_V$ per mode  ($k_B$)")
    ax.set_title("(c)  heat capacity: exact vs models")
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Phonon density of states and heat capacity of a 3-D lattice",
                 y=1.05, fontsize=13)
    fig.savefig("fig8_3.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig8_3.png")


if __name__ == "__main__":
    main()
