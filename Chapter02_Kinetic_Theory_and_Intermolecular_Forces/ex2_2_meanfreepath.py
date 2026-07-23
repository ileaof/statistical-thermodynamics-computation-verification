#!/usr/bin/env python3
"""
Example 2.2 (direct numerical) -- Relative speed, collision frequency and the
mean free path, with the sqrt(2) factor verified by Monte Carlo.

Kinetic theory predicts, for a dilute gas of hard spheres of diameter d and
number density n,

    mean speed          <v>       = sqrt(8 k_B T / (pi m)),
    mean relative speed <v_rel>   = sqrt(2) <v>   (Maxwell average),
    collision frequency z         = sqrt(2) n sigma <v>,   sigma = pi d^2,
    mean free path      lambda    = <v> / z = 1 / (sqrt(2) n sigma).

The famous sqrt(2) comes from averaging the relative speed of two molecules that
BOTH sample the Maxwell-Boltzmann distribution.  This script draws pairs of 3-D
Maxwellian velocities, measures <|v1 - v2|> / <|v|>, and shows it converges to
sqrt(2) with the 1/sqrt(N) statistical error of Monte Carlo.  It then tabulates
the collision frequency and mean free path of air over a range of pressures.

Only numpy, scipy, matplotlib are used; the seed is fixed.
"""

import numpy as np
import matplotlib.pyplot as plt

k_B = 1.380649e-23          # J/K
u   = 1.660539e-27          # kg
RNG = np.random.default_rng(20260723)


def sample_maxwell_velocities(n_samples, m, T):
    """Draw 3-D velocity vectors: each Cartesian component is N(0, k_B T/m)."""
    sigma_v = np.sqrt(k_B * T / m)          # per-component standard deviation
    return RNG.normal(0.0, sigma_v, size=(n_samples, 3))


def main():
    print("=" * 70)
    print("Example 2.2  Relative speed, collision frequency, mean free path")
    print("=" * 70)

    m = 28.0134 * u                          # N2
    T = 300.0
    v_avg = np.sqrt(8.0 * k_B * T / (np.pi * m))
    print(f"\nN2 at {T:.0f} K:  <v> (exact) = {v_avg:.2f} m/s")

    # --- Monte Carlo: <v_rel>/<v> -> sqrt(2) ----------------------------
    print("\n(1) Monte Carlo convergence of <|v1-v2|>/<|v|> to sqrt(2):")
    print(f"    (mean over R independent runs)")
    print(f"    {'N pairs':>10} {'ratio':>10} {'|error|':>12} {'run s.d.':>11}")
    R = 8
    sizes = [1000, 4000, 16000, 64000, 256000]
    err, err_sd = [], []
    for N in sizes:
        vals = []
        for _ in range(R):
            v1 = sample_maxwell_velocities(N, m, T)
            v2 = sample_maxwell_velocities(N, m, T)
            speed1 = np.linalg.norm(v1, axis=1)
            rel    = np.linalg.norm(v1 - v2, axis=1)
            vals.append(rel.mean() / speed1.mean())
        vals = np.array(vals)
        mean_ratio = vals.mean()
        err.append(abs(mean_ratio - np.sqrt(2)))
        err_sd.append(vals.std())
        print(f"    {N:>10d} {mean_ratio:>10.5f} "
              f"{abs(mean_ratio-np.sqrt(2)):>12.2e} {vals.std():>11.2e}")
    err = np.array(err); err_sd = np.array(err_sd)
    # The statistical (per-run) error, not the residual mean bias, is the clean
    # 1/sqrt(N) quantity; fit its decay to confirm the Monte Carlo error law.
    order = np.polyfit(np.log(sizes), np.log(err_sd), 1)[0]
    print(f"    target sqrt(2) = {np.sqrt(2):.5f};  per-run error order = "
          f"{order:.3f} (expected -0.5)")

    # --- Collision frequency and mean free path of air vs pressure ------
    d = 3.7e-10                              # effective N2 diameter, m
    sigma = np.pi * d ** 2                   # collision cross-section
    print("\n(2) Air (N2) collision frequency and mean free path, T = 300 K:")
    print(f"    {'P (Pa)':>12} {'n (1/m^3)':>14} {'z (1/s)':>12} "
          f"{'lambda (m)':>13}")
    pressures = [1e0, 1e2, 1e3, 1e4, 1.01325e5]
    lam_list, n_list = [], []
    for P in pressures:
        n = P / (k_B * T)                    # ideal gas number density
        z = np.sqrt(2.0) * n * sigma * v_avg
        lam = v_avg / z
        n_list.append(n); lam_list.append(lam)
        print(f"    {P:>12.3e} {n:>14.4e} {z:>12.4e} {lam:>13.4e}")

    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)

    # (a) speed vs relative-speed distributions
    N = 200000
    v1 = sample_maxwell_velocities(N, m, T)
    v2 = sample_maxwell_velocities(N, m, T)
    s1  = np.linalg.norm(v1, axis=1)
    rel = np.linalg.norm(v1 - v2, axis=1)
    bins = np.linspace(0, 2500, 80)
    axes[0].hist(s1,  bins=bins, density=True, alpha=0.55, color="#2c3e50",
                 label=r"speed $|v|$")
    axes[0].hist(rel, bins=bins, density=True, alpha=0.55, color="#c0392b",
                 label=r"relative $|v_1-v_2|$")
    axes[0].axvline(s1.mean(),  color="#2c3e50", ls=":", lw=1.5)
    axes[0].axvline(rel.mean(), color="#c0392b", ls=":", lw=1.5)
    axes[0].set_xlabel("speed (m/s)")
    axes[0].set_ylabel("probability density (s/m)")
    axes[0].set_title("(a)  speed and relative-speed")
    axes[0].legend(frameon=False)

    # (b) convergence to sqrt(2)
    axes[1].loglog(sizes, err_sd, "o", color="#2c3e50", ms=7,
                   label="per-run statistical error")
    axes[1].loglog(sizes, err_sd[0] * np.sqrt(sizes[0] / np.array(sizes)), "-",
                   color="#c0392b", lw=2, label=r"$\propto N^{-1/2}$")
    axes[1].set_xlabel("number of pairs  $N$")
    axes[1].set_ylabel(r"s.d. of $\langle v_{\rm rel}\rangle/\langle v\rangle$")
    axes[1].set_title(r"(b)  ratio $\to \sqrt{2}$, error $\sim N^{-1/2}$")
    axes[1].legend(frameon=False)

    # (c) mean free path vs pressure
    axes[2].loglog(pressures, lam_list, "o-", color="#2c3e50", ms=6)
    axes[2].set_xlabel("pressure  $P$  (Pa)")
    axes[2].set_ylabel(r"mean free path  $\lambda$  (m)")
    axes[2].set_title(r"(c)  $\lambda = 1/(\sqrt{2}\,n\sigma)\propto 1/P$")

    fig.suptitle("Relative speed, collision frequency and mean free path",
                 y=1.05, fontsize=13)
    fig.savefig("fig2_2.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig2_2.png")


if __name__ == "__main__":
    main()
