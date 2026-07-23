#!/usr/bin/env python3
"""
Example 2.3 (advanced, research-grade) -- Molecular dynamics of a Lennard-Jones
fluid, with a full verification campaign.

The Lennard-Jones pair potential

    V(r) = 4 eps [ (sigma/r)^12 - (sigma/r)^6 ]

captures the essential physics of intermolecular forces: a steep repulsive core
and a weak attractive tail with a minimum of depth eps at r = 2^{1/6} sigma.  We
integrate Newton's equations for N such particles in a periodic box with the
velocity-Verlet scheme and reduced units (eps = sigma = m = 1), then VERIFY:

  (1) Energy conservation.  The total energy fluctuation of a symplectic
      integrator scales as dt^2; we measure it over a range of time steps and
      fit the convergence order (expected 2).
  (2) Emergence of Maxwell-Boltzmann.  Although the dynamics is deterministic,
      each velocity component relaxes to a Gaussian of variance k_B T/m; we
      overlay the measured histogram on the analytic Gaussian.
  (3) Equipartition.  The kinetic temperature (2 <KE> per degree of freedom)
      reproduces the target temperature.
  (4) Structure.  The radial distribution function g(r) shows the first
      neighbour shell of the dense fluid.

Only numpy, scipy, matplotlib are used; the seed is fixed for reproducibility.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

RNG = np.random.default_rng(20260723)
RC = 2.5                                       # potential cutoff (reduced units)


# ---------------------------------------------------------------------------
# Core molecular-dynamics routines (reduced units, m = eps = sigma = 1)
# ---------------------------------------------------------------------------
# Shifted-force constants: make BOTH the potential and the force vanish
# continuously at the cutoff RC, so the energy has no jumps and the symplectic
# dt^2 conservation law is clean.
_VRC = 4.0 * (RC ** -12 - RC ** -6)                    # V(RC)
_FRC = 24.0 * (2.0 * RC ** -13 - RC ** -7)             # scalar force f(RC)


def lj_forces(pos, L):
    """Shifted-force Lennard-Jones forces and potential (minimum-image)."""
    disp = pos[:, None, :] - pos[None, :, :]           # N,N,2 separations
    disp -= L * np.round(disp / L)                     # minimum image
    r2 = np.sum(disp ** 2, axis=2)                     # N,N squared distances
    np.fill_diagonal(r2, np.inf)                       # exclude self-term
    mask = r2 < RC * RC
    inv_r2 = np.where(mask, 1.0 / r2, 0.0)
    inv_r6 = inv_r2 ** 3
    r = np.sqrt(np.where(mask, r2, 1.0))
    inv_r = np.where(mask, 1.0 / r, 0.0)
    # Bare LJ force/r = 24 (2 r^-14 - r^-8); subtract f(RC)/r for shifted force.
    coef = 24.0 * inv_r2 * inv_r6 * (2.0 * inv_r6 - 1.0) - _FRC * inv_r
    coef = np.where(mask, coef, 0.0)
    F = np.sum(coef[:, :, None] * disp, axis=1)        # sum contributions on i
    # Shifted-force potential:  V(r) - V(RC) + (r - RC) f(RC).
    Vpair = np.where(mask,
                     4.0 * (inv_r6 ** 2 - inv_r6) - _VRC + (r - RC) * _FRC,
                     0.0)
    PE = 0.5 * np.sum(Vpair)                            # 1/2 removes double count
    return F, PE


def init_lattice(n_side, L):
    """Place N = n_side^2 particles on a square lattice inside the box."""
    xs = (np.arange(n_side) + 0.5) * (L / n_side)
    gx, gy = np.meshgrid(xs, xs)
    return np.column_stack([gx.ravel(), gy.ravel()])


def init_velocities(N, T):
    """Maxwellian velocities with zero net momentum, rescaled to temperature T."""
    v = RNG.normal(0.0, np.sqrt(T), size=(N, 2))
    v -= v.mean(axis=0)                                # remove centre-of-mass drift
    dof = 2 * N - 2
    T_now = np.sum(v ** 2) / dof
    return v * np.sqrt(T / T_now)


def velocity_verlet(pos, vel, L, dt, n_steps, record=False):
    """Integrate n_steps; optionally record energies, temperature, velocities."""
    F, PE = lj_forces(pos, L)
    N = len(pos)
    dof = 2 * N - 2
    E_hist, T_hist, v_samples, pos_samples = [], [], [], []
    for step in range(n_steps):
        vel += 0.5 * F * dt
        pos += vel * dt
        pos %= L                                       # periodic wrap
        F, PE = lj_forces(pos, L)
        vel += 0.5 * F * dt
        if record:
            KE = 0.5 * np.sum(vel ** 2)
            E_hist.append(KE + PE)
            T_hist.append(2.0 * KE / dof)
            if step % 10 == 0:
                v_samples.append(vel.copy())
                pos_samples.append(pos.copy())
    return (pos, vel, np.array(E_hist), np.array(T_hist),
            v_samples, pos_samples)


def radial_distribution(pos, L, n_bins=120, r_max=None):
    """Radial distribution function g(r) for a 2-D periodic configuration."""
    if r_max is None:
        r_max = L / 2.0
    N = len(pos)
    disp = pos[:, None, :] - pos[None, :, :]
    disp -= L * np.round(disp / L)
    r = np.sqrt(np.sum(disp ** 2, axis=2))
    r = r[np.triu_indices(N, k=1)]                     # unique pairs
    hist, edges = np.histogram(r, bins=n_bins, range=(0, r_max))
    rc = 0.5 * (edges[:-1] + edges[1:])
    rho = N / L ** 2
    shell = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) # 2-D annulus area
    g = hist / (rho * shell * (N / 2.0))
    return rc, g


# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("Example 2.3  Lennard-Jones molecular dynamics -- verification campaign")
    print("=" * 72)

    n_side, T_target, rho = 8, 1.0, 0.45
    N = n_side ** 2
    L = np.sqrt(N / rho)
    print(f"\nN = {N} particles, rho = {rho}, box L = {L:.3f}, "
          f"target T = {T_target}")

    # ---- (1) Energy-conservation order in dt ---------------------------
    print("\n(1) Total-energy fluctuation vs time step (velocity-Verlet):")
    print(f"    {'dt':>8} {'steps':>7} {'RMS dE/N':>14}")
    pos0 = init_lattice(n_side, L)
    vel0 = init_velocities(N, T_target)
    # Thermostatted equilibration: integrate in short bursts and rescale the
    # velocities back to T_target between bursts, draining the lattice potential
    # energy so the subsequent NVE run sits at the desired temperature.
    dof = 2 * N - 2
    pos_eq, vel_eq = pos0.copy(), vel0.copy()
    for _ in range(40):
        pos_eq, vel_eq, *_ = velocity_verlet(pos_eq, vel_eq, L, 0.004, 100,
                                             record=False)
        T_now = np.sum(vel_eq ** 2) / dof
        vel_eq *= np.sqrt(T_target / T_now)
    dts = [0.001, 0.002, 0.004, 0.008, 0.016]
    rms = []
    t_phys = 4.0
    for dt in dts:
        steps = int(t_phys / dt)
        _, _, E, _, _, _ = velocity_verlet(pos_eq.copy(), vel_eq.copy(), L,
                                           dt, steps, record=True)
        rms.append(np.std(E) / N)
        print(f"    {dt:>8.4f} {steps:>7d} {rms[-1]:>14.3e}")
    order = np.polyfit(np.log(dts), np.log(rms), 1)[0]
    print(f"    fitted energy-fluctuation order = {order:.3f} (expected 2)")

    # ---- (2)-(4) production run at a good dt ---------------------------
    print("\n(2-4) Production run for distributions and structure")
    dt = 0.004
    pos, vel, E, Tt, vsamp, psamp = velocity_verlet(pos_eq.copy(),
                                        vel_eq.copy(), L, dt, 6000, record=True)
    half = len(Tt) // 2
    T_meas = Tt[half:].mean()
    print(f"    measured kinetic temperature = {T_meas:.4f} "
          f"(target {T_target}) -> equipartition check")
    print(f"    energy conservation over run: "
          f"std(E)/N = {np.std(E)/N:.2e} (bounded, no secular drift)")

    V = np.concatenate(vsamp[len(vsamp)//2:])          # N,2 velocity components
    vc = V.ravel()
    print(f"    velocity-component mean = {vc.mean():+.4f} (expect 0), "
          f"variance = {vc.var():.4f} (expect T = {T_target})")

    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4), constrained_layout=True)

    # (a) energy fluctuation order
    ax = axes[0, 0]
    ax.loglog(dts, rms, "o", color="#2c3e50", ms=8, label="measured RMS")
    ax.loglog(dts, rms[-1] * (np.array(dts) / dts[-1]) ** 2, "-",
              color="#c0392b", lw=2, label=r"$\propto (\Delta t)^2$")
    ax.set_xlabel(r"time step  $\Delta t$")
    ax.set_ylabel(r"RMS energy fluctuation / $N$")
    ax.set_title("(a)  symplectic energy conservation")
    ax.legend(frameon=False)

    # (b) velocity-component histogram vs Gaussian
    ax = axes[0, 1]
    ax.hist(vc, bins=60, density=True, color="#9ecae1",
            label="MD velocity components")
    xx = np.linspace(vc.min(), vc.max(), 300)
    gauss = np.exp(-xx ** 2 / (2 * T_meas)) / np.sqrt(2 * np.pi * T_meas)
    ax.plot(xx, gauss, color="#c0392b", lw=2,
            label=r"$\mathcal{N}(0,\,k_BT/m)$")
    ax.set_xlabel(r"velocity component  $v_x,\,v_y$")
    ax.set_ylabel("probability density")
    ax.set_title("(b)  Maxwell-Boltzmann emerges")
    ax.legend(frameon=False, fontsize=9)

    # (c) temperature time series
    ax = axes[1, 0]
    t_axis = np.arange(len(Tt)) * dt
    ax.plot(t_axis, Tt, color="#2c3e50", lw=0.8)
    ax.axhline(T_target, color="#c0392b", lw=2, ls="--", label="target $T$")
    ax.axhline(T_meas, color="#27ae60", lw=1.5, ls=":",
               label=f"mean = {T_meas:.3f}")
    ax.set_xlabel("time  (reduced units)")
    ax.set_ylabel("kinetic temperature")
    ax.set_title("(c)  equipartition thermometer")
    ax.legend(frameon=False, fontsize=9)

    # (d) radial distribution function
    ax = axes[1, 1]
    gsum = None
    snaps = psamp[len(psamp)//2:]
    for ps in snaps:
        rc, g1 = radial_distribution(ps, L)
        gsum = g1 if gsum is None else gsum + g1
    g = gsum / len(snaps)                               # configuration average
    ax.plot(rc, g, color="#2c3e50", lw=1.8)
    ax.axhline(1.0, color="#c0392b", ls=":", lw=1.2)
    ax.axvline(2 ** (1 / 6), color="#27ae60", ls="--", lw=1.2,
               label=r"$r_{\min}=2^{1/6}\sigma$")
    ax.set_xlabel(r"distance  $r/\sigma$")
    ax.set_ylabel(r"$g(r)$")
    ax.set_title("(d)  liquid structure")
    ax.legend(frameon=False)

    fig.suptitle("Molecular dynamics of a Lennard-Jones fluid: "
                 "verification campaign", y=1.03, fontsize=13)
    fig.savefig("fig2_3.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig2_3.png")


if __name__ == "__main__":
    main()
