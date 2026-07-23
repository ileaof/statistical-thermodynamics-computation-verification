#!/usr/bin/env python3
"""
Example 1.3 (advanced, research-grade) -- Emergence of the Boltzmann
distribution from equal a priori probabilities, with a full verification
campaign.

Idea
----
Place a *small* system -- a single quantum harmonic oscillator with levels
E_i = i*eps, i = 0,1,2,... -- in thermal contact with a reservoir of N identical
oscillators (an Einstein solid).  The composite of N+1 oscillators is isolated
with a fixed total of Q quanta.  Counting reservoir microstates gives the EXACT
probability that the small system holds i quanta:

        P(i) = Omega_res(N, Q - i) / Omega_tot,
        Omega_res(N, q) = C(q + N - 1, q),  Omega_tot = C(Q + N, Q).

Two limits are examined and verified numerically:

 1.  As the reservoir grows (N -> infinity at fixed mean occupancy, i.e. Q = N),
     P(i) approaches the geometric Boltzmann law  P(i) = (1 - x) x^i  with
     x = Q/(Q + N) = exp(-eps/kT).  We measure the sup-norm error vs N and show
     it falls like 1/N.

 2.  A Monte Carlo experiment samples composite microstates *uniformly* (the
     literal statement of equal a priori probabilities) using the stars-and-bars
     bijection: the small system's energy equals the position of the first
     divider among Q + N slots.  The histogram converges to the exact P(i) with
     the expected 1/sqrt(n_samples) statistical error, complete with error bars.

The campaign reports: convergence order of the reservoir->Boltzmann limit, the
Monte Carlo error law, the temperature recovered by fitting ln P(i) vs i
compared with the thermodynamic 1/T = dS/dE, and CPU timings.

Only numpy, scipy and matplotlib are used.  All randomness uses a fixed seed.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

k_B = 1.380649e-23                          # J/K
RNG = np.random.default_rng(20260723)       # deterministic


# ---------------------------------------------------------------------------
# Exact small-system distribution and the Boltzmann reference
# ---------------------------------------------------------------------------
def log_omega(N, q):
    q = np.asarray(q, dtype=float)
    return gammaln(q + N) - gammaln(q + 1.0) - gammaln(float(N))


def exact_distribution(N, Q):
    """Exact P(i), i = 0..Q, for the small oscillator + N-oscillator reservoir."""
    i = np.arange(0, Q + 1)
    logP = log_omega(N, Q - i) - log_omega(N + 1, Q)     # Omega_tot = C(Q+N,Q)
    return i, np.exp(logP)


def boltzmann_reference(N, Q):
    """Geometric Boltzmann law that P(i) tends to; returns (x, P_geom(i))."""
    x = Q / (Q + N)                          # = exp(-eps/kT)
    i = np.arange(0, Q + 1)
    P = (1.0 - x) * x ** i
    return x, P


# ---------------------------------------------------------------------------
# Monte Carlo: sample composite microstates uniformly (equal a priori prob.)
# ---------------------------------------------------------------------------
def monte_carlo_small_energy(N, Q, n_samples):
    """Uniformly sample microstates; return the small system's quantum number.

    Stars-and-bars: distributing Q quanta among N+1 oscillators <-> choosing N
    divider positions among Q + N slots.  The count of quanta in oscillator 0
    equals the index of the first divider, i.e. the minimum of the N chosen
    positions.  Drawing N distinct positions uniformly makes every microstate
    equally likely -- the postulate implemented literally.
    """
    slots = Q + N
    out = np.empty(n_samples, dtype=np.int64)
    for s in range(n_samples):
        bars = RNG.choice(slots, size=N, replace=False)
        out[s] = bars.min()                  # quanta before the first divider
    return out


# ---------------------------------------------------------------------------
# Verification campaign
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("Example 1.3  Boltzmann distribution from equal a priori probability")
    print("=" * 72)

    # ---- (1) reservoir -> Boltzmann convergence, order estimate ---------
    print("\n(1) Convergence of exact P(i) to the geometric Boltzmann law")
    print(f"    (fixed x = Q/(Q+N) = 0.5 by setting Q = N)")
    print(f"    {'N':>7} {'sup-norm error':>18} {'ratio':>8}")
    Ns = [16, 32, 64, 128, 256, 512, 1024]
    errs = []
    for N in Ns:
        Q = N                                # keeps x = 0.5
        i, P = exact_distribution(N, Q)
        _, Pgeo = boltzmann_reference(N, Q)
        errs.append(np.max(np.abs(P - Pgeo)))
    for k, N in enumerate(Ns):
        ratio = errs[k - 1] / errs[k] if k else float("nan")
        print(f"    {N:>7d} {errs[k]:>18.6e} {ratio:>8.3f}")
    # A 1/N law doubles-halves: ratio ~ 2 as N doubles.
    p_order = np.polyfit(np.log(Ns), np.log(errs), 1)[0]
    print(f"    fitted convergence order  d(log err)/d(log N) = {p_order:.3f}"
          f"   (Boltzmann limit ~ N^-1)")

    # ---- (2) Monte Carlo convergence with error bars --------------------
    print("\n(2) Monte Carlo histogram vs exact P(i):  RMS error and 1/sqrt(n)")
    N, Q = 64, 64
    i_exact, P_exact = exact_distribution(N, Q)
    R = 6                                     # independent repetitions
    sample_sizes = [1000, 4000, 16000, 64000, 128000]
    rms = []          # mean RMS error over repetitions
    rms_sd = []       # spread across repetitions (an error bar on the error)
    t_mc = []
    for n in sample_sizes:
        vals = []
        t0 = time.perf_counter()
        for _ in range(R):
            draws = monte_carlo_small_energy(N, Q, n)
            hist = np.bincount(draws, minlength=Q + 1)[:Q + 1] / n
            vals.append(np.sqrt(np.mean((hist - P_exact) ** 2)))
        t_mc.append((time.perf_counter() - t0) / R)
        rms.append(np.mean(vals))
        rms_sd.append(np.std(vals))
    print(f"    (mean over R={R} independent runs)")
    print(f"    {'n_samples':>10} {'RMS error':>14} {'ratio':>8} {'CPU/run(s)':>11}")
    for k, n in enumerate(sample_sizes):
        ratio = rms[k - 1] / rms[k] if k else float("nan")
        print(f"    {n:>10d} {rms[k]:>14.6e} {ratio:>8.3f} {t_mc[k]:>11.3f}")
    mc_order = np.polyfit(np.log(sample_sizes), np.log(rms), 1)[0]
    print(f"    fitted MC error order = {mc_order:.3f}   (expected -0.5)")

    # ---- (3) temperature recovered from the slope of ln P(i) ------------
    print("\n(3) Statistical temperature check")
    # Fit ln P(i) = const + i ln x over the well-sampled range.
    N, Q = 2000, 2000
    i, P = exact_distribution(N, Q)
    fit_hi = max(8, int(0.03 * Q))            # clean exponential body near i=0
    slope = np.polyfit(i[:fit_hi], np.log(P[:fit_hi]), 1)[0]
    x_fit = np.exp(slope)
    x_theory = Q / (Q + N)
    # Thermodynamic beta*eps from the reservoir: 1/kT = dS/dE gives, for an
    # Einstein solid, eps/kT = d ln Omega/dq = ln(1 + N/q) evaluated at q = Q.
    beta_eps_thermo = np.log(1.0 + N / Q)
    print(f"    x from log-slope fit of P(i)   = {x_fit:.6f}")
    print(f"    x = Q/(Q+N) (equal a priori)   = {x_theory:.6f}")
    print(f"    eps/kT from fit  (-ln x_fit)   = {-np.log(x_fit):.6f}")
    print(f"    eps/kT from reservoir dS/dE    = {beta_eps_thermo:.6f}")
    print(f"    agreement                      = "
          f"{abs(-np.log(x_fit) - beta_eps_thermo):.2e}")

    # -----------------------------------------------------------------
    # Figure: 4 panels -- distribution, reservoir convergence,
    #          MC convergence, MC histogram with error bars.
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4), constrained_layout=True)

    # (a) exact vs Boltzmann for two reservoir sizes
    ax = axes[0, 0]
    for N_, c, lab in [(16, "#9ecae1", "N = 16"), (256, "#c0392b", "N = 256")]:
        ii, PP = exact_distribution(N_, N_)
        ax.semilogy(ii[:40], PP[:40], "o-", color=c, ms=3, lw=1.2,
                    label=f"exact, {lab}")
    _, Pg = boltzmann_reference(256, 256)
    ax.semilogy(np.arange(40), Pg[:40], "k--", lw=1.5, label="Boltzmann $(1-x)x^i$")
    ax.set_xlabel("small-system quantum number  $i$")
    ax.set_ylabel("$P(i)$")
    ax.set_title("(a)  exact counting $\\to$ Boltzmann")
    ax.legend(frameon=False, fontsize=9)

    # (b) reservoir -> Boltzmann convergence
    ax = axes[0, 1]
    ax.loglog(Ns, errs, "o", color="#2c3e50", ms=7, label="sup-norm error")
    ax.loglog(Ns, errs[0] * Ns[0] / np.array(Ns), "-", color="#c0392b",
              lw=2, label=r"$\propto N^{-1}$")
    ax.set_xlabel("reservoir size  $N$")
    ax.set_ylabel("max$_i\\,|P-P_{\\rm Boltz}|$")
    ax.set_title("(b)  reservoir limit converges as  $N^{-1}$")
    ax.legend(frameon=False)

    # (c) Monte Carlo error law
    ax = axes[1, 0]
    ax.errorbar(sample_sizes, rms, yerr=rms_sd, fmt="s", color="#2c3e50",
                ms=7, capsize=3, label="MC RMS error (mean $\\pm$ s.d.)")
    ax.loglog(sample_sizes, rms[0] * np.sqrt(sample_sizes[0] / np.array(sample_sizes)),
              "-", color="#c0392b", lw=2, label=r"$\propto n^{-1/2}$")
    ax.set_xlabel("number of sampled microstates  $n$")
    ax.set_ylabel("RMS$(\\hat P - P)$")
    ax.set_title("(c)  Monte Carlo error  $\\sim n^{-1/2}$")
    ax.legend(frameon=False)

    # (d) histogram with Poisson-like error bars vs exact
    ax = axes[1, 1]
    N, Q, n = 64, 64, 64000
    draws = monte_carlo_small_energy(N, Q, n)
    hist = np.bincount(draws, minlength=Q + 1)[:Q + 1] / n
    err = np.sqrt(hist * (1 - hist) / n)     # binomial standard error
    i_ex, P_ex = exact_distribution(N, Q)
    m = 25
    ax.errorbar(i_ex[:m], hist[:m], yerr=err[:m], fmt="o", color="#2c3e50",
                ms=4, capsize=2, label=f"MC ($n=${n:,})")
    ax.plot(i_ex[:m], P_ex[:m], "-", color="#c0392b", lw=2, label="exact $P(i)$")
    ax.set_xlabel("small-system quantum number  $i$")
    ax.set_ylabel("$P(i)$")
    ax.set_title("(d)  sampled microstates reproduce  $P(i)$")
    ax.legend(frameon=False)

    fig.suptitle("Emergence and verification of the Boltzmann distribution",
                 y=1.03, fontsize=13)
    fig.savefig("fig1_3.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig1_3.png")


if __name__ == "__main__":
    main()
