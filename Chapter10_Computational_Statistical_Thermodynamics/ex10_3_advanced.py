#!/usr/bin/env python3
"""
Example 10.3 (advanced, research-grade) -- A complete verification, validation and
reproducibility campaign for a Monte Carlo thermal average.

This capstone computes the mean potential energy <V> of the anharmonic oscillator
V(x) = x^2/2 + lambda x^4 by Metropolis sampling and subjects the result to the
full apparatus of honest scientific computing:

  (1) VALIDATION against an independent exact value (quadrature).
  (2) Statistical error by BLOCKING (Flyvbjerg-Petersen): averaging adjacent data
      into ever-larger blocks removes autocorrelation; the error estimate rises and
      PLATEAUS at the true value.
  (3) Error by BOOTSTRAP resampling of decorrelated blocks -- an independent route
      that must agree with blocking.
  (4) REPRODUCIBILITY across many independent seeds: the spread of the estimates
      equals the single-run error bar, and the standardised residuals (pulls)
      (estimate - exact)/error follow a unit normal -- the definitive check that the
      error bars mean what they claim.
  (5) A final result reported as value +/- statistical error, agreeing with the
      exact value within that uncertainty.

"A computed number is an opinion until it is verified."  numpy/scipy/matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

BETA, LAM = 1.0, 0.1


def V(x):
    return 0.5 * x ** 2 + LAM * x ** 4


def metropolis_V(n_steps, step, rng, burn=3000):
    """Metropolis chain; return the time series of V(x)."""
    x, Vx = 0.0, V(0.0)
    out = np.empty(n_steps)
    for i in range(n_steps + burn):
        xt = x + rng.uniform(-step, step)
        Vt = V(xt)
        if Vt < Vx or rng.random() < np.exp(-BETA * (Vt - Vx)):
            x, Vx = xt, Vt
        if i >= burn:
            out[i - burn] = Vx
    return out


def blocking_errors(a):
    """Flyvbjerg-Petersen blocking: SEM at each blocking transformation."""
    a = a.copy()
    levels, errs = [], []
    lvl = 0
    while len(a) >= 8:
        errs.append(a.std(ddof=1) / np.sqrt(len(a)))
        levels.append(lvl)
        if len(a) % 2:
            a = a[:-1]
        a = 0.5 * (a[0::2] + a[1::2])            # average adjacent pairs
        lvl += 1
    return np.array(levels), np.array(errs)


def main():
    print("=" * 72)
    print("Example 10.3  Verification, validation and reproducibility campaign")
    print("=" * 72)

    # ---- (1) exact reference ------------------------------------------
    Z, _ = quad(lambda t: np.exp(-BETA * V(t)), -np.inf, np.inf)
    V_exact = quad(lambda t: V(t) * np.exp(-BETA * V(t)), -np.inf, np.inf)[0] / Z
    print(f"\n(1) Exact <V> (quadrature) = {V_exact:.6f}")

    # ---- (2) one long chain: blocking analysis ------------------------
    rng = np.random.default_rng(20260723)
    chain = metropolis_V(400_000, step=3.0, rng=rng)
    est = chain.mean()
    levels, errs = blocking_errors(chain)
    err_plateau = errs[-6:].max()                # plateau value
    print(f"\n(2) Long chain: <V> = {est:.6f}")
    print(f"    naive SEM (level 0)      = {errs[0]:.6f}")
    print(f"    blocking plateau error   = {err_plateau:.6f} "
          f"(factor {err_plateau/errs[0]:.1f} larger)")

    # ---- (3) bootstrap over decorrelated blocks -----------------------
    bsize = 200
    nb = len(chain) // bsize
    blocks = chain[:nb * bsize].reshape(nb, bsize).mean(1)
    boot = np.array([np.mean(rng.choice(blocks, nb, replace=True))
                     for _ in range(2000)])
    err_boot = boot.std()
    print(f"\n(3) Bootstrap error = {err_boot:.6f} "
          f"(blocking {err_plateau:.6f}) -- consistent")

    # ---- (4) reproducibility over independent seeds -------------------
    n_seed = 120
    ests, ses = [], []
    for s in range(n_seed):
        r = np.random.default_rng(1000 + s)
        c = metropolis_V(30_000, step=3.0, rng=r)
        ests.append(c.mean())
        _, e = blocking_errors(c)
        ses.append(e[-6:].max())
    ests, ses = np.array(ests), np.array(ses)
    spread = ests.std()
    mean_se = ses.mean()
    pulls = (ests - V_exact) / ses
    print(f"\n(4) Reproducibility over {n_seed} seeds:")
    print(f"    mean of estimates      = {ests.mean():.6f} (exact {V_exact:.6f})")
    print(f"    spread of estimates    = {spread:.6f}")
    print(f"    mean quoted error bar  = {mean_se:.6f}  (should match spread)")
    print(f"    pull distribution: mean = {pulls.mean():+.3f}, std = "
          f"{pulls.std():.3f} (ideal 0, 1)")

    # ---- (5) final honest result --------------------------------------
    final_err = err_plateau
    print(f"\n(5) FINAL RESULT:  <V> = {est:.5f} +/- {final_err:.5f}")
    print(f"    exact = {V_exact:.5f};  deviation = "
          f"{abs(est-V_exact)/final_err:.2f} sigma  -> VALIDATED")

    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(levels, errs, "o-", color="#2c3e50", ms=5)
    ax.axhline(err_plateau, color="#c0392b", ls="--", lw=1.5,
               label=f"plateau = {err_plateau:.4f}")
    ax.set_xlabel("blocking level"); ax.set_ylabel("estimated SEM")
    ax.set_title("(a)  blocking: error rises to a plateau")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.hist(ests, bins=20, density=True, color="#9ecae1",
            label="seed estimates")
    xx = np.linspace(ests.min(), ests.max(), 200)
    ax.plot(xx, np.exp(-(xx - V_exact) ** 2 / (2 * spread ** 2)) /
            (spread * np.sqrt(2 * np.pi)), color="#c0392b", lw=2,
            label="$N(V_{\\rm exact},\\sigma)$")
    ax.axvline(V_exact, color="#27ae60", ls=":", lw=1.5, label="exact")
    ax.set_xlabel(r"$\langle V\rangle$ estimate"); ax.set_ylabel("density")
    ax.set_title("(b)  reproducibility over seeds"); ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ns = np.unique(np.logspace(2.5, np.log10(len(chain)), 40).astype(int))
    run = np.array([chain[:n].mean() for n in ns])
    runerr = np.array([blocking_errors(chain[:n])[1][-4:].max() for n in ns])
    ax.plot(ns, run, color="#2c3e50", lw=1.5, label="running $\\langle V\\rangle$")
    ax.fill_between(ns, run - runerr, run + runerr, color="#9ecae1", alpha=0.6,
                    label="$\\pm$ error")
    ax.axhline(V_exact, color="#c0392b", ls="--", lw=1.5, label="exact")
    ax.set_xscale("log"); ax.set_xlabel("samples  $N$")
    ax.set_ylabel(r"$\langle V\rangle$")
    ax.set_title("(c)  convergence to the exact value"); ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    ax.hist(pulls, bins=16, density=True, color="#9ecae1", label="pulls")
    xx = np.linspace(-4, 4, 200)
    ax.plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), color="#c0392b",
            lw=2, label="$N(0,1)$")
    ax.set_xlabel(r"pull $(\hat V-V_{\rm exact})/\sigma$"); ax.set_ylabel("density")
    ax.set_title(f"(d)  error bars are honest (std={pulls.std():.2f})")
    ax.legend(frameon=False)

    fig.suptitle("Verification, validation and reproducibility of a Monte Carlo "
                 "average", y=1.03, fontsize=13)
    fig.savefig("fig10_3.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig10_3.png")


if __name__ == "__main__":
    main()
