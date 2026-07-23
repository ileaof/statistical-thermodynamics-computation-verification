#!/usr/bin/env python3
"""
Example 10.2 (direct numerical) -- The Metropolis algorithm: sampling a Boltzmann
distribution, autocorrelation, and honest error bars.

The Metropolis algorithm builds a Markov chain whose stationary distribution is the
Boltzmann distribution p(x) ~ exp(-beta V(x)).  A trial move x -> x + delta is
accepted with probability min(1, exp(-beta[V(x')-V(x)])), which satisfies detailed
balance and therefore samples p(x) correctly.  Using the anharmonic oscillator
V(x) = x^2/2 + lambda x^4 of Example 10.1, this script:

  * verifies that the sampled histogram matches the exact Boltzmann distribution;
  * measures the integrated autocorrelation time tau of the chain;
  * shows how the step size controls the acceptance rate and tau (there is an
    optimum near 40-50% acceptance);
  * demonstrates block averaging, which restores the correct 1/sqrt(N) error once
    blocks are longer than the autocorrelation time.

numpy/scipy/matplotlib only; fixed seed.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

RNG = np.random.default_rng(20260723)
BETA, LAM = 1.0, 0.1


def V(x):
    return 0.5 * x ** 2 + LAM * x ** 4


def metropolis(n_steps, step, x0=0.0, burn=5000):
    """Metropolis chain; returns the sampled trajectory and acceptance rate."""
    x = x0
    Vx = V(x)
    traj = np.empty(n_steps)
    n_acc = 0
    for i in range(n_steps + burn):
        xt = x + RNG.uniform(-step, step)
        Vt = V(xt)
        if Vt < Vx or RNG.random() < np.exp(-BETA * (Vt - Vx)):
            x, Vx = xt, Vt
            if i >= burn:
                n_acc += 1
        if i >= burn:
            traj[i - burn] = x
    return traj, n_acc / n_steps


def autocorr_time(a, cutoff=200):
    """Integrated autocorrelation time of series a (normalised ACF sum)."""
    a = a - a.mean()
    n = len(a)
    var = np.dot(a, a) / n
    tau = 0.5
    for k in range(1, cutoff):
        c = np.dot(a[:-k], a[k:]) / (n - k) / var
        if c <= 0:
            break
        tau += c
    return tau


def main():
    print("=" * 70)
    print("Example 10.2  Metropolis sampling, autocorrelation, block averaging")
    print("=" * 70)

    # exact reference
    Z, _ = quad(lambda t: np.exp(-BETA * V(t)), -np.inf, np.inf)
    x2_exact = quad(lambda t: t**2*np.exp(-BETA*V(t)), -np.inf, np.inf)[0] / Z
    print(f"\nExact <x^2> = {x2_exact:.6f}")

    traj, acc = metropolis(200_000, step=3.0)
    print(f"\nChain of {len(traj)} steps, step=3.0, acceptance = {acc:.3f}")
    print(f"  sample mean <x^2> = {np.mean(traj**2):.6f}")
    tau = autocorr_time(traj ** 2)
    print(f"  integrated autocorrelation time tau = {tau:.1f} steps")
    n_eff = len(traj) / (2 * tau)
    naive = np.std(traj ** 2) / np.sqrt(len(traj))
    corrected = np.std(traj ** 2) / np.sqrt(n_eff)
    print(f"  naive error = {naive:.5f}, autocorrelation-corrected = "
          f"{corrected:.5f}  (factor sqrt(2 tau) = {np.sqrt(2*tau):.1f})")

    # --- step-size study -----------------------------------------------
    print("\nStep-size study (acceptance and autocorrelation time):")
    print(f"  {'step':>6} {'acceptance':>12} {'tau':>8}")
    steps = [0.3, 1.0, 3.0, 8.0, 20.0]
    accs, taus = [], []
    for st in steps:
        tr, ac = metropolis(60_000, step=st)
        t = autocorr_time(tr ** 2)
        accs.append(ac); taus.append(t)
        print(f"  {st:>6.1f} {ac:>12.3f} {t:>8.1f}")

    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)

    # (a) histogram vs exact Boltzmann
    ax = axes[0, 0]
    ax.hist(traj, bins=80, density=True, color="#9ecae1", label="Metropolis")
    xx = np.linspace(-5, 5, 300)
    ax.plot(xx, np.exp(-BETA * V(xx)) / Z, color="#c0392b", lw=2,
            label="exact Boltzmann")
    ax.set_xlabel("$x$"); ax.set_ylabel("probability density")
    ax.set_title("(a)  sampled distribution vs exact"); ax.legend(frameon=False)

    # (b) autocorrelation function
    ax = axes[0, 1]
    a = traj ** 2 - np.mean(traj ** 2)
    var = np.dot(a, a) / len(a)
    ks = np.arange(0, 60)
    acf = [np.dot(a[:len(a)-k], a[k:]) / (len(a) - k) / var for k in ks]
    ax.plot(ks, acf, "o-", color="#2c3e50", ms=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("lag  $k$"); ax.set_ylabel(r"autocorrelation of $x^2$")
    ax.set_title(rf"(b)  ACF, $\tau={tau:.1f}$")

    # (c) acceptance and tau vs step size
    ax = axes[1, 0]
    ax.semilogx(steps, accs, "o-", color="#2980b9", lw=2, label="acceptance")
    ax.set_xlabel("step size"); ax.set_ylabel("acceptance rate", color="#2980b9")
    ax.tick_params(axis="y", labelcolor="#2980b9")
    ax2 = ax.twinx()
    ax2.semilogx(steps, taus, "s--", color="#c0392b", lw=2, label=r"$\tau$")
    ax2.set_ylabel(r"autocorrelation time $\tau$", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax.set_title("(c)  step size: acceptance vs autocorrelation")

    # (d) block averaging: error estimate vs block size
    ax = axes[1, 1]
    obs = traj ** 2
    block_sizes = np.unique(np.logspace(0, 3.3, 25).astype(int))
    errs = []
    for b in block_sizes:
        nb = len(obs) // b
        means = obs[:nb * b].reshape(nb, b).mean(1)
        errs.append(means.std() / np.sqrt(nb))
    ax.semilogx(block_sizes, errs, "o-", color="#2c3e50", ms=4)
    ax.axhline(corrected, color="#c0392b", ls="--", lw=1.5,
               label="autocorr-corrected error")
    ax.set_xlabel("block size"); ax.set_ylabel(r"estimated error in $\langle x^2\rangle$")
    ax.set_title("(d)  block averaging reaches a plateau")
    ax.legend(frameon=False)

    fig.suptitle("The Metropolis algorithm: sampling, autocorrelation, error bars",
                 y=1.03, fontsize=13)
    fig.savefig("fig10_2.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig10_2.png")


if __name__ == "__main__":
    main()
