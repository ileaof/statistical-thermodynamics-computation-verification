#!/usr/bin/env python3
"""
Example 1.2 (direct numerical) -- Two Einstein solids in thermal contact.

Model
-----
An Einstein solid is a set of N independent quantum harmonic oscillators, each
of energy-level spacing (hbar omega).  A macrostate holding q energy quanta has
multiplicity (a stars-and-bars count of distributing q indistinguishable quanta
among N oscillators)

        Omega(N, q) = C(q + N - 1, q).

Two solids A and B share a fixed total  q = q_A + q_B  of quanta and can only
exchange energy with each other (isolated composite).  By the postulate of
equal a priori probabilities every one of the

        Omega_tot = C(q + N_A + N_B - 1, q)

microstates of the composite is equally likely, so the probability of finding
q_A quanta in A is

        P(q_A) = Omega_A(N_A, q_A) * Omega_B(N_B, q - q_A) / Omega_tot.

This script computes that distribution exactly with log-gamma arithmetic (no
overflow), locates the most probable partition, and *verifies* three exact
statements: (i) the multiplicities sum to Omega_tot (Vandermonde identity),
(ii) the peak sits at q_A* = q N_A/(N_A+N_B), and (iii) at that peak the two
solids share a common statistical temperature  1/T = dS/dE.  The distribution
sharpens as the solids grow -- the microscopic content of the second law.

Only numpy, scipy and matplotlib are used.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

k_B = 1.380649e-23                          # J/K


def log_omega(N, q):
    """ln Omega(N, q) = ln C(q + N - 1, q), vectorised over integer q."""
    q = np.asarray(q, dtype=float)
    return gammaln(q + N) - gammaln(q + 1.0) - gammaln(float(N))


def distribution(NA, NB, q):
    """Exact probability P(q_A) for q_A = 0..q and the composite log-count."""
    qA = np.arange(0, q + 1)
    log_wA = log_omega(NA, qA)
    log_wB = log_omega(NB, q - qA)
    log_joint = log_wA + log_wB
    log_tot = log_omega(NA + NB, q)         # closed form for the denominator
    P = np.exp(log_joint - log_tot)         # stable: subtract before exp
    return qA, P, log_tot, log_joint


def main():
    print("=" * 70)
    print("Example 1.2  Two Einstein solids exchanging energy quanta")
    print("=" * 70)

    # ---- Verification (i): Vandermonde sum rule -------------------------
    NA, NB, q = 300, 200, 400
    qA, P, _log_tot, _log_joint = distribution(NA, NB, q)
    print(f"\nComposite: N_A={NA}, N_B={NB}, q={q}")
    print(f"  sum_qA Omega_A Omega_B / Omega_tot = {P.sum():.12f}  (exact 1)")
    print(f"  => Vandermonde identity verified to {abs(P.sum()-1):.2e}")

    # ---- Verification (ii): location of the peak ------------------------
    qA_star_predicted = q * NA / (NA + NB)
    qA_star_numeric = qA[np.argmax(P)]
    print(f"\n  most probable q_A: predicted q N_A/(N_A+N_B) = "
          f"{qA_star_predicted:.2f}")
    print(f"                     numerical argmax           = "
          f"{qA_star_numeric:d}")

    # ---- Verification (iii): equal temperatures at the peak -------------
    # Statistical temperature of an Einstein solid:
    #   S = k_B ln Omega,  1/T = dS/dE,  E = q hbar omega
    #   => k_B/(hbar omega) * d(ln Omega)/dq = 1/T.
    # A high-q oscillator solid has  d ln Omega/dq ~ ln(1 + N/q); equal T means
    # (N_A/q_A*)-matched log-derivatives.  Compare discrete derivatives:
    def dlnOmega_dq(N, qval):
        return log_omega(N, qval + 1) - log_omega(N, qval)      # forward diff
    qA_s = int(round(qA_star_predicted))
    betaA = dlnOmega_dq(NA, qA_s)
    betaB = dlnOmega_dq(NB, q - qA_s)
    print(f"\n  d(lnOmega_A)/dq_A at peak = {betaA:.6f}")
    print(f"  d(lnOmega_B)/dq_B at peak = {betaB:.6f}")
    print(f"  temperatures match to     {abs(betaA-betaB):.2e}  "
          f"(=> T_A = T_B at equilibrium)")

    # ---- Sharpening of the peak with system size ------------------------
    print("\n  Relative width sigma(q_A)/q_A* vs size (equal solids, q=2N):")
    for N in [20, 100, 500, 2500]:
        qq = 2 * N
        x, p, *_ = distribution(N, N, qq)
        mean = np.sum(x * p)
        var = np.sum((x - mean) ** 2 * p)
        print(f"    N={N:>5d}:  sigma/q_A* = {np.sqrt(var)/mean:.5f}")

    # -----------------------------------------------------------------
    # Figure: (a) exact distribution; (b) sharpening with size
    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4),
                                   constrained_layout=True)

    for (NA_, NB_, q_, c, lab) in [
        (30, 30, 60, "#9ecae1", "small: $N_A=N_B=30$"),
        (300, 300, 600, "#c0392b", "large: $N_A=N_B=300$")]:
        x, p, *_ = distribution(NA_, NB_, q_)
        ax1.plot(x / q_, p / p.max(), color=c, lw=2, label=lab)
    ax1.axvline(0.5, color="k", ls=":", lw=1)
    ax1.set_xlabel(r"energy fraction in solid A,  $q_A/q$")
    ax1.set_ylabel("probability (peak-normalised)")
    ax1.set_title("(a)  equilibrium is overwhelmingly probable")
    ax1.legend(frameon=False)

    Ns = np.array([10, 30, 100, 300, 1000, 3000, 10000])
    rel = []
    for N in Ns:
        x, p, *_ = distribution(N, N, 2 * N)
        mean = np.sum(x * p)
        rel.append(np.sqrt(np.sum((x - mean) ** 2 * p)) / mean)
    ax2.loglog(Ns, rel, "o", color="#2c3e50", ms=7, label="exact width")
    ax2.loglog(Ns, rel[0] * np.sqrt(Ns[0] / Ns), "-", color="#c0392b",
               lw=2, label=r"$\propto N^{-1/2}$")
    ax2.set_xlabel("oscillators per solid  $N$")
    ax2.set_ylabel(r"relative width  $\sigma_{q_A}/q_A^{*}$")
    ax2.set_title("(b)  fluctuations vanish as  $N^{-1/2}$")
    ax2.legend(frameon=False)

    fig.suptitle("Energy sharing between two Einstein solids", y=1.06,
                 fontsize=13)
    fig.savefig("fig1_2.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig1_2.png")


if __name__ == "__main__":
    main()
