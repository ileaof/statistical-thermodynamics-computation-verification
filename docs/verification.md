# The verification philosophy

> *A computed number is an opinion until it has been verified.*

This single principle shapes every program in the repository. A simulation or a
numerical evaluation is not finished when it produces a number; it is finished
when that number has been shown to be correct, and its uncertainty honestly
quantified. This page explains the practices the code follows so that you can
apply them to your own work.

## Every result is checked against something independent

No program reports a number without a reference. The reference is one of:

1. **An exact analytical result.** The Sackur-Tetrode entropy is compared with
   experiment; the Stefan-Boltzmann constant recovered from the Planck spectrum
   is compared with CODATA; the 2-D Ising magnetization is compared with
   Onsager's exact solution.
2. **A limiting case.** Heat capacities are checked against the Dulong-Petit and
   equipartition limits; quantum statistics are checked against their classical
   (dilute) limit; the continuum partition function is checked against the exact
   sum.
3. **A second, independent method.** Heat capacities are computed from a closed
   form, from a finite difference of the energy, *and* from energy fluctuations,
   and the three must agree. Monte Carlo error bars are computed by blocking
   *and* by bootstrap.

Where the checks pass, the console output says so explicitly, printing the
computed value, the reference value, and the discrepancy.

## Convergence is measured, not assumed

When a result depends on a discretization (a time step, a grid, a number of
retained levels) or on a sample count, the program measures how the error
*scales* and confirms the expected order:

| Method | Expected scaling | Where it appears |
|---|---|---|
| Monte Carlo estimate | error ∝ N<sup>&minus;1/2</sup> | Chapters 1, 2, 7, 10 |
| Symplectic (velocity-Verlet) integrator | energy drift ∝ (Δt)<sup>2</sup> | Chapter 2 |
| Continuum vs. discrete partition sum | error ∝ √α | Chapter 4 |
| Truncated level sum | geometric convergence | Chapter 3 |
| Finite-size scaling | χ ∝ L<sup>γ/ν</sup> | Chapter 9 |

The empirical order is obtained by fitting the slope of log(error) versus
log(size) &mdash; exactly what
[`utilities.convergence_order`](../src/statistical_thermodynamics/utilities.py)
does.

## Stochastic programs are reproducible

Every program that uses random numbers fixes its seed through
`numpy.random.default_rng(20260723)`. Re-running any program therefore gives
**identical** output, down to the last digit. Reproducibility is not a nicety;
it is a precondition for verification, because a result you cannot reproduce you
cannot check.

## Uncertainty is quantified honestly

Monte Carlo results are meaningless without error bars, and error bars are
misleading if correlations are ignored. The code therefore:

- measures the **integrated autocorrelation time** of a Markov chain and inflates
  the naive error by √(2τ);
- uses **Flyvbjerg-Petersen blocking**, whose error estimate rises and then
  *plateaus* at the true value once blocks exceed the correlation time;
- cross-checks with a **bootstrap** over decorrelated blocks;
- validates the error bars themselves by the **pull distribution**: over many
  independent seeds, (estimate &minus; exact) / error must follow a unit normal.

The capstone program, [`ex10_3_advanced.py`](../Chapter10_Computational_Statistical_Thermodynamics/ex10_3_advanced.py),
demonstrates all four together.

## Verification, validation and reproducibility

The three words are not synonyms:

- **Verification** &mdash; *are we solving the equations right?* (numerics)
- **Validation** &mdash; *are we solving the right equations?* (physics, vs.
  exact results or experiment)
- **Reproducibility** &mdash; *can anyone obtain the same result?* (fixed seeds,
  pinned dependencies, self-contained code)

Chapter 10 is devoted to making all three routine. The rest of the book puts
them into practice on every problem.
