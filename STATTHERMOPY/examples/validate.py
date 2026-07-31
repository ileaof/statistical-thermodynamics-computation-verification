"""Validate the StatThermoPy engine against embedded NIST/JANAF reference data.

Runs :func:`statthermopy.validation.validate` for every shipped reference species and prints
a per-species table of mean/max percent error for Cp and S. The reference tables contain only
*values* (curated from the NIST Chemistry WebBook Shomate equations, evaluated at a T grid,
standard state 1 bar) — no empirical correlation coefficients ship in the package, so the
calculation core stays pure statistical mechanics.
"""

from __future__ import annotations

from statthermopy.validation import list_references, validate


def main() -> None:
    species = list_references()
    print(f"Embedded reference data for {len(species)} species: {', '.join(species)}\n")
    header = f"{'Species':7s} {'Cp MAE%':>8s} {'Cp max%':>8s} {'S MAE%':>8s} {'S max%':>8s}"
    print(header)
    print("-" * len(header))
    worst_cp = worst_s = 0.0
    for sp in species:
        cp = validate(sp, "Cp")
        s = validate(sp, "S")
        worst_cp = max(worst_cp, cp.max_abs_error_percent)
        worst_s = max(worst_s, s.max_abs_error_percent)
        print(
            f"{sp:7s} {cp.mean_abs_error_percent:8.3f} {cp.max_abs_error_percent:8.3f} "
            f"{s.mean_abs_error_percent:8.3f} {s.max_abs_error_percent:8.3f}"
        )
    print(
        f"\nWorst-case error across all species: Cp {worst_cp:.2f}%, S {worst_s:.2f}% "
        f"(rigid-rotor / harmonic-oscillator model; deviations grow toward high T due to "
        f"neglected anharmonicity and internal-rotor treatments)."
    )


if __name__ == "__main__":
    main()
