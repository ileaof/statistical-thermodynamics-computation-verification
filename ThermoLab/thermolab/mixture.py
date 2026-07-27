"""Multicomponent mixtures."""

from __future__ import annotations

from .fluid import _FluidBase, _DEFAULT_BACKEND


class Mixture(_FluidBase):
    """A binary or multicomponent mixture.

    Parameters
    ----------
    components:
        List of component names (case-insensitive), e.g.
        ``["N2", "O2", "CO2", "H2O"]``.
    fractions:
        Mole fractions, same length as ``components``. Normalized to sum to 1.
    backend:
        Thermodynamic backend name (default ``"thermopack"``).
    eos:
        Optional explicit EOS override; otherwise GERG2008 is used when every
        component is in the GERG core (air/combustion gases) and SRK otherwise
        (e.g. refrigerant blends).

    Examples
    --------
    >>> mix = Mixture(["N2", "O2", "CO2", "H2O"],
    ...               [0.78, 0.21, 0.005, 0.005], backend="thermopack")
    >>> st = mix.state(T=1200, P=3e5)
    """

    def __init__(self, components: list[str], fractions, *,
                 backend: str = _DEFAULT_BACKEND,
                 reference_state: str = "DEFAULT", eos: str | None = None):
        if len(components) < 1:
            raise ValueError("Mixture requires at least one component.")
        super().__init__(backend, list(components), fractions,
                         reference_state=reference_state, eos=eos)

    @property
    def nc(self) -> int:
        """Number of components."""
        return len(self._components)