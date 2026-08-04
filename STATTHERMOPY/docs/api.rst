API reference
=============

Core data structures
--------------------

.. automodule:: statthermopy.core.molecule
   :members:

.. automodule:: statthermopy.core.state
   :members:

.. automodule:: statthermopy.core.contribution
   :members:

Partition-function modes
------------------------

.. automodule:: statthermopy.modes.translational
   :members:
.. automodule:: statthermopy.modes.rotational
   :members:
.. automodule:: statthermopy.modes.vibrational
   :members:
.. automodule:: statthermopy.modes.electronic
   :members:

Assembly and thermodynamics
---------------------------

.. automodule:: statthermopy.partition
   :members:
.. automodule:: statthermopy.thermodynamics
   :members:
.. automodule:: statthermopy.mixture
   :members:

Database
--------

.. automodule:: statthermopy.database.registry
   :members:

Statistical transport properties
-------------------------------

Transport and thermophysical coefficients of a pure gas from the **Chapman–Enskog** first-order
solution of the Boltzmann equation with the **Lennard–Jones** pair potential, plus the ideal-gas
thermophysical coefficients and the derived dimensionless groups (Prandtl/Schmidt/Lewis). Fully
first-principles: heat capacities and ``γ`` come from the partition-function engine, the only
molecular inputs are ``σ`` and ``ε`` (:class:`~statthermopy.core.molecule.LennardJones`); no
external property database (REFPROP/CoolProp) is used.

.. automodule:: statthermopy.transport.transport
   :members:
.. automodule:: statthermopy.transport.collision
   :members:
.. automodule:: statthermopy.transport.lennard_jones
   :members:
.. automodule:: statthermopy.transport.plots
   :members:
.. automodule:: statthermopy.transport.export
   :members:

Utilities
--------

.. automodule:: statthermopy.constants
   :members:
.. automodule:: statthermopy.units
   :members:
.. automodule:: statthermopy.io.exporters
   :members:
.. automodule:: statthermopy.plots.plotting
   :members:
.. automodule:: statthermopy.validation.base
   :members:
.. automodule:: statthermopy.validation.reference
   :members:
.. automodule:: statthermopy.backend.executor
   :members:
.. automodule:: statthermopy.backend.numba_backend
   :members:
.. automodule:: statthermopy.backend.openmp_backend
   :members:
.. automodule:: statthermopy.backend.cuda_backend
   :members:

GUI (optional, requires PySide6)
-------------------------------

.. automodule:: statthermopy.gui.app
   :members:
.. automodule:: statthermopy.gui.mainwindow
   :members: