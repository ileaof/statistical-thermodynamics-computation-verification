StatThermoPy Documentation
==========================

StatThermoPy computes thermodynamic properties of ideal gases **exclusively from Statistical
Mechanics** — via the molecular partition function ``Q = Q_t Q_r Q_v Q_e`` — without any
empirical property correlations (NASA polynomials, JANAF, Shomate, CoolProp, REFPROP).

.. toctree::
   :maxdepth: 2

   theory
   api

Quick start
-----------

.. code-block:: python

   from statthermopy import Thermodynamics, State
   from statthermopy.database import get

   n2 = get("N2")
   th = Thermodynamics(n2, State(T=298.15, P=101325.0)).compute()
   print(th.Cp_m, th.gamma, th.S_m)

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`