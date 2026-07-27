"""Optimization tests."""

from __future__ import annotations

import pytest

from thermolab import cycles as C
from thermolab import optimization as opt


def test_brayton_optimize_pressure_ratio():
    res = opt.optimize_cycle(
        lambda rp: C.brayton(pressure_ratio=rp[0] if hasattr(rp, "__len__") else rp, T3=1400),
        bounds=[(4, 30)], objective="eta", x0=[10.0],
    )
    # efficiency increases with pressure ratio in this range -> optimum near upper bound
    assert res.success
    assert res.x[0] > 10


def test_sweep():
    df = opt.sweep(lambda pressure_ratio: C.brayton(pressure_ratio=pressure_ratio, T3=1400),
                   "pressure_ratio", [4, 8, 12, 16, 20])
    assert len(df) == 5
    assert df["eta"].iloc[-1] > df["eta"].iloc[0]  # rises with rp here