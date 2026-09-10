import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pressure_weighting import linear_upper_pressure_weight  # noqa: E402


def test_linear_upper_pressure_weight_is_continuous_and_monotonic():
    pressure = np.array([700.0, 600.0, 500.0, 450.0, 400.0, 350.0, 300.0])
    weight = linear_upper_pressure_weight(
        pressure,
        start_hpa=500.0,
        top_hpa=350.0,
        top_factor=3.0,
    )
    expected = np.array([1.0, 1.0, 1.0, 5.0 / 3.0, 7.0 / 3.0, 3.0, 3.0])
    assert np.allclose(weight, expected)
    assert np.all(np.diff(weight) >= 0.0)


def test_top_factor_one_disables_pressure_reweighting():
    pressure = np.array([700.0, 500.0, 400.0, 350.0])
    weight = linear_upper_pressure_weight(
        pressure,
        start_hpa=500.0,
        top_hpa=350.0,
        top_factor=1.0,
    )
    assert np.allclose(weight, 1.0)
