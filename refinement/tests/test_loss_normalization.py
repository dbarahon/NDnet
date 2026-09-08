from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# The refinement directory is script-oriented rather than an installed Python
# package. Add it explicitly so these tests exercise the same imports used by
# NDref_prob.py when run from refinement/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loss_normalization import (  # noqa: E402
    build_nd_gram,
    estimate_base_coefficient_scales,
    logp_cell_weights,
)


def test_logp_weights_are_positive_and_normalized():
    pressure = np.array([1000.0, 900.0, 700.0, 500.0, 300.0])
    weight = logp_cell_weights(pressure)
    assert np.all(weight > 0.0)
    assert np.isclose(weight.sum(), 1.0)


def test_nd_gram_matches_direct_reconstruction():
    pressure = np.array([1000.0, 800.0, 500.0])
    eof = np.array([[1.0, 2.0, 0.0], [0.5, -1.0, 3.0]])
    nd_scale = 10.0
    gram, weight = build_nd_gram(eof, pressure, nd_scale)

    coeff = np.array([2.0, -0.5])
    mode_space = coeff @ gram @ coeff
    direct = np.sum(weight * ((coeff @ eof) / nd_scale) ** 2)
    assert np.isclose(mode_space, direct)


def test_coefficient_scale_estimator_returns_area_weighted_rms():
    mu0 = torch.zeros((2, 2, 2, 1), dtype=torch.float32)
    mu0[:, 0, :, :] = 2.0
    mu0[:, 1, :, :] = 4.0
    lat = np.array([-30.0, 30.0])

    scale = estimate_base_coefficient_scales(
        mu0,
        lat,
        n_time=2,
        floor=0.1,
        chunk_size=1,
    )
    assert np.allclose(scale, [2.0, 4.0])
