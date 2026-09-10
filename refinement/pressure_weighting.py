"""Fixed pressure weights for the direct site Nd objective.

The weighting is intentionally simple and deterministic. It changes the
relative importance of observed pressure levels within each site while the
site-level loss remains normalized and each site retains equal weight.
"""

from __future__ import annotations

import numpy as np


def linear_upper_pressure_weight(
    pressure_hpa: np.ndarray,
    start_hpa: float,
    top_hpa: float,
    top_factor: float,
) -> np.ndarray:
    """Return a continuous linear pressure weight for upper-level site Nd.

    The weight is 1 at and below the lower-atmosphere side of the transition
    (``pressure >= start_hpa``), increases linearly as pressure decreases, and
    reaches ``top_factor`` at ``top_hpa``. It remains at ``top_factor`` at
    lower pressures.

    Linear interpolation is performed in pressure, not log-pressure. This is
    deliberate: the goal is a transparent monotonic reweighting of the native
    site levels without introducing a step at a particular model level.
    """
    p = np.asarray(pressure_hpa, dtype=np.float64)
    start_hpa = float(start_hpa)
    top_hpa = float(top_hpa)
    top_factor = float(top_factor)

    if p.ndim != 1 or p.size < 1:
        raise ValueError("pressure_hpa must be a non-empty 1D array")
    if np.any(~np.isfinite(p)) or np.any(p <= 0.0):
        raise ValueError("pressure_hpa must contain finite positive values")
    if not np.isfinite(start_hpa) or not np.isfinite(top_hpa):
        raise ValueError("start_hpa and top_hpa must be finite")
    if not start_hpa > top_hpa:
        raise ValueError(
            "start_hpa must exceed top_hpa for upper-level weighting; "
            f"got {start_hpa:g} and {top_hpa:g}"
        )
    if not np.isfinite(top_factor) or top_factor < 1.0:
        raise ValueError("top_factor must be finite and >= 1")

    fraction = (start_hpa - p) / (start_hpa - top_hpa)
    fraction = np.clip(fraction, 0.0, 1.0)
    return 1.0 + (top_factor - 1.0) * fraction
