"""Pressure-space post-processing for the final Nd product.

The trained network and EOF coefficients are left untouched. This module
only modifies reconstructed Nd/sigmaNd during diagnostics or production
inference.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

VALID_TAPER_MODES = ("none", "climatology", "zero", "base")


def normalize_taper_mode(mode: str | None) -> str:
    if mode is None:
        return "none"
    mode = str(mode).strip().lower()
    aliases = {
        "off": "none",
        "false": "none",
        "clim": "climatology",
        "mean": "climatology",
        "nd_mean": "climatology",
        "0": "zero",
        "reanalysis": "base",
        "baseline": "base",
    }
    mode = aliases.get(mode, mode)
    if mode not in VALID_TAPER_MODES:
        raise ValueError(
            f"Unknown upper-taper mode {mode!r}; expected one of "
            f"{VALID_TAPER_MODES}."
        )
    return mode


def upper_taper_weight(pressure_hpa, p_start=500.0, p_top=350.0):
    """Quintic smootherstep weight in pressure coordinates.

    The weight is exactly 1 for p >= p_start, exactly 0 for p <= p_top,
    and has zero first and second derivative at both joins.
    """
    p_start = float(p_start)
    p_top = float(p_top)
    if not p_start > p_top:
        raise ValueError(
            f"p_start must be greater than p_top; got {p_start} and {p_top}."
        )

    x = (pressure_hpa - p_top) / (p_start - p_top)
    if isinstance(x, xr.DataArray):
        x = x.clip(min=0.0, max=1.0)
    else:
        x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)

    return 6.0 * x**5 - 15.0 * x**4 + 10.0 * x**3


def interp_profile_logp(profile, pressure_hpa, lev_dim="lev"):
    """Interpolate a 1-D climatological profile in log-pressure space."""
    if not isinstance(profile, xr.DataArray):
        raise TypeError("profile must be an xarray.DataArray with a lev coordinate")
    if lev_dim not in profile.dims:
        raise ValueError(f"profile must contain {lev_dim!r}; got dims={profile.dims}")

    lev = np.asarray(profile[lev_dim].values, dtype=np.float64)
    val = np.asarray(profile.values, dtype=np.float64)
    good = np.isfinite(lev) & np.isfinite(val) & (lev > 0.0)
    if good.sum() < 2:
        raise ValueError("climatological profile has fewer than two valid pressure levels")

    lev = lev[good]
    val = val[good]
    order = np.argsort(np.log(lev))
    xlev = np.log(lev[order])
    ylev = val[order]

    is_da = isinstance(pressure_hpa, xr.DataArray)
    p = np.asarray(
        pressure_hpa.values if is_da else pressure_hpa,
        dtype=np.float64,
    )
    out = np.full(p.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(p) & (p > 0.0)
    if np.any(ok):
        out[ok] = np.interp(
            np.log(p[ok]),
            xlev,
            ylev,
            left=np.nan,
            right=np.nan,
        )

    if is_da:
        return xr.DataArray(
            out.astype(np.float32),
            dims=pressure_hpa.dims,
            coords=pressure_hpa.coords,
            name="Nd_climatology",
        )
    if out.ndim == 0:
        return float(out)
    return out.astype(np.float32)


def apply_nd_pressure_taper(
    nd_raw,
    Nd_mean,
    pressure_hpa=None,
    mode="climatology",
    p_start=500.0,
    p_top=350.0,
    base_nd=None,
    clip_min=0.0,
    name=None,
):
    """Apply the selected upper-pressure taper to reconstructed Nd.

    ``climatology`` relaxes the NN anomaly toward the EOF climatology::

        Nd_final = Nd_mean(p) + T(p) * [Nd_raw - Nd_mean(p)]

    ``zero`` relaxes total Nd toward zero::

        Nd_final = T(p) * Nd_raw

    ``base`` relaxes the refined NN product toward the physically based
    reanalysis-emulation product::

        Nd_final = Nd_base + T(p) * [Nd_raw - Nd_base]

    ``none`` returns the raw NN reconstruction unchanged.

    In base mode ``base_nd`` is required and is never modified by this
    function.
    """
    mode = normalize_taper_mode(mode)
    if mode == "none":
        out = (
            nd_raw.copy(deep=False)
            if isinstance(nd_raw, xr.DataArray)
            else np.array(nd_raw, copy=True)
        )
        if name is not None and isinstance(out, xr.DataArray):
            out = out.rename(name)
        return out

    if pressure_hpa is None:
        if not isinstance(nd_raw, xr.DataArray) or "lev" not in nd_raw.coords:
            raise ValueError("pressure_hpa is required unless nd_raw has a lev coordinate")
        pressure_hpa = nd_raw["lev"]

    T = upper_taper_weight(pressure_hpa, p_start=p_start, p_top=p_top)

    if mode == "climatology":
        target = interp_profile_logp(Nd_mean, pressure_hpa)
    elif mode == "zero":
        target = 0.0
    else:
        if base_nd is None:
            raise ValueError("base_nd is required when upper-taper mode is 'base'")
        target = base_nd

    out = target + T * (nd_raw - target)

    if clip_min is not None:
        if isinstance(out, xr.DataArray):
            out = out.clip(min=float(clip_min))
        else:
            out = np.maximum(out, float(clip_min))

    if isinstance(out, xr.DataArray):
        out.attrs = dict(getattr(nd_raw, "attrs", {}))
        out.attrs.update(
            upper_taper_mode=mode,
            upper_taper_start_hpa=float(p_start),
            upper_taper_top_hpa=float(p_top),
        )
        if name is not None:
            out = out.rename(name)

    return out


def apply_sigma_pressure_taper(
    sigma_raw,
    pressure_hpa=None,
    mode="climatology",
    p_start=500.0,
    p_top=350.0,
    name=None,
):
    """Apply the uncertainty transformation consistent with the Nd taper.

    For every active taper mode, sigma_final = T(p) * sigma_raw.  In
    ``climatology`` mode the climatology is treated as deterministic; in
    ``base`` mode the reanalysis-emulation fallback is treated as the fixed
    reference and this sigma represents uncertainty in the refined departure
    from it.
    """
    mode = normalize_taper_mode(mode)
    if mode == "none":
        out = (
            sigma_raw.copy(deep=False)
            if isinstance(sigma_raw, xr.DataArray)
            else np.array(sigma_raw, copy=True)
        )
        if name is not None and isinstance(out, xr.DataArray):
            out = out.rename(name)
        return out

    if pressure_hpa is None:
        if not isinstance(sigma_raw, xr.DataArray) or "lev" not in sigma_raw.coords:
            raise ValueError("pressure_hpa is required unless sigma_raw has a lev coordinate")
        pressure_hpa = sigma_raw["lev"]

    T = upper_taper_weight(pressure_hpa, p_start=p_start, p_top=p_top)
    out = T * sigma_raw

    if isinstance(out, xr.DataArray):
        out = out.clip(min=0.0)
        out.attrs = dict(getattr(sigma_raw, "attrs", {}))
        out.attrs.update(
            upper_taper_mode=mode,
            upper_taper_start_hpa=float(p_start),
            upper_taper_top_hpa=float(p_top),
        )
        if name is not None:
            out = out.rename(name)
    else:
        out = np.maximum(out, 0.0)

    return out
