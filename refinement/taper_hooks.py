"""Install final-product pressure taper hooks for refinement diagnostics.

The trained model, coefficient inference, and reanalysis-emulation base product
are left unchanged.  Tapering is applied immediately before diagnostic NetCDF
output, when the refined NN product and the untouched base product are both
available in the same Dataset.

This is particularly important for ``UPPER_TAPER_MODE = 'base'``:

    Nd_final = Nd_base + T(p) * (Nd_nn_raw - Nd_base)

The base field itself is never modified.
"""

from __future__ import annotations

import xarray as xr

import config as cfg
import utils

from profile_taper import (
    apply_nd_pressure_taper,
    apply_sigma_pressure_taper,
    normalize_taper_mode,
)


def _settings():
    return (
        normalize_taper_mode(getattr(cfg, "UPPER_TAPER_MODE", "none")),
        float(getattr(cfg, "UPPER_TAPER_START_HPA", 500.0)),
        float(getattr(cfg, "UPPER_TAPER_TOP_HPA", 350.0)),
        bool(getattr(cfg, "SAVE_RAW_TAPER_FIELDS", True)),
    )


def _apply_final_product_taper(ds):
    """Mutate a recognized diagnostic Dataset so plots see the final product.

    Recognized refinement outputs
    -----------------------------
    site:
        Nd_rec, Nd_base
    campaign:
        Nd_nn, Nd_base, p_merra, optional sigma_nn
    global:
        Nd_nn, Nd_base, optional sigmaNd_nn

    ``Nd_base`` is always used only as a target/reference and is never changed.
    """
    if not isinstance(ds, xr.Dataset):
        return ds

    mode, p_start, p_top, save_raw = _settings()
    if mode == "none":
        return ds

    # Avoid applying the taper twice if the same Dataset is written twice.
    if bool(ds.attrs.get("upper_taper_applied", False)):
        return ds

    applied = False

    # --------------------------------------------------
    # SITE diagnostic: Nd_rec is the refined NN product.
    # --------------------------------------------------
    if "Nd_rec" in ds and "Nd_base" in ds:
        raw = ds["Nd_rec"].copy(deep=False)
        if save_raw and "Nd_rec_raw" not in ds:
            ds["Nd_rec_raw"] = raw.rename("Nd_rec_raw")

        ds["Nd_rec"] = apply_nd_pressure_taper(
            raw,
            Nd_mean=cfg.Nd_mean,
            mode=mode,
            p_start=p_start,
            p_top=p_top,
            base_nd=ds["Nd_base"] if mode == "base" else None,
            name="Nd_rec",
        )
        applied = True

    # --------------------------------------------------
    # CAMPAIGN / GLOBAL diagnostics: Nd_nn is refined NN.
    # --------------------------------------------------
    elif "Nd_nn" in ds and "Nd_base" in ds:
        raw = ds["Nd_nn"].copy(deep=False)
        if save_raw and "Nd_nn_raw" not in ds:
            ds["Nd_nn_raw"] = raw.rename("Nd_nn_raw")

        pressure = ds["p_merra"] if "p_merra" in ds else None

        ds["Nd_nn"] = apply_nd_pressure_taper(
            raw,
            Nd_mean=cfg.Nd_mean,
            pressure_hpa=pressure,
            mode=mode,
            p_start=p_start,
            p_top=p_top,
            base_nd=ds["Nd_base"] if mode == "base" else None,
            name="Nd_nn",
        )

        # Campaign uncertainty is already interpolated to each segment pressure.
        if "sigma_nn" in ds:
            sig_raw = ds["sigma_nn"].copy(deep=False)
            if save_raw and "sigma_nn_raw" not in ds:
                ds["sigma_nn_raw"] = sig_raw.rename("sigma_nn_raw")
            ds["sigma_nn"] = apply_sigma_pressure_taper(
                sig_raw,
                pressure_hpa=pressure,
                mode=mode,
                p_start=p_start,
                p_top=p_top,
                name="sigma_nn",
            )

        # Global propagated uncertainty retains a lev coordinate.
        if "sigmaNd_nn" in ds:
            sig_raw = ds["sigmaNd_nn"].copy(deep=False)
            if save_raw and "sigmaNd_nn_raw" not in ds:
                ds["sigmaNd_nn_raw"] = sig_raw.rename("sigmaNd_nn_raw")
            ds["sigmaNd_nn"] = apply_sigma_pressure_taper(
                sig_raw,
                mode=mode,
                p_start=p_start,
                p_top=p_top,
                name="sigmaNd_nn",
            )

        applied = True

    if applied:
        ds.attrs.update(
            upper_taper_applied=True,
            upper_taper_mode=mode,
            upper_taper_start_hpa=p_start,
            upper_taper_top_hpa=p_top,
            upper_taper_base_unchanged=True,
        )

    return ds


def install_refinement_taper_hooks():
    """Patch diagnostic writers exactly once.

    NDref_prob_taper.py calls this before importing/running NDref_prob.py, so
    the ``from utils import *`` inside NDref_prob.py receives these wrapped
    writer functions.  The taper therefore changes only final diagnostic
    products, never training or coefficient inference.
    """
    if getattr(utils, "_pressure_taper_installed", False):
        return

    original_safe_to_netcdf = utils.safe_to_netcdf
    original_safe_to_netcdf_chunked = utils.safe_to_netcdf_chunked

    def safe_to_netcdf_tapered(ds, path, *args, **kwargs):
        _apply_final_product_taper(ds)
        return original_safe_to_netcdf(ds, path, *args, **kwargs)

    def safe_to_netcdf_chunked_tapered(ds, path, *args, **kwargs):
        _apply_final_product_taper(ds)
        return original_safe_to_netcdf_chunked(ds, path, *args, **kwargs)

    utils.safe_to_netcdf = safe_to_netcdf_tapered
    utils.safe_to_netcdf_chunked = safe_to_netcdf_chunked_tapered
    utils._pressure_taper_installed = True

    mode, p_start, p_top, _ = _settings()
    print(
        f"[TAPER] installed final-product hooks: mode={mode} "
        f"start={p_start:g} hPa top={p_top:g} hPa | "
        "NN only; base reconstruction unchanged"
    )
