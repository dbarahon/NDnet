"""Install post-inference pressure taper hooks for refinement diagnostics.

This keeps the trained model and all training losses unchanged.  The hooks only
replace the reconstruction/sampling functions used after coefficient inference,
so site, campaign, and global diagnostics all evaluate the same final product.
"""

from __future__ import annotations

import config as cfg
import data_handling
import campaign

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
    )


def install_refinement_taper_hooks():
    """Patch post-training reconstruction paths exactly once."""
    if getattr(data_handling, "_pressure_taper_installed", False):
        return

    original_grid = data_handling.reconstruct_Nd_from_coeffs_chunked
    original_sites = data_handling.reconstruct_Nd_from_A_sites
    original_campaign = campaign.sample_campaign_from_dense_predictions

    def reconstruct_Nd_from_coeffs_chunked_tapered(*args, **kwargs):
        ds = original_grid(*args, **kwargs)
        mode, p_start, p_top = _settings()
        if mode == "none":
            return ds

        ds = ds.copy()
        ds["Nd_rec"] = apply_nd_pressure_taper(
            ds["Nd_rec"],
            Nd_mean=cfg.Nd_mean,
            mode=mode,
            p_start=p_start,
            p_top=p_top,
            name="Nd_rec",
        )
        if "sigmaNd_rec" in ds:
            ds["sigmaNd_rec"] = apply_sigma_pressure_taper(
                ds["sigmaNd_rec"],
                mode=mode,
                p_start=p_start,
                p_top=p_top,
                name="sigmaNd_rec",
            )
        ds.attrs.update(
            upper_taper_mode=mode,
            upper_taper_start_hpa=p_start,
            upper_taper_top_hpa=p_top,
        )
        return ds

    def reconstruct_Nd_from_A_sites_tapered(*args, **kwargs):
        Nd_rec, A_abs_sites, A_mean_sites = original_sites(*args, **kwargs)
        mode, p_start, p_top = _settings()
        if mode != "none":
            Nd_rec = apply_nd_pressure_taper(
                Nd_rec,
                Nd_mean=cfg.Nd_mean,
                mode=mode,
                p_start=p_start,
                p_top=p_top,
                name="Nd_rec",
            )
        return Nd_rec, A_abs_sites, A_mean_sites

    def sample_campaign_from_dense_predictions_tapered(*args, **kwargs):
        ds = original_campaign(*args, **kwargs)
        mode, p_start, p_top = _settings()
        if mode == "none":
            return ds

        ds = ds.copy()
        ds["Nd_model"] = apply_nd_pressure_taper(
            ds["Nd_model"],
            Nd_mean=cfg.Nd_mean,
            pressure_hpa=ds["p_merra"],
            mode=mode,
            p_start=p_start,
            p_top=p_top,
            name="Nd_model",
        )
        if "sigma_model" in ds:
            ds["sigma_model"] = apply_sigma_pressure_taper(
                ds["sigma_model"],
                pressure_hpa=ds["p_merra"],
                mode=mode,
                p_start=p_start,
                p_top=p_top,
                name="sigma_model",
            )
        ds.attrs.update(
            upper_taper_mode=mode,
            upper_taper_start_hpa=p_start,
            upper_taper_top_hpa=p_top,
        )
        return ds

    data_handling.reconstruct_Nd_from_coeffs_chunked = reconstruct_Nd_from_coeffs_chunked_tapered
    data_handling.reconstruct_Nd_from_A_sites = reconstruct_Nd_from_A_sites_tapered
    campaign.sample_campaign_from_dense_predictions = sample_campaign_from_dense_predictions_tapered
    data_handling._pressure_taper_installed = True

    mode, p_start, p_top = _settings()
    print(
        f"[TAPER] installed refinement hooks: mode={mode} "
        f"start={p_start:g} hPa top={p_top:g} hPa"
    )
