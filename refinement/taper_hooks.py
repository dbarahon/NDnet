"""Install post-inference pressure taper hooks for refinement diagnostics.

The trained model, coefficient inference, and reanalysis-emulation base product
are left unchanged. Only the refined NN reconstruction (and its propagated
uncertainty) receives the optional upper-level taper.
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


def _site_input_name(args, kwargs):
    """Return the name of the site residual-coefficient input when available."""
    if "A_sites_resid" in kwargs:
        arr = kwargs["A_sites_resid"]
    elif args:
        arr = args[0]
    else:
        arr = None
    return str(getattr(arr, "name", "") or "")


def install_refinement_taper_hooks():
    """Patch post-training NN reconstruction paths exactly once.

    Important scientific distinction:
      - NN reconstruction: tapered according to config.
      - base/reanalysis-emulation reconstruction: always left untouched.
    """
    if getattr(data_handling, "_pressure_taper_installed", False):
        return

    original_grid = data_handling.reconstruct_Nd_from_coeffs_chunked
    original_sites = data_handling.reconstruct_Nd_from_A_sites
    original_campaign = campaign.sample_campaign_from_dense_predictions

    def reconstruct_Nd_from_coeffs_chunked_tapered(*args, **kwargs):
        ds = original_grid(*args, **kwargs)
        mode, p_start, p_top = _settings()

        # In NDref_prob.py the refined NN call provides sigma_coeffs_abs,
        # whereas the base/reanalysis-emulation call does not. This keeps
        # Nd_base exactly as reconstructed by the base EOF emulation.
        sigma_coeffs = kwargs.get("sigma_coeffs_abs", None)
        is_refined_nn = sigma_coeffs is not None

        if mode == "none" or not is_refined_nn:
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

        # A_sites_from_Ares_base() explicitly names the base field
        # "A_base_sites_resid". Do not alter that physically based comparator.
        input_name = _site_input_name(args, kwargs)
        is_base = input_name == "A_base_sites_resid"

        if mode != "none" and not is_base:
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

        # The NN diagnostic supplies sigma_grid; the base comparator is called
        # with sigma_grid=None. Taper only the final NN product.
        if "sigma_grid" in kwargs:
            sigma_grid = kwargs["sigma_grid"]
        elif len(args) >= 2:
            sigma_grid = args[1]
        else:
            sigma_grid = None
        is_refined_nn = sigma_grid is not None

        if mode == "none" or not is_refined_nn:
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
        f"start={p_start:g} hPa top={p_top:g} hPa | "
        "NN only; base reconstruction unchanged"
    )
