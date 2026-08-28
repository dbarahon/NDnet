"""Evaluation-only site and campaign diagnostics for refinement.

This module is used when ``config.do_train`` is False. It never configures an
optimizer and never calls ``training_val`` or ``configure_training_stage``.
The trained refinement model is loaded from ``./<MODEL_NAME>_weights.pt`` and
used in eval mode for site/campaign diagnostics.
"""

from __future__ import annotations

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

import config as cfg
from config import (
    A_mean,
    CAMPAIGN_TRAIN_DIR,
    EOF,
    Nd_mean,
    SITE_JSON,
    hp,
    modis_source,
    prefix_dense,
    prefix_sites,
    pth_all,
    sitemaskmode,
    vars_in_2d,
    vars_out,
)
from campaign import (
    EOFNdMultiObsDatasetTorch,
    load_campaign_training_files,
    make_multiobs_collate_fn,
    sample_campaign_from_dense_predictions,
    tensorize_campaign_inputs_from_existing,
)
from data_handling import (
    A_sites_from_Ares_base,
    A_sites_from_infer_grid,
    get_data,
    reconstruct_Nd_from_A_sites,
    tensorize_sites_inputs_v2,
)
from NDref_prob import active_model_name, build_model
from plotting import plot_campaign_scatter_panels, plot_vertical_profiles_sites
from train_infer import load_model_weights_for_inference, run_infer_x_mu0
from utils import make_canonical_da, safe_to_netcdf


def _load_eval_data():
    """Load the full site/campaign evaluation window without a training split."""
    Nin_2d, A_sites, Nd_sites, Err_sites, Ares_base, site_meta = get_data(
        vars_out=vars_out,
        pth_all=pth_all,
        vars_in_2d=vars_in_2d,
        modis_source=modis_source,
        dense_prefix=prefix_dense,
        site_prefix=prefix_sites,
        cache_folder="cache",
        minP=250,
        site_json_path=SITE_JSON,
    )

    X2d_t, MU0_t, YS_t, VS_t, Nds, Eds = tensorize_sites_inputs_v2(
        Nin_2d=Nin_2d,
        A_sites=A_sites,
        Ares_base=Ares_base,
        Nd_sites=Nd_sites,
        Err_sites=Err_sites,
        pin_memory=True,
        verbose=True,
    )

    nt = Nin_2d.sizes["time"]
    all_idx = np.arange(nt, dtype=np.int64)

    campaign_ds = None
    camp_bundle = None
    if bool(getattr(cfg, "do_campaigns", False)):
        campaign_train_dir = getattr(cfg, "CAMPAIGN_TRAIN_DIR", CAMPAIGN_TRAIN_DIR)
        campaign_pattern = getattr(cfg, "CAMPAIGN_PATTERN", "*_training.nc")
        if campaign_train_dir:
            campaign_ds = load_campaign_training_files(
                campaign_train_dir=campaign_train_dir,
                template_time=Ares_base["time"].values,
                campaign_pattern=campaign_pattern,
                require_valid_obs=True,
            )
            camp_bundle = tensorize_campaign_inputs_from_existing(
                campaign_ds=campaign_ds,
                template_time=Ares_base["time"].values,
                pin_memory=True,
                verbose=True,
            )

    full_ds = EOFNdMultiObsDatasetTorch(
        X2d_t=X2d_t,
        MU0_t=MU0_t,
        YS_t=YS_t,
        VS_t=VS_t,
        Nds_t=Nds,
        Eds_t=Eds,
        campaign_bundle=camp_bundle,
        time_idx=all_idx,
    )

    # Match the existing post-training inference path exactly: no campaign
    # augmentation and the same ragged-segment collator.
    full_loader = DataLoader(
        full_ds,
        batch_size=hp["batch_sz"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=make_multiobs_collate_fn(
            use_campaign_aug=False,
            aug_mode="none",
            seed=1234,
        ),
    )

    return {
        "Nin_2d": Nin_2d,
        "A_sites": A_sites,
        "Nd_sites": Nd_sites,
        "Ares_base": Ares_base,
        "site_meta": site_meta,
        "camp_bundle": camp_bundle,
        "campaign_ds": campaign_ds,
        "full_loader": full_loader,
    }


def _load_frozen_model(Nin_2d, device, numgpu):
    """Build the architecture and load the already-trained local checkpoint."""
    in_channels = len(list(Nin_2d.data_vars))
    model = build_model(in_channels, numgpu, device, hp)
    run_model_name = active_model_name(hp)
    weights_path = "./" + run_model_name + "_weights.pt"

    print("=" * 70)
    print("[EVAL-ONLY] do_train=False")
    print("[EVAL-ONLY] training and sigma refinement are SKIPPED")
    print(f"[EVAL-ONLY] loading frozen model: {weights_path}")
    print("=" * 70)

    load_model_weights_for_inference(
        model,
        weights_path,
        device=device,
        strict=True,
        eval_mode=True,
    )
    return model


def _run_dense_inference(state, model, device):
    """Run one frozen dense pass and reuse it for sites and campaigns."""
    if bool(getattr(cfg, "bypass_infer", False)):
        print("[EVAL-ONLY] bypass_infer=True; using base residual coefficients")
        coeffs_resid_nn = state["Ares_base"].values.astype(np.float32)
        sig_grid = None
    else:
        with torch.no_grad():
            coeffs_resid_nn, sig_grid = run_infer_x_mu0(
                device,
                state["full_loader"],
                model,
            )
    return coeffs_resid_nn, sig_grid


def _run_site_diagnostics(state, coeffs_resid_nn, sig_grid):
    if not bool(getattr(cfg, "do_sites", False)):
        return

    print("=========== site evaluation (no training) ===========")

    Nd_sites = state["Nd_sites"]
    A_sites = state["A_sites"]
    Ares_base = state["Ares_base"]
    site_meta = state["site_meta"]

    Nd_true = Nd_sites.rename("Nd_true")
    site_mask = np.isfinite(Nd_true).any("lev").astype("uint8").rename("site_mask")

    for c in ["site", "ilat", "ilon", "site_lat", "site_lon"]:
        if c in site_meta and c not in Nd_true.coords:
            Nd_true = Nd_true.assign_coords({c: ("site", site_meta[c].values)})

    if bool(getattr(cfg, "bypass_infer", False)):
        assert np.all(A_sites["site"].values == site_meta["site"].values)
        A_sites_resid = A_sites.rename("A_sites_resid")
        sig_sites = None
    else:
        A_sites_resid, sig_sites = A_sites_from_infer_grid(
            coeffs_resid_nn=coeffs_resid_nn,
            sig_grid=sig_grid,
            site_meta=site_meta,
            time_coord=Nd_true["time"],
            mode_coord=Ares_base["mode"].values,
        )

    Nd_rec_lts, A_abs_sites, A_mean_sites = reconstruct_Nd_from_A_sites(
        A_sites_resid=A_sites_resid,
        site_meta=site_meta,
        EOF=EOF,
        Nd_mean=Nd_mean,
        A_mean=A_mean,
        valid_ref=Nd_sites,
        mask_mode=sitemaskmode,
    )
    Nd_rec_tls = Nd_rec_lts.rename("Nd_rec").transpose("time", "lev", "site")

    # Base/reanalysis emulation remains untouched.
    A_base_sites_resid = A_sites_from_Ares_base(
        Ares_base=Ares_base,
        site_meta=site_meta,
    )
    Nd_base_lts, A_base_abs_sites, A_mean_sites = reconstruct_Nd_from_A_sites(
        A_sites_resid=A_base_sites_resid,
        site_meta=site_meta,
        EOF=EOF,
        Nd_mean=Nd_mean,
        A_mean=A_mean,
        valid_ref=Nd_sites,
        mask_mode=sitemaskmode,
    )
    Nd_base_tls = Nd_base_lts.transpose("time", "lev", "site").rename("Nd_base")
    Nd_base_tls = Nd_base_tls.reset_coords(
        names=[c for c in ["lat", "lon"] if c in Nd_base_tls.coords],
        drop=True,
    )

    Nd_true = Nd_true.reset_coords(drop=True)
    site_mask = site_mask.reset_coords(drop=True)

    ref_coords = {
        "time": Nd_true["time"],
        "lev": Nd_true["lev"],
        "site": Nd_true["site"],
        "mode": A_sites_resid["mode"],
    }

    dsS = xr.Dataset(
        {
            "Nd_true": make_canonical_da(
                Nd_true,
                name="Nd_true",
                dims=("time", "lev", "site"),
                ref_coords=ref_coords,
                dtype=np.float32,
            ),
            "Nd_base": make_canonical_da(
                Nd_base_tls,
                name="Nd_base",
                dims=("time", "lev", "site"),
                ref_coords=ref_coords,
                dtype=np.float32,
            ),
            "Nd_rec": make_canonical_da(
                Nd_rec_tls,
                name="Nd_rec",
                dims=("time", "lev", "site"),
                ref_coords=ref_coords,
                dtype=np.float32,
            ),
            "site_mask": make_canonical_da(
                site_mask,
                name="site_mask",
                dims=("time", "site"),
                ref_coords=ref_coords,
                dtype="uint8",
            ),
            "A_sites_resid": make_canonical_da(
                A_sites_resid,
                name="A_sites_resid",
                dims=("time", "mode", "site"),
                ref_coords=ref_coords,
                dtype=np.float32,
            ),
            "A_base_sites_resid": make_canonical_da(
                A_base_sites_resid,
                name="A_base_sites_resid",
                dims=("time", "mode", "site"),
                ref_coords=ref_coords,
                dtype=np.float32,
            ),
            "A_sites_abs": make_canonical_da(
                A_abs_sites,
                name="A_sites_abs",
                dims=("time", "mode", "site"),
                ref_coords=ref_coords,
                dtype=np.float32,
            ),
            "A_base_sites_abs": make_canonical_da(
                A_base_abs_sites,
                name="A_base_sites_abs",
                dims=("time", "mode", "site"),
                ref_coords=ref_coords,
                dtype=np.float32,
            ),
            "A_mean_sites": make_canonical_da(
                A_mean_sites,
                name="A_mean_sites",
                dims=("mode", "site"),
                ref_coords=ref_coords,
                dtype=np.float32,
            ),
        }
    )

    if sig_sites is not None:
        dsS["sigma_sites"] = make_canonical_da(
            sig_sites,
            name="sigma_sites",
            dims=("time", "mode", "site"),
            ref_coords=ref_coords,
            dtype=np.float32,
        )

    for c in [
        "ilat",
        "ilon",
        "site_lat",
        "site_lon",
        "obs_lat",
        "obs_lon",
        "file_lat",
        "file_lon",
    ]:
        if c in site_meta:
            dsS = dsS.assign_coords({c: ("site", site_meta[c].values)})

    # When launched through NDref_prob_taper.py, safe_to_netcdf is already
    # patched so Nd_rec becomes the selected final tapered product here.
    safe_to_netcdf(dsS, "NDnet_site_test.nc")
    print("[WRITE] NDnet_site_test.nc")

    with xr.open_dataset("NDnet_site_test.nc") as dsS_plot:
        plot_vertical_profiles_sites(
            Nd_true_s=dsS_plot["Nd_true"],
            Nd_base_s=dsS_plot["Nd_base"],
            Nd_nn_s=dsS_plot["Nd_rec"].transpose("lev", "time", "site"),
            out_png="NDnet_site_profiles.png",
            min_samples=5,
        )
    print("[PLOT] NDnet_site_profiles.png")


def _run_campaign_diagnostics(state, coeffs_resid_nn, sig_grid):
    if not bool(getattr(cfg, "do_campaigns", False)):
        return

    camp_bundle = state["camp_bundle"]
    campaign_ds = state["campaign_ds"]
    if camp_bundle is None:
        print("[INFO] do_campaigns=True but no campaign bundle was loaded; skipping")
        return

    print("=========== campaign evaluation (no training) ===========")

    dsC_nn = sample_campaign_from_dense_predictions(
        coeffs_resid_nn=coeffs_resid_nn,
        sigma_grid=sig_grid,
        bundle=camp_bundle,
        EOF=EOF,
        Nd_mean=Nd_mean,
        A_mean=A_mean,
    )

    dsC_base = sample_campaign_from_dense_predictions(
        coeffs_resid_nn=state["Ares_base"].values.astype(np.float32),
        sigma_grid=None,
        bundle=camp_bundle,
        EOF=EOF,
        Nd_mean=Nd_mean,
        A_mean=A_mean,
    )

    dsC = xr.Dataset(
        {
            "Nd_obs": dsC_nn["Nd_obs_mean"].astype(np.float32),
            "Nd_nn": dsC_nn["Nd_model"].astype(np.float32),
            "Nd_base": dsC_base["Nd_model"].astype(np.float32),
            "p_merra": dsC_nn["p_merra"].astype(np.float32),
            "time_idx": dsC_nn["time_idx"].astype(np.int64),
            "flight_id": dsC_nn["flight_id"].astype(np.int64),
            "segment_gid": dsC_nn["segment_gid"].astype(np.int64),
            "A_nn_resid": dsC_nn["A_seg_resid"].astype(np.float32),
            "A_nn_abs": dsC_nn["A_seg_abs"].astype(np.float32),
            "A_base_resid": dsC_base["A_seg_resid"].astype(np.float32),
            "A_base_abs": dsC_base["A_seg_abs"].astype(np.float32),
        }
    )

    if "sigma_model" in dsC_nn:
        dsC["sigma_nn"] = dsC_nn["sigma_model"].astype(np.float32)
    if "day" in dsC_nn:
        dsC["time"] = dsC_nn["day"]
    if "campaign" in dsC_nn.coords:
        dsC = dsC.assign_coords(
            {"campaign": ("segment", dsC_nn["campaign"].values)}
        )
    if "campaign_id" in dsC_nn:
        dsC["campaign_id"] = dsC_nn["campaign_id"].astype(np.int32)
    if campaign_ds is not None and "modis_ok" in campaign_ds:
        dsC["modis_ok"] = xr.DataArray(
            campaign_ds["modis_ok"].values.astype(np.int8),
            dims=("segment",),
        )

    # Taper hook, if active, transforms only Nd_nn/sigma_nn and leaves Nd_base
    # untouched immediately before this write.
    safe_to_netcdf(dsC, "NDnet_campaign_test.nc")
    print("[WRITE] NDnet_campaign_test.nc")

    with xr.open_dataset("NDnet_campaign_test.nc") as dsC_plot:
        plot_campaign_scatter_panels(
            dsC=dsC_plot,
            out_png="NDnet_campaign_scatter.png",
            obs_col="Nd_obs",
            nn_col="Nd_nn",
            base_col="Nd_base",
            log_axes=bool(hp.get("campaign_log_scatter", True)),
            min_nd_compare=hp.get("campaign_min_nd_compare", 1.0),
            use_modis_mask=bool(hp.get("campaign_use_modis_mask", False)),
            xlim=None,
            ylim=None,
        )
    print("[PLOT] NDnet_campaign_scatter.png")


def run_eval_sites_campaigns():
    """Run requested site/campaign diagnostics with a frozen existing model."""
    if bool(getattr(cfg, "do_train", False)):
        raise RuntimeError(
            "run_eval_sites_campaigns() is for do_train=False only; refusing "
            "to mix evaluation-only and training paths."
        )

    if not (
        bool(getattr(cfg, "do_sites", False))
        or bool(getattr(cfg, "do_campaigns", False))
    ):
        print("[EVAL-ONLY] neither do_sites nor do_campaigns is enabled")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    numgpu = torch.cuda.device_count()

    state = _load_eval_data()
    model = _load_frozen_model(state["Nin_2d"], device, numgpu)
    coeffs_resid_nn, sig_grid = _run_dense_inference(state, model, device)

    _run_site_diagnostics(state, coeffs_resid_nn, sig_grid)
    _run_campaign_diagnostics(state, coeffs_resid_nn, sig_grid)


if __name__ == "__main__":
    run_eval_sites_campaigns()
