from config import *
import config as cfg
from data_handling import *
from utils import *
from NN_construct import *
from train_infer import *
from plotting import *
from campaign import *
from profile_taper import (
    apply_nd_pressure_taper,
    apply_sigma_pressure_taper,
    normalize_taper_mode,
)
from holdout import site_holdout_mask, campaign_holdout_mask, save_manifest
from diagnostic_cache import (
    diagnostic_cache_valid,
    diagnostic_time_indices,
    load_diagnostic_cache,
    remap_campaign_bundle_for_sparse,
    sample_sites_from_sparse_grid,
    write_diagnostic_cache,
)

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import dask.array as da
import matplotlib.pyplot as plt


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def build_model(in_channels, numgpu, device, hp):
    """Build the refinement model."""
    model = ProbRefineFromBaseAndX(
        in_channels=in_channels,
        num_modes=num_modes,
        feat_ch=32,
        head_ch=64,
        sigma_min=float(hp.get("model_sigma_min", 1e-3)),
        detach_sigma_features=bool(hp.get("detach_sigma_features", True)),
    ).to(device)

    if numgpu > 1:
        model = nn.DataParallel(model)
    return model


def active_model_name(hp):
    """Model name produced by the selected training stage."""
    if bool(hp.get("fine_tune_sigma", False)):
        return hp.get(
            "sigma_finetune_model_name",
            hp["model_name"] + "_sigmaFT",
        )
    return hp["model_name"]


def configure_training_stage(model, criterion, hp, device):
    """Configure normal training or sigma-only fine tuning.

    This function is called only when ``do_train=True``. Evaluation-only
    runs never construct an optimizer and never enter sigma fine tuning.
    """
    fine_tune_sigma = bool(hp.get("fine_tune_sigma", False))
    base_model_name = hp["model_name"]

    hp_run = hp.copy()
    hp_run["model_name"] = active_model_name(hp)

    if fine_tune_sigma:
        print("=" * 70)
        print("[SIGMA-FT] SIGMA-ONLY FINE TUNING")
        print("=" * 70)

        source_weights = hp.get(
            "sigma_finetune_source_weights",
            base_model_name + "_weights.pt",
        )

        load_model_weights_for_inference(
            model,
            source_weights,
            device=device,
            strict=True,
            eval_mode=False,
        )
        model.train()

        core_model = model.module if isinstance(model, nn.DataParallel) else model
        for p in core_model.parameters():
            p.requires_grad = False
        for p in core_model.sigma_head.parameters():
            p.requires_grad = True

        criterion.lambda_anchor = 0.0
        criterion.lambda_grad = 0.0
        criterion.lambda_nd = 0.0
        criterion.lambda_campaign = 0.0

        criterion.lambda_sites = float(
            hp.get("sigma_finetune_lambda_sites", criterion.lambda_sites)
        )
        criterion.lambda_Ndgauss = float(
            hp.get("sigma_finetune_lambda_Ndgauss", criterion.lambda_Ndgauss)
        )
        criterion.lambda_campaign_gauss = float(
            hp.get(
                "sigma_finetune_lambda_campaign_gauss",
                criterion.lambda_campaign_gauss,
            )
        )
        criterion.lambda_sigma = float(
            hp.get("sigma_finetune_lambda_sigma", 5.0)
        )
        criterion.lambda_campaign_sigma = float(
            hp.get("sigma_finetune_lambda_campaign_sigma", 2.0)
        )

        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("[SIGMA-FT] No trainable sigma_head parameters found.")

        optimizer = optim.Adam(
            params,
            lr=float(hp.get("sigma_finetune_lr", 1e-5)),
            weight_decay=float(hp.get("sigma_finetune_weight_decay", 1e-5)),
        )

        start_epoch = 0
        hp_run["nepochs"] = int(hp.get("sigma_finetune_epochs", 30))
        hp_run["patience"] = int(
            hp.get("sigma_finetune_patience", hp.get("patience", 20))
        )
        hp_run["use_campaign_monitor"] = bool(
            hp.get("sigma_finetune_use_campaign_monitor", False)
        )
        hp_run["campaign_monitor_beta"] = float(
            hp.get("sigma_finetune_campaign_monitor_beta", 0.0)
        )

        print(f"[SIGMA-FT] source weights          = {source_weights}")
        print(f"[SIGMA-FT] output model            = {hp_run['model_name']}")
        print(f"[SIGMA-FT] epochs                  = {hp_run['nepochs']}")
        print(f"[SIGMA-FT] lr                      = {optimizer.param_groups[0]['lr']:.3e}")
        print(f"[SIGMA-FT] lambda_sites            = {criterion.lambda_sites:g}")
        print(f"[SIGMA-FT] lambda_Ndgauss          = {criterion.lambda_Ndgauss:g}")
        print(f"[SIGMA-FT] lambda_campaign_gauss   = {criterion.lambda_campaign_gauss:g}")
        print(f"[SIGMA-FT] lambda_sigma            = {criterion.lambda_sigma:g}")
        print(f"[SIGMA-FT] lambda_campaign_sigma   = {criterion.lambda_campaign_sigma:g}")
        print(f"[SIGMA-FT] campaign monitor        = {hp_run['use_campaign_monitor']}")

    else:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(
            params,
            lr=hp["l_r"],
            weight_decay=1e-5,
        )
        start_epoch = load_training_checkpoint(
            model,
            optimizer,
            base_model_name + ".ckpt",
            device,
        )

    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    params = [p for p in model.parameters() if p.requires_grad]
    print("[DBG] #trainable tensors:", len(params))
    print("[DBG] #trainable params:", sum(p.numel() for p in params))
    for name in trainable_names:
        print("      ", name)

    if fine_tune_sigma:
        bad = [n for n in trainable_names if "sigma_head" not in n]
        if bad:
            raise RuntimeError(
                "[SIGMA-FT] Non-sigma parameters are trainable: " + ", ".join(bad)
            )

    return optimizer, start_epoch, hp_run


def _taper_settings():
    return (
        normalize_taper_mode(getattr(cfg, "UPPER_TAPER_MODE", "none")),
        float(getattr(cfg, "UPPER_TAPER_START_HPA", 500.0)),
        float(getattr(cfg, "UPPER_TAPER_TOP_HPA", 350.0)),
        bool(getattr(cfg, "SAVE_RAW_TAPER_FIELDS", True)),
    )


def finalize_nd_product(
    ds,
    *,
    nd_name,
    base_name,
    sigma_name=None,
    pressure_name=None,
    raw_nd_name=None,
    raw_sigma_name=None,
):
    """Apply the configured pressure taper to the final NN product.

    The base/reanalysis-emulation field is a reference only and is never
    modified. This function is used by site, campaign, and global diagnostics.
    """
    mode, p_start, p_top, save_raw = _taper_settings()

    nd_raw = ds[nd_name].copy(deep=False)
    pressure = ds[pressure_name] if pressure_name is not None else None
    base_nd = ds[base_name] if mode == "base" else None

    if save_raw and mode != "none" and raw_nd_name:
        ds[raw_nd_name] = nd_raw.rename(raw_nd_name)

    ds[nd_name] = apply_nd_pressure_taper(
        nd_raw,
        Nd_mean=Nd_mean,
        pressure_hpa=pressure,
        mode=mode,
        p_start=p_start,
        p_top=p_top,
        base_nd=base_nd,
        name=nd_name,
    )

    if sigma_name is not None and sigma_name in ds:
        sigma_raw = ds[sigma_name].copy(deep=False)
        if save_raw and mode != "none" and raw_sigma_name:
            ds[raw_sigma_name] = sigma_raw.rename(raw_sigma_name)
        ds[sigma_name] = apply_sigma_pressure_taper(
            sigma_raw,
            pressure_hpa=pressure,
            mode=mode,
            p_start=p_start,
            p_top=p_top,
            name=sigma_name,
        )

    ds.attrs.update(
        upper_taper_mode=mode,
        upper_taper_start_hpa=p_start,
        upper_taper_top_hpa=p_top,
        upper_taper_base_unchanged=True,
    )
    return ds


def load_site_campaign_state(
    need_campaign_bundle=False,
    *,
    evaluation_only=False,
    need_sites=False,
    need_campaigns=False,
):
    """Load site/campaign data, subsetting dense tensors before materialization.

    Training keeps the historical full-period tensorization.  Evaluation-only
    sparse diagnostics first identify the union of site/campaign observation
    days, then materialize MODIS/base tensors only for those days.
    """
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

    nt = Nin_2d.sizes["time"]
    all_idx = np.arange(nt, dtype=np.int64)

    campaign_ds = None
    camp_bundle = None
    campaign_train_dir = getattr(cfg, "CAMPAIGN_TRAIN_DIR", None)
    campaign_pattern = getattr(cfg, "CAMPAIGN_PATTERN", "*_training.nc")

    # Load campaign metadata before choosing diagnostic days so campaign-only
    # days can be included in the sparse inference union.
    if need_campaign_bundle and campaign_train_dir:
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

    site_test_mask = site_holdout_mask(Nd_sites, cfg.HOLDOUT_FRACTION, cfg.HOLDOUT_SEED)
    campaign_test_mask = (
        campaign_holdout_mask(campaign_ds, cfg.HOLDOUT_FRACTION, cfg.HOLDOUT_SEED)
        if campaign_ds is not None else None
    )
    if not evaluation_only:
        save_manifest(Nd_sites, site_test_mask, campaign_ds, campaign_test_mask,
                      cfg.HOLDOUT_MANIFEST, cfg.HOLDOUT_SEED, cfg.HOLDOUT_FRACTION)
    print(f"[HOLDOUT] site profiles: {site_test_mask.sum()}")
    if campaign_test_mask is not None:
        print(f"[HOLDOUT] campaign segments: {campaign_test_mask.sum()}")
    tensor_time_idx = all_idx
    tensors_sparse = False

    if evaluation_only and bool(getattr(cfg, "SPARSE_DIAGNOSTIC_INFERENCE", True)):
        # A_sites is already a small (time, mode, site) product at this point.
        # Derive the site-valid mask here instead of first materializing the
        # enormous global MODIS/base tensors via tensorize_sites_inputs_v2().
        if need_sites:
            site_valid = np.isfinite(np.asarray(A_sites.values)).any(axis=1)
        else:
            site_valid = np.zeros((nt, 1), dtype=bool)

        tensor_time_idx = diagnostic_time_indices(
            all_idx=all_idx,
            VS_t=site_valid,
            campaign_bundle=camp_bundle,
            need_sites=need_sites,
            need_campaigns=need_campaigns,
        )
        if tensor_time_idx.size == 0:
            raise RuntimeError("No site/campaign days selected for diagnostic tensorization")

        tensors_sparse = tensor_time_idx.size < all_idx.size
        print(
            f"[DIAG-LOAD] tensorizing {tensor_time_idx.size}/{all_idx.size} days "
            "before NumPy/PyTorch materialization"
        )

    if tensors_sparse:
        Nin_tensor = Nin_2d.isel(time=tensor_time_idx)
        A_sites_tensor = A_sites.isel(time=tensor_time_idx)
        Ares_tensor = Ares_base.isel(time=tensor_time_idx)
        Nd_sites_tensor = Nd_sites.isel(time=tensor_time_idx)
        Err_sites_tensor = Err_sites.isel(time=tensor_time_idx)
    else:
        Nin_tensor = Nin_2d
        A_sites_tensor = A_sites
        Ares_tensor = Ares_base
        Nd_sites_tensor = Nd_sites
        Err_sites_tensor = Err_sites

    X2d_t, MU0_t, YS_t, VS_t, Nds, Eds = tensorize_sites_inputs_v2(
        Nin_2d=Nin_tensor,
        A_sites=A_sites_tensor,
        Ares_base=Ares_tensor,
        Nd_sites=Nd_sites_tensor,
        Err_sites=Err_sites_tensor,
        pin_memory=True,
        verbose=True,
    )

    return {
        "Nin_2d": Nin_2d,
        "A_sites": A_sites,
        "Nd_sites": Nd_sites,
        "Err_sites": Err_sites,
        "Ares_base": Ares_base,
        "site_meta": site_meta,
        "X2d_t": X2d_t,
        "MU0_t": MU0_t,
        "YS_t": YS_t,
        "VS_t": VS_t,
        "Nds": Nds,
        "Eds": Eds,
        "all_idx": all_idx,
        "tensor_time_idx": np.asarray(tensor_time_idx, dtype=np.int64),
        "tensors_sparse": bool(tensors_sparse),
        "campaign_ds": campaign_ds,
        "camp_bundle": camp_bundle,
        "site_test_mask": site_test_mask,
        "campaign_test_mask": campaign_test_mask,
    }


def make_full_loader(state, time_idx=None):
    if bool(state.get("tensors_sparse", False)):
        # Tensors have already been compacted to the selected diagnostic days.
        # Use local [0,K) indices and omit campaign metadata: inference only
        # consumes x2d and mu0, while campaign sampling happens afterward with
        # the original full-period bundle and explicit time-index remapping.
        local_idx = np.arange(state["X2d_t"].shape[0], dtype=np.int64)
        full_ds = EOFNdMultiObsDatasetTorch(
            X2d_t=state["X2d_t"],
            MU0_t=state["MU0_t"],
            YS_t=state["YS_t"],
            VS_t=state["VS_t"],
            Nds_t=state["Nds"],
            Eds_t=state["Eds"],
            campaign_bundle=None,
            time_idx=local_idx,
        )
    else:
        if time_idx is None:
            time_idx = state["all_idx"]
        time_idx = np.asarray(time_idx, dtype=np.int64)
        full_ds = EOFNdMultiObsDatasetTorch(
            X2d_t=state["X2d_t"],
            MU0_t=state["MU0_t"],
            YS_t=state["YS_t"],
            VS_t=state["VS_t"],
            Nds_t=state["Nds"],
            Eds_t=state["Eds"],
            campaign_bundle=state["camp_bundle"],
            time_idx=time_idx,
        )

    return DataLoader(
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


def make_training_loaders(state, numwork):
    nt = state["Nin_2d"].sizes["time"]
    all_idx = state["all_idx"]
    split = int(0.9 * nt)
    train_idx = all_idx[:split]
    val_idx = all_idx[split:]

    hold = state["site_test_mask"]
    nd_train = state["Nds"].clone()
    ed_train = state["Eds"].clone()
    ys_train = state["YS_t"].clone()
    vs_train = state["VS_t"].clone()
    hold_t = torch.as_tensor(hold, dtype=torch.bool, device=vs_train.device)
    nd_train = nd_train.masked_fill(hold_t[:, None, :], float("nan"))
    ed_train = ed_train.masked_fill(hold_t[:, None, :], float("nan"))
    ys_train = ys_train.masked_fill(hold_t[:, None, :], float("nan"))
    vs_train = vs_train.masked_fill(hold_t, False)
    camp_bundle = state["camp_bundle"]
    if camp_bundle is not None and state["campaign_test_mask"] is not None:
        from dataclasses import replace
        keep = torch.as_tensor(~state["campaign_test_mask"], dtype=torch.bool,
                               device=camp_bundle.valid_obs.device)
        camp_bundle = replace(
            camp_bundle,
            valid_obs=camp_bundle.valid_obs & keep,
            aug_valid=camp_bundle.aug_valid & keep.unsqueeze(0),
            y_mean=camp_bundle.y_mean.masked_fill(~keep, float("nan")),
            y_aug=camp_bundle.y_aug.masked_fill(~keep.unsqueeze(0), float("nan")),
        )
    use_campaigns = camp_bundle is not None

    if use_campaigns:
        train_idx_c = filter_times_with_campaign_segments(
            camp_bundle,
            train_idx,
            min_segments=hp.get("campaign_min_segments_per_day", 1),
        )
        val_idx_c = filter_times_with_campaign_segments(
            camp_bundle,
            val_idx,
            min_segments=1,
        )
        train_idx_use = np.union1d(train_idx, train_idx_c)
        val_idx_use = np.union1d(val_idx, val_idx_c)
        print(f"[INFO] campaign training enabled: {CAMPAIGN_TRAIN_DIR}")
        print(f"[INFO] train days with campaign segments: {train_idx_c.size}")
        print(f"[INFO] val days with campaign segments:   {val_idx_c.size}")
    else:
        train_idx_use = train_idx
        val_idx_use = val_idx
        print("[INFO] campaign training disabled")

    use_sampler = bool(hp.get("use_site_sampler", True)) and not use_campaigns
    if use_sampler:
        train_idx_final, train_sampler = make_equal_site_time_sampler(
            VS_t=state["VS_t"],
            time_idx=train_idx_use,
            min_sites=hp["min_sites_for_sampling"],
            beta=hp["site_sampling_beta"],
            eps=1.0,
        )
    else:
        train_sampler = None
        train_idx_final = train_idx_use
        if use_campaigns and bool(hp.get("use_site_sampler", True)):
            print("[INFO] disabling site-only sampler because it would miss campaign-only days")

    ds_train = EOFNdMultiObsDatasetTorch(
        X2d_t=state["X2d_t"],
        MU0_t=state["MU0_t"],
        YS_t=ys_train,
        VS_t=vs_train,
        Nds_t=nd_train,
        Eds_t=ed_train,
        campaign_bundle=camp_bundle,
        time_idx=train_idx_final,
    )
    ds_val = EOFNdMultiObsDatasetTorch(
        X2d_t=state["X2d_t"],
        MU0_t=state["MU0_t"],
        YS_t=ys_train,
        VS_t=vs_train,
        Nds_t=nd_train,
        Eds_t=ed_train,
        campaign_bundle=camp_bundle,
        time_idx=val_idx_use,
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=hp["batch_sz"],
        sampler=train_sampler,
        shuffle=False if train_sampler is not None else True,
        num_workers=numwork,
        pin_memory=True,
        drop_last=False,
        collate_fn=make_multiobs_collate_fn(
            use_campaign_aug=bool(hp.get("use_campaign_aug", True)),
            aug_mode=hp.get("campaign_aug_mode", "random_valid"),
            seed=int(hp.get("campaign_aug_seed", 1234)),
        ),
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=hp["batch_sz"],
        shuffle=False,
        num_workers=numwork,
        pin_memory=True,
        drop_last=False,
        collate_fn=make_multiobs_collate_fn(
            use_campaign_aug=False,
            aug_mode="none",
            seed=1234,
        ),
    )
    return train_loader, val_loader


def train_refinement_model(state, device, numgpu, numwork):
    train_loader, val_loader = make_training_loaders(state, numwork)
    in_channels = len(list(state["Nin_2d"].data_vars))
    model = build_model(in_channels, numgpu, device, hp)

    criterion = SparseSiteCampaignNLLRefineLoss(
        lat_grid=state["Nin_2d"].lat.values,
        site_meta=state["site_meta"],
        mode_weights=mode_weights,
        EOF=EOF,
        Nd_mean=Nd_mean,
        A_mean=A_mean,
        lambda_grad=float(hp.get("lambda_grad", 0.005)),
        lambda_anchor=float(hp.get("lambda_anchor", 0.01)),
        lambda_sites=float(hp.get("lambda_sites", 1.0)),
        lambda_nd=float(hp.get("lambda_nd", 1.0)),
        lambda_Ndgauss=float(hp.get("lambda_Ndgauss", 1.0)),
        lambda_campaign=float(hp.get("lambda_campaign", 1.0)),
        lambda_campaign_gauss=float(hp.get("lambda_campaign_gauss", 1.0)),
        min_sigma=hp.get("min_sigma", 1e-3),
        err_mix=hp.get("err_mix", 0.15),
        lambda_sigma=hp["lambda_sigma"],
        lambda_campaign_sigma=hp["lambda_campaign_sigma"],
        sigma_mode_var_floor=hp["sigma_mode_var_floor"],
    ).to(device)

    optimizer, start_epoch, hp_run = configure_training_stage(
        model=model,
        criterion=criterion,
        hp=hp,
        device=device,
    )
    train_losses, val_losses = training_val(
        model,
        criterion,
        train_loader,
        val_loader,
        optimizer,
        hp_run,
        device,
        start_epoch=start_epoch,
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Val")
    ax.legend()
    ax.set_xlim(left=5)
    fig.savefig(hp_run["model_name"] + "_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("========== DONE training")
    return model, hp_run["model_name"]


def load_frozen_model(state, device, numgpu, model_name):
    in_channels = len(list(state["Nin_2d"].data_vars))
    model = build_model(in_channels, numgpu, device, hp)
    weights_path = f"./{model_name}_weights.pt"
    print("=" * 70)
    print("[EVAL-ONLY] do_train=False: no optimizer, no training, no sigma fine tuning")
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


def dense_inference_for_diagnostics(
    state,
    model,
    device,
    *,
    need_sites,
    need_campaigns,
):
    if bool(state.get("tensors_sparse", False)):
        infer_idx = np.asarray(state["tensor_time_idx"], dtype=np.int64)
        print(
            f"[DIAG-INFER] using pre-subset tensors: "
            f"{infer_idx.size}/{state['all_idx'].size} days"
        )
    else:
        infer_idx = diagnostic_time_indices(
            all_idx=state["all_idx"],
            VS_t=state["VS_t"],
            campaign_bundle=state["camp_bundle"],
            need_sites=need_sites,
            need_campaigns=need_campaigns,
        )

    if infer_idx.size == 0:
        raise RuntimeError("No site/campaign days selected for diagnostic inference")

    if bypass_infer:
        print("===== ByPassing Inference: using base residual coefficients =====")
        coeffs = state["Ares_base"].isel(time=infer_idx).values.astype(np.float32)
        return coeffs, None, infer_idx

    model.eval()
    loader = make_full_loader(
        state,
        time_idx=None if bool(state.get("tensors_sparse", False)) else infer_idx,
    )
    with torch.no_grad():
        coeffs, sig = run_infer_x_mu0(device, loader, model)
    return coeffs, sig, infer_idx


def render_site_diagnostics(dsS_raw):
    """Apply current taper to an untapered site dataset, then write and plot."""
    dsS = dsS_raw.copy(deep=False)
    dsS = finalize_nd_product(
        dsS,
        nd_name="Nd_rec",
        base_name="Nd_base",
        raw_nd_name="Nd_rec_raw",
    )
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


def run_site_diagnostics(
    state,
    coeffs_resid_nn,
    sig_grid,
    infer_idx,
    model_name,
):
    print("=========== site evaluation ===========")
    Nd_sites = state["Nd_sites"]
    A_sites = state["A_sites"]
    Ares_base = state["Ares_base"]
    site_meta = state["site_meta"]

    hold_da = xr.DataArray(state["site_test_mask"], dims=("time", "site"),
                           coords={"time": Nd_sites.time, "site": Nd_sites.site})
    Nd_true = Nd_sites.where(hold_da).rename("Nd_true")
    site_mask = np.isfinite(Nd_true).any("lev").astype("uint8").rename("site_mask")

    for c in ["site", "ilat", "ilon", "site_lat", "site_lon"]:
        if c in site_meta and c not in Nd_true.coords:
            Nd_true = Nd_true.assign_coords({c: ("site", site_meta[c].values)})

    if bypass_infer:
        assert np.all(A_sites["site"].values == site_meta["site"].values)
        A_sites_resid = A_sites.rename("A_sites_resid")
        sig_sites = None
    else:
        A_sites_resid, sig_sites = sample_sites_from_sparse_grid(
            coeffs_resid_nn=coeffs_resid_nn,
            sig_grid=sig_grid,
            infer_idx=infer_idx,
            site_meta=site_meta,
            full_time_coord=Nd_true["time"],
            mode_coord=Ares_base["mode"].values,
        )

    Nd_rec_lts, A_abs_sites, A_mean_sites = reconstruct_Nd_from_A_sites(
        A_sites_resid=A_sites_resid,
        site_meta=site_meta,
        EOF=EOF,
        Nd_mean=Nd_mean,
        A_mean=A_mean,
        valid_ref=Nd_true,
        mask_mode=sitemaskmode,
    )
    Nd_rec_tls = Nd_rec_lts.rename("Nd_rec").transpose("time", "lev", "site")

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
        valid_ref=Nd_true,
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

    dsS_raw = xr.Dataset(
        {
            "Nd_true": make_canonical_da(
                Nd_true, name="Nd_true", dims=("time", "lev", "site"),
                ref_coords=ref_coords, dtype=np.float32,
            ),
            "Nd_base": make_canonical_da(
                Nd_base_tls, name="Nd_base", dims=("time", "lev", "site"),
                ref_coords=ref_coords, dtype=np.float32,
            ),
            "Nd_rec": make_canonical_da(
                Nd_rec_tls, name="Nd_rec", dims=("time", "lev", "site"),
                ref_coords=ref_coords, dtype=np.float32,
            ),
            "site_mask": make_canonical_da(
                site_mask, name="site_mask", dims=("time", "site"),
                ref_coords=ref_coords, dtype="uint8",
            ),
            "A_sites_resid": make_canonical_da(
                A_sites_resid, name="A_sites_resid", dims=("time", "mode", "site"),
                ref_coords=ref_coords, dtype=np.float32,
            ),
            "A_base_sites_resid": make_canonical_da(
                A_base_sites_resid, name="A_base_sites_resid", dims=("time", "mode", "site"),
                ref_coords=ref_coords, dtype=np.float32,
            ),
            "A_sites_abs": make_canonical_da(
                A_abs_sites, name="A_sites_abs", dims=("time", "mode", "site"),
                ref_coords=ref_coords, dtype=np.float32,
            ),
            "A_base_sites_abs": make_canonical_da(
                A_base_abs_sites, name="A_base_sites_abs", dims=("time", "mode", "site"),
                ref_coords=ref_coords, dtype=np.float32,
            ),
            "A_mean_sites": make_canonical_da(
                A_mean_sites, name="A_mean_sites", dims=("mode", "site"),
                ref_coords=ref_coords, dtype=np.float32,
            ),
        }
    )

    if sig_sites is not None:
        dsS_raw["sigma_sites"] = make_canonical_da(
            sig_sites, name="sigma_sites", dims=("time", "mode", "site"),
            ref_coords=ref_coords, dtype=np.float32,
        )

    for c in [
        "ilat", "ilon", "site_lat", "site_lon",
        "obs_lat", "obs_lon", "file_lat", "file_lon",
    ]:
        if c in site_meta:
            dsS_raw = dsS_raw.assign_coords({c: ("site", site_meta[c].values)})

    dsS_raw.attrs["observation_partition"] = "independent_holdout20"
    dsS_raw.attrs["diagnostic_inference_days"] = int(len(infer_idx))
    dsS_raw.attrs["diagnostic_total_days"] = int(len(state["all_idx"]))
    write_diagnostic_cache(dsS_raw, "site", model_name)
    render_site_diagnostics(dsS_raw)


def render_campaign_diagnostics(dsC_raw):
    """Apply current taper to an untapered campaign dataset, then write and plot."""
    dsC = dsC_raw.copy(deep=False)
    dsC = finalize_nd_product(
        dsC,
        nd_name="Nd_nn",
        base_name="Nd_base",
        sigma_name="sigma_nn",
        pressure_name="p_merra",
        raw_nd_name="Nd_nn_raw",
        raw_sigma_name="sigma_nn_raw",
    )
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


def run_campaign_diagnostics(
    state,
    coeffs_resid_nn,
    sig_grid,
    infer_idx,
    model_name,
):
    camp_bundle = state["camp_bundle"]
    campaign_ds = state["campaign_ds"]
    if camp_bundle is None:
        print("[INFO] campaign diagnostics requested but no campaign bundle is loaded")
        return

    print("=========== campaign evaluation ===========")

    sparse_bundle = remap_campaign_bundle_for_sparse(camp_bundle, infer_idx)

    dsC_nn = sample_campaign_from_dense_predictions(
        coeffs_resid_nn=coeffs_resid_nn,
        sigma_grid=sig_grid,
        bundle=sparse_bundle,
        EOF=EOF,
        Nd_mean=Nd_mean,
        A_mean=A_mean,
    )

    base_sparse = state["Ares_base"].isel(time=infer_idx).values.astype(np.float32)
    dsC_base = sample_campaign_from_dense_predictions(
        coeffs_resid_nn=base_sparse,
        sigma_grid=None,
        bundle=sparse_bundle,
        EOF=EOF,
        Nd_mean=Nd_mean,
        A_mean=A_mean,
    )

    original_time_idx = camp_bundle.time_idx.detach().cpu().numpy().astype(np.int64)

    dsC_raw = xr.Dataset(
        {
            "Nd_obs": dsC_nn["Nd_obs_mean"].astype(np.float32),
            "Nd_nn": dsC_nn["Nd_model"].astype(np.float32),
            "Nd_base": dsC_base["Nd_model"].astype(np.float32),
            "p_merra": dsC_nn["p_merra"].astype(np.float32),
            "time_idx": xr.DataArray(original_time_idx, dims=("segment",)),
            "flight_id": dsC_nn["flight_id"].astype(np.int64),
            "segment_gid": dsC_nn["segment_gid"].astype(np.int64),
            "A_nn_resid": dsC_nn["A_seg_resid"].astype(np.float32),
            "A_nn_abs": dsC_nn["A_seg_abs"].astype(np.float32),
            "A_base_resid": dsC_base["A_seg_resid"].astype(np.float32),
            "A_base_abs": dsC_base["A_seg_abs"].astype(np.float32),
        }
    )

    if "sigma_model" in dsC_nn:
        dsC_raw["sigma_nn"] = dsC_nn["sigma_model"].astype(np.float32)
    if "day" in dsC_nn:
        dsC_raw["time"] = dsC_nn["day"]
    if "campaign" in dsC_nn.coords:
        dsC_raw = dsC_raw.assign_coords(
            {"campaign": ("segment", dsC_nn["campaign"].values)}
        )
    if "campaign_id" in dsC_nn:
        dsC_raw["campaign_id"] = dsC_nn["campaign_id"].astype(np.int32)
    if campaign_ds is not None and "modis_ok" in campaign_ds:
        dsC_raw["modis_ok"] = xr.DataArray(
            campaign_ds["modis_ok"].values.astype(np.int8),
            dims=("segment",),
        )

    if state["campaign_test_mask"] is not None:
        dsC_raw = dsC_raw.isel(segment=np.flatnonzero(state["campaign_test_mask"]))
    dsC_raw.attrs["observation_partition"] = "independent_holdout20"
    dsC_raw.attrs["diagnostic_inference_days"] = int(len(infer_idx))
    dsC_raw.attrs["diagnostic_total_days"] = int(len(state["all_idx"]))
    write_diagnostic_cache(dsC_raw, "campaign", model_name)
    render_campaign_diagnostics(dsC_raw)


def run_site_campaign_diagnostics(
    state,
    model,
    device,
    model_name,
    *,
    need_sites,
    need_campaigns,
):
    if not (need_sites or need_campaigns):
        return

    coeffs_resid_nn, sig_grid, infer_idx = dense_inference_for_diagnostics(
        state,
        model,
        device,
        need_sites=need_sites,
        need_campaigns=need_campaigns,
    )

    if need_sites:
        run_site_diagnostics(
            state,
            coeffs_resid_nn,
            sig_grid,
            infer_idx,
            model_name,
        )
    if need_campaigns:
        run_campaign_diagnostics(
            state,
            coeffs_resid_nn,
            sig_grid,
            infer_idx,
            model_name,
        )


def run_global_test(device, numgpu, model_name):
    if not do_test:
        return

    print("=========== global testing ============")
    Nin_2d, A_sites, Nd_sites, Err_sites, Ares_base, site_meta = get_data(
        vars_out=vars_out,
        pth_all=pth_all,
        vars_in_2d=vars_in_2d,
        modis_source=modis_source,
        dense_prefix=test_pr,
        site_prefix=test_pr,
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

    all_idx = np.arange(Nin_2d.sizes["time"], dtype=np.int64)
    test_ds = EOFNdDatasetSitesTorch(
        X2d_t, MU0_t, YS_t, VS_t, Nds, Eds, all_idx
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=hp["batch_sz"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = build_model(len(list(Nin_2d.data_vars)), numgpu, device, hp)
    load_model_weights_for_inference(
        model,
        f"./{model_name}_weights.pt",
        device=device,
        strict=True,
        eval_mode=True,
    )

    if bypass_infer:
        coeffs_resid_nn = Ares_base.values.astype(np.float32)
        sig = None
    else:
        with torch.no_grad():
            coeffs_resid_nn, sig = run_infer_x_mu0(device, test_loader, model)

    coeffs_abs_nn = coeffs_resid_nn + Amean4
    coeffs_abs_base = Ares_base + Amean4

    coeffs_abs_nn_da = da.from_array(
        coeffs_abs_nn.astype(np.float32), chunks=(64, -1, -1, -1)
    )
    sig_nn_da = None
    if sig is not None:
        sig_nn_da = da.from_array(
            sig.astype(np.float32), chunks=(64, -1, -1, -1)
        )

    coeffs_abs_base_da = (
        coeffs_abs_base.data
        if isinstance(coeffs_abs_base, xr.DataArray)
        else coeffs_abs_base
    )
    if not isinstance(coeffs_abs_base_da, da.Array):
        coeffs_abs_base_da = da.from_array(
            np.asarray(coeffs_abs_base_da, dtype=np.float32),
            chunks=(64, -1, -1, -1),
        )

    ds_nn = reconstruct_Nd_from_coeffs_chunked(
        coeffs_abs_nn_da,
        EOF,
        Nd_mean,
        Ares_base["time"],
        Ares_base["lat"],
        Ares_base["lon"],
        sigma_coeffs_abs=sig_nn_da,
        chunk_t=64,
    )
    Nd_nn_da = ds_nn["Nd_rec"].rename("Nd_nn").load()

    Nd_base_da = reconstruct_Nd_from_coeffs_chunked(
        coeffs_abs_base_da,
        EOF,
        Nd_mean,
        Ares_base["time"],
        Ares_base["lat"],
        Ares_base["lon"],
        chunk_t=64,
    )["Nd_rec"].rename("Nd_base").load()

    dsG = xr.Dataset({"Nd_base": Nd_base_da, "Nd_nn": Nd_nn_da})
    if "sigmaNd_rec" in ds_nn:
        dsG["sigmaNd_nn"] = ds_nn["sigmaNd_rec"].rename("sigmaNd_nn").load()

    dsG = finalize_nd_product(
        dsG,
        nd_name="Nd_nn",
        base_name="Nd_base",
        sigma_name="sigmaNd_nn",
        raw_nd_name="Nd_nn_raw",
        raw_sigma_name="sigmaNd_nn_raw",
    )

    out_nc = f"{model_name}_global_test.nc"
    safe_to_netcdf_chunked(dsG, out_nc)
    print(f"[WRITE] {out_nc}")

    plot_global_level_comparison(
        Nd_base=dsG["Nd_base"],
        Nd_nn=dsG["Nd_nn"],
        Nd_mean=Nd_mean,
        levels=[900, 700, 500],
        out_png="NDnet_global_test_levels.png",
        minN=MIN_ND,
    )
    print("[PLOT] NDnet_global_test_levels.png")

    if "sigmaNd_nn" in dsG:
        plot_global_level_sigma_rel_inference(
            sigmaNd_nn=dsG["sigmaNd_nn"],
            Nd_nn=dsG["Nd_nn"],
            Nd_mean=Nd_mean,
            levels=[500, 700, 900],
            out_png="NDnet_global_relative_sigma_levels.png",
            minNd=MIN_ND,
            vmin=0,
            vmax=100,
        )
        print("[PLOT] NDnet_global_relative_sigma_levels.png")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    numgpu = torch.cuda.device_count()
    mp.set_start_method("spawn", force=True)
    numwork = 0

    print("Using DataParallel on", numgpu, "GPUs")
    taper_mode, taper_start, taper_top, _ = _taper_settings()
    print(
        f"[TAPER] mode={taper_mode} start={taper_start:g} hPa "
        f"top={taper_top:g} hPa | NN only; base unchanged"
    )

    if do_train:
        run_model_name = active_model_name(hp)
        print(
            f"[MODE] training enabled; "
            f"fine_tune_sigma={bool(hp.get('fine_tune_sigma', False))}"
        )
        print(f"[MODE] active training/output model = {run_model_name}")
    else:
        run_model_name = MODEL_NAME
        print("[MODE] do_train=False -> frozen evaluation only")
        print("[MODE] training and sigma fine tuning are disabled")
        print(f"[MODE] evaluation model = {run_model_name}")

    site_cache_hit = False
    campaign_cache_hit = False
    if not do_train:
        if do_sites:
            site_cache_hit = diagnostic_cache_valid("site", run_model_name)
        if do_campaigns:
            campaign_cache_hit = diagnostic_cache_valid("campaign", run_model_name)

    if site_cache_hit:
        render_site_diagnostics(
            load_diagnostic_cache("site", run_model_name)
        )
    if campaign_cache_hit:
        render_campaign_diagnostics(
            load_diagnostic_cache("campaign", run_model_name)
        )

    need_sites_compute = bool(do_sites and (do_train or not site_cache_hit))
    need_campaigns_compute = bool(
        do_campaigns and (do_train or not campaign_cache_hit)
    )

    state = None
    model = None

    if do_train or need_sites_compute or need_campaigns_compute:
        state = load_site_campaign_state(
            need_campaign_bundle=bool(do_train or need_campaigns_compute),
            evaluation_only=not bool(do_train),
            need_sites=need_sites_compute,
            need_campaigns=need_campaigns_compute,
        )

    if do_train:
        model, run_model_name = train_refinement_model(
            state, device, numgpu, numwork
        )
        need_sites_compute = bool(do_sites)
        need_campaigns_compute = bool(do_campaigns)
    elif need_sites_compute or need_campaigns_compute:
        model = load_frozen_model(state, device, numgpu, run_model_name)

    if need_sites_compute or need_campaigns_compute:
        run_site_campaign_diagnostics(
            state,
            model,
            device,
            run_model_name,
            need_sites=need_sites_compute,
            need_campaigns=need_campaigns_compute,
        )

    run_global_test(device, numgpu, run_model_name)
    print("DONE.")
