from config import *
import config as cfg
from data_handling import *
from utils import *
from NN_construct import *
from train_infer import *
from plotting import * #plot_global_level_comparison, plot_vertical_profiles_sites, plot_global_level_sigma_inference

from campaign import *

import numpy as np
import os
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
import torch.multiprocessing as mp
import dask.array as da
import matplotlib.pyplot as plt

from torch.utils.data import WeightedRandomSampler
import torch.multiprocessing as m


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  # PyTorch 2.x



def build_model(in_channels, numgpu, device, hp):
    """Build the refinement model with optional sigma-feature detachment."""
    model = ProbRefineFromBaseAndX(
        in_channels=in_channels,
        num_modes=num_modes,
        feat_ch=32,
        head_ch=64,
        # Keep the historical model-output floor unless explicitly changed.
        sigma_min=float(hp.get("model_sigma_min", 1e-3)),
        detach_sigma_features=bool(hp.get("detach_sigma_features", True)),
    ).to(device)

    if numgpu > 1:
        model = nn.DataParallel(model)
    return model



def active_model_name(hp):
    """Return the model name used for outputs/inference for the selected mode."""
    if bool(hp.get("fine_tune_sigma", False)):
        return hp.get(
            "sigma_finetune_model_name",
            hp["model_name"] + "_sigmaFT",
        )
    return hp["model_name"]



def configure_training_stage(model, criterion, hp, device):
    """
    Configure normal training or optional sigma-only fine tuning.

    Normal mode:
      - all currently trainable parameters are optimized
      - checkpoint + optimizer state are resumed as before

    Sigma fine-tune mode:
      - load the best weights from the completed base run
      - freeze the whole model
      - re-enable only sigma_head
      - disable mean-only loss terms
      - keep Gaussian likelihood terms active
      - strengthen variance-relative sigma penalties
      - start a fresh sigma-only optimizer and epoch count

    Returns
    -------
    optimizer, start_epoch, hp_run
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

        # Load only model weights from the good full-model solution.
        # Do not restore the old optimizer: its parameter groups contain
        # the full model, whereas this stage trains sigma_head only.
        load_model_weights_for_inference(
            model,
            source_weights,
            device=device,
            strict=True,
            eval_mode=False,
        )
        model.train()

        # DataParallel-safe access to model submodules.
        core_model = model.module if isinstance(model, nn.DataParallel) else model

        # Freeze everything, then re-enable only the uncertainty head.
        for p in core_model.parameters():
            p.requires_grad = False
        for p in core_model.sigma_head.parameters():
            p.requires_grad = True

        # Mean-only terms cannot improve a frozen mean predictor.
        criterion.lambda_anchor = 0.0
        criterion.lambda_grad = 0.0
        criterion.lambda_nd = 0.0
        criterion.lambda_campaign = 0.0

        # Optional sigma-FT overrides for likelihood terms.  If absent,
        # retain the normal-training values already stored in criterion.
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

        # Stronger variance-relative sigma regularization.
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

        # Treat calibration as a new short training stage.
        start_epoch = 0
        hp_run["nepochs"] = int(hp.get("sigma_finetune_epochs", 30))
        hp_run["patience"] = int(
            hp.get("sigma_finetune_patience", hp.get("patience", 20))
        )

        # The normal campaign monitor was designed for mean/campaign skill.
        # For sigma calibration, validation loss already contains the site,
        # Nd-Gaussian, campaign-Gaussian, and sigma penalties.  Default to
        # monitoring that complete validation objective directly.
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
        # Original normal-training/resume behavior.
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

    # Common sanity diagnostics.
    trainable_names = [
        name for name, p in model.named_parameters() if p.requires_grad
    ]
    params = [p for p in model.parameters() if p.requires_grad]

    print("[DBG] #trainable tensors:", len(params))
    print("[DBG] #trainable params:", sum(p.numel() for p in params))
    print("[DBG] trainable parameter tensors:")
    for name in trainable_names:
        print("      ", name)

    if fine_tune_sigma:
        bad = [
            n for n in trainable_names
            if "sigma_head" not in n
        ]
        if bad:
            raise RuntimeError(
                "[SIGMA-FT] Non-sigma parameters are trainable: "
                + ", ".join(bad)
            )

    return optimizer, start_epoch, hp_run



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    numgpu = torch.cuda.device_count()
    mp.set_start_method("spawn", force=True)

    # IMPORTANT: for dask/xarray streaming, start with 0 workers (avoid duplicate IO)
    numwork = 0

    print("Using DataParallel on", numgpu, "GPUs")

    fine_tune_sigma = bool(hp.get("fine_tune_sigma", False))
    run_model_name = active_model_name(hp)
    print(f"[MODE] fine_tune_sigma = {fine_tune_sigma}")
    print(f"[MODE] active model    = {run_model_name}")

    if do_train:

        # ---------- DATA (site-only targets) ----------
        Nin_2d, A_sites, Nd_sites, Err_sites, Ares_base, site_meta = get_data(
            vars_out=vars_out,
            pth_all=pth_all,
            vars_in_2d=vars_in_2d,
            modis_source=modis_source,
            dense_prefix=prefix_dense,   # full dense window, e.g. starts before 2011
            site_prefix=prefix_sites,    # site-file token / glob
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
        
        var_obs = list(vars_out.keys())[0]

        nt = Nin_2d.sizes["time"]
        all_idx = np.arange(nt, dtype=np.int64)

        # deterministic split by time
        split = int(0.9 * nt)
        train_idx = all_idx[:split]
        val_idx   = all_idx[split:]

        # --------------------------------------------------
        # CAMPAIGN TRAINING FILES (reuse Ares_base/Nin_2d time axis)
        # --------------------------------------------------
        campaign_train_dir = getattr(cfg, "CAMPAIGN_TRAIN_DIR", None)
        campaign_pattern   = getattr(cfg, "CAMPAIGN_PATTERN", "*_training.nc")
        use_campaigns = bool(campaign_train_dir)

        if use_campaigns:
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

            # include all site days plus campaign-only days
            train_idx_use = np.union1d(train_idx, train_idx_c)
            val_idx_use   = np.union1d(val_idx, val_idx_c)

            print(f"[INFO] campaign training enabled: {campaign_train_dir}")
            print(f"[INFO] train days with campaign segments: {train_idx_c.size}")
            print(f"[INFO] val days with campaign segments:   {val_idx_c.size}")
        else:
            camp_bundle = None
            train_idx_use = train_idx
            val_idx_use = val_idx
            print("[INFO] campaign training disabled")

        # --------------------------------------------------
        # IMPORTANT:
        # old site sampler excludes campaign-only days, so disable it
        # whenever campaign training is active
        # --------------------------------------------------
        use_sampler = bool(hp.get("use_site_sampler", True)) and (not use_campaigns)

        if use_sampler:
            train_idx_f, train_sampler = make_equal_site_time_sampler(
                VS_t=VS_t,
                time_idx=train_idx_use,
                min_sites=hp["min_sites_for_sampling"],
                beta=hp["site_sampling_beta"],
                eps=1.0,
            )
            train_idx_final = train_idx_f
        else:
            train_sampler = None
            train_idx_final = train_idx_use
            if use_campaigns and bool(hp.get("use_site_sampler", True)):
                print("[INFO] disabling site-only sampler because it would miss campaign-only days")

        # --------------------------------------------------
        # MERGED daily dataset: site + optional campaign branch
        # --------------------------------------------------
        ds_train = EOFNdMultiObsDatasetTorch(
            X2d_t=X2d_t,
            MU0_t=MU0_t,
            YS_t=YS_t,
            VS_t=VS_t,
            Nds_t=Nds,
            Eds_t=Eds,
            campaign_bundle=camp_bundle,
            time_idx=train_idx_final,
        )

        ds_val = EOFNdMultiObsDatasetTorch(
            X2d_t=X2d_t,
            MU0_t=MU0_t,
            YS_t=YS_t,
            VS_t=VS_t,
            Nds_t=Nds,
            Eds_t=Eds,
            campaign_bundle=camp_bundle,
            time_idx=val_idx_use,
        )

        collate_train = make_multiobs_collate_fn(
            use_campaign_aug=bool(hp.get("use_campaign_aug", True)),
            aug_mode=hp.get("campaign_aug_mode", "random_valid"),
            seed=int(hp.get("campaign_aug_seed", 1234)),
        )

        collate_val = make_multiobs_collate_fn(
            use_campaign_aug=False,
            aug_mode="none",
            seed=1234,
        )

        train_loader = DataLoader(
            ds_train,
            batch_size=hp["batch_sz"],
            sampler=train_sampler,
            shuffle=False if train_sampler is not None else True,
            num_workers=numwork,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_train,
        )

        val_loader = DataLoader(
            ds_val,
            batch_size=hp["batch_sz"],
            shuffle=False,
            num_workers=numwork,
            pin_memory=True,
            collate_fn=collate_val,
        )
              # ---------- MODEL ----------
        in_channels = len(list(Nin_2d.data_vars))

        model = build_model(in_channels, numgpu, device, hp)
        # ---------- LOSS ----------
        latitudes = Nin_2d.lat.values
        
        
        criterion = SparseSiteCampaignNLLRefineLoss(
            lat_grid=Nin_2d.lat.values,
            site_meta=site_meta,
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
 
        # DataParallel if needed
        #if numgpu > 1:
        #@    model = nn.DataParallel(model)

        # ---------- OPTIMIZER / OPTIONAL SIGMA-ONLY FINE TUNING ----------
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
            start_epoch=start_epoch
        )


        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Val")        
        plt.legend()
        plt.xlim(left=5)
        plt.savefig(hp_run["model_name"] + "_loss.png")
        print("========== DONE training")

        # --------------------------------------------------
        # SITE TEST (optional)
        # --------------------------------------------------
        if do_sites:

            # --------------------------------------------------
            # SITE TEST (NDnet_site_test.nc)  -- simplified
            # --------------------------------------------------
            var_obs = list(vars_out.keys())[0]

            # truth as-is (no Nout): Nd_sites is (time, lev, site)
            Nd_true = Nd_sites.rename("Nd_true")                       # (time, lev, site)
            site_mask = xr.ufuncs.isfinite(Nd_true).any("lev").astype("uint8").rename("site_mask")  # (time, site)

            # attach coords for plotting
            for c in ["site", "ilat", "ilon", "site_lat", "site_lon"]:
                if c in site_meta and c not in Nd_true.coords:
                    Nd_true = Nd_true.assign_coords({c: ("site", site_meta[c].values)})

            # build full loader once (only needed if not bypassing)
          
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

            full_loader = DataLoader(
                full_ds,
                batch_size=hp["batch_sz"],
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=make_multiobs_collate_fn(
                    use_campaign_aug=False,
                    aug_mode="none",
                    seed=1234,
                ),
            )

            # -----------------------------------------
            # Get A_sites_resid (time, mode, site)
            # -----------------------------------------
            if bypass_infer:
                
                print('=====ByPassing Inference=====')
                assert np.all(A_sites["site"].values == site_meta["site"].values)
                A_sites_resid = A_sites.rename("A_sites_resid")  # already (time, mode, site)
                sig_sites = None
            else:
                # infer full grid residual coeffs and sigma
                coeffs_resid_nn, sig_grid = run_infer_x_mu0(device, full_loader, model)  # (T,M,H,W) numpy
                A_sites_resid, sig_sites = A_sites_from_infer_grid(
                    coeffs_resid_nn=coeffs_resid_nn,
                    sig_grid=sig_grid,
                    site_meta=site_meta,
                    time_coord=Nd_true["time"],
                    mode_coord=Ares_base["mode"].values if isinstance(Ares_base, xr.DataArray) else np.arange(coeffs_resid_nn.shape[1]),
                )

            # -----------------------------------------
            # Reconstruct Nd from A_sites (no gridded coeffs needed)
            # -----------------------------------------
            Nd_rec_lts, A_abs_sites, A_mean_sites = reconstruct_Nd_from_A_sites(
                A_sites_resid=A_sites_resid,
                site_meta=site_meta,
                EOF=EOF,
                Nd_mean=Nd_mean,
                A_mean=A_mean,
                valid_ref=Nd_sites,
                mask_mode=sitemaskmode

            )

            # plotting wants (time,lev,site) OR now accepts (lev,time,site)
            Nd_rec = Nd_rec_lts.rename("Nd_rec")  # (lev,time,site)

            # also keep a copy in (time,lev,site) for netcdf consistency if you prefer
            Nd_rec_tls = Nd_rec.transpose("time", "lev", "site")

            # -----------------------------------------
            # BASE at sites (always)
            # -----------------------------------------
            A_base_sites_resid = A_sites_from_Ares_base(Ares_base=Ares_base, site_meta=site_meta)

            Nd_base_lts, A_base_abs_sites, A_mean_sites = reconstruct_Nd_from_A_sites(
                A_sites_resid=A_base_sites_resid,
                site_meta=site_meta,
                EOF=EOF,
                Nd_mean=Nd_mean,
                A_mean=A_mean,
                valid_ref=Nd_sites,
                mask_mode=sitemaskmode
            )

            Nd_base_tls = Nd_base_lts.transpose("time", "lev", "site").rename("Nd_base")
            # right after you compute Nd_base_tls
            Nd_base_tls = Nd_base_tls.reset_coords(names=[c for c in ["lat", "lon"] if c in Nd_base_tls.coords], drop=True)


            # kill any CF aux coords like lat/lon attached by per-site files
            Nd_true = Nd_true.reset_coords(drop=True)
            site_mask = site_mask.reset_coords(drop=True)
            # -----------------------------------------
            # Save netcdf (separate variables)
            # -----------------------------------------
            ref_coords = {
                "time": Nd_true["time"],
                "lev":  Nd_true["lev"],
                "site": Nd_true["site"],
                "mode": A_sites_resid["mode"],
            }
            
            dsS = xr.Dataset(
                {
                    "Nd_true": make_canonical_da(Nd_true,        name="Nd_true", dims=("time","lev","site"), ref_coords=ref_coords, dtype=np.float32, verbose=False),
                    "Nd_base": make_canonical_da(Nd_base_tls,    name="Nd_base", dims=("time","lev","site"), ref_coords=ref_coords, dtype=np.float32, verbose=False),
                    "Nd_rec":  make_canonical_da(Nd_rec_tls,     name="Nd_rec",  dims=("time","lev","site"), ref_coords=ref_coords, dtype=np.float32, verbose=False),

                    "site_mask": make_canonical_da(site_mask,    name="site_mask", dims=("time","site"), ref_coords=ref_coords, dtype="uint8", verbose=False),

                    "A_sites_resid":      make_canonical_da(A_sites_resid,      name="A_sites_resid",      dims=("time","mode","site"), ref_coords=ref_coords, dtype=np.float32, verbose=False),
                    "A_base_sites_resid": make_canonical_da(A_base_sites_resid, name="A_base_sites_resid", dims=("time","mode","site"), ref_coords=ref_coords, dtype=np.float32, verbose=False),

                    "A_sites_abs":      make_canonical_da(A_abs_sites,      name="A_sites_abs",      dims=("time","mode","site"), ref_coords=ref_coords, dtype=np.float32, verbose=False),
                    "A_base_sites_abs": make_canonical_da(A_base_abs_sites, name="A_base_sites_abs", dims=("time","mode","site"), ref_coords=ref_coords, dtype=np.float32, verbose=False),

                    "A_mean_sites": make_canonical_da(A_mean_sites, name="A_mean_sites", dims=("mode","site"), ref_coords=ref_coords, dtype=np.float32, verbose=False),
                }
            )

            if sig_sites is not None:
                dsS["sigma_sites"] = make_canonical_da(sig_sites, name="sigma_sites", dims=("time","mode","site"), ref_coords=ref_coords, dtype=np.float32, verbose=False)

            # Attach site metadata ONCE (coords only; will not conflict)
            for c in ["ilat","ilon","site_lat","site_lon","obs_lat","obs_lon","file_lat","file_lon"]:
                if c in site_meta:
                    dsS = dsS.assign_coords({c: ("site", site_meta[c].values)})

            safe_to_netcdf(dsS, "NDnet_site_test.nc")
            print("[WRITE] NDnet_site_test.nc")

            # -----------------------------------------
            # Plot (update plotting function to accept lev,time,site too)
            # -----------------------------------------
            dsS_plot = xr.open_dataset("NDnet_site_test.nc")

            plot_vertical_profiles_sites(
                Nd_true_s=dsS_plot["Nd_true"],         # (time,lev,site)
                Nd_base_s=dsS_plot["Nd_base"],                        # no base in this simplified test
                Nd_nn_s=dsS_plot["Nd_rec"].transpose("lev", "time", "site"),  # pass (lev,time,site) to prove robustness
                out_png="NDnet_site_profiles.png",
                min_samples=5,
            )
            print("[PLOT] NDnet_site_profiles.png")
            
            
        # --------------------------------------------------
        # CAMPAIGN TEST (optional)
        # --------------------------------------------------
        if do_campaigns:
            if camp_bundle is None:
                print("[INFO] do_campaigns requested but no campaign bundle is loaded; skipping.")
            else:
                print("===========campaign evaluation============")

                # Full daily loader using the same merged daily dataset
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

                full_loader = DataLoader(
                    full_ds,
                    batch_size=hp["batch_sz"],
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    collate_fn=make_multiobs_collate_fn(
                        use_campaign_aug=False,
                        aug_mode="none",
                        seed=1234,
                    ),
                )

                # -----------------------------------------
                # Dense inference
                # -----------------------------------------
                if bypass_infer:
                    print("=====ByPassing Inference for campaigns=====")
                    coeffs_resid_nn = Ares_base.values.astype(np.float32)
                    sig_grid = None
                else:
                    coeffs_resid_nn, sig_grid = run_infer_x_mu0(device, full_loader, model)

                # -----------------------------------------
                # Sample campaign points from NN and base
                # -----------------------------------------
                dsC_nn = sample_campaign_from_dense_predictions(
                    coeffs_resid_nn=coeffs_resid_nn,
                    sigma_grid=sig_grid,
                    bundle=camp_bundle,
                    EOF=EOF,
                    Nd_mean=Nd_mean,
                    A_mean=A_mean,
                )

                dsC_base = sample_campaign_from_dense_predictions(
                    coeffs_resid_nn=Ares_base.values.astype(np.float32),
                    sigma_grid=None,
                    bundle=camp_bundle,
                    EOF=EOF,
                    Nd_mean=Nd_mean,
                    A_mean=A_mean,
                )

                # -----------------------------------------
                # Assemble output dataset
                # -----------------------------------------
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
                    dsC = dsC.assign_coords({"campaign": ("segment", dsC_nn["campaign"].values)})
                if "campaign_id" in dsC_nn:
                    dsC["campaign_id"] = dsC_nn["campaign_id"].astype(np.int32)

                # optional MODIS mask from preprocessed campaign files, if you later add it there
                if "modis_ok" in campaign_ds:
                    modis_ok_sorted = campaign_ds["modis_ok"].values.astype(np.int8)
                    dsC["modis_ok"] = xr.DataArray(modis_ok_sorted, dims=("segment",))

                safe_to_netcdf(dsC, "NDnet_campaign_test.nc")
                print("[WRITE] NDnet_campaign_test.nc")

                # -----------------------------------------
                # Plot per-campaign scatter panels
                # -----------------------------------------
                dsC_plot = xr.open_dataset("NDnet_campaign_test.nc")

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





    # --------------------------------------------------
    # TEST (global reconstruction)
    # --------------------------------------------------
    if do_test:
        print("===========testing============")

        
        
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

        
        nt = Nin_2d.sizes["time"]
        all_idx = np.arange(nt, dtype=np.int64)
        
        test_ds = EOFNdDatasetSitesTorch(X2d_t, MU0_t, YS_t, VS_t, Nds, Eds, all_idx)
        test_loader   = DataLoader(test_ds, batch_size=hp["batch_sz"],
              shuffle=False, num_workers=0, pin_memory=True)
              


                # Build model first (same architecture as training)
       
        in_channels = len(list(Nin_2d.data_vars))
        model = build_model(in_channels, numgpu, device, hp)

        # Load weights
        load_model_weights_for_inference(
            model,
            run_model_name + "_weights.pt",
            device=device,
            strict=True,     # set False if you changed heads
            eval_mode=True
        )

        # Run inference
        
    # ============================================================
    #  inference + Nd and sigmaNd reconstruction
    # ============================================================

    coeffs_resid_nn, sig = run_infer_x_mu0(device, test_loader, model)  # numpy (T,M,lat,lon)
    # IMPORTANT ASSUMPTION:
    #   `sig` is std-dev of coeff residuals per mode (same shape as coeffs_resid_nn).
    #   If `sig` is log-sigma or variance, convert BEFORE using below.

    # absolute coeffs = residual + Amean
    coeffs_abs_nn   = coeffs_resid_nn + Amean4               # numpy (T,M,lat,lon)
    coeffs_abs_base = (Ares_base + Amean4)                   # xarray (time,mode,lat,lon)

    # daskify NN coeffs (+ sigma)
    coeffs_abs_nn_da = da.from_array(coeffs_abs_nn.astype(np.float32), chunks=(64, -1, -1, -1))
    sig_nn_da        = da.from_array(sig.astype(np.float32),           chunks=(64, -1, -1, -1))

    # base coeffs: keep dask-backed if possible
    coeffs_abs_base_da = coeffs_abs_base.data if isinstance(coeffs_abs_base, xr.DataArray) else coeffs_abs_base
    if not isinstance(coeffs_abs_base_da, da.Array):
        coeffs_abs_base_da = da.from_array(np.asarray(coeffs_abs_base_da, dtype=np.float32), chunks=(64, -1, -1, -1))

    # coords
    time_coord = Ares_base["time"]
    lat_coord  = Ares_base["lat"]
    lon_coord  = Ares_base["lon"]

    # reconstruct Nd globally + sigmaNd for NN
    ds_nn = reconstruct_Nd_from_coeffs_chunked(
        coeffs_abs_nn_da,
        EOF,
        Nd_mean,
        time_coord,
        lat_coord,
        lon_coord,
        sigma_coeffs_abs=sig_nn_da,
        chunk_t=64,
    )

    Nd_nn_da      = ds_nn["Nd_rec"].rename("Nd_nn")
    sigmaNd_nn_da = ds_nn["sigmaNd_rec"].rename("sigmaNd_nn")

    # reconstruct Nd for base (no sigma unless you have one)
    Nd_base_da = reconstruct_Nd_from_coeffs_chunked(
        coeffs_abs_base_da,
        EOF,
        Nd_mean,
        time_coord,
        lat_coord,
        lon_coord,
        chunk_t=64,
    )["Nd_rec"].rename("Nd_base")

    # materialize
    Nd_nn_da      = Nd_nn_da.load()
    sigmaNd_nn_da = sigmaNd_nn_da.load()
    Nd_base_da    = Nd_base_da.load()

    # save global fields
    dsG = xr.Dataset(
        {
            "Nd_base": Nd_base_da,
            "Nd_nn": Nd_nn_da,
            "sigmaNd_nn": sigmaNd_nn_da,
        }
    )

    # also save global-level sigma summary (handy for debugging/plots)
    #dsSigma = global_level_sigma_stats(dsG["sigmaNd_nn"])
    #dsG = xr.merge([dsG, dsSigma])

    print(dsG)
    safe_to_netcdf_chunked(dsG, f"{run_model_name}_global_test.nc")
    print("[WRITE] NDnet_global_test.nc")

    plot_global_level_comparison(
        Nd_base=dsG["Nd_base"],
        Nd_nn=dsG["Nd_nn"],
        Nd_mean=Nd_mean,
        levels=[900, 700, 500],
        out_png="NDnet_global_test_levels.png",
    )
    print("[PLOT] NDnet_global_test_levels.png")
    
    #plot_global_level_sigma_inference(
    #sigmaNd_nn=dsG["sigmaNd_nn"],
    #Nd_mean=Nd_mean,
    #levels=[500, 700, 900],
    #out_png=f"{run_model_name}_global_sigma_levels.png",
	#)

    
    plot_global_level_sigma_rel_inference(
        sigmaNd_nn=dsG["sigmaNd_nn"],
        Nd_nn=dsG["Nd_nn"],
        Nd_mean=Nd_mean,
        levels=[500, 700, 900],
        out_png="NDnet_global_relative_sigma_levels.png",
        minNd=1.0,
        vmin=0,
        vmax=100,
    )

    print("DONE.")
