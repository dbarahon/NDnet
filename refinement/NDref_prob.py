"""NDnet probabilistic refinement workflow with dimensionless loss scaling.

Training initialization is explicit: no runtime monkey-patching of driver
functions is used. Site, campaign, and global diagnostics continue to use the
shared refinement driver and raw network output.
"""

from __future__ import annotations

import torch
import torch.multiprocessing as mp

import config as cfg
import refinement_driver as core
from dimensionless_training import train_dimensionless_refinement_model


def _training_state(state):
    """Return a shallow training view with campaigns optionally removed."""
    use_campaigns = bool(getattr(cfg, "TRAIN_WITH_CAMPAIGNS", True))
    if use_campaigns:
        print("[TRAIN-DATA] fixed sites + campaigns")
        return state

    out = dict(state)
    out["camp_bundle"] = None
    out["campaign_ds"] = None
    print("[TRAIN-DATA] fixed sites only; campaign tensors excluded from training")
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    numgpu = torch.cuda.device_count()
    mp.set_start_method("spawn", force=True)
    numwork = 0

    print("Using DataParallel on", numgpu, "GPUs")
    print(
        f"[LOSS-SCALE] Nd={cfg.LOSS_ND_SCALE:g} cm-3; "
        f"coefficient floor={cfg.LOSS_COEFF_SCALE_FLOOR:g}"
    )
    print(
        f"[RELAX] lambda={cfg.UPPER_RELAX_LAMBDA:g} "
        f"start={cfg.UPPER_RELAX_START_HPA:g} hPa "
        f"top={cfg.UPPER_RELAX_TOP_HPA:g} hPa; "
        "same Nd scale as observational losses"
    )
    print("[RELAX] raw NN output is used at diagnostics and production inference")
    print(
        f"[TRAIN-DATA] TRAIN_WITH_CAMPAIGNS="
        f"{bool(getattr(cfg, 'TRAIN_WITH_CAMPAIGNS', True))}"
    )
    print(
        f"[INIT] TRAIN_INIT_MODE="
        f"{str(getattr(cfg, 'TRAIN_INIT_MODE', 'resume')).lower()}"
    )

    run_model_name = cfg.MODEL_NAME
    if cfg.do_train:
        print("[MODE] dimensionless full-model training enabled")
    else:
        print("[MODE] do_train=False -> frozen evaluation only")
    print(f"[MODE] model = {run_model_name}")

    site_cache_hit = False
    campaign_cache_hit = False
    if not cfg.do_train:
        if cfg.do_sites:
            site_cache_hit = core.diagnostic_cache_valid("site", run_model_name)
        if cfg.do_campaigns:
            campaign_cache_hit = core.diagnostic_cache_valid(
                "campaign", run_model_name
            )

    if site_cache_hit:
        core.render_site_diagnostics(
            core.load_diagnostic_cache("site", run_model_name)
        )
    if campaign_cache_hit:
        core.render_campaign_diagnostics(
            core.load_diagnostic_cache("campaign", run_model_name)
        )

    need_sites_compute = bool(
        cfg.do_sites and (cfg.do_train or not site_cache_hit)
    )
    need_campaigns_compute = bool(
        cfg.do_campaigns and (cfg.do_train or not campaign_cache_hit)
    )

    state = None
    model = None

    if cfg.do_train or need_sites_compute or need_campaigns_compute:
        need_campaign_bundle = bool(
            need_campaigns_compute
            or (
                cfg.do_train
                and bool(getattr(cfg, "TRAIN_WITH_CAMPAIGNS", True))
            )
        )
        state = core.load_site_campaign_state(
            need_campaign_bundle=need_campaign_bundle,
            evaluation_only=not bool(cfg.do_train),
            need_sites=need_sites_compute,
            need_campaigns=need_campaigns_compute,
        )

    if cfg.do_train:
        model, run_model_name = train_dimensionless_refinement_model(
            _training_state(state),
            device,
            numgpu,
            numwork,
        )
        need_sites_compute = bool(cfg.do_sites)
        need_campaigns_compute = bool(cfg.do_campaigns)
    elif need_sites_compute or need_campaigns_compute:
        model = core.load_frozen_model(
            state,
            device,
            numgpu,
            run_model_name,
        )

    if need_sites_compute or need_campaigns_compute:
        core.run_site_campaign_diagnostics(
            state,
            model,
            device,
            run_model_name,
            need_sites=need_sites_compute,
            need_campaigns=need_campaigns_compute,
        )

    core.run_global_test(device, numgpu, run_model_name)
    print("DONE.")


if __name__ == "__main__":
    main()
