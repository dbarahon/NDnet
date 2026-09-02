"""Learned-upper-relax refinement driver.

The heavy site/campaign diagnostic implementation lives in
``refinement_driver.py``. This module keeps experiment controls explicit:
training initialization and whether campaigns participate in training are
independent from the site/campaign diagnostic switches.
"""

from __future__ import annotations

import os

import torch
import torch.multiprocessing as mp

import config as cfg
import refinement_driver as core


_VALID_INIT_MODES = {"weights", "scratch", "resume"}


def _install_training_initialization():
    """Apply the configured initialization policy without altering V2 code."""
    mode = str(getattr(cfg, "TRAIN_INIT_MODE", "resume")).strip().lower()
    if mode not in _VALID_INIT_MODES:
        raise ValueError(
            f"Unknown TRAIN_INIT_MODE={mode!r}; expected one of "
            f"{sorted(_VALID_INIT_MODES)}"
        )

    if mode == "resume":
        print("[INIT] mode=resume: use current checkpoint when available")
        return

    if bool(cfg.hp.get("fine_tune_sigma", False)):
        raise ValueError(
            "TRAIN_INIT_MODE='weights'/'scratch' is for full-model training; "
            "disable fine_tune_sigma or use TRAIN_INIT_MODE='resume'."
        )

    if mode == "scratch":
        def _start_fresh(model, optimizer, ckpt_path, device):
            print(
                f"[INIT] mode=scratch: ignoring checkpoint {ckpt_path}; "
                "using random model initialization"
            )
            return 0

        core.load_training_checkpoint = _start_fresh
        return

    source = str(getattr(cfg, "TRAIN_INIT_WEIGHTS", "")).strip()
    if not source:
        raise ValueError(
            "TRAIN_INIT_MODE='weights' requires TRAIN_INIT_WEIGHTS."
        )
    if not os.path.exists(source):
        raise FileNotFoundError(
            f"[INIT] requested initialization weights not found: {source}"
        )

    def _start_from_weights(model, optimizer, ckpt_path, device):
        print(
            f"[INIT] mode=weights: ignoring checkpoint {ckpt_path}; "
            f"loading {source}"
        )
        core.load_model_weights_for_inference(
            model,
            source,
            device=device,
            strict=True,
            eval_mode=False,
        )
        return 0

    core.load_training_checkpoint = _start_from_weights


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
        f"[RELAX] lambda={cfg.UPPER_RELAX_LAMBDA:g} "
        f"start={cfg.UPPER_RELAX_START_HPA:g} hPa "
        f"top={cfg.UPPER_RELAX_TOP_HPA:g} hPa "
        f"scale={cfg.UPPER_RELAX_ND_SCALE:g} cm-3"
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

    if cfg.do_train:
        run_model_name = core.active_model_name(cfg.hp)
        print(
            f"[MODE] training enabled; "
            f"fine_tune_sigma={bool(cfg.hp.get('fine_tune_sigma', False))}"
        )
        print(f"[MODE] active training/output model = {run_model_name}")
    else:
        run_model_name = cfg.MODEL_NAME
        print("[MODE] do_train=False -> frozen evaluation only")
        print(f"[MODE] evaluation model = {run_model_name}")

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
        _install_training_initialization()
        model, run_model_name = core.train_refinement_model(
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
