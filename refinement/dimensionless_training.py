"""Training entry point for the dimensionless NDnet refinement objective.

This module intentionally uses direct function calls for initialization and
training. It does not monkey-patch functions in ``refinement_driver`` at
runtime, which keeps experiment behavior explicit and reproducible.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

import config as cfg
import refinement_driver as core
from NN_relax import SparseSiteCampaignNLLRefineLossRelax
from loss_normalization import estimate_base_coefficient_scales
from train_infer import (
    load_model_weights_for_inference,
    load_training_checkpoint,
    training_val,
)


_VALID_INIT_MODES = {"weights", "scratch", "resume"}


def initialize_training_model(model, optimizer, *, device, model_name):
    """Apply the configured initialization policy explicitly and return epoch."""
    mode = str(getattr(cfg, "TRAIN_INIT_MODE", "resume")).strip().lower()
    if mode not in _VALID_INIT_MODES:
        raise ValueError(
            f"Unknown TRAIN_INIT_MODE={mode!r}; expected one of {sorted(_VALID_INIT_MODES)}"
        )

    if mode == "scratch":
        print("[INIT] mode=scratch: using random model initialization")
        return 0

    if mode == "weights":
        source = str(getattr(cfg, "TRAIN_INIT_WEIGHTS", "")).strip()
        if not source:
            raise ValueError("TRAIN_INIT_MODE='weights' requires TRAIN_INIT_WEIGHTS")
        if not os.path.exists(source):
            raise FileNotFoundError(f"Initialization weights not found: {source}")
        print(f"[INIT] mode=weights: loading {source}; starting at epoch 0")
        load_model_weights_for_inference(
            model,
            source,
            device=device,
            strict=True,
            eval_mode=False,
        )
        return 0

    checkpoint = model_name + ".ckpt"
    print(f"[INIT] mode=resume: checkpoint={checkpoint}")
    return load_training_checkpoint(model, optimizer, checkpoint, device)


def _write_loss_scale_metadata(model_name, coeff_scales, n_training_days):
    """Persist normalization constants and objective settings for the run."""
    lambdas = {
        key: float(value)
        for key, value in cfg.hp.items()
        if key.startswith("lambda_") and isinstance(value, (int, float))
    }
    lambdas["lambda_upper_relax"] = float(cfg.UPPER_RELAX_LAMBDA)

    payload = {
        "schema": "NDnet dimensionless loss scales v1",
        "model_name": str(model_name),
        "training_init_mode": str(getattr(cfg, "TRAIN_INIT_MODE", "resume")),
        "nd_loss_scale_cm3": float(cfg.LOSS_ND_SCALE),
        "coefficient_scale_definition": (
            "latitude-area-weighted RMS of base residual EOF coefficients "
            "over the training-period days"
        ),
        "coefficient_scale_floor": float(cfg.LOSS_COEFF_SCALE_FLOOR),
        "coefficient_scales_by_mode": [float(x) for x in coeff_scales],
        "n_training_days_for_scales": int(n_training_days),
        "anchor_definition": (
            "log-pressure-weighted whole-column mean squared Nd departure "
            "from base, divided by nd_loss_scale_cm3^2"
        ),
        "upper_relax_definition": (
            "log-pressure-weighted upper-column mean squared Nd departure "
            "from base, divided by the same nd_loss_scale_cm3^2"
        ),
        "site_coeff_gaussian_definition": (
            "optional Gaussian likelihood against coefficients projected from "
            "site profiles; disabled by default because coefficients are not "
            "direct observations and this NLL has no bounded order-one scale"
        ),
        "upper_relax_start_hpa": float(cfg.UPPER_RELAX_START_HPA),
        "upper_relax_top_hpa": float(cfg.UPPER_RELAX_TOP_HPA),
        "lambdas": lambdas,
    }
    path = model_name + "_loss_scales.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[LOSS-SCALE] wrote {path}")


def train_dimensionless_refinement_model(state, device, numgpu, numwork):
    """Train the full refinement model with fixed dimensionless loss scales."""
    train_loader, val_loader = core.make_training_loaders(state, numwork)
    in_channels = len(list(state["Nin_2d"].data_vars))
    model = core.build_model(in_channels, numgpu, device, cfg.hp)

    n_training_days = max(1, int(0.9 * state["MU0_t"].shape[0]))
    coeff_scales = estimate_base_coefficient_scales(
        state["MU0_t"],
        state["Nin_2d"].lat.values,
        n_time=n_training_days,
        floor=float(cfg.LOSS_COEFF_SCALE_FLOOR),
        chunk_size=int(getattr(cfg, "LOSS_SCALE_CHUNK_SIZE", 16)),
    )

    print(f"[LOSS-SCALE] Nd reference = {cfg.LOSS_ND_SCALE:g} cm-3")
    print(
        "[LOSS-SCALE] prior coefficient RMS by mode = "
        + np.array2string(coeff_scales, precision=4, separator=", ")
    )
    print(
        "[LOSS] weights: "
        f"siteNd={cfg.hp.get('lambda_nd', 1.0):g} "
        f"siteNdG={cfg.hp.get('lambda_Ndgauss', 0.25):g} "
        f"campNd={cfg.hp.get('lambda_campaign', 1.0):g} "
        f"campNdG={cfg.hp.get('lambda_campaign_gauss', 0.20):g} "
        f"anchor={cfg.hp.get('lambda_anchor', 1.0):g} "
        f"upper={cfg.UPPER_RELAX_LAMBDA:g} "
        f"siteCoeffG={cfg.hp.get('lambda_site_coeff_gauss', 0.0):g} "
        f"sigSite={cfg.hp.get('lambda_sigma', 1.0):g} "
        f"sigCamp={cfg.hp.get('lambda_campaign_sigma', 1.0):g}"
    )
    _write_loss_scale_metadata(cfg.MODEL_NAME, coeff_scales, n_training_days)

    criterion = SparseSiteCampaignNLLRefineLossRelax(
        lat_grid=state["Nin_2d"].lat.values,
        site_meta=state["site_meta"],
        # Dimensionless coefficient terms use coeff_scales, so no additional
        # EOF-energy weighting is required.
        mode_weights=np.ones(cfg.num_modes, dtype=np.float32),
        EOF=cfg.EOF,
        Nd_mean=cfg.Nd_mean,
        A_mean=cfg.A_mean,
        coeff_scales=coeff_scales,
        coeff_scale_floor=float(cfg.LOSS_COEFF_SCALE_FLOOR),
        nd_loss_scale=float(cfg.LOSS_ND_SCALE),
        lambda_grad=float(cfg.hp.get("lambda_grad", 0.0)),
        lambda_anchor=float(cfg.hp.get("lambda_anchor", 1.0)),
        # Historical internal name retained by NN_construct_v2. In the
        # dimensionless/public configuration this is explicitly the optional
        # projected-coefficient Gaussian site likelihood.
        lambda_sites=float(cfg.hp.get("lambda_site_coeff_gauss", 0.0)),
        lambda_nd=float(cfg.hp.get("lambda_nd", 1.0)),
        lambda_Ndgauss=float(cfg.hp.get("lambda_Ndgauss", 0.25)),
        lambda_campaign=float(cfg.hp.get("lambda_campaign", 1.0)),
        lambda_campaign_gauss=float(cfg.hp.get("lambda_campaign_gauss", 0.20)),
        min_sigma=float(cfg.hp.get("min_sigma", 0.1)),
        err_mix=float(cfg.hp.get("err_mix", 0.15)),
        lambda_sigma=float(cfg.hp.get("lambda_sigma", 1.0)),
        lambda_campaign_sigma=float(cfg.hp.get("lambda_campaign_sigma", 1.0)),
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(
        params,
        lr=float(cfg.hp["l_r"]),
        weight_decay=float(cfg.hp.get("weight_decay", 1e-5)),
    )
    start_epoch = initialize_training_model(
        model,
        optimizer,
        device=device,
        model_name=cfg.MODEL_NAME,
    )

    train_losses, val_losses = training_val(
        model,
        criterion,
        train_loader,
        val_loader,
        optimizer,
        cfg.hp,
        device,
        start_epoch=start_epoch,
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Validation monitor")
    ax.legend()
    if len(train_losses) > 5:
        ax.set_xlim(left=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dimensionless objective")
    fig.savefig(cfg.MODEL_NAME + "_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("========== DONE dimensionless training")
    return model, cfg.MODEL_NAME
