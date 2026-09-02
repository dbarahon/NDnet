"""Launcher for the learned-upper-relax refinement experiment.

The substantial site/campaign diagnostic refactor is preserved byte-for-byte
in ``NDref_prob_refactor.py``.  This thin launcher only seeds a fresh training
checkpoint from the selected V2 weights, then executes that refactored driver.
"""

from __future__ import annotations

import os
import runpy

import torch
import torch.optim as optim

import config as cfg
from NN_construct import ProbRefineFromBaseAndX
from train_infer import load_model_weights_for_inference, save_training_checkpoint


def _seed_checkpoint_from_v2_if_needed():
    if not bool(getattr(cfg, "do_train", False)):
        return

    source = cfg.hp.get("init_from_weights", None)
    if not source:
        return

    ckpt_path = cfg.hp["model_name"] + ".ckpt"
    if os.path.exists(ckpt_path):
        print(f"[RELAX-INIT] existing checkpoint found: {ckpt_path}")
        return

    if not os.path.exists(source):
        raise FileNotFoundError(
            f"[RELAX-INIT] requested V2 initialization weights not found: {source}"
        )

    print("=" * 70)
    print("[RELAX-INIT] seeding learned-relax run from V2 weights")
    print(f"[RELAX-INIT] source = {source}")
    print(f"[RELAX-INIT] target = {ckpt_path}")
    print("=" * 70)

    # Four MODIS channels are defined by vars_in_2d.  Constructing this on CPU
    # is sufficient because the universal checkpoint stores unprefixed model
    # keys and an empty Adam state; the normal driver then resumes it on GPU.
    model = ProbRefineFromBaseAndX(
        in_channels=len(cfg.vars_in_2d),
        num_modes=cfg.num_modes,
        feat_ch=32,
        head_ch=64,
        sigma_min=float(cfg.hp.get("model_sigma_min", 1e-3)),
        detach_sigma_features=bool(cfg.hp.get("detach_sigma_features", True)),
    )
    load_model_weights_for_inference(
        model,
        source,
        device=torch.device("cpu"),
        strict=True,
        eval_mode=False,
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(cfg.hp["l_r"]),
        weight_decay=1e-5,
    )
    save_training_checkpoint(model, optimizer, -1, ckpt_path)
    print("[RELAX-INIT] checkpoint seeded at epoch 0")


if __name__ == "__main__":
    _seed_checkpoint_from_v2_if_needed()
    runpy.run_module("NDref_prob_refactor", run_name="__main__")
