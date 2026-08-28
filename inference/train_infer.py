import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
import config as cfg


class StepTimer:
    def __init__(self, device="cuda"):
        self.device = device
        self.reset()

    def reset(self):
        self.t = {}
        self._t0 = None

    def tic(self, key):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self._t0 = time.time()
        self._key = key

    def toc(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.t[self._key] = self.t.get(self._key, 0.0) + (time.time() - self._t0)

    def report(self, prefix="", denom=1):
        msg = " | ".join([f"{k}:{self.t[k]/denom:.3f}s" for k in self.t])
        print(f"{prefix}{msg}")

def move_to(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    return x


def save_model_weights(model, path):
    sd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(sd, path)
    print(f"[SAVE] Model weights saved to {path}")



def _strip_module_prefix(sd):
    if any(k.startswith("module.") for k in sd.keys()):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd

def save_training_checkpoint(model, optimizer, epoch, path):
    # always save *unwrapped* keys
    sd = model.state_dict()
    sd = _strip_module_prefix(sd)

    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": sd,              # <-- universal (no module.)
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path
    )
    print(f"[CKPT] Saved universal checkpoint at epoch {epoch}: {path}")



def _add_module_prefix(sd):
    if not any(k.startswith("module.") for k in sd.keys()):
        return {"module." + k: v for k, v in sd.items()}
    return sd

def load_training_checkpoint(model, optimizer, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        print("[CKPT] No training checkpoint found")
        return 0

    print(f"[CKPT] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    sd = ckpt["model_state_dict"]

    # adapt to current wrapper
    if isinstance(model, nn.DataParallel):
        sd = _add_module_prefix(sd)          # DP expects module.*
    else:
        sd = _strip_module_prefix(sd)        # non-DP expects no module.*

    model.load_state_dict(sd, strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    start_epoch = int(ckpt["epoch"]) + 1
    print(f"[CKPT] Resuming from epoch {start_epoch}")
    return start_epoch




def load_model_weights_for_inference(
    model: nn.Module,
    weights_path: str,
    device: torch.device | str = "cuda",
    strict: bool = True,
    eval_mode: bool = True,
    verbose: bool = True,
):
    """
    Load weights into `model` for inference, adapting automatically to DataParallel.

    Parameters
    ----------
    model : nn.Module
        Your model, possibly wrapped with nn.DataParallel.
    weights_path : str
        Path to a weights-only file saved with torch.save(state_dict, path).
    device : torch.device | str
        Target device for map_location and model placement.
    strict : bool
        Whether to enforce exact key match.
    eval_mode : bool
        If True, calls model.eval().
    verbose : bool
        If True, prints status.

    Returns
    -------
    model : nn.Module
        The model with loaded weights (same object).
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Ensure model is on the target device before loading (good practice for DP as well)
    model = model.to(device)

    if verbose:
        dp_infer = isinstance(model, nn.DataParallel)
        print(f"[WEIGHTS] Loading: {weights_path}")
        print(f"[WEIGHTS] Target device: {device}, DataParallel model: {dp_infer}, strict={strict}")

    sd = torch.load(weights_path, map_location=device, weights_only=False)

    # Some people save full checkpoints; handle both.
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    # Adapt keys to wrapper
    if isinstance(model, nn.DataParallel):
        sd = _add_module_prefix(sd)
    else:
        sd = _strip_module_prefix(sd)

    missing, unexpected = model.load_state_dict(sd, strict=strict)

    if verbose:
        if missing:
            print(f"[WEIGHTS] Missing keys ({len(missing)}): {missing[:8]}{' ...' if len(missing)>8 else ''}")
        if unexpected:
            print(f"[WEIGHTS] Unexpected keys ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")
        print("[WEIGHTS] Load OK")

    if eval_mode:
        model.eval()

    return model


@torch.no_grad()
def _eval_loop(model, criterion, loader, device, use_amp=True):
    model.eval()
    val_sum = 0.0
    val_n = 0
    n_zero = 0
    amp_enabled = (use_amp and device.type == "cuda")

    for x2d, mu0, yS, vS, Nds in loader:
        x2d = x2d.to(device, non_blocking=True)
        yS  = yS.to(device, non_blocking=True)
        vS  = vS.to(device, non_blocking=True)
        mu0 = mu0.to(device, non_blocking=True)
        Nds = Nds.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=amp_enabled):
            mu, sigma = model(x2d, mu0)
            loss = criterion(mu, sigma, yS, vS, mu0, Nds, return_parts=False)

        lv = float(loss.detach().item())
        if lv == 0.0:
            n_zero += 1
            
        val_sum += float(loss.detach().item())
        val_n += 1
        #if val_n > 0:
        #	print(f"[VALDBG] batches={val_n} zero_batches={n_zero}", flush=True)

    return val_sum / max(val_n, 1)


def training_val(model, criterion, train_loader, val_loader, optimizer, hp, device,
                 start_epoch=0, use_amp=True, clip_grad=0):

    scaler = GradScaler(enabled=(use_amp and device.type == "cuda"))
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)
    # ----------------  anchor schedule knobs ----------------
    lambda_anchor0 = float(getattr(criterion, "lambda_anchor", 0.0))
    anchor_decay   = float(hp.get("lambda_anchor_decay", 1.0))   # 1.0 => no decay
    anchor_min     = float(hp.get("lambda_anchor_min", 0.0))
    # ------------------------------------------------------------

    best_val = float("inf")
    bad_epochs = 0
    train_losses, val_losses = [], []

    for epoch in range(start_epoch, hp["nepochs"]):

        # ---------------- NEW: anneal anchor each epoch -----------
        if hasattr(criterion, "lambda_anchor") and anchor_decay < 1.0:
            new_anchor = max(anchor_min, lambda_anchor0 * (anchor_decay ** epoch))
            criterion.lambda_anchor = float(new_anchor)
        # ---------------------------------------------------------
        
        model.train()
        t0 = time.perf_counter()
        train_sum, train_n = 0.0, 0
        amp_enabled = (use_amp and device.type == "cuda")

        step= 0
        
        for bi, (x2d, mu0, yS, vS, Nds) in enumerate(train_loader):
            step += 1
            x2d = x2d.to(device, non_blocking=True)
            yS  = yS.to(device, non_blocking=True)
            vS  = vS.to(device, non_blocking=True)
            mu0 = mu0.to(device, non_blocking=True)
            Nds = Nds.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=amp_enabled):
                mu, sigma = model(x2d, mu0)
                if not torch.isfinite(mu).all():
                    print("[DBG] nonfinite mu", flush=True)
                if not torch.isfinite(sigma).all():
                    print("[DBG] nonfinite sigma", flush=True)
                loss, parts = criterion(mu, sigma, yS, vS, mu0, Nds, return_parts=True)
                if not torch.isfinite(loss):
                    print("[ERR] non-finite loss detected!", loss.detach().item(), flush=True)


            scaler.scale(loss).backward()

            
            if (step % cfg.print_every == 0):
                print(
                    f"[LOSS] step={step:06d} "
                    f"tot={parts['loss']:.5f} "
                    f"site={parts['L_site']:.5f} "
                    f"nd={parts['L_nd']:.5f} "
                    f"anc={parts['L_anchor']:.5f} "
                    f"grad={parts['L_grad']:.5f} "
                    f"sig={parts['L_sig']:.5f} "
                    f"sig_mean={parts['sig_mean_sites']:.3f} "
                    f"vS={parts['vS_frac']:.3f}", 
                    flush =  True
                )
            if clip_grad and clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))

            scaler.step(optimizer)
            # after scaler.step and scaler.update
            if bi == 0:
                print(f"[DBG] grad scaler scale={scaler.get_scale()}", flush=True)

            scaler.update()

            train_sum += float(loss.detach().item())
            train_n += 1

        t_train = time.perf_counter() - t0
        train_loss = train_sum / max(train_n, 1)

        t1 = time.perf_counter()
        val_loss = _eval_loop(model, criterion, val_loader, device, use_amp=use_amp)
        t_val = time.perf_counter() - t1

        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch+1}/{hp['nepochs']}], "
            f"Train: {train_loss:.6f}, Val: {val_loss:.6f}, "
            f"T(train): {t_train:.2f}s, T(val): {t_val:.2f}s, "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}",
            flush=True
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            save_model_weights(model, hp["model_name"] + "_weights.pt")
            save_training_checkpoint(model, optimizer, epoch, hp["model_name"] + ".ckpt")
        else:
            bad_epochs += 1
            if bad_epochs >= hp["patience"]:
                print("[EARLY] Early stopping", flush=True)
                break

    return train_losses, val_losses

    
@torch.no_grad()
def run_infer_x_mu0(device, loader, model, use_amp=True, return_sigma=True):
    model.eval()
    mu_list, sig_list = [], []
    amp_enabled = (use_amp and device.type == "cuda")

    for batch in loader:
        # dataset may yield (x2d, mu0) or (x2d, yS, vS, mu0)
        x2d, mu0 = batch
        
        x2d = x2d.to(device, non_blocking=True)
        mu0 = mu0.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=amp_enabled):
            mu, sigma = model(x2d, mu0)

        mu_list.append(mu.detach().cpu())
        if return_sigma:
            sig_list.append(sigma.detach().cpu())

    mu = torch.cat(mu_list, dim=0).numpy().astype("float32")
    if not return_sigma:
        return mu
    sig = torch.cat(sig_list, dim=0).numpy().astype("float32")
    return mu, sig



