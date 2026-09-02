import os
import time

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config as cfg
from utils import log_train_event


run_id = os.environ.get("RUN_ID", "run0")
job_id = os.environ.get("SLURM_JOB_ID", os.environ.get("PBS_JOBID", "nojid"))
rank = int(os.environ.get("RANK", "0"))


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
        msg = " | ".join([f"{k}:{self.t[k] / denom:.3f}s" for k in self.t])
        print(f"{prefix}{msg}")


def move_to(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    return x


def _move_batch_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: move_to(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return tuple(move_to(x, device) for x in batch)
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def _extract_model_inputs(batch):
    """Return x2d and mu0 from legacy tuples or multi-observation dicts."""
    if isinstance(batch, dict):
        return batch["x2d"], batch["mu0"]

    x2d, mu0, _, _, _, _ = batch
    return x2d, mu0


def _criterion_from_batch(criterion, mu, sigma, batch, return_parts=False):
    """Bridge legacy site tuples and site/campaign dictionary batches."""
    if isinstance(batch, dict):
        yS = batch.get("yS")
        Nds = batch.get("Nds")
        Eds = batch.get("Eds")
        seg_y = batch.get("seg_y")
        seg_y_std = batch.get("seg_y_std")

        return criterion(
            mu.float(),
            sigma.float(),
            yS.float() if isinstance(yS, torch.Tensor) else yS,
            batch.get("vS"),
            batch["mu0"].float(),
            Nds.float() if isinstance(Nds, torch.Tensor) else Nds,
            Eds.float() if isinstance(Eds, torch.Tensor) else Eds,
            seg_batch_idx=batch.get("seg_batch_idx"),
            seg_ilat=batch.get("seg_ilat"),
            seg_ilon=batch.get("seg_ilon"),
            seg_p=batch.get("seg_p"),
            seg_y=seg_y.float() if isinstance(seg_y, torch.Tensor) else seg_y,
            seg_y_std=(
                seg_y_std.float()
                if isinstance(seg_y_std, torch.Tensor)
                else seg_y_std
            ),
            seg_valid_obs=batch.get("seg_valid_obs"),
            seg_campaign_id=batch.get("seg_campaign_id"),
            seg_group_id=batch.get("seg_flight_id"),
            return_parts=return_parts,
        )

    x2d, mu0, yS, vS, Nds, Eds = batch
    return criterion(
        mu.float(),
        sigma.float(),
        yS.float(),
        vS,
        mu0.float(),
        Nds.float(),
        Eds.float(),
        return_parts=return_parts,
    )


def save_model_weights(model, path):
    sd = (
        model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict()
    )
    torch.save(sd, path)
    print(f"[SAVE] Model weights saved to {path}")


def _strip_module_prefix(sd):
    if any(k.startswith("module.") for k in sd):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


def _add_module_prefix(sd):
    if not any(k.startswith("module.") for k in sd):
        return {"module." + k: v for k, v in sd.items()}
    return sd


def save_training_checkpoint(model, optimizer, epoch, path):
    sd = _strip_module_prefix(model.state_dict())
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": sd,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    print(f"[CKPT] Saved universal checkpoint at epoch {epoch}: {path}")


def load_training_checkpoint(model, optimizer, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        print("[CKPT] No training checkpoint found")
        return 0

    print(f"[CKPT] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]

    if isinstance(model, nn.DataParallel):
        sd = _add_module_prefix(sd)
    else:
        sd = _strip_module_prefix(sd)

    model.load_state_dict(sd, strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    start_epoch = int(ckpt["epoch"]) + 1
    print(f"[CKPT] Resuming from epoch {start_epoch}")
    return start_epoch


def load_model_weights_for_inference(
    model,
    weights_path,
    device="cuda",
    strict=True,
    eval_mode=True,
    verbose=True,
):
    device = (
        torch.device(device)
        if not isinstance(device, torch.device)
        else device
    )

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    model = model.to(device)

    if verbose:
        dp_infer = isinstance(model, nn.DataParallel)
        print(f"[WEIGHTS] Loading: {weights_path}")
        print(
            f"[WEIGHTS] Target device: {device}, "
            f"DataParallel model: {dp_infer}, strict={strict}"
        )

    sd = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    if isinstance(model, nn.DataParallel):
        sd = _add_module_prefix(sd)
    else:
        sd = _strip_module_prefix(sd)

    missing, unexpected = model.load_state_dict(sd, strict=strict)

    if verbose:
        if missing:
            suffix = " ..." if len(missing) > 8 else ""
            print(f"[WEIGHTS] Missing keys ({len(missing)}): {missing[:8]}{suffix}")
        if unexpected:
            suffix = " ..." if len(unexpected) > 8 else ""
            print(
                f"[WEIGHTS] Unexpected keys ({len(unexpected)}): "
                f"{unexpected[:8]}{suffix}"
            )
        print("[WEIGHTS] Load OK")

    if eval_mode:
        model.eval()

    return model


@torch.no_grad()
def _eval_loop(model, criterion, loader, device, use_amp=True):
    """Return full validation loss averaged over batches."""
    model.eval()
    val_sum = 0.0
    val_n = 0
    amp_enabled = use_amp and device.type == "cuda"

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        x2d, mu0 = _extract_model_inputs(batch)

        with autocast(device_type="cuda", enabled=amp_enabled):
            mu, sigma = model(x2d, mu0)

        with autocast(device_type="cuda", enabled=False):
            loss = _criterion_from_batch(
                criterion,
                mu,
                sigma,
                batch,
                return_parts=False,
            )

        val_sum += float(loss.detach().item())
        val_n += 1

    return val_sum / max(val_n, 1)


@torch.no_grad()
def _campaign_only_eval(model, criterion, loader, device, use_amp=True):
    """Evaluate only campaign MSE + Gaussian terms on batches that have them."""
    model.eval()
    amp_enabled = use_amp and device.type == "cuda"
    camp_sum = 0.0
    camp_n = 0

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        x2d, mu0 = _extract_model_inputs(batch)

        with autocast(device_type="cuda", enabled=amp_enabled):
            mu, sigma = model(x2d, mu0)

        with autocast(device_type="cuda", enabled=False):
            _, parts = _criterion_from_batch(
                criterion,
                mu,
                sigma,
                batch,
                return_parts=True,
            )

        if float(parts.get("n_campaign_valid", 0.0)) > 0:
            camp_sum += (
                float(parts.get("L_campaign", 0.0))
                + float(parts.get("L_campaign_gauss", 0.0))
            )
            camp_n += 1

    if camp_n == 0:
        return 0.0, 0
    return camp_sum / camp_n, camp_n


def training_val(
    model,
    criterion,
    train_loader,
    val_loader,
    optimizer,
    hp,
    device,
    start_epoch=0,
    use_amp=True,
    clip_grad=0,
):
    scaler = GradScaler(enabled=(use_amp and device.type == "cuda"))
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20,
    )

    lambda_anchor0 = float(getattr(criterion, "lambda_anchor", 0.0))
    anchor_decay = float(hp.get("lambda_anchor_decay", 1.0))
    anchor_min = float(hp.get("lambda_anchor_min", 0.0))

    use_campaign_monitor = bool(hp.get("use_campaign_monitor", True))
    campaign_monitor_beta = float(hp.get("campaign_monitor_beta", 1.0))

    best_monitor = float("inf")
    bad_epochs = 0
    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, hp["nepochs"]):
        if hasattr(criterion, "lambda_anchor") and anchor_decay < 1.0:
            new_anchor = max(
                anchor_min,
                lambda_anchor0 * (anchor_decay ** epoch),
            )
            criterion.lambda_anchor = float(new_anchor)

        model.train()
        t0 = time.perf_counter()
        train_sum = 0.0
        train_upper_sum = 0.0
        train_n = 0
        amp_enabled = use_amp and device.type == "cuda"

        step = 0

        for bi, batch in enumerate(train_loader):
            step += 1
            batch = _move_batch_to_device(batch, device)
            x2d, mu0 = _extract_model_inputs(batch)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=amp_enabled):
                mu, sigma = model(x2d, mu0)

            if not torch.isfinite(mu).all():
                print("[DBG] nonfinite mu", flush=True)
            if not torch.isfinite(sigma).all():
                print("[DBG] nonfinite sigma", flush=True)

            with autocast(device_type="cuda", enabled=False):
                loss, parts = _criterion_from_batch(
                    criterion,
                    mu,
                    sigma,
                    batch,
                    return_parts=True,
                )

            if not torch.isfinite(loss):
                print(
                    "[ERR] non-finite loss detected!",
                    loss.detach().item(),
                    flush=True,
                )

            scaler.scale(loss).backward()

            if step % cfg.print_every == 0:
                msg = (
                    f"[LOSS] step={step:06d} "
                    f"tot={parts['loss']:.5f} "
                    f"site={parts['L_site']:.5f} "
                    f"nd={parts['L_nd']:.5f} "
                    f"ndg={parts['L_nd_gauss']:.5f} "
                    f"camp={parts.get('L_campaign', 0.0):.5f} "
                    f"campg={parts.get('L_campaign_gauss', 0.0):.5f} "
                    f"anc={parts['L_anchor']:.5f} "
                    f"upper={parts.get('L_upper_relax', 0.0):.5f} "
                    f"grad={parts['L_grad']:.5f} "
                    f"sig={parts['L_sig']:.5f} "
                    f"csig={parts.get('L_campaign_sig', 0.0):.5f} "
                    f"sigmean={parts['sig_mean_sites']:.3f} "
                    f"vS={parts['vS_frac']:.3f} "
                    f"ncamp={parts.get('n_campaign_valid', 0.0):.0f}"
                )
                print(msg, flush=True)

                rec = {
                    "event": "loss_step",
                    "run_id": run_id,
                    "job_id": job_id,
                    "rank": rank,
                    "epoch": int(epoch),
                    "step": int(step),
                    "bi": int(bi),
                    "loss": float(parts["loss"]),
                    "L_site": float(parts["L_site"]),
                    "L_nd": float(parts["L_nd"]),
                    "L_nd_gauss": float(parts["L_nd_gauss"]),
                    "L_campaign": float(parts.get("L_campaign", 0.0)),
                    "L_campaign_gauss": float(
                        parts.get("L_campaign_gauss", 0.0)
                    ),
                    "L_anchor": float(parts["L_anchor"]),
                    "L_upper_relax": float(
                        parts.get("L_upper_relax", 0.0)
                    ),
                    "L_upper_relax_raw": float(
                        parts.get("L_upper_relax_raw", 0.0)
                    ),
                    "L_grad": float(parts["L_grad"]),
                    "L_sig": float(parts["L_sig"]),
                    "L_campaign_sig": float(
                        parts.get("L_campaign_sig", 0.0)
                    ),
                    "sig_mean_sites": float(parts["sig_mean_sites"]),
                    "vS_frac": float(parts["vS_frac"]),
                    "n_campaign_valid": float(
                        parts.get("n_campaign_valid", 0.0)
                    ),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
                log_train_event(rec, write_csv=True, write_jsonl=True)

            if clip_grad and clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(clip_grad),
                )

            scaler.step(optimizer)

            if bi == 0:
                print(
                    f"[DBG] grad scaler scale={scaler.get_scale()}",
                    flush=True,
                )

            scaler.update()

            train_sum += float(loss.detach().item())
            train_upper_sum += float(parts.get("L_upper_relax", 0.0))
            train_n += 1

        t_train = time.perf_counter() - t0
        train_loss = train_sum / max(train_n, 1)
        train_upper = train_upper_sum / max(train_n, 1)

        t1 = time.perf_counter()
        val_loss = _eval_loop(
            model,
            criterion,
            val_loader,
            device,
            use_amp=use_amp,
        )
        t_val = time.perf_counter() - t1

        if use_campaign_monitor:
            t2 = time.perf_counter()
            campaign_train_eval_loss, n_campaign_batches = (
                _campaign_only_eval(
                    model,
                    criterion,
                    train_loader,
                    device,
                    use_amp=use_amp,
                )
            )
            t_camp_eval = time.perf_counter() - t2
        else:
            campaign_train_eval_loss = 0.0
            n_campaign_batches = 0
            t_camp_eval = 0.0

        monitor_loss = (
            val_loss
            + campaign_monitor_beta * campaign_train_eval_loss
        )
        scheduler.step(monitor_loss)

        print(
            f"Epoch [{epoch + 1}/{hp['nepochs']}], "
            f"Train: {train_loss:.6f}, "
            f"UpperRelax: {train_upper:.6f}, "
            f"Val: {val_loss:.6f}, "
            f"CampTrainEval: {campaign_train_eval_loss:.6f}, "
            f"Monitor: {monitor_loss:.6f}, "
            f"T(train): {t_train:.2f}s, "
            f"T(val): {t_val:.2f}s, "
            f"T(camp_eval): {t_camp_eval:.2f}s, "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )

        log_train_event(
            {
                "event": "epoch",
                "run_id": run_id,
                "job_id": job_id,
                "rank": rank,
                "epoch": int(epoch),
                "step": int(step),
                "train_loss": float(train_loss),
                "train_upper_relax": float(train_upper),
                "val_loss": float(val_loss),
                "campaign_train_eval_loss": float(
                    campaign_train_eval_loss
                ),
                "campaign_monitor_beta": float(campaign_monitor_beta),
                "monitor_loss": float(monitor_loss),
                "n_campaign_monitor_batches": int(n_campaign_batches),
                "t_train_s": float(t_train),
                "t_val_s": float(t_val),
                "t_campaign_eval_s": float(t_camp_eval),
                "lr": float(optimizer.param_groups[0]["lr"]),
            },
            write_csv=True,
            write_jsonl=True,
        )

        train_losses.append(train_loss)
        val_losses.append(monitor_loss)

        if monitor_loss < best_monitor:
            best_monitor = monitor_loss
            bad_epochs = 0
            save_model_weights(
                model,
                hp["model_name"] + "_weights.pt",
            )
            save_training_checkpoint(
                model,
                optimizer,
                epoch,
                hp["model_name"] + ".ckpt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= hp["patience"]:
                print("[EARLY] Early stopping", flush=True)
                break

    return train_losses, val_losses


@torch.no_grad()
def run_infer_x_mu0(
    device,
    loader,
    model,
    use_amp=True,
    return_sigma=True,
):
    model.eval()
    mu_list = []
    sig_list = []
    amp_enabled = use_amp and device.type == "cuda"

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        x2d, mu0 = _extract_model_inputs(batch)

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
