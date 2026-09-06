#!/usr/bin/env python3
"""Single-mode refinement ablation on the configured global test period.

The experiment follows the same test-data path used by refinement_driver.run_global_test
(default: cfg.test_pr, currently *2009*). Inference is run once. For each EOF mode m,
the refined residual coefficient is replaced by the corresponding base/prior residual
coefficient while all other refined modes are left unchanged:

    A_resid^(-m)[:,m,:,:] = A_resid_base[:,m,:,:]

Thus this removes only the learned refinement contribution of mode m; it does NOT set
the absolute EOF coefficient to zero and does not remove that mode from NDnet-prior.

Outputs
-------
mode_ablation_zonal.png
    3 x 4 latitude-pressure panels of annual/time-mean zonal Nd, one panel for
    each ablated mode. This is the main diagnostic for the vertical banding.

mode_ablation_zonal_impact.png
    3 x 4 panels of (ablated - full refined) zonal Nd, using one common
    diverging scale. This isolates the actual contribution removed in each panel.

mode_ablation_zonal.nc
    Full refined zonal mean, all 12 ablated zonal means, and their differences.

mode_ablation_summary.csv
    Mode ranking metrics, including latitude-weighted RMS impact above 700 hPa.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.amp import autocast
from torch.utils.data import DataLoader

import config as cfg
import refinement_driver as core


NROWS = 3
NCOLS = 4
EXPECTED_MODES = NROWS * NCOLS


def _resolve_weights(path_arg: str | None) -> Path:
    if path_arg:
        path = Path(path_arg)
        if not path.exists():
            raise FileNotFoundError(f"Weights not found: {path}")
        return path

    candidates = []
    for name in [
        getattr(cfg, "MODEL_NAME", None),
        cfg.hp.get("model_name") if hasattr(cfg, "hp") else None,
        "NDnet",
    ]:
        if name:
            p = Path(f"{name}_weights.pt")
            if p not in candidates:
                candidates.append(p)

    for path in candidates:
        if path.exists():
            return path

    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not resolve model weights. Pass --weights explicitly. "
        f"Tried: {tried}"
    )


def _weighted_rms_lat(field_zlat: np.ndarray, lat: np.ndarray, lev: np.ndarray, pmax=None) -> float:
    """Latitude-area-weighted RMS, with equal weight across retained EOF levels."""
    f = np.asarray(field_zlat, dtype=np.float64)
    wlat = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
    p = np.asarray(lev, dtype=np.float64)

    if pmax is not None:
        keep = p <= float(pmax)
        f = f[keep, :]

    w = np.broadcast_to(wlat[None, :], f.shape)
    good = np.isfinite(f) & np.isfinite(w) & (w > 0)
    if not np.any(good):
        return np.nan
    return float(np.sqrt(np.sum(w[good] * f[good] ** 2) / np.sum(w[good])))


def _weighted_mean_lat(field_zlat: np.ndarray, lat: np.ndarray, lev: np.ndarray, pmax=None) -> float:
    """Latitude-area-weighted mean, with equal weight across retained EOF levels."""
    f = np.asarray(field_zlat, dtype=np.float64)
    wlat = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
    p = np.asarray(lev, dtype=np.float64)

    if pmax is not None:
        keep = p <= float(pmax)
        f = f[keep, :]

    w = np.broadcast_to(wlat[None, :], f.shape)
    good = np.isfinite(f) & np.isfinite(w) & (w > 0)
    if not np.any(good):
        return np.nan
    return float(np.sum(w[good] * f[good]) / np.sum(w[good]))


def _panel_axes(figsize=(13.0, 10.0)):
    fig, axes = plt.subplots(
        NROWS,
        NCOLS,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    return fig, axes


def _pressure_ticks(lev: np.ndarray) -> np.ndarray:
    standard = np.array([1000, 900, 800, 700, 600, 500, 400, 300, 200], dtype=float)
    lo = float(np.nanmin(lev))
    hi = float(np.nanmax(lev))
    return standard[(standard >= lo) & (standard <= hi)]


def plot_ablation_fields(
    zonal_ablation: np.ndarray,
    lat: np.ndarray,
    lev: np.ndarray,
    out_png: Path,
    *,
    vmin: float = 10.0,
    vmax: float = 1000.0,
):
    """Plot the actual zonal Nd field after removing each refinement mode."""
    fig, axes = _panel_axes()
    norm = mcolors.LogNorm(vmin=float(vmin), vmax=float(vmax))
    p_ticks = _pressure_ticks(lev)
    mappable = None

    for m, ax in enumerate(axes.ravel()):
        z = np.asarray(zonal_ablation[m], dtype=float)
        z = np.where(np.isfinite(z) & (z > 0.0), z, np.nan)
        mappable = ax.pcolormesh(
            lat,
            lev,
            z,
            shading="auto",
            cmap="turbo",
            norm=norm,
        )

        ax.set_ylim(float(np.nanmax(lev)), float(np.nanmin(lev)))
        ax.set_xlim(float(np.nanmin(lat)), float(np.nanmax(lat)))
        ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
        ax.set_yticks(p_ticks)
        ax.tick_params(axis="both", labelsize=8, direction="out", length=3)
        ax.grid(False)

        letter = chr(ord("a") + m)
        ax.set_title(f"({letter}) remove mode {m}", loc="left", fontsize=9, pad=3)

        row, col = divmod(m, NCOLS)
        if row == NROWS - 1:
            ax.set_xlabel("Latitude (°)", fontsize=9)
        if col == 0:
            ax.set_ylabel("Pressure (hPa)", fontsize=9)

    cb = fig.colorbar(
        mappable,
        ax=list(axes.ravel()),
        orientation="horizontal",
        fraction=0.035,
        pad=0.07,
        aspect=45,
    )
    cb.set_label(r"$N_d$ (cm$^{-3}$)", fontsize=10)
    cb.ax.tick_params(labelsize=8)

    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_impact(
    impact: np.ndarray,
    lat: np.ndarray,
    lev: np.ndarray,
    out_png: Path,
    *,
    impact_vmax: float | None = None,
):
    """Plot zonal (ablated - full) Nd for each removed refinement mode."""
    vals = np.asarray(impact, dtype=float)
    finite_abs = np.abs(vals[np.isfinite(vals)])
    if impact_vmax is None:
        if finite_abs.size:
            impact_vmax = float(np.nanpercentile(finite_abs, 99.0))
        else:
            impact_vmax = 1.0
    if not np.isfinite(impact_vmax) or impact_vmax <= 0:
        impact_vmax = 1.0

    fig, axes = _panel_axes()
    norm = mcolors.TwoSlopeNorm(vmin=-impact_vmax, vcenter=0.0, vmax=impact_vmax)
    p_ticks = _pressure_ticks(lev)
    mappable = None

    for m, ax in enumerate(axes.ravel()):
        mappable = ax.pcolormesh(
            lat,
            lev,
            vals[m],
            shading="auto",
            cmap="RdBu_r",
            norm=norm,
        )
        ax.set_ylim(float(np.nanmax(lev)), float(np.nanmin(lev)))
        ax.set_xlim(float(np.nanmin(lat)), float(np.nanmax(lat)))
        ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
        ax.set_yticks(p_ticks)
        ax.tick_params(axis="both", labelsize=8, direction="out", length=3)
        ax.grid(False)

        letter = chr(ord("a") + m)
        ax.set_title(f"({letter}) remove mode {m}", loc="left", fontsize=9, pad=3)

        row, col = divmod(m, NCOLS)
        if row == NROWS - 1:
            ax.set_xlabel("Latitude (°)", fontsize=9)
        if col == 0:
            ax.set_ylabel("Pressure (hPa)", fontsize=9)

    cb = fig.colorbar(
        mappable,
        ax=list(axes.ravel()),
        orientation="horizontal",
        fraction=0.035,
        pad=0.07,
        aspect=45,
    )
    cb.set_label(r"Ablated $-$ full $N_d$ (cm$^{-3}$)", fontsize=10)
    cb.ax.tick_params(labelsize=8)

    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_ablation(
    weights_path: Path,
    test_prefix: str,
    batch_size: int,
    device: torch.device,
):
    print("=" * 72)
    print("GLOBAL EOF-MODE ABLATION")
    print(f"test prefix : {test_prefix}")
    print(f"weights     : {weights_path}")
    print("ablation    : replace one refined residual mode by base residual mode")
    print("=" * 72)

    Nin_2d, A_sites, Nd_sites, Err_sites, Ares_base, site_meta = core.get_data(
        vars_out=cfg.vars_out,
        pth_all=cfg.pth_all,
        vars_in_2d=cfg.vars_in_2d,
        modis_source=cfg.modis_source,
        dense_prefix=test_prefix,
        site_prefix=test_prefix,
        cache_folder="cache",
        minP=250,
        site_json_path=cfg.SITE_JSON,
    )

    X2d_t, MU0_t, YS_t, VS_t, Nds, Eds = core.tensorize_sites_inputs_v2(
        Nin_2d=Nin_2d,
        A_sites=A_sites,
        Ares_base=Ares_base,
        Nd_sites=Nd_sites,
        Err_sites=Err_sites,
        pin_memory=True,
        verbose=True,
    )

    nt = int(Nin_2d.sizes["time"])
    all_idx = np.arange(nt, dtype=np.int64)
    test_ds = core.EOFNdDatasetSitesTorch(
        X2d_t, MU0_t, YS_t, VS_t, Nds, Eds, all_idx
    )
    loader = DataLoader(
        test_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    numgpu = torch.cuda.device_count() if device.type == "cuda" else 0
    model = core.build_model(len(list(Nin_2d.data_vars)), numgpu, device, cfg.hp)
    core.load_model_weights_for_inference(
        model,
        str(weights_path),
        device=device,
        strict=True,
        eval_mode=True,
    )

    eof = np.asarray(cfg.EOF.values, dtype=np.float32)
    nd_mean = np.asarray(cfg.Nd_mean.values, dtype=np.float32)
    a_mean = np.asarray(cfg.A_mean, dtype=np.float32)
    lev = np.asarray(cfg.Nd_mean["lev"].values, dtype=np.float32)
    lat = np.asarray(Nin_2d["lat"].values, dtype=np.float32)

    nmodes, nlev = eof.shape
    if nmodes != EXPECTED_MODES:
        raise ValueError(
            f"This diagnostic is specified as a 3x4 panel and expects 12 modes; got {nmodes}."
        )
    if a_mean.shape[0] < nmodes:
        raise ValueError(f"A_mean has {a_mean.shape[0]} modes, EOF has {nmodes}")

    eof_t = torch.as_tensor(eof, dtype=torch.float32, device=device)
    nd_mean_t = torch.as_tensor(nd_mean, dtype=torch.float32, device=device)
    a_mean_t = torch.as_tensor(a_mean[:nmodes], dtype=torch.float32, device=device)

    # Sum over time and longitude as batches arrive. Keeping only (mode,lev,lat)
    # accumulators avoids materializing 12 full 4-D ablated products.
    sum_full = np.zeros((nlev, lat.size), dtype=np.float64)
    sum_ab = np.zeros((nmodes, nlev, lat.size), dtype=np.float64)
    n_time_seen = 0
    nlon = int(Nin_2d.sizes["lon"])

    amp_enabled = device.type == "cuda"
    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x2d, mu0 = batch[0], batch[1]
            x2d = x2d.to(device, non_blocking=True)
            mu0 = mu0.to(device, non_blocking=True)

            with autocast(device_type="cuda", enabled=amp_enabled):
                mu, _ = model(x2d, mu0)

            # Do the reconstruction in float32 even if model inference used AMP.
            mu = mu.float()
            mu0 = mu0.float()
            a_abs = mu + a_mean_t.unsqueeze(0)
            raw_full = (
                torch.einsum("bmhw,mz->bzhw", a_abs, eof_t)
                + nd_mean_t.view(1, nlev, 1, 1)
            )

            if not torch.isfinite(raw_full).all():
                raise RuntimeError(f"Non-finite full reconstruction in batch {bi}")

            nd_full = torch.clamp(raw_full, min=0.0)
            sum_full += nd_full.sum(dim=(0, 3)).cpu().numpy().astype(np.float64)

            # Exact leave-one-refinement-mode-out reconstruction:
            # replacing mu_m by mu0_m changes raw Nd by (mu0_m-mu_m)*EOF_m(z).
            for m in range(nmodes):
                delta_raw = (
                    (mu0[:, m] - mu[:, m]).unsqueeze(1)
                    * eof_t[m].view(1, nlev, 1, 1)
                )
                nd_ab = torch.clamp(raw_full + delta_raw, min=0.0)
                sum_ab[m] += nd_ab.sum(dim=(0, 3)).cpu().numpy().astype(np.float64)

            n_time_seen += int(mu.shape[0])
            if (bi + 1) % 5 == 0 or n_time_seen == nt:
                print(f"[ABLATE] processed {n_time_seen}/{nt} test days", flush=True)

    if n_time_seen != nt:
        raise RuntimeError(f"Processed {n_time_seen} days but expected {nt}")

    denom = float(n_time_seen * nlon)
    zonal_full = (sum_full / denom).astype(np.float32)
    zonal_ab = (sum_ab / denom).astype(np.float32)
    impact = (zonal_ab - zonal_full[None, :, :]).astype(np.float32)

    return zonal_full, zonal_ab, impact, lev, lat


def build_summary(impact: np.ndarray, lev: np.ndarray, lat: np.ndarray) -> pd.DataFrame:
    rows = []
    for m in range(impact.shape[0]):
        d = impact[m]
        rows.append(
            {
                "mode": m,
                "rms_impact_all_cm3": _weighted_rms_lat(d, lat, lev),
                "rms_impact_p_le_700_cm3": _weighted_rms_lat(d, lat, lev, pmax=700),
                "mean_impact_all_cm3": _weighted_mean_lat(d, lat, lev),
                "mean_impact_p_le_700_cm3": _weighted_mean_lat(d, lat, lev, pmax=700),
                "max_abs_zonal_impact_cm3": float(np.nanmax(np.abs(d))),
            }
        )
    return pd.DataFrame(rows).sort_values("rms_impact_p_le_700_cm3", ascending=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--weights",
        default=None,
        help="Model weights. Default tries config MODEL_NAME/hp model_name and NDnet_weights.pt.",
    )
    p.add_argument(
        "--test-prefix",
        default=getattr(cfg, "test_pr", "*2009*"),
        help="Dense/site file prefix. Default is cfg.test_pr, i.e. the normal global-test period.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=int(getattr(cfg, "hp", {}).get("batch_sz", 16)),
    )
    p.add_argument("--out-dir", type=Path, default=Path("mode_ablation"))
    p.add_argument("--vmin", type=float, default=10.0, help="Lower Nd color limit for ablated fields.")
    p.add_argument("--vmax", type=float, default=1000.0, help="Upper Nd color limit for ablated fields.")
    p.add_argument(
        "--impact-vmax",
        type=float,
        default=None,
        help="Symmetric impact color limit; default is pooled 99th percentile |impact|.",
    )
    args = p.parse_args()

    if args.vmin <= 0 or args.vmax <= args.vmin:
        raise ValueError("Require 0 < --vmin < --vmax for logarithmic Nd plotting")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = _resolve_weights(args.weights)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    zonal_full, zonal_ab, impact, lev, lat = run_ablation(
        weights_path=weights,
        test_prefix=args.test_prefix,
        batch_size=args.batch_size,
        device=device,
    )

    coords = {
        "mode": np.arange(EXPECTED_MODES, dtype=np.int32),
        "lev": lev,
        "lat": lat,
    }
    ds = xr.Dataset(
        {
            "Nd_zonal_full": xr.DataArray(
                zonal_full,
                dims=("lev", "lat"),
                coords={"lev": lev, "lat": lat},
            ),
            "Nd_zonal_ablation": xr.DataArray(
                zonal_ab,
                dims=("mode", "lev", "lat"),
                coords=coords,
            ),
            "Nd_zonal_impact": xr.DataArray(
                impact,
                dims=("mode", "lev", "lat"),
                coords=coords,
            ),
        }
    )
    ds["Nd_zonal_full"].attrs["units"] = "cm-3"
    ds["Nd_zonal_ablation"].attrs["units"] = "cm-3"
    ds["Nd_zonal_impact"].attrs.update(
        units="cm-3",
        long_name="single-mode ablation minus full refined zonal mean Nd",
    )
    ds.attrs.update(
        test_prefix=str(args.test_prefix),
        weights_file=str(weights),
        ablation_definition=(
            "For mode m only, replace refined residual coefficient with base/prior residual; "
            "all other refined residual modes unchanged."
        ),
    )

    nc_path = args.out_dir / "mode_ablation_zonal.nc"
    ds.to_netcdf(nc_path)

    fields_png = args.out_dir / "mode_ablation_zonal.png"
    impact_png = args.out_dir / "mode_ablation_zonal_impact.png"
    plot_ablation_fields(
        zonal_ab,
        lat,
        lev,
        fields_png,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    plot_ablation_impact(
        impact,
        lat,
        lev,
        impact_png,
        impact_vmax=args.impact_vmax,
    )

    summary = build_summary(impact, lev, lat)
    csv_path = args.out_dir / "mode_ablation_summary.csv"
    summary.to_csv(csv_path, index=False, float_format="%.6g")

    print("\nModes ranked by upper-tropospheric zonal RMS impact (p <= 700 hPa):")
    print(summary[["mode", "rms_impact_p_le_700_cm3", "max_abs_zonal_impact_cm3"]].to_string(index=False))
    print("\n[WROTE]")
    print(f"  {fields_png}")
    print(f"  {impact_png}")
    print(f"  {nc_path}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
