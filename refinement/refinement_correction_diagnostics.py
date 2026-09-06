#!/usr/bin/env python3
"""Diagnostics for the learned refinement correction using already-saved NetCDFs.

This script is deliberately inference-free. It works from the diagnostic files
written by refinement_driver.py and is intended to answer two questions:

1) Where, in absolute Nd units, does the refined solution depart from the base?
2) Does the refinement improve the site target by using unusually large, rough,
   or cross-mode-incoherent coefficient corrections?

Typical inputs produced by the learned-upper-relax branch are:

  NDnet_global_test.nc
  NDnet_site_test.nc

Either input is optional. If present, the corresponding diagnostics are made.

Global outputs
--------------
global_absolute_correction_zonal.png
    Time-mean, zonal-mean (Nd_refined - Nd_base) in cm-3. No percentage
    normalization is used.

global_absolute_correction_zonal.nc
    Zonal mean correction and zonal RMS correction.

global_absolute_correction_pressure.csv
    Latitude-area-weighted absolute correction metrics by pressure.

Site outputs
------------
site_pressure_training_diagnostics.csv
    Pressure-by-pressure fit and correction metrics. Site-balanced RMSE/bias
    are computed by first evaluating each site and then averaging sites equally.

site_mode_correction_summary.csv
    Per-mode size of the learned coefficient correction, relative to the base
    coefficient scale and, when available, the predicted sigma.

site_coeff_corr_base.csv
site_coeff_corr_refined.csv
site_coeff_corr_delta.csv
site_coeff_corr_difference.csv
    Cross-mode correlation matrices on matched site/time samples.

site_coefficient_manifold_summary.csv
    Global diagnostics of how far refined coefficients move from the empirical
    base-coefficient distribution, including Mahalanobis displacement.

site_vertical_roughness_summary.csv
    RMS first- and second-log-pressure derivatives of base, refined, and
    correction profiles over a configurable pressure range.

site_training_diagnostics.png
    Compact summary of pressure fit, absolute correction, and mode correction.

site_coefficient_correlation.png
    Base/refined/correlation-difference matrices.

Important
---------
NDnet_site_test.nc stores Nd_true from the conditioned/fill-completed site field
used by refinement diagnostics/training, not the original raw sparse observations.
Thus the site-fit diagnostics below describe the training target represented in
that saved file. They should not be interpreted as raw observational coverage.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import xarray as xr


GLOBAL_BASE_CANDIDATES = ("Nd_base",)
GLOBAL_REFINED_CANDIDATES = ("Nd_nn", "Nd_rec", "Nd_ref", "Nd_refined")
SITE_REQUIRED = (
    "Nd_true",
    "Nd_base",
    "Nd_rec",
    "A_sites_resid",
    "A_base_sites_resid",
    "A_sites_abs",
    "A_base_sites_abs",
)


def _first_present(ds: xr.Dataset, names, label: str) -> str:
    for name in names:
        if name in ds:
            return name
    raise KeyError(f"Could not find {label}; tried {list(names)}. Variables={list(ds.data_vars)}")


def _as_hpa(lev: np.ndarray) -> np.ndarray:
    p = np.asarray(lev, dtype=float)
    good = p[np.isfinite(p) & (p > 0)]
    if good.size and float(np.nanmedian(good)) > 2000.0:
        p = p / 100.0
    return p


def _pressure_ticks(lev: np.ndarray) -> np.ndarray:
    std = np.array([1000, 900, 800, 700, 600, 500, 400, 300, 200], dtype=float)
    lo = float(np.nanmin(lev))
    hi = float(np.nanmax(lev))
    return std[(std >= lo) & (std <= hi)]


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good) < 3:
        return np.nan
    xx = x[good]
    yy = y[good]
    if np.nanstd(xx) == 0 or np.nanstd(yy) == 0:
        return np.nan
    return float(np.corrcoef(xx, yy)[0, 1])


def _rmse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x * x))) if x.size else np.nan


def _mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else np.nan


def _mae(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(np.abs(x))) if x.size else np.nan


def _open_lazy(path: Path) -> xr.Dataset:
    try:
        return xr.open_dataset(path, chunks={"time": 16})
    except Exception:
        return xr.open_dataset(path)


def analyze_global(path: Path, out_dir: Path, vmax: float | None, percentile: float) -> None:
    print(f"[GLOBAL] reading {path}")
    ds = _open_lazy(path)
    base_name = _first_present(ds, GLOBAL_BASE_CANDIDATES, "global base Nd")
    ref_name = _first_present(ds, GLOBAL_REFINED_CANDIDATES, "global refined Nd")

    base = ds[base_name]
    ref = ds[ref_name]
    for d in ("time", "lev", "lat", "lon"):
        if d not in base.dims or d not in ref.dims:
            raise ValueError(f"Global Nd fields must contain {d!r}; base={base.dims}, refined={ref.dims}")

    delta = (ref - base).rename("Nd_refined_minus_base")

    zonal_mean = delta.mean(("time", "lon"), skipna=True).compute()
    zonal_rms = np.sqrt((delta * delta).mean(("time", "lon"), skipna=True)).compute()

    lev = _as_hpa(zonal_mean["lev"].values)
    lat = np.asarray(zonal_mean["lat"].values, dtype=float)
    zm = np.asarray(zonal_mean.values, dtype=float)

    finite_abs = np.abs(zm[np.isfinite(zm)])
    if vmax is None:
        vmax_use = float(np.nanpercentile(finite_abs, percentile)) if finite_abs.size else 1.0
    else:
        vmax_use = float(vmax)
    if not np.isfinite(vmax_use) or vmax_use <= 0:
        vmax_use = 1.0

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax_use, vcenter=0.0, vmax=vmax_use)
    im = ax.pcolormesh(lat, lev, zm, shading="auto", cmap="RdBu_r", norm=norm)
    ax.set_ylim(float(np.nanmax(lev)), float(np.nanmin(lev)))
    ax.set_xlim(float(np.nanmin(lat)), float(np.nanmax(lat)))
    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_yticks(_pressure_ticks(lev))
    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_title(r"Time-mean zonal $N_d^{refined}-N_d^{base}$", loc="left")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.12, aspect=35)
    cb.set_label(r"Absolute correction (cm$^{-3}$)")
    fig.savefig(out_dir / "global_absolute_correction_zonal.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    xr.Dataset(
        {
            "Nd_correction_zonal_mean": zonal_mean.assign_coords(lev=("lev", lev)),
            "Nd_correction_zonal_rms": zonal_rms.assign_coords(lev=("lev", lev)),
        }
    ).to_netcdf(out_dir / "global_absolute_correction_zonal.nc")

    wlat = xr.DataArray(np.cos(np.deg2rad(ds["lat"].values)), dims=("lat",), coords={"lat": ds["lat"]})
    mean_delta = delta.weighted(wlat).mean(("time", "lat", "lon"), skipna=True).compute()
    mean_abs = np.abs(delta).weighted(wlat).mean(("time", "lat", "lon"), skipna=True).compute()
    rms_delta = np.sqrt((delta * delta).weighted(wlat).mean(("time", "lat", "lon"), skipna=True)).compute()
    mean_base = base.weighted(wlat).mean(("time", "lat", "lon"), skipna=True).compute()
    mean_ref = ref.weighted(wlat).mean(("time", "lat", "lon"), skipna=True).compute()

    gdf = pd.DataFrame(
        {
            "pressure_hpa": _as_hpa(mean_delta["lev"].values),
            "mean_base_cm3": np.asarray(mean_base.values, float),
            "mean_refined_cm3": np.asarray(mean_ref.values, float),
            "mean_correction_cm3": np.asarray(mean_delta.values, float),
            "mean_abs_correction_cm3": np.asarray(mean_abs.values, float),
            "rms_correction_cm3": np.asarray(rms_delta.values, float),
        }
    ).sort_values("pressure_hpa", ascending=False)
    gdf.to_csv(out_dir / "global_absolute_correction_pressure.csv", index=False, float_format="%.6g")
    ds.close()
    print("[GLOBAL] wrote zonal absolute-correction diagnostics")


def _site_pressure_metrics(ds: xr.Dataset) -> pd.DataFrame:
    true = np.asarray(ds["Nd_true"].values, dtype=float)
    base = np.asarray(ds["Nd_base"].values, dtype=float)
    ref = np.asarray(ds["Nd_rec"].values, dtype=float)
    lev = _as_hpa(ds["lev"].values)
    _, Z, S = true.shape

    if "site_mask" in ds:
        site_mask = np.asarray(ds["site_mask"].values, dtype=bool)
    else:
        site_mask = np.isfinite(true).any(axis=1)

    rows = []
    for iz in range(Z):
        site_base_bias = []
        site_ref_bias = []
        site_base_rmse = []
        site_ref_rmse = []
        site_corr_mean = []
        site_corr_rms = []
        site_corr_mae = []
        npoints = 0
        nsites = 0
        pooled_true = []
        pooled_base = []
        pooled_ref = []

        for s in range(S):
            good = (
                site_mask[:, s]
                & np.isfinite(true[:, iz, s])
                & np.isfinite(base[:, iz, s])
                & np.isfinite(ref[:, iz, s])
            )
            n = int(np.count_nonzero(good))
            if n < 1:
                continue
            nsites += 1
            npoints += n
            y = true[good, iz, s]
            b = base[good, iz, s]
            r = ref[good, iz, s]
            d = r - b
            site_base_bias.append(_mean(b - y))
            site_ref_bias.append(_mean(r - y))
            site_base_rmse.append(_rmse(b - y))
            site_ref_rmse.append(_rmse(r - y))
            site_corr_mean.append(_mean(d))
            site_corr_rms.append(_rmse(d))
            site_corr_mae.append(_mae(d))
            pooled_true.append(y)
            pooled_base.append(b)
            pooled_ref.append(r)

        if pooled_true:
            yp = np.concatenate(pooled_true)
            bp = np.concatenate(pooled_base)
            rp = np.concatenate(pooled_ref)
            pooled_base_rmse = _rmse(bp - yp)
            pooled_ref_rmse = _rmse(rp - yp)
        else:
            yp = bp = rp = np.array([], dtype=float)
            pooled_base_rmse = pooled_ref_rmse = np.nan

        sb = _mean(np.asarray(site_base_rmse))
        sr = _mean(np.asarray(site_ref_rmse))
        improvement = 100.0 * (sb - sr) / sb if np.isfinite(sb) and sb > 0 else np.nan
        rows.append(
            {
                "pressure_hpa": lev[iz],
                "n_sites": nsites,
                "n_points": npoints,
                "base_bias_site_balanced_cm3": _mean(np.asarray(site_base_bias)),
                "refined_bias_site_balanced_cm3": _mean(np.asarray(site_ref_bias)),
                "base_rmse_site_balanced_cm3": sb,
                "refined_rmse_site_balanced_cm3": sr,
                "rmse_improvement_site_balanced_pct": improvement,
                "base_rmse_pooled_cm3": pooled_base_rmse,
                "refined_rmse_pooled_cm3": pooled_ref_rmse,
                "obs_base_corr_pooled": _safe_corr(yp, bp),
                "obs_refined_corr_pooled": _safe_corr(yp, rp),
                "correction_mean_site_balanced_cm3": _mean(np.asarray(site_corr_mean)),
                "correction_rms_site_balanced_cm3": _mean(np.asarray(site_corr_rms)),
                "correction_mae_site_balanced_cm3": _mean(np.asarray(site_corr_mae)),
            }
        )

    return pd.DataFrame(rows).sort_values("pressure_hpa", ascending=False)


def _flatten_mode_samples(da: xr.DataArray, mask_ts: np.ndarray | None = None) -> np.ndarray:
    a = np.asarray(da.transpose("time", "mode", "site").values, dtype=float)
    if mask_ts is not None:
        a = np.where(mask_ts[:, None, :], a, np.nan)
    return np.transpose(a, (0, 2, 1)).reshape(-1, a.shape[1])


def _site_mode_metrics(ds: xr.Dataset, mask_ts: np.ndarray) -> pd.DataFrame:
    ref = np.asarray(ds["A_sites_resid"].transpose("time", "mode", "site").values, dtype=float)
    base = np.asarray(ds["A_base_sites_resid"].transpose("time", "mode", "site").values, dtype=float)
    delta = ref - base
    sigma = None
    if "sigma_sites" in ds:
        sigma = np.asarray(ds["sigma_sites"].transpose("time", "mode", "site").values, dtype=float)

    modes = np.asarray(ds["mode"].values)
    rows = []
    for im in range(ref.shape[1]):
        good = mask_ts & np.isfinite(ref[:, im, :]) & np.isfinite(base[:, im, :])
        d = delta[:, im, :][good]
        b = base[:, im, :][good]
        r = ref[:, im, :][good]
        base_rms = _rmse(b)
        delta_rms = _rmse(d)
        row = {
            "mode": modes[im],
            "n_samples": int(d.size),
            "delta_mean": _mean(d),
            "delta_std": float(np.nanstd(d)) if d.size else np.nan,
            "delta_rms": delta_rms,
            "delta_mae": _mae(d),
            "delta_abs_p95": float(np.nanpercentile(np.abs(d), 95.0)) if d.size else np.nan,
            "base_resid_rms": base_rms,
            "refined_resid_rms": _rmse(r),
            "delta_rms_over_base_resid_rms": delta_rms / base_rms if np.isfinite(base_rms) and base_rms > 0 else np.nan,
            "corr_base_refined_resid": _safe_corr(b, r),
            "corr_base_delta": _safe_corr(b, d),
        }
        if sigma is not None:
            sg = sigma[:, im, :][good]
            good_sig = np.isfinite(sg) & (sg > 0) & np.isfinite(d)
            if np.any(good_sig):
                ratio = np.abs(d[good_sig]) / sg[good_sig]
                row["median_abs_delta_over_sigma"] = float(np.nanmedian(ratio))
                row["p95_abs_delta_over_sigma"] = float(np.nanpercentile(ratio, 95.0))
            else:
                row["median_abs_delta_over_sigma"] = np.nan
                row["p95_abs_delta_over_sigma"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _covariance_manifold_diagnostics(ds: xr.Dataset, mask_ts: np.ndarray, out_dir: Path) -> pd.DataFrame:
    Xb = _flatten_mode_samples(ds["A_base_sites_abs"], mask_ts)
    Xr = _flatten_mode_samples(ds["A_sites_abs"], mask_ts)
    paired = np.all(np.isfinite(Xb), axis=1) & np.all(np.isfinite(Xr), axis=1)
    Xb = Xb[paired]
    Xr = Xr[paired]
    M = Xb.shape[1] if Xb.ndim == 2 else int(ds.sizes["mode"])
    modes = [str(v) for v in np.asarray(ds["mode"].values)]

    if Xb.shape[0] < max(10, M + 2):
        warnings.warn("Too few complete coefficient vectors for covariance diagnostics")
        nanm = np.full((M, M), np.nan)
        for name in ("base", "refined", "delta", "difference"):
            pd.DataFrame(nanm, index=modes, columns=modes).to_csv(out_dir / f"site_coeff_corr_{name}.csv")
        return pd.DataFrame([{"n_complete_vectors": int(Xb.shape[0])}])

    Xd = Xr - Xb
    Cb = np.cov(Xb, rowvar=False)
    Cr = np.cov(Xr, rowvar=False)
    Cb_corr = np.corrcoef(Xb, rowvar=False)
    Cr_corr = np.corrcoef(Xr, rowvar=False)
    Cd_corr = np.corrcoef(Xd, rowvar=False)
    CorrDiff = Cr_corr - Cb_corr

    pd.DataFrame(Cb_corr, index=modes, columns=modes).to_csv(out_dir / "site_coeff_corr_base.csv")
    pd.DataFrame(Cr_corr, index=modes, columns=modes).to_csv(out_dir / "site_coeff_corr_refined.csv")
    pd.DataFrame(Cd_corr, index=modes, columns=modes).to_csv(out_dir / "site_coeff_corr_delta.csv")
    pd.DataFrame(CorrDiff, index=modes, columns=modes).to_csv(out_dir / "site_coeff_corr_difference.csv")

    ridge = 1e-6 * float(np.trace(Cb) / max(M, 1))
    Cinv = np.linalg.pinv(Cb + ridge * np.eye(M), rcond=1e-8)
    d2 = np.einsum("ni,ij,nj->n", Xd, Cinv, Xd)
    d2 = np.where(np.isfinite(d2) & (d2 >= 0), d2, np.nan)
    md = np.sqrt(d2)

    mu = np.mean(Xb, axis=0)
    rb = np.sqrt(np.maximum(0.0, np.einsum("ni,ij,nj->n", Xb - mu, Cinv, Xb - mu)))
    rr = np.sqrt(np.maximum(0.0, np.einsum("ni,ij,nj->n", Xr - mu, Cinv, Xr - mu)))

    denom_cov = np.linalg.norm(Cb, ord="fro")
    denom_corr = np.linalg.norm(Cb_corr, ord="fro")
    row = {
        "n_complete_vectors": int(Xb.shape[0]),
        "n_modes": M,
        "covariance_frobenius_relative_change": float(np.linalg.norm(Cr - Cb, ord="fro") / denom_cov) if denom_cov > 0 else np.nan,
        "correlation_frobenius_relative_change": float(np.linalg.norm(CorrDiff, ord="fro") / denom_corr) if denom_corr > 0 else np.nan,
        "mahalanobis_delta_median": float(np.nanmedian(md)),
        "mahalanobis_delta_p90": float(np.nanpercentile(md, 90.0)),
        "mahalanobis_delta_p95": float(np.nanpercentile(md, 95.0)),
        "mahalanobis_delta_p99": float(np.nanpercentile(md, 99.0)),
        "base_mahalanobis_radius_median": float(np.nanmedian(rb)),
        "refined_mahalanobis_radius_median": float(np.nanmedian(rr)),
        "refined_to_base_median_radius_ratio": float(np.nanmedian(rr) / np.nanmedian(rb)) if np.nanmedian(rb) > 0 else np.nan,
    }

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    mats = [Cb_corr, Cr_corr, CorrDiff]
    titles = ["Base coefficient correlation", "Refined coefficient correlation", "Refined - base correlation"]
    norms = [mcolors.Normalize(-1, 1), mcolors.Normalize(-1, 1), mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)]
    ims = []
    for ax, mat, title, norm in zip(axes, mats, titles, norms):
        im = ax.imshow(mat, origin="lower", cmap="RdBu_r", norm=norm, aspect="equal")
        ims.append(im)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Mode")
        ax.set_ylabel("Mode")
        ax.set_xticks(range(M))
        ax.set_yticks(range(M))
        ax.tick_params(labelsize=7)
    fig.colorbar(ims[1], ax=axes[:2], orientation="horizontal", fraction=0.07, pad=0.13, label="Correlation")
    fig.colorbar(ims[2], ax=axes[2], orientation="horizontal", fraction=0.07, pad=0.13, label="Correlation change")
    fig.savefig(out_dir / "site_coefficient_correlation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame([row])


def _vertical_roughness(ds: xr.Dataset, mask_ts: np.ndarray, pmin: float, pmax: float) -> pd.DataFrame:
    lev = _as_hpa(ds["lev"].values)
    order = np.argsort(np.log(lev))
    p = lev[order]
    x = np.log(p)
    keep = (p >= float(pmin)) & (p <= float(pmax))

    out = {}
    for label, var in [("base", "Nd_base"), ("refined", "Nd_rec")]:
        a = np.asarray(ds[var].values, dtype=float)[:, order, :]
        a = np.where(mask_ts[:, None, :], a, np.nan)
        d1 = np.gradient(a, x, axis=1, edge_order=1)
        d2 = np.gradient(d1, x, axis=1, edge_order=1)
        out[f"{label}_rms_dNd_dlnp"] = _rmse(d1[:, keep, :])
        out[f"{label}_rms_d2Nd_dlnp2"] = _rmse(d2[:, keep, :])

    a_ref = np.asarray(ds["Nd_rec"].values, dtype=float)[:, order, :]
    a_base = np.asarray(ds["Nd_base"].values, dtype=float)[:, order, :]
    d = np.where(mask_ts[:, None, :], a_ref - a_base, np.nan)
    d1 = np.gradient(d, x, axis=1, edge_order=1)
    d2 = np.gradient(d1, x, axis=1, edge_order=1)
    out["correction_rms_dNd_dlnp"] = _rmse(d1[:, keep, :])
    out["correction_rms_d2Nd_dlnp2"] = _rmse(d2[:, keep, :])
    out["refined_to_base_d1_rms_ratio"] = out["refined_rms_dNd_dlnp"] / out["base_rms_dNd_dlnp"] if out["base_rms_dNd_dlnp"] > 0 else np.nan
    out["refined_to_base_d2_rms_ratio"] = out["refined_rms_d2Nd_dlnp2"] / out["base_rms_d2Nd_dlnp2"] if out["base_rms_d2Nd_dlnp2"] > 0 else np.nan
    out["pressure_min_hpa"] = float(pmin)
    out["pressure_max_hpa"] = float(pmax)
    return pd.DataFrame([out])


def _plot_site_summary(pressure_df: pd.DataFrame, mode_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), constrained_layout=True)

    p = pressure_df["pressure_hpa"].to_numpy(float)
    ax = axes[0]
    ax.plot(pressure_df["base_rmse_site_balanced_cm3"], p, label="Base")
    ax.plot(pressure_df["refined_rmse_site_balanced_cm3"], p, label="Refined")
    ax.set_ylim(np.nanmax(p), np.nanmin(p))
    ax.set_xlabel(r"Site-balanced RMSE (cm$^{-3}$)")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_title("Fit to saved site target", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(pressure_df["correction_rms_site_balanced_cm3"], p)
    ax.set_ylim(np.nanmax(p), np.nanmin(p))
    ax.set_xlabel(r"RMS refined - base (cm$^{-3}$)")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_title("Absolute site correction", loc="left", fontsize=9)

    ax = axes[2]
    ax.bar(mode_df["mode"].astype(str), mode_df["delta_rms"])
    ax.set_xlabel("Mode")
    ax.set_ylabel("RMS coefficient correction")
    ax.set_title("Learned correction by mode", loc="left", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)

    fig.savefig(out_dir / "site_training_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_sites(path: Path, out_dir: Path, rough_pmin: float, rough_pmax: float) -> None:
    print(f"[SITE] reading {path}")
    with xr.open_dataset(path) as ds:
        missing = [v for v in SITE_REQUIRED if v not in ds]
        if missing:
            raise KeyError(f"{path} missing required variables: {missing}")

        if "site_mask" in ds:
            mask_ts = np.asarray(ds["site_mask"].values, dtype=bool)
        else:
            mask_ts = np.isfinite(np.asarray(ds["Nd_true"].values, dtype=float)).any(axis=1)

        pressure_df = _site_pressure_metrics(ds)
        pressure_df.to_csv(out_dir / "site_pressure_training_diagnostics.csv", index=False, float_format="%.6g")

        mode_df = _site_mode_metrics(ds, mask_ts)
        mode_df.to_csv(out_dir / "site_mode_correction_summary.csv", index=False, float_format="%.6g")

        manifold_df = _covariance_manifold_diagnostics(ds, mask_ts, out_dir)
        manifold_df.to_csv(out_dir / "site_coefficient_manifold_summary.csv", index=False, float_format="%.6g")

        rough_df = _vertical_roughness(ds, mask_ts, rough_pmin, rough_pmax)
        rough_df.to_csv(out_dir / "site_vertical_roughness_summary.csv", index=False, float_format="%.6g")

        _plot_site_summary(pressure_df, mode_df, out_dir)

    print("[SITE] wrote training-formulation diagnostics")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--global-test", type=Path, default=Path("NDnet_global_test.nc"),
                    help="Saved global test NetCDF. Missing file is skipped.")
    ap.add_argument("--site-test", type=Path, default=Path("NDnet_site_test.nc"),
                    help="Saved site diagnostic NetCDF. Missing file is skipped.")
    ap.add_argument("--out-dir", type=Path, default=Path("refinement_correction_diagnostics"))
    ap.add_argument("--global-vmax", type=float, default=None,
                    help="Symmetric cm-3 limit for absolute correction zonal plot. Default uses percentile.")
    ap.add_argument("--global-percentile", type=float, default=99.0,
                    help="Percentile of |zonal mean correction| used when --global-vmax is omitted.")
    ap.add_argument("--rough-pmin", type=float, default=400.0,
                    help="Lower pressure bound (hPa) for site vertical-roughness summary.")
    ap.add_argument("--rough-pmax", type=float, default=900.0,
                    help="Upper pressure bound (hPa) for site vertical-roughness summary.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    did_any = False

    if args.global_test.exists():
        analyze_global(args.global_test, args.out_dir, args.global_vmax, args.global_percentile)
        did_any = True
    else:
        print(f"[SKIP] global file not found: {args.global_test}")

    if args.site_test.exists():
        analyze_sites(args.site_test, args.out_dir, args.rough_pmin, args.rough_pmax)
        did_any = True
    else:
        print(f"[SKIP] site file not found: {args.site_test}")

    if not did_any:
        raise FileNotFoundError("Neither global nor site diagnostic file was found")

    print(f"[DONE] outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
