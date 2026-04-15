#!/usr/bin/env python3
"""
prep_campaign_training.py

Create training-ready campaign NetCDF files from segment files like:
  *_segments_1deg_merraP.nc

This version makes each campaign file self-contained for campaign-only training.
In addition to observation-side variables, it also collocates:

  - x2d(channel, segment): MODIS predictor inputs sampled at each campaign segment
  - mu0(mode, segment):    base residual-coefficient input sampled at each segment

So later the campaign loader can train on all campaign dates independently of
the dense/site date window.

What it adds:
  - day(segment): normalized daily timestamp
  - day_index(segment): 0..Nday-1 within file
  - ilat(segment), ilon(segment): nearest template-grid indices from lat_box/lon_box
  - valid_obs(segment): raw observation QC
  - aug(aug): augmentation index
  - z_aug(aug, segment): antithetic standardized perturbations
  - Nd_obs_aug(aug, segment): augmented Nd targets preserving Nd_mean in expectation
  - aug_valid(aug, segment): validity mask for augmented targets

New collocated inputs:
  - channel(channel): names of sampled MODIS predictor variables
  - x2d(channel, segment): collocated MODIS inputs
  - x2d_valid(channel, segment): finite mask for x2d
  - x2d_all_valid(segment): all sampled MODIS channels finite
  - mode(mode): mode coordinate from base coeff file
  - mu0(mode, segment): collocated base residual coefficients
  - mu0_valid(segment): finite mask across modes
  - campaign_input_valid(segment): valid_obs & x2d_all_valid & mu0_valid

Minimal example:
  python prep_campaign_training.py \
    --segments-dir /path/to/field_campaigns/data \
    --out-dir /path/to/campaign_training \
    --modis-dir /path/to/MCD06COSP_D3_MODIS/on_Noutgrid \
    --base-dir /path/to/NDnet_infer_daily
"""

from __future__ import annotations

import argparse
import os
import zlib
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr


# ============================================================
# helpers
# ============================================================

def campaign_name_from_path(path: Path) -> str:
    stem = path.stem
    suffixes = [
        "_segments_1deg_merraP",
        "_segments",
        "_flightdata",
    ]
    for s in suffixes:
        if stem.endswith(s):
            stem = stem[:-len(s)]
    return stem


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_day_ns(time_vals: np.ndarray) -> np.ndarray:
    t = np.asarray(time_vals).astype("datetime64[ns]")
    day = t.astype("datetime64[D]").astype("datetime64[ns]")
    return day


def make_day_index(day_vals: np.ndarray) -> np.ndarray:
    _, inv = np.unique(day_vals, return_inverse=True)
    return inv.astype(np.int32)


def _wrap_longitudes_to_grid(lon: np.ndarray, lon_centers: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, dtype=float).copy()
    lon_centers = np.asarray(lon_centers, dtype=float)

    if lon_centers.min() >= 0.0 and lon_centers.max() > 180.0:
        lon = np.mod(lon, 360.0)
    else:
        lon = ((lon + 180.0) % 360.0) - 180.0

    return lon


def nearest_indices_1d(grid_vals: np.ndarray, x: np.ndarray) -> np.ndarray:
    gv = np.asarray(grid_vals, dtype=float)
    xv = np.asarray(x, dtype=float)
    out = np.empty(xv.shape, dtype=np.int32)
    for i, xi in enumerate(xv):
        out[i] = int(np.argmin(np.abs(gv - xi)))
    return out


def map_segments_to_grid(
    lat_box: np.ndarray,
    lon_box: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_box = np.asarray(lat_box, dtype=float)
    lon_box = np.asarray(lon_box, dtype=float)

    lon_wrapped = _wrap_longitudes_to_grid(lon_box, lon_grid)
    ilat = nearest_indices_1d(lat_grid, lat_box)
    ilon = nearest_indices_1d(lon_grid, lon_wrapped)
    return ilat, ilon, lon_wrapped


def resolve_template_grid_path(
    user_template_grid: Optional[str],
    ds_in: xr.Dataset,
    seg_file: Path,
) -> Path:
    if user_template_grid:
        p = Path(user_template_grid)
        if not p.exists():
            raise FileNotFoundError(f"--template-grid not found: {p}")
        return p

    attr = ds_in.attrs.get("template_grid_file", None)
    if attr:
        p = Path(str(attr))
        if p.exists():
            return p

    raise FileNotFoundError(
        f"No template grid provided for {seg_file.name}. "
        f"Pass --template-grid or include a valid template_grid_file attribute."
    )


def load_template_grid(template_grid: Path) -> Tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(template_grid, decode_times=False) as ds:
        if "lat" not in ds.coords or "lon" not in ds.coords:
            raise KeyError(f"{template_grid} must contain 1D lat/lon coordinates")
        lat = np.asarray(ds["lat"].values, dtype=float)
        lon = np.asarray(ds["lon"].values, dtype=float)
    return lat, lon


def stable_file_seed(base_seed: int, path: Path) -> int:
    crc = zlib.crc32(str(path).encode("utf-8")) & 0xFFFFFFFF
    return int((int(base_seed) + crc) % (2**32 - 1))


def parse_csv_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    out = [x.strip() for x in str(s).split(",") if x.strip()]
    return out if out else None


def make_antithetic_z(
    n_aug: int,
    n_seg: int,
    rng: np.random.Generator,
    zmax: float = 2.0,
) -> np.ndarray:
    if n_aug < 1:
        raise ValueError("n_aug must be >= 1")

    z_list = []

    if n_aug % 2 == 1:
        z_list.append(np.zeros((1, n_seg), dtype=np.float32))
        n_pairs = (n_aug - 1) // 2
    else:
        n_pairs = n_aug // 2

    if n_pairs > 0:
        u = rng.standard_normal(size=(n_pairs, n_seg)).astype(np.float32)
        if zmax is not None and zmax > 0:
            u = np.clip(u, -float(zmax), float(zmax))
        z_list.append(u)
        z_list.append(-u)

    z = np.concatenate(z_list, axis=0)
    assert z.shape == (n_aug, n_seg), f"Unexpected z shape {z.shape}, expected {(n_aug, n_seg)}"
    return z.astype(np.float32)


def build_valid_obs_mask(
    ds: xr.Dataset,
    min_raw: int = 20,
    min_nd: float = 0.0,
) -> np.ndarray:
    required = ["Nd_mean", "Nd_std", "p_merra", "lat_box", "lon_box", "time"]
    for v in required:
        if v not in ds:
            raise KeyError(f"Missing required variable {v!r}")

    Nd_mean = np.asarray(ds["Nd_mean"].values, dtype=float)
    Nd_std = np.asarray(ds["Nd_std"].values, dtype=float)
    p_merra = np.asarray(ds["p_merra"].values, dtype=float)
    lat_box = np.asarray(ds["lat_box"].values, dtype=float)
    lon_box = np.asarray(ds["lon_box"].values, dtype=float)

    m = (
        np.isfinite(Nd_mean) &
        np.isfinite(Nd_std) &
        np.isfinite(p_merra) &
        np.isfinite(lat_box) &
        np.isfinite(lon_box) &
        np.isfinite(np.asarray(ds["time"].values).astype("datetime64[ns]"))
    )

    m &= (Nd_std >= 0.0)
    m &= (Nd_mean >= float(min_nd))

    if "n_raw" in ds:
        n_raw = np.asarray(ds["n_raw"].values, dtype=float)
        m &= np.isfinite(n_raw)
        m &= (n_raw >= int(min_raw))

    return m.astype(bool)


def maybe_copy_var(ds_out: xr.Dataset, ds_in: xr.Dataset, name: str) -> None:
    if name in ds_in:
        ds_out[name] = ds_in[name]


def infer_outfile_name(seg_file: Path, out_dir: Path) -> Path:
    return out_dir / f"{seg_file.stem}_training.nc"


# ============================================================
# MODIS collocation
# ============================================================

def find_modis_month_file(modis_dir: Path, day) -> Optional[Path]:
    yyyymm = np.datetime_as_string(np.asarray(day).astype("datetime64[M]"), unit="M").replace("-", "")
    patterns = [
        f"MCD06COSP_D3_MODIS.{yyyymm}.Noutgrid.nc",
        f"MCD06COSP_D3_MODIS.{yyyymm}.Noutgrid.nc4",
        f"MCD06COSP_D3_MODIS.{yyyymm}.Noutgrid*",
    ]
    for pat in patterns:
        hits = sorted(modis_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def auto_detect_modis_vars(ds: xr.Dataset) -> List[str]:
    vars_out = []
    for v in ds.data_vars:
        dims = tuple(ds[v].dims)
        if len(dims) == 3 and ("time" in dims) and ("lat" in dims) and ("lon" in dims):
            vars_out.append(v)
    if not vars_out:
        raise RuntimeError("No 3D (time,lat,lon) MODIS variables found for auto-detection.")
    return sorted(vars_out)


def sample_modis_inputs_for_segments(
    time_vals,
    lat_box,
    lon_box,
    modis_dir: Path,
    modis_vars: Optional[List[str]] = None,
):
    """
    Sample MODIS inputs at each segment day/lat_box/lon_box.

    Returns
    -------
    x2d : [C,Nseg] float32
    x2d_valid : [C,Nseg] bool
    x2d_all_valid : [Nseg] bool
    channel_names : list[str]
    """
    time_vals = np.asarray(time_vals).astype("datetime64[ns]")
    day_vals = normalize_day_ns(time_vals)
    lat_box = np.asarray(lat_box, dtype=float)
    lon_box = np.asarray(lon_box, dtype=float)

    nseg = day_vals.size
    channel_names = None
    x2d = None

    unique_months = np.unique(day_vals.astype("datetime64[M]"))

    for month in unique_months:
        idx = np.where(day_vals.astype("datetime64[M]") == month)[0]
        if idx.size == 0:
            continue

        modis_file = find_modis_month_file(modis_dir, month)
        if modis_file is None:
            print(f"[WARN] no MODIS monthly file for {str(month)}")
            continue

        with xr.open_dataset(modis_file, decode_times=True) as ds_modis:
            if modis_vars is None:
                vars_use = auto_detect_modis_vars(ds_modis)
            else:
                vars_use = list(modis_vars)
                for v in vars_use:
                    if v not in ds_modis:
                        raise KeyError(f"{v!r} not found in {modis_file}")

            if channel_names is None:
                channel_names = vars_use
                x2d = np.full((len(channel_names), nseg), np.nan, dtype=np.float32)
            else:
                if vars_use != channel_names:
                    raise RuntimeError(
                        f"Inconsistent MODIS variable list across files.\n"
                        f"First file vars={channel_names}\n"
                        f"This file vars={vars_use}"
                    )

            lon_sel = _wrap_longitudes_to_grid(lon_box[idx], np.asarray(ds_modis["lon"].values, dtype=float))
            time_sel = day_vals[idx]

            ds_sel = ds_modis[channel_names].sel(
                time=xr.DataArray(time_sel, dims="segment"),
                lat=xr.DataArray(lat_box[idx], dims="segment"),
                lon=xr.DataArray(lon_sel, dims="segment"),
                method="nearest",
            )

            arr = np.stack(
                [np.asarray(ds_sel[v].values, dtype=np.float32) for v in channel_names],
                axis=0
            )  # [C,Nmonth_seg]

            x2d[:, idx] = arr

    if channel_names is None or x2d is None:
        raise RuntimeError(f"Could not sample any MODIS inputs from {modis_dir}")

    x2d_valid = np.isfinite(x2d)
    x2d_all_valid = x2d_valid.all(axis=0)
    return x2d, x2d_valid, x2d_all_valid, channel_names


# ============================================================
# Base-coefficient collocation
# ============================================================

def find_base_day_file(base_dir: Path, day) -> Optional[Path]:
    ymd = np.datetime_as_string(np.asarray(day).astype("datetime64[D]"), unit="D").replace("-", "")
    patterns = [
        f"NDnet_infer_{ymd}.nc4",
        f"NDnet_infer_{ymd}.nc",
        f"NDnet_infer_{ymd}",
        f"NDnet_infer_{ymd}*.nc4",
        f"NDnet_infer_{ymd}*.nc",
        f"NDnet_infer_{ymd}*",
    ]
    for pat in patterns:
        hits = sorted(base_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def auto_detect_base_var(ds: xr.Dataset) -> str:
    """
    Pick the first variable that contains mode/lat/lon, optionally with time.
    """
    candidates = []
    for v in ds.data_vars:
        dims = set(ds[v].dims)
        if {"mode", "lat", "lon"}.issubset(dims):
            candidates.append(v)

    if not candidates:
        raise RuntimeError("Could not auto-detect a base coeff variable with mode/lat/lon dims.")

    # prefer a variable with time too, but fall back otherwise
    for v in candidates:
        if "time" in ds[v].dims:
            return v
    return candidates[0]


def sample_mu0_for_segments(
    day_vals,
    ilat,
    ilon,
    base_dir: Path,
    base_var: Optional[str] = None,
):
    """
    Sample base residual coefficients mu0(mode,segment) from daily files.

    Returns
    -------
    mu0 : [M,Nseg] float32
    mu0_valid : [Nseg] bool
    mode_coord : [M]
    base_var_used : str
    """
    day_vals = np.asarray(day_vals).astype("datetime64[ns]")
    ilat = np.asarray(ilat, dtype=np.int64)
    ilon = np.asarray(ilon, dtype=np.int64)

    nseg = day_vals.size
    mu0 = None
    mode_coord = None
    base_var_used = None

    unique_days = np.unique(day_vals)

    for day in unique_days:
        idx = np.where(day_vals == day)[0]
        if idx.size == 0:
            continue

        base_file = find_base_day_file(base_dir, day)
        if base_file is None:
            print(f"[WARN] no base coeff file for {str(day)}")
            continue

        with xr.open_dataset(base_file, decode_times=True) as ds_base:
            if base_var is None:
                var_use = auto_detect_base_var(ds_base)
            else:
                var_use = str(base_var)
                if var_use not in ds_base:
                    raise KeyError(f"{var_use!r} not found in {base_file}")

            da = ds_base[var_use]
            if "time" in da.dims:
                da = da.isel(time=0)

            if not {"mode", "lat", "lon"}.issubset(set(da.dims)):
                raise ValueError(
                    f"Base variable {var_use!r} in {base_file} must have mode/lat/lon dims. "
                    f"Got dims={da.dims}"
                )

            if mu0 is None:
                M = int(da.sizes["mode"])
                mu0 = np.full((M, nseg), np.nan, dtype=np.float32)
                if "mode" in da.coords:
                    mode_coord = np.asarray(da["mode"].values)
                else:
                    mode_coord = np.arange(M, dtype=np.int64)
                base_var_used = var_use
            else:
                if da.sizes["mode"] != mu0.shape[0]:
                    raise RuntimeError(
                        f"Inconsistent mode count across base files: first={mu0.shape[0]}, "
                        f"this={da.sizes['mode']} in {base_file}"
                    )

            sel = da.isel(
                lat=xr.DataArray(ilat[idx], dims="segment"),
                lon=xr.DataArray(ilon[idx], dims="segment"),
            ).transpose("mode", "segment")

            mu0[:, idx] = np.asarray(sel.values, dtype=np.float32)

    if mu0 is None or mode_coord is None or base_var_used is None:
        raise RuntimeError(f"Could not sample any base coeff inputs from {base_dir}")

    mu0_valid = np.isfinite(mu0).all(axis=0)
    return mu0, mu0_valid, mode_coord, base_var_used


# ============================================================
# main processing
# ============================================================

def process_one_file(
    seg_file: Path,
    out_dir: Path,
    template_grid_path: Optional[str],
    modis_dir: Path,
    base_dir: Path,
    modis_vars: Optional[List[str]],
    base_var: Optional[str],
    n_aug: int,
    sigma_scale: float,
    zmax: float,
    seed: int,
    min_raw: int,
    min_nd: float,
    std_floor: float,
    overwrite: bool,
) -> Path:
    with xr.open_dataset(seg_file, decode_times=True) as ds_in:
        ds_in = ds_in.load()

    out_file = infer_outfile_name(seg_file, out_dir)
    if out_file.exists() and not overwrite:
        print(f"[SKIP] exists: {out_file}")
        return out_file

    campaign = campaign_name_from_path(seg_file)

    # Resolve template grid
    tpl = resolve_template_grid_path(template_grid_path, ds_in, seg_file)
    lat_grid, lon_grid = load_template_grid(tpl)

    # Basic segment arrays
    n_seg = int(ds_in.sizes["segment"])
    time_vals = np.asarray(ds_in["time"].values).astype("datetime64[ns]")
    day_vals = normalize_day_ns(time_vals)
    day_index = make_day_index(day_vals)

    lat_box = np.asarray(ds_in["lat_box"].values, dtype=np.float64)
    lon_box = np.asarray(ds_in["lon_box"].values, dtype=np.float64)

    ilat, ilon, lon_box_wrapped = map_segments_to_grid(lat_box, lon_box, lat_grid, lon_grid)

    # Raw observation QC
    valid_obs = build_valid_obs_mask(ds_in, min_raw=min_raw, min_nd=min_nd)

    # Augmentation
    file_seed = stable_file_seed(seed, seg_file)
    rng = np.random.default_rng(file_seed)

    z_aug = make_antithetic_z(
        n_aug=n_aug,
        n_seg=n_seg,
        rng=rng,
        zmax=zmax,
    )

    Nd_mean = np.asarray(ds_in["Nd_mean"].values, dtype=np.float32)
    Nd_std = np.asarray(ds_in["Nd_std"].values, dtype=np.float32)

    aug_std = np.maximum(Nd_std, np.float32(std_floor)).astype(np.float32)
    Nd_obs_aug = Nd_mean[None, :] + np.float32(sigma_scale) * aug_std[None, :] * z_aug

    aug_valid = (
        valid_obs[None, :] &
        np.isfinite(Nd_obs_aug) &
        (Nd_obs_aug >= np.float32(min_nd))
    )

    # --------------------------------------------------------
    # NEW: collocated campaign inputs
    # --------------------------------------------------------
    x2d, x2d_valid, x2d_all_valid, channel_names = sample_modis_inputs_for_segments(
        time_vals=time_vals,
        lat_box=lat_box,
        lon_box=lon_box,
        modis_dir=modis_dir,
        modis_vars=modis_vars,
    )

    mu0, mu0_valid, mode_coord, base_var_used = sample_mu0_for_segments(
        day_vals=day_vals,
        ilat=ilat,
        ilon=ilon,
        base_dir=base_dir,
        base_var=base_var,
    )

    campaign_input_valid = valid_obs & x2d_all_valid & mu0_valid

    # --------------------------------------------------------
    # Build output dataset
    # --------------------------------------------------------
    ds_out = xr.Dataset()

    if "segment" in ds_in.coords:
        ds_out = ds_out.assign_coords(segment=ds_in["segment"])
    else:
        ds_out = ds_out.assign_coords(segment=("segment", np.arange(n_seg, dtype=np.int32)))

    ds_out = ds_out.assign_coords(aug=("aug", np.arange(n_aug, dtype=np.int32)))
    ds_out = ds_out.assign_coords(channel=("channel", np.asarray(channel_names, dtype="U")))
    ds_out = ds_out.assign_coords(mode=("mode", np.asarray(mode_coord)))

    for v in [
        "flightID", "lat", "lon", "lat_box", "lon_box",
        "alt", "p_merra", "Nd_mean", "Nd_std", "LWC_mean",
        "n_raw", "time"
    ]:
        maybe_copy_var(ds_out, ds_in, v)

    # Original training helpers
    ds_out["day"] = xr.DataArray(day_vals, dims=("segment",))
    ds_out["day_index"] = xr.DataArray(day_index.astype(np.int32), dims=("segment",))
    ds_out["ilat"] = xr.DataArray(ilat.astype(np.int32), dims=("segment",))
    ds_out["ilon"] = xr.DataArray(ilon.astype(np.int32), dims=("segment",))
    ds_out["lon_box_wrapped"] = xr.DataArray(lon_box_wrapped.astype(np.float32), dims=("segment",))
    ds_out["valid_obs"] = xr.DataArray(valid_obs.astype(np.uint8), dims=("segment",))
    ds_out["z_aug"] = xr.DataArray(z_aug.astype(np.float32), dims=("aug", "segment"))
    ds_out["Nd_obs_aug"] = xr.DataArray(Nd_obs_aug.astype(np.float32), dims=("aug", "segment"))
    ds_out["aug_valid"] = xr.DataArray(aug_valid.astype(np.uint8), dims=("aug", "segment"))

    # NEW: self-contained campaign inputs
    ds_out["x2d"] = xr.DataArray(
        x2d.astype(np.float32),
        dims=("channel", "segment"),
    )
    ds_out["x2d_valid"] = xr.DataArray(
        x2d_valid.astype(np.uint8),
        dims=("channel", "segment"),
    )
    ds_out["x2d_all_valid"] = xr.DataArray(
        x2d_all_valid.astype(np.uint8),
        dims=("segment",),
    )

    ds_out["mu0"] = xr.DataArray(
        mu0.astype(np.float32),
        dims=("mode", "segment"),
    )
    ds_out["mu0_valid"] = xr.DataArray(
        mu0_valid.astype(np.uint8),
        dims=("segment",),
    )
    ds_out["campaign_input_valid"] = xr.DataArray(
        campaign_input_valid.astype(np.uint8),
        dims=("segment",),
    )

    # Attributes
    ds_out.attrs = dict(ds_in.attrs)
    ds_out.attrs.update({
        "campaign_name": campaign,
        "source_segment_file": str(seg_file),
        "template_grid_file_used": str(tpl),
        "preprocessing_script": "prep_campaign_training.py",
        "augmentation_method": "antithetic_gaussian",
        "augmentation_target": "Nd_mean",
        "augmentation_spread": "Nd_std",
        "augmentation_formula": "Nd_obs_aug = Nd_mean + sigma_scale * max(Nd_std,std_floor) * z_aug",
        "augmentation_mean_preserving": "true",
        "sigma_scale": float(sigma_scale),
        "std_floor": float(std_floor),
        "n_aug": int(n_aug),
        "zmax": float(zmax),
        "seed": int(seed),
        "file_seed": int(file_seed),
        "min_raw": int(min_raw),
        "min_nd": float(min_nd),
        "modis_dir": str(modis_dir),
        "base_dir": str(base_dir),
        "modis_vars_sampled": ",".join(channel_names),
        "base_var_sampled": str(base_var_used),
        "note": (
            "This file is self-contained for campaign-only training. "
            "x2d(channel,segment) and mu0(mode,segment) are collocated inputs sampled "
            "at each campaign segment."
        ),
    })

    # Variable attrs
    ds_out["day"].attrs.update({
        "long_name": "normalized day at midnight UTC-compatible wall-clock day",
    })
    ds_out["day_index"].attrs.update({
        "long_name": "0-based day index within campaign file",
    })
    ds_out["ilat"].attrs.update({
        "long_name": "nearest template-grid latitude index from lat_box",
    })
    ds_out["ilon"].attrs.update({
        "long_name": "nearest template-grid longitude index from lon_box",
    })
    ds_out["lon_box_wrapped"].attrs.update({
        "long_name": "lon_box wrapped to template-grid longitude convention",
        "units": "degrees_east",
    })
    ds_out["valid_obs"].attrs.update({
        "long_name": "1 if raw segment passes observation QC, else 0",
        "units": "1",
    })
    ds_out["z_aug"].attrs.update({
        "long_name": "standardized antithetic augmentation perturbation",
        "description": "zero-centered across aug by construction for each segment",
        "units": "1",
    })
    ds_out["Nd_obs_aug"].attrs.update({
        "long_name": "augmented Nd observation target",
        "units": "cm-3",
    })
    ds_out["aug_valid"].attrs.update({
        "long_name": "1 if augmented target is valid for training under preprocessing rules, else 0",
        "units": "1",
    })
    ds_out["x2d"].attrs.update({
        "long_name": "collocated MODIS predictor inputs sampled at campaign segments",
    })
    ds_out["x2d_valid"].attrs.update({
        "long_name": "1 where the sampled MODIS predictor value is finite",
        "units": "1",
    })
    ds_out["x2d_all_valid"].attrs.update({
        "long_name": "1 where all sampled MODIS predictor channels are finite",
        "units": "1",
    })
    ds_out["mu0"].attrs.update({
        "long_name": "collocated base residual-coefficient input sampled at campaign segments",
    })
    ds_out["mu0_valid"].attrs.update({
        "long_name": "1 where all sampled mu0 modes are finite",
        "units": "1",
    })
    ds_out["campaign_input_valid"].attrs.update({
        "long_name": "1 where valid_obs & x2d_all_valid & mu0_valid",
        "units": "1",
    })

    # Encodings
    encoding = {}
    for name, da in ds_out.data_vars.items():
        if np.issubdtype(da.dtype, np.floating):
            encoding[name] = {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}
        elif np.issubdtype(da.dtype, np.integer):
            if da.dtype == np.uint8:
                encoding[name] = {"zlib": True, "complevel": 4}
            elif da.dtype == np.int32:
                encoding[name] = {"zlib": True, "complevel": 4, "_FillValue": np.int32(-1)}
            elif da.dtype == np.int64:
                encoding[name] = {"zlib": True, "complevel": 4, "_FillValue": np.int64(-1)}
            else:
                encoding[name] = {"zlib": True, "complevel": 4}

    ds_out.to_netcdf(out_file, encoding=encoding)

    n_valid = int(valid_obs.sum())
    n_input_valid = int(campaign_input_valid.sum())
    n_aug_valid = int(aug_valid.sum())
    total_aug = int(np.prod(aug_valid.shape))

    print(
        f"[WROTE] {out_file}\n"
        f"        campaign={campaign}  n_seg={n_seg}  n_day={int(day_index.max()) + 1 if n_seg > 0 else 0}\n"
        f"        valid_obs={n_valid}/{n_seg}  input_valid={n_input_valid}/{n_seg}  aug_valid={n_aug_valid}/{total_aug}\n"
        f"        n_chan={len(channel_names)}  n_mode={len(mode_coord)}  base_var={base_var_used}"
    )

    return out_file


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create self-contained campaign training files with collocated inputs and mean-preserving Nd augmentation."
    )
    p.add_argument(
        "--segments-dir",
        type=str,
        default = '/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/field_campaigns/data',
        help="Directory containing *_segments_1deg_merraP.nc files",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default = '/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/field_campaigns/training',
        help="Output directory for campaign training files",
    )
    p.add_argument(
        "--modis-dir",
        type=str,
        default = '/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/dailies/MCD06COSP_D3_MODIS/on_Noutgrid',
        help="Directory with monthly MODIS on_Noutgrid files",
    )
    p.add_argument(
        "--base-dir",
        type=str,
        default='/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/inference/NDnet_infer_daily',
        help="Directory with daily base coeff files",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="*_segments_1deg_merraP.nc",
        help="Glob pattern inside --segments-dir",
    )
    p.add_argument(
        "--template-grid",
        type=str,
        default=None,
        help="Template grid file with 1D lat/lon coords; if omitted, uses input-file global attr template_grid_file",
    )
    p.add_argument(
        "--modis-vars",
        type=str,
        default=None,
        help="Optional comma-separated MODIS variable list. Default: auto-detect all 3D time/lat/lon vars.",
    )
    p.add_argument(
        "--base-var",
        type=str,
        default=None,
        help="Optional base coeff variable name. Default: auto-detect first var with mode/lat/lon dims.",
    )
    p.add_argument(
        "--n-aug",
        type=int,
        default=9,
        help="Number of augmentation members per segment",
    )
    p.add_argument(
        "--sigma-scale",
        type=float,
        default=0.5,
        help="Scale factor applied to Nd_std in augmentation",
    )
    p.add_argument(
        "--zmax",
        type=float,
        default=2.0,
        help="Clip standardized perturbations z to [-zmax, zmax]",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed; each file gets a stable derived seed",
    )
    p.add_argument(
        "--min-raw",
        type=int,
        default=20,
        help="Minimum n_raw required in valid_obs if n_raw exists",
    )
    p.add_argument(
        "--min-nd",
        type=float,
        default=0.0,
        help="Minimum Nd threshold for valid_obs and aug_valid",
    )
    p.add_argument(
        "--std-floor",
        type=float,
        default=0.0,
        help="Floor applied to Nd_std only for augmentation amplitude",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    segments_dir = Path(args.segments_dir)
    out_dir = Path(args.out_dir)
    modis_dir = Path(args.modis_dir)
    base_dir = Path(args.base_dir)

    if not segments_dir.exists():
        raise FileNotFoundError(f"--segments-dir not found: {segments_dir}")
    if not modis_dir.exists():
        raise FileNotFoundError(f"--modis-dir not found: {modis_dir}")
    if not base_dir.exists():
        raise FileNotFoundError(f"--base-dir not found: {base_dir}")

    ensure_dir(out_dir)

    seg_files = sorted(segments_dir.glob(args.pattern))
    if not seg_files:
        raise FileNotFoundError(f"No files matching {args.pattern!r} in {segments_dir}")

    modis_vars = parse_csv_list(args.modis_vars)

    print(f"[INFO] Found {len(seg_files)} files in {segments_dir}")
    for sf in seg_files:
        process_one_file(
            seg_file=sf,
            out_dir=out_dir,
            template_grid_path=args.template_grid,
            modis_dir=modis_dir,
            base_dir=base_dir,
            modis_vars=modis_vars,
            base_var=args.base_var,
            n_aug=args.n_aug,
            sigma_scale=args.sigma_scale,
            zmax=args.zmax,
            seed=args.seed,
            min_raw=args.min_raw,
            min_nd=args.min_nd,
            std_floor=args.std_floor,
            overwrite=args.overwrite,
        )

    print("[DONE] preprocessing complete.")


if __name__ == "__main__":
    main()
