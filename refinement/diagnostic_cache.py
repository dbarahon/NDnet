"""Utilities for fast repeated site/campaign diagnostics.

The expensive refinement model is independent of the post-inference pressure
closure. These helpers therefore cache *raw*, untapered diagnostic products
and support inference only on days actually needed by site/campaign
diagnostics.

No training behavior is changed here.
"""

from __future__ import annotations

from dataclasses import replace
import glob
import json
import os
import re

import numpy as np
import torch
import xarray as xr

import config as cfg


CACHE_VERSION = "2"
_RAW_SITE_MASK_CACHE = {}


def _safe_token(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _file_stat_attrs(prefix, path):
    """Return cheap file identity metadata suitable for NetCDF attrs."""
    out = {f"{prefix}_path": str(path)}
    try:
        st = os.stat(path)
        out[f"{prefix}_size"] = str(int(st.st_size))
        out[f"{prefix}_mtime_ns"] = str(int(st.st_mtime_ns))
    except OSError:
        out[f"{prefix}_size"] = "missing"
        out[f"{prefix}_mtime_ns"] = "missing"
    return out


def diagnostic_cache_signature(model_name):
    """Metadata that must match before a raw diagnostic cache is reused."""
    weights = os.path.abspath(f"./{model_name}_weights.pt")
    eof_file = os.path.join(str(cfg.BASEROOT), "Nd_EOF.nc")
    mean_file = os.path.join(
        str(cfg.BASEROOT),
        str(cfg.MODEL_BASE) + "_coeff_mean.npy",
    )

    sig = {
        "diagnostic_cache_version": CACHE_VERSION,
        "diagnostic_model_name": str(model_name),
        "diagnostic_model_base": str(cfg.MODEL_BASE),
        "diagnostic_baseroot": str(cfg.BASEROOT),
        "diagnostic_prefix_dense": str(cfg.prefix_dense),
        "diagnostic_prefix_sites": str(cfg.prefix_sites),
        "diagnostic_site_mask_mode": str(cfg.sitemaskmode),
        "diagnostic_holdout_fraction": str(getattr(cfg, "HOLDOUT_FRACTION", 0)),
        "diagnostic_holdout_seed": str(getattr(cfg, "HOLDOUT_SEED", 0)),
        "diagnostic_observation_partition": "holdout_only",
        "diagnostic_bypass_infer": str(bool(cfg.bypass_infer)),
        "diagnostic_site_json": str(cfg.SITE_JSON),
        "diagnostic_campaign_dir": str(
            getattr(cfg, "CAMPAIGN_TRAIN_DIR", "")
        ),
        "diagnostic_campaign_pattern": str(
            getattr(cfg, "CAMPAIGN_PATTERN", "*_training.nc")
        ),
    }
    sig.update(_file_stat_attrs("diagnostic_weights", weights))
    sig.update(_file_stat_attrs("diagnostic_eof", eof_file))
    sig.update(_file_stat_attrs("diagnostic_coeff_mean", mean_file))
    return sig


def diagnostic_cache_path(kind, model_name):
    if kind not in ("site", "campaign"):
        raise ValueError(f"unknown diagnostic cache kind {kind!r}")
    root = str(getattr(cfg, "DIAGNOSTIC_CACHE_DIR", "./cache/diagnostics"))
    return os.path.join(
        root,
        f"{_safe_token(model_name)}_{kind}_raw.nc",
    )


def diagnostic_cache_valid(kind, model_name, verbose=True):
    """Return True only when the cache exists and its scientific inputs match."""
    if not bool(getattr(cfg, "REUSE_DIAGNOSTIC_CACHE", True)):
        return False

    path = diagnostic_cache_path(kind, model_name)
    if not os.path.exists(path):
        if verbose:
            print(f"[DIAG-CACHE] miss: {path}")
        return False

    expected = diagnostic_cache_signature(model_name)
    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            attrs = dict(ds.attrs)
    except Exception as exc:
        if verbose:
            print(f"[DIAG-CACHE] unreadable {path}: {exc!r}")
        return False

    mismatches = []
    for key, value in expected.items():
        if str(attrs.get(key, "")) != str(value):
            mismatches.append(key)

    if mismatches:
        if verbose:
            print(
                f"[DIAG-CACHE] stale: {path}; changed keys: "
                + ", ".join(mismatches)
            )
        return False

    if verbose:
        print(f"[DIAG-CACHE] hit: {path}")
    return True


def load_diagnostic_cache(kind, model_name):
    path = diagnostic_cache_path(kind, model_name)
    with xr.open_dataset(path) as ds:
        out = ds.load()
    print(f"[DIAG-CACHE] loaded raw {kind} diagnostics: {path}")
    return out


def write_diagnostic_cache(ds, kind, model_name):
    """Write an untapered diagnostic dataset atomically enough for cluster use."""
    if not bool(getattr(cfg, "WRITE_DIAGNOSTIC_CACHE", True)):
        return

    path = diagnostic_cache_path(kind, model_name)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    out = ds.copy(deep=False)
    out.attrs = dict(out.attrs)
    out.attrs.update(diagnostic_cache_signature(model_name))
    out.attrs["diagnostic_cache_kind"] = kind
    out.attrs["diagnostic_cache_is_raw_untapered"] = "true"

    tmp = path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    out.to_netcdf(tmp, engine="netcdf4", format="NETCDF4")
    os.replace(tmp, path)
    print(f"[DIAG-CACHE] wrote raw {kind} diagnostics: {path}")


def _resolve_site_nc(var_dir, site, varname, prefix):
    """Resolve the same per-site input file convention used by data_handling."""
    if prefix and (("*" in prefix) or ("?" in prefix) or str(prefix).endswith(".nc")):
        hits = sorted(glob.glob(os.path.join(var_dir, prefix)))
        if hits:
            for h in hits:
                bn = os.path.basename(h)
                if f".{site}." in bn and f".{varname}." in bn:
                    return h
            for h in hits:
                if f".{site}." in os.path.basename(h):
                    return h
            return hits[0]

    if prefix:
        fn = os.path.join(
            var_dir,
            f"Cloudnet.{site}.{varname}.tavg1D.{prefix}.nc",
        )
        if os.path.exists(fn):
            return fn

    hits = sorted(
        glob.glob(os.path.join(var_dir, f"Cloudnet.{site}.{varname}.tavg1D.*.nc"))
    )
    if not hits:
        raise FileNotFoundError(
            f"No per-site file for site={site!r}, var={varname!r} under {var_dir}"
        )
    return hits[0]


def _site_names_from_catalog(path):
    names = []
    if str(path).endswith(".jsonl"):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if "site" in d:
                    names.append(str(d["site"]))
    else:
        with open(path, "r") as f:
            obj = json.load(f)
        names.extend(str(k) for k in obj.keys())
    return np.asarray(names, dtype="U")


def _time_values_from_files(files):
    times = []
    for fn in files:
        try:
            ds = xr.open_dataset(fn, use_cftime=False)
        except TypeError:
            ds = xr.open_dataset(fn)
        try:
            if "time" in ds.coords and ds.sizes.get("time", 0) > 0:
                times.append(np.asarray(ds["time"].values).astype("datetime64[ns]"))
        finally:
            ds.close()
    if not times:
        return np.empty((0,), dtype="datetime64[ns]")
    return np.unique(np.concatenate(times).astype("datetime64[ns]"))


def _diagnostic_template_time_from_config():
    """Reconstruct the same dense daily time intersection used by get_data()."""
    base_var = list(cfg.vars_out.keys())[1]
    base_pattern = os.path.join(cfg.vars_out[base_var], str(cfg.prefix_dense))
    base_files = sorted(glob.glob(base_pattern))
    if not base_files:
        raise FileNotFoundError(f"No base files matching {base_pattern}")
    t_base = _time_values_from_files(base_files)

    modis_pattern = os.path.join(
        str(cfg.pth_all),
        "dailies",
        str(cfg.modis_source),
        "on_Noutgrid",
        f"{cfg.modis_source}.{cfg.prefix_dense}.nc",
    )
    modis_files = sorted(glob.glob(modis_pattern))
    if modis_files:
        t_modis = _time_values_from_files(modis_files)
        if t_modis.size:
            t_base = np.intersect1d(t_base, t_modis)

    return np.asarray(t_base, dtype="datetime64[ns]")


def raw_site_observation_day_mask(template_time, site_meta=None, min_pressure_hpa=250.0):
    """Return a (time, site) mask from the *unfilled* CloudNet site files.

    The training/projection path fills missing site days before projecting to
    EOF coefficients. Consequently ``A_sites``/``VS_t`` are intentionally much
    denser than the actual observations and must not be used to decide which
    days need NN inference for an observation-only diagnostic.
    """
    varname = list(cfg.vars_out.keys())[0]
    var_dir = cfg.vars_out[varname]
    prefix = cfg.prefix_sites
    min_valid_lev = int(getattr(cfg, "min_valid_lev", 1))

    template_values = np.asarray(
        getattr(template_time, "values", template_time)
    ).astype("datetime64[ns]")
    t_template = xr.DataArray(template_values, dims=("time",), name="time")
    ntime = t_template.size

    if site_meta is None:
        site_names = _site_names_from_catalog(cfg.SITE_JSON)
    else:
        site_names = np.asarray(site_meta["site"].values)

    mask = np.zeros((ntime, site_names.size), dtype=bool)

    for si, site in enumerate(site_names):
        fn = _resolve_site_nc(var_dir, str(site), varname, prefix)
        try:
            ds = xr.open_dataset(fn, use_cftime=False)
        except TypeError:
            ds = xr.open_dataset(fn)

        try:
            if varname not in ds:
                raise KeyError(f"{fn} missing variable {varname!r}")

            obs = ds[varname]
            if "lev" in obs.dims and "lev" in ds.coords:
                obs = obs.sel(
                    lev=ds["lev"].where(ds["lev"] >= float(min_pressure_hpa), drop=True)
                )

            finite = np.isfinite(obs)
            valid = finite.sum("lev") >= min_valid_lev if "lev" in finite.dims else finite
            valid = valid.astype("uint8").reindex(time=t_template.values, fill_value=0)
            mask[:, si] = np.asarray(valid.values, dtype=bool)
        finally:
            ds.close()

    n_union = int(mask.any(axis=1).sum())
    print(
        f"[DIAG-INFER] raw site-observation days: {n_union}/{ntime} "
        f"across {site_names.size} sites"
    )
    return mask


def _raw_site_mask_for_current_config(expected_nt):
    key = (
        str(cfg.prefix_dense),
        str(cfg.prefix_sites),
        str(cfg.SITE_JSON),
        int(getattr(cfg, "min_valid_lev", 1)),
        int(expected_nt),
    )
    if key in _RAW_SITE_MASK_CACHE:
        return _RAW_SITE_MASK_CACHE[key]

    template_time = _diagnostic_template_time_from_config()
    if template_time.size != int(expected_nt):
        raise RuntimeError(
            "Raw site-mask template length does not match aligned diagnostic period: "
            f"{template_time.size} vs {expected_nt}"
        )

    mask = raw_site_observation_day_mask(template_time, site_meta=None)
    _RAW_SITE_MASK_CACHE[key] = mask
    return mask


def diagnostic_time_indices(
    *,
    all_idx,
    VS_t,
    campaign_bundle,
    need_sites,
    need_campaigns,
):
    """Return sorted original time indices needed by requested diagnostics.

    Site diagnostics default to actual unfilled CloudNet observation dates.
    This is consistent with ``sitemaskmode='obs_dates'`` and avoids treating
    stochastic fill values used for EOF projection/training as observations.
    Set ``SITE_DIAGNOSTIC_OBS_DAYS_ONLY=False`` in config to restore the dense
    training-mask behavior.
    """
    all_idx = np.asarray(all_idx, dtype=np.int64)
    if not bool(getattr(cfg, "SPARSE_DIAGNOSTIC_INFERENCE", True)):
        print(f"[DIAG-INFER] sparse inference disabled: {all_idx.size} days")
        return all_idx

    selected = []

    if need_sites:
        use_raw_obs = bool(getattr(cfg, "SITE_DIAGNOSTIC_OBS_DAYS_ONLY", True))
        if use_raw_obs:
            try:
                arr = _raw_site_mask_for_current_config(all_idx.size)
            except Exception as exc:
                print(
                    "[DIAG-INFER] WARNING: raw site-day mask failed; "
                    f"falling back to training validity mask: {exc!r}"
                )
                arr = None
        else:
            arr = None

        if arr is None:
            if isinstance(VS_t, torch.Tensor):
                arr = VS_t.detach().cpu().numpy().astype(bool)
            else:
                arr = np.asarray(VS_t).astype(bool)

        site_valid = arr if arr.ndim == 1 else arr.any(axis=1)
        site_idx = all_idx[site_valid[all_idx]]
        selected.append(site_idx)
        print(
            f"[DIAG-INFER] selected site days: {site_idx.size}/{all_idx.size}"
        )

    if need_campaigns:
        if campaign_bundle is None:
            camp_idx = np.empty((0,), dtype=np.int64)
        else:
            counts = campaign_bundle.time_counts.detach().cpu().numpy()
            camp_idx = all_idx[counts[all_idx] > 0]
        selected.append(camp_idx)
        print(
            f"[DIAG-INFER] campaign days: {camp_idx.size}/{all_idx.size}"
        )

    if not selected:
        return np.empty((0,), dtype=np.int64)

    infer_idx = np.unique(np.concatenate(selected)).astype(np.int64)
    print(
        f"[DIAG-INFER] dense NN pass: {infer_idx.size}/{all_idx.size} days "
        f"({100.0 * infer_idx.size / max(all_idx.size, 1):.1f}%)"
    )
    return infer_idx


def sample_sites_from_sparse_grid(
    *,
    coeffs_resid_nn,
    sig_grid,
    infer_idx,
    site_meta,
    full_time_coord,
    mode_coord,
):
    """Sample sparse-day dense predictions at sites into small full-time arrays."""
    coeffs = np.asarray(coeffs_resid_nn, dtype=np.float32)
    infer_idx = np.asarray(infer_idx, dtype=np.int64)
    if coeffs.shape[0] != infer_idx.size:
        raise ValueError(
            "sparse site sampling: prediction count does not match infer_idx: "
            f"{coeffs.shape[0]} vs {infer_idx.size}"
        )

    ilat = site_meta["ilat"].values.astype(np.int64)
    ilon = site_meta["ilon"].values.astype(np.int64)
    site = site_meta["site"].values

    K, M, _, _ = coeffs.shape
    T = len(full_time_coord)
    S = ilat.size

    sampled = coeffs[:, :, ilat, ilon]
    full = np.full((T, M, S), np.nan, dtype=np.float32)
    full[infer_idx, :, :] = sampled

    coords = {
        "time": full_time_coord,
        "mode": mode_coord,
        "site": site,
    }
    A_da = xr.DataArray(
        full,
        dims=("time", "mode", "site"),
        coords=coords,
        name="A_sites_resid",
    )

    if sig_grid is None:
        return A_da, None

    sig = np.asarray(sig_grid, dtype=np.float32)
    if sig.shape[0] != K:
        raise ValueError("sparse site sampling: sigma count does not match predictions")
    sig_sampled = sig[:, :, ilat, ilon]
    sig_full = np.full((T, M, S), np.nan, dtype=np.float32)
    sig_full[infer_idx, :, :] = sig_sampled
    S_da = xr.DataArray(
        sig_full,
        dims=("time", "mode", "site"),
        coords=coords,
        name="sigma_sites",
    )
    return A_da, S_da


def remap_campaign_bundle_for_sparse(bundle, infer_idx):
    """Return a shallow bundle copy whose segment time indices address K predictions."""
    if bundle is None:
        return None

    infer_idx = np.asarray(infer_idx, dtype=np.int64)
    if infer_idx.size == 0:
        raise ValueError("cannot remap campaign bundle with no inference days")

    original_t = bundle.time_idx.detach().cpu().numpy().astype(np.int64)
    max_t = int(max(np.max(infer_idx), np.max(original_t)))
    lookup = np.full((max_t + 1,), -1, dtype=np.int64)
    lookup[infer_idx] = np.arange(infer_idx.size, dtype=np.int64)
    local_t = lookup[original_t]

    if np.any(local_t < 0):
        missing = np.unique(original_t[local_t < 0])
        raise RuntimeError(
            "sparse inference omitted campaign days: "
            + ", ".join(str(int(v)) for v in missing[:20])
        )

    local_tensor = torch.as_tensor(
        local_t,
        dtype=bundle.time_idx.dtype,
        device=bundle.time_idx.device,
    )
    return replace(bundle, time_idx=local_tensor)
