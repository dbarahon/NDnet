#!/usr/bin/env python3
from pathlib import Path
import math
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# User settings
# ============================================================
exp= "push8_sigma2"
SEGMENTS_DIR = Path("/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/field_campaigns/data/")          # files from prior script

MODEL_DIR    = Path(f"/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/inference/after_refinement/taper/NDnet_infer_daily/")      # NDnet_infer_YYYYMMDD.nc4 files
#MODEL_DIR    = Path(f"/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/inference/after_refinement/NDnet_infer_daily")      # NDnet_infer_YYYYMMDD.nc4 files
#MODEL_DIR    = Path(f"/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/inference/NDnet_infer_daily/")      # NDnet_infer_YYYYMMDD.nc4 files
print(MODEL_DIR)


MODIS_DIR = Path("/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/dailies/MCD06COSP_D3_MODIS/on_Noutgrid")   # monthly MCD06COSP_D3_MODIS.YYYYMM.Noutgrid.nc files
USE_MODIS_RETRIEVAL_MASK = True        # True -> only compare where MODIS retrieval exists
MODIS_COD_VAR = "Cloud_Optical_Thickness_Liquid_Mean"


INTERP_DIR   = Path("interp_segments")  # output interpolated segment files
PLOT_DIR     = Path("plots")

OUTPUT_DIR   = Path(f"segment_compare_{exp}")  # interpolated files + plots go here

SEG_PATTERN  = "*_segments_1deg_merraP.nc"
MODEL_VAR    = "NDinc_cf01p"
#MODEL_VAR    = "NDinc_cf01p_pred"
SIGMA_VAR    = "sigmaNd"

LOG_SCATTER_AXES = True   # True -> log-log scatter axes
MIN_ND_COMPARE   = 1.0    # set to None to disable Nd minimum for comparison/stats

# Optional plot limits. Set to None for automatic.
XMAX_SCATTER = 1000
YMAX_SCATTER = XMAX_SCATTER
XMIN_SCATTER = 10
YMIN_SCATTER = XMIN_SCATTER



USE_MODIS_RETRIEVAL_MASK = True
MIN_COD_COMPARE = 0.1

# ============================================================
# Helpers
# ============================================================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def campaign_name_from_path(path: Path) -> str:
    stem = path.stem
    suffixes = [
        "_segments_1deg_merraP_modelinterp",
        "_segments_1deg_merraP",
        "_flightdata",
    ]
    for s in suffixes:
        if stem.endswith(s):
            stem = stem[:-len(s)]
    return stem


def _wrap_longitudes_to_grid(lon, lon_centers):
    lon = np.asarray(lon, dtype=float).copy()
    lon_centers = np.asarray(lon_centers, dtype=float)

    if lon_centers.min() >= 0 and lon_centers.max() > 180:
        lon = np.mod(lon, 360.0)
    else:
        lon = ((lon + 180.0) % 360.0) - 180.0

    return lon


def interp_profile_logp_scalar(p_target, p_levels, y_levels):
    """
    Linear interpolation in log-pressure space.
    Returns NaN outside the model pressure range or if insufficient valid data.
    """
    p_target = float(p_target)
    p_levels = np.asarray(p_levels, dtype=float)
    y_levels = np.asarray(y_levels, dtype=float)

    good = np.isfinite(p_levels) & np.isfinite(y_levels) & (p_levels > 0)
    if good.sum() < 2 or not np.isfinite(p_target) or p_target <= 0:
        return np.nan

    pp = p_levels[good]
    yy = y_levels[good]

    pmin = np.min(pp)
    pmax = np.max(pp)
    if p_target < pmin or p_target > pmax:
        return np.nan

    order = np.argsort(np.log(pp))
    x = np.log(pp[order])
    y = yy[order]

    return float(np.interp(np.log(p_target), x, y))


def find_model_file(model_dir: Path, day) -> Path | None:
    ymd = pd.Timestamp(day).strftime("%Y%m%d")
    patterns = [
        f"NDnet_infer_{ymd}.nc4",
        f"NDnet_infer_{ymd}.nc",
        f"NDnet_infer_{ymd}",
        f"NDnet_infer_{ymd}*.nc4",
        f"NDnet_infer_{ymd}*.nc",
        f"NDnet_infer_{ymd}*",
    ]
    for pat in patterns:
        hits = sorted(model_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def build_campaign_day_index(times):
    """
    Map actual dates to campaign-day index 1..N, ignoring calendar gaps.
    """
    day = pd.to_datetime(times).normalize()
    uniq = np.sort(day.unique())
    mapping = {d: i + 1 for i, d in enumerate(uniq)}
    return np.array([mapping[d] for d in day], dtype=int)


def add_within_day_offsets(df: pd.DataFrame, width=0.6):
    """
    Spread multiple same-day segments slightly around the integer campaign day.
    """
    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True)

    grp = df.groupby("campaign_day", sort=False)
    within = grp.cumcount().to_numpy(dtype=float)
    counts = grp["campaign_day"].transform("count").to_numpy(dtype=float)

    offsets = np.zeros(len(df), dtype=float)
    ok = counts > 1
    offsets[ok] = ((within[ok] - 0.5 * (counts[ok] - 1)) / counts[ok]) * width

    df["xplot"] = df["campaign_day"].to_numpy(dtype=float) + offsets
    return df


def layout_for_n(n):
    ncols = 2 if n <= 4 else 3
    nrows = math.ceil(n / ncols)
    return nrows, ncols

def subset_pair(
    df: pd.DataFrame,
    obs_col: str,
    model_col: str,
    require_positive=False,
    min_nd_compare=None,
):
    m = np.isfinite(df[obs_col].values) & np.isfinite(df[model_col].values)

    if require_positive:
        m &= (df[obs_col].values > 0) & (df[model_col].values > 0)

    if min_nd_compare is not None:
        if "Nd_obs" in df.columns:
            m &= np.isfinite(df["Nd_obs"].values) & (df["Nd_obs"].values >= min_nd_compare)
        if "Nd_model" in df.columns:
            m &= np.isfinite(df["Nd_model"].values) & (df["Nd_model"].values >= min_nd_compare)

    if USE_MODIS_RETRIEVAL_MASK and "modis_ok" in df.columns:
        m &= (df["modis_ok"].values == 1)

    return df.loc[m].copy()



def compute_stats(df: pd.DataFrame, obs_col: str, model_col: str):
    x = df[obs_col].to_numpy(dtype=float)
    y = df[model_col].to_numpy(dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan, np.nan, int(m.sum())

    x = x[m]
    y = y[m]

    r = np.corrcoef(x, y)[0, 1]
    bias = np.mean(y - x)   # model - obs
    return float(r), float(bias), int(m.sum())


def global_scatter_limits(dfs, obs_col: str, model_col: str, log_axes=False):
    vals = []
    for df in dfs:
        sub = subset_pair(
            df,
            obs_col,
            model_col,
            require_positive=log_axes,
            min_nd_compare=MIN_ND_COMPARE if obs_col == "Nd_obs" else None,
        )
        if not sub.empty:
            vals.append(sub[obs_col].values)
            vals.append(sub[model_col].values)

    if not vals:
        return (1e-2, 1.0) if log_axes else (0.0, 1.0)

    vals = np.concatenate(vals)
    vals = vals[np.isfinite(vals)]

    if log_axes:
        vals = vals[vals > 0]
        lo = np.nanmin(vals)
        hi = np.nanmax(vals)
        return lo * 0.9, hi * 1.1
    else:
        hi = np.nanmax(vals)
        if XMAX_SCATTER is not None:
            hi = max(hi, XMAX_SCATTER)
        if YMAX_SCATTER is not None:
            hi = max(hi, YMAX_SCATTER)
        return 0.0, hi * 1.05


def find_modis_month_file(modis_dir: Path, day) -> Path | None:
    yyyymm = pd.Timestamp(day).strftime("%Y%m")
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


def sample_modis_cod_for_segments(
    time_vals,
    lat_box,
    lon_box,
    modis_dir: Path,
    cod_var: str = MODIS_COD_VAR,
    min_cod: float = MIN_COD_COMPARE,
):
    """
    Sample MODIS monthly on_Noutgrid files at each segment day/lat_box/lon_box.
    Returns:
        cod_modis : sampled COD
        modis_ok  : COD > min_cod
    """
    time_vals = pd.to_datetime(time_vals)
    day_vals = time_vals.normalize()

    n = len(time_vals)
    cod_modis = np.full(n, np.nan, dtype=np.float32)
    modis_ok = np.zeros(n, dtype=np.int8)

    unique_months = np.unique(day_vals.strftime("%Y-%m"))

    for ym in unique_months:
        month_mask = day_vals.strftime("%Y-%m") == ym
        idx_month = np.where(month_mask)[0]
        if idx_month.size == 0:
            continue

        month_day = pd.Timestamp(day_vals[idx_month[0]])
        modis_file = find_modis_month_file(modis_dir, month_day)
        if modis_file is None:
            print(f"[WARN] No MODIS monthly file for {ym}")
            continue

        with xr.open_dataset(modis_file, decode_times=True) as ds_modis:
            if cod_var not in ds_modis:
                raise KeyError(f"{cod_var!r} not found in {modis_file}")

            # Wrap longitudes to MODIS file convention
            lon_sel = _wrap_longitudes_to_grid(lon_box[idx_month], ds_modis["lon"].values)

            cod = (
                ds_modis[cod_var]
                .sel(
                    time=xr.DataArray(day_vals[idx_month].to_numpy(), dims="segment"),
                    lat=xr.DataArray(lat_box[idx_month], dims="segment"),
                    lon=xr.DataArray(lon_sel, dims="segment"),
                    method="nearest",
                )
                .values
            )

            cod_modis[idx_month] = cod.astype(np.float32)
            modis_ok[idx_month] = (
                np.isfinite(cod_modis[idx_month]) & (cod_modis[idx_month] > min_cod)
            ).astype(np.int8)

    return cod_modis, modis_ok

# ============================================================
# Interpolation to segment files
# ============================================================
def interpolate_one_segment_file(
    segment_file: Path,
    model_dir: Path,
    out_dir: Path,
    model_var: str = MODEL_VAR,
    sigma_var: str = SIGMA_VAR,
):
    campaign = campaign_name_from_path(segment_file)
    out_file = out_dir / f"{segment_file.stem}_modelinterp.nc"

    with xr.open_dataset(segment_file, decode_times=True) as ds_seg:
        time_vals = pd.to_datetime(ds_seg["time"].values)
        lat_box   = ds_seg["lat_box"].values.astype(float)
        lon_box   = ds_seg["lon_box"].values.astype(float)
        p_merra   = ds_seg["p_merra"].values.astype(float)

        nseg = ds_seg.sizes["segment"]

        nd_model    = np.full(nseg, np.nan, dtype=np.float32)
        sigma_model = np.full(nseg, np.nan, dtype=np.float32)

        day_vals = pd.to_datetime(time_vals).normalize()
        unique_days = np.unique(day_vals)

        for day in unique_days:
            idx = np.where(day_vals == day)[0]
            if idx.size == 0:
                continue

            model_file = find_model_file(model_dir, day)
            if model_file is None:
                print(f"[WARN] {campaign}: no model file for {pd.Timestamp(day).strftime('%Y-%m-%d')}")
                continue

            with xr.open_dataset(model_file, decode_times=True) as ds_mod:
                if model_var not in ds_mod:
                    raise KeyError(f"{model_var!r} not found in {model_file}")

                lev = ds_mod["lev"].values.astype(float)
                lon_sel = _wrap_longitudes_to_grid(lon_box[idx], ds_mod["lon"].values)

                prof_nd = (
                    ds_mod[model_var]
                    .isel(time=0)
                    .sel(
                        lat=xr.DataArray(lat_box[idx], dims="segment"),
                        lon=xr.DataArray(lon_sel, dims="segment"),
                        method="nearest",
                    )
                    .transpose("segment", "lev")
                    .values
                )

                has_sigma = sigma_var in ds_mod
                if has_sigma:
                    prof_sig = (
                        ds_mod[sigma_var]
                        .isel(time=0)
                        .sel(
                            lat=xr.DataArray(lat_box[idx], dims="segment"),
                            lon=xr.DataArray(lon_sel, dims="segment"),
                            method="nearest",
                        )
                        .transpose("segment", "lev")
                        .values
                    )

                for j, ii in enumerate(idx):
                    nd_model[ii] = interp_profile_logp_scalar(
                        p_merra[ii], lev, prof_nd[j, :]
                    )
                    if has_sigma:
                        sigma_model[ii] = interp_profile_logp_scalar(
                            p_merra[ii], lev, prof_sig[j, :]
                        )

        # MODIS retrieval-existence mask
        cod_modis, modis_ok = sample_modis_cod_for_segments(
            time_vals=time_vals,
            lat_box=lat_box,
            lon_box=lon_box,
            modis_dir=MODIS_DIR,
            cod_var=MODIS_COD_VAR,
            min_cod=MIN_COD_COMPARE,
        )

        campaign_day = build_campaign_day_index(time_vals).astype(np.int32)

        out = ds_seg.load()
        out["Nd_model"] = (("segment",), nd_model)
        out["sigma_model"] = (("segment",), sigma_model)
        out["campaign_day"] = (("segment",), campaign_day)
        out["cod_modis"] = (("segment",), cod_modis)
        out["modis_ok"] = (("segment",), modis_ok)

        out["Nd_model"].attrs.update({
            "long_name": "Model Nd interpolated to segment pressure",
            "units": "cm-3",
            "source_variable": model_var,
            "interpolation": "same 1-degree cell, vertical linear interpolation in log-pressure",
        })
        out["sigma_model"].attrs.update({
            "long_name": "Model sigma interpolated to segment pressure",
            "units": "cm-3",
            "source_variable": sigma_var,
            "interpolation": "same 1-degree cell, vertical linear interpolation in log-pressure",
        })
        out["campaign_day"].attrs.update({
            "long_name": "Campaign day index (1..N) ignoring calendar gaps"
        })
        out["cod_modis"].attrs.update({
            "long_name": "MODIS Cloud Optical Thickness Liquid Mean sampled at segment day and 1-degree box",
            "units": "none",
        })
        out["modis_ok"].attrs.update({
            "long_name": f"1 if MODIS retrieval exists (COD > {MIN_COD_COMPARE}), else 0",
            "units": "1",
        })

        encoding = {
            "Nd_model": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)},
            "sigma_model": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)},
            "campaign_day": {"zlib": True, "complevel": 4, "_FillValue": np.int32(-1)},
            "cod_modis": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)},
            "modis_ok": {"zlib": True, "complevel": 4, "_FillValue": np.int8(-1)},
        }

        out.to_netcdf(out_file, encoding=encoding)
        out.close()

    print(f"[WROTE] {out_file}")
    return out_file
    

def interpolate_all_segment_files(
    segments_dir: Path,
    model_dir: Path,
    out_dir: Path,
    seg_pattern: str = SEG_PATTERN,
):
    ensure_dir(out_dir)
    seg_files = sorted(segments_dir.glob(seg_pattern))
    if not seg_files:
        raise FileNotFoundError(f"No files matching {seg_pattern} in {segments_dir}")

    out_files = []
    for sf in seg_files:
        print(f"[INTERP] {sf.name}")
        out_files.append(interpolate_one_segment_file(sf, model_dir, out_dir))
    return out_files


# ============================================================
# Plot prep
# ============================================================
def interp_file_to_df(path: Path) -> pd.DataFrame:
    campaign = campaign_name_from_path(path)

    with xr.open_dataset(path, decode_times=True) as ds:
        n = ds.sizes["segment"]

        if "Nd_std" in ds:
            sigma_obs = ds["Nd_std"].values.astype(float)
        else:
            sigma_obs = np.full(n, np.nan, dtype=float)

        if "sigma_model" in ds:
            sigma_model = ds["sigma_model"].values.astype(float)
        else:
            sigma_model = np.full(n, np.nan, dtype=float)

        if "cod_modis" in ds:
            cod_modis = ds["cod_modis"].values.astype(float)
        else:
            cod_modis = np.full(n, np.nan, dtype=float)

        if "modis_ok" in ds:
            modis_ok = ds["modis_ok"].values.astype(int)
        else:
            modis_ok = np.zeros(n, dtype=int)

        df = pd.DataFrame({
            "campaign": campaign,
            "time": pd.to_datetime(ds["time"].values),
            "campaign_day": ds["campaign_day"].values.astype(int),
            "p_merra": ds["p_merra"].values.astype(float),
            "Nd_obs": ds["Nd_mean"].values.astype(float),
            "Nd_model": ds["Nd_model"].values.astype(float),
            "sigma_obs": sigma_obs,
            "sigma_model": sigma_model,
            "cod_modis": cod_modis,
            "modis_ok": modis_ok,
        })

    df = df.sort_values("time").reset_index(drop=True)
    df = add_within_day_offsets(df, width=0.6)
    return df


def load_all_interp_data(interp_files):
    dfs = [interp_file_to_df(p) for p in interp_files]
    dfs = [d for d in dfs if not d.empty]
    if not dfs:
        raise ValueError("No valid interpolated points found for plotting.")
    return dfs


# ============================================================
# Plotting
# ============================================================
def plot_timeseries_panel(
    dfs,
    out_png: Path,
    obs_col: str,
    model_col: str,
    ylabel: str,
    title_text: str,
):
    n = len(dfs)
    nrows, ncols = layout_for_n(n)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6.2 * ncols, 4.0 * nrows),
        sharex=False, sharey=False, constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    p_all = []
    for df in dfs:
        sub = subset_pair(
            df,
            obs_col,
            model_col,
            require_positive=False,
            min_nd_compare=MIN_ND_COMPARE if obs_col == "Nd_obs" else None,
        )
        if not sub.empty:
            p_all.append(sub["p_merra"].values)
    p_all = np.concatenate(p_all) if p_all else np.array([1000.0])

    pmin = np.nanmin(p_all)
    pmax = np.nanmax(p_all)
    last_sc = None

    for ax, df in zip(axes, dfs):
        campaign = df["campaign"].iloc[0]
        sub = subset_pair(
            df,
            obs_col,
            model_col,
            require_positive=False,
            min_nd_compare=MIN_ND_COMPARE if obs_col == "Nd_obs" else None,
        )
        sub = sub.sort_values("xplot").reset_index(drop=True)

        if sub.empty:
            ax.set_title(f"{campaign}\nno valid points")
            ax.set_xlabel("Campaign day")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            continue

        r, bias, npts = compute_stats(sub, obs_col, model_col)

        ax.plot(
            sub["xplot"].values,
            sub[obs_col].values,
            linewidth=1.0,
            alpha=0.6,
            label="Obs",
        )
        ax.plot(
            sub["xplot"].values,
            sub[model_col].values,
            linewidth=1.0,
            alpha=0.6,
            label="Model",
        )

        ax.scatter(
            sub["xplot"].values - 0.04,
            sub[obs_col].values,
            c=sub["p_merra"].values,
            s=24,
            cmap="viridis",
            vmin=pmin,
            vmax=pmax,
            marker="o",
            alpha=0.95,
        )
        last_sc = ax.scatter(
            sub["xplot"].values + 0.04,
            sub[model_col].values,
            c=sub["p_merra"].values,
            s=28,
            cmap="viridis",
            vmin=pmin,
            vmax=pmax,
            marker="x",
            alpha=0.95,
        )

        ax.set_title(f"{campaign}\nr={r:.2f}, bias={bias:.1f}, n={npts}")
        ax.set_xlabel("Campaign day")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

        xmax = int(np.nanmax(sub["campaign_day"].values))
        ax.set_xlim(0.5, xmax + 0.5)

    for ax in axes[n:]:
        ax.remove()

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes[:n], shrink=0.95)
        cbar.set_label("Pressure (hPa)")
        cbar.ax.invert_yaxis()

    fig.suptitle(title_text, fontsize=14)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[WROTE] {out_png}")


def plot_scatter_panel(
    dfs,
    out_png: Path,
    obs_col: str,
    model_col: str,
    xlabel: str,
    ylabel: str,
    title_text: str,
    log_axes: bool = False,
):
    n = len(dfs)
    nrows, ncols = layout_for_n(n)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.4 * ncols, 4.8 * nrows),
        sharex=True, sharey=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    lo, hi = global_scatter_limits(dfs, obs_col, model_col, log_axes=log_axes)
    
    lo, hi = [XMIN_SCATTER, XMAX_SCATTER]

    p_all = []
    for df in dfs:
        sub = subset_pair(
            df,
            obs_col,
            model_col,
            require_positive=log_axes,
            min_nd_compare=MIN_ND_COMPARE if obs_col == "Nd_obs" else None,
        )
        if not sub.empty:
            p_all.append(sub["p_merra"].values)
    p_all = np.concatenate(p_all) if p_all else np.array([1000.0])

    pmin = np.nanmin(p_all)
    pmax = np.nanmax(p_all)
    last_sc = None

    for ax, df in zip(axes, dfs):
        campaign = df["campaign"].iloc[0]
        sub = subset_pair(
            df,
            obs_col,
            model_col,
            require_positive=log_axes,
            min_nd_compare=MIN_ND_COMPARE if obs_col == "Nd_obs" else None,
        )

        if sub.empty:
            ax.set_title(f"{campaign}\nno valid points")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            continue

        r, bias, npts = compute_stats(sub, obs_col, model_col)

        last_sc = ax.scatter(
            sub[obs_col].values,
            sub[model_col].values,
            c=sub["p_merra"].values,
            s=26,
            cmap="viridis",
            vmin=pmin,
            vmax=pmax,
            alpha=0.85,
        )

        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="k")
        ax.set_title(f"{campaign}\nr={r:.2f}, bias={bias:.1f}, n={npts}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        if log_axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    for ax in axes[n:]:
        ax.remove()

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes[:n], shrink=0.95)
        cbar.set_label("Pressure (hPa)")
        cbar.ax.invert_yaxis()

    fig.suptitle(title_text, fontsize=14)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[WROTE] {out_png}")


def plot_scatter_all(
    dfs,
    out_png: Path,
    obs_col: str,
    model_col: str,
    xlabel: str,
    ylabel: str,
    title_text: str,
    log_axes: bool = False,
):
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = subset_pair(
        all_df,
        obs_col,
        model_col,
        require_positive=log_axes,
        min_nd_compare=MIN_ND_COMPARE if obs_col == "Nd_obs" else None,
    )

    if all_df.empty:
        print(f"[INFO] No valid points for {out_png.name}; skipping.")
        return

    r, bias, npts = compute_stats(all_df, obs_col, model_col)
    lo, hi = global_scatter_limits(dfs, obs_col, model_col, log_axes=log_axes)
    lo, hi = [XMIN_SCATTER, XMAX_SCATTER]

    pmin = np.nanmin(all_df["p_merra"].values)
    pmax = np.nanmax(all_df["p_merra"].values)

    fig, ax = plt.subplots(figsize=(6.8, 5.8), constrained_layout=True)

    sc = ax.scatter(
        all_df[obs_col].values,
        all_df[model_col].values,
        c=all_df["p_merra"].values,
        s=22,
        cmap="viridis",
        vmin=pmin,
        vmax=pmax,
        alpha=0.75,
    )

    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="k")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_text}\nr={r:.2f}, bias={bias:.1f}, n={npts}")
    ax.grid(True, alpha=0.3)

    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Pressure (hPa)")
    cbar.ax.invert_yaxis()

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[WROTE] {out_png}")


# ============================================================
# Main
# ============================================================
def main():
    ensure_dir(OUTPUT_DIR)

    interp_files = interpolate_all_segment_files(
        segments_dir=SEGMENTS_DIR,
        model_dir=MODEL_DIR,
        out_dir=OUTPUT_DIR,
        seg_pattern=SEG_PATTERN,
    )

    dfs = load_all_interp_data(interp_files)

    # Nd plots
    plot_timeseries_panel(
        dfs,
        OUTPUT_DIR / "campaign_timeseries_panel.png",
        obs_col="Nd_obs",
        model_col="Nd_model",
        ylabel=r"$N_d$ (cm$^{-3}$)",
        title_text="Segment-wise time series: aircraft vs model",
    )
    plot_scatter_panel(
        dfs,
        OUTPUT_DIR / "campaign_scatter_panel.png",
        obs_col="Nd_obs",
        model_col="Nd_model",
        xlabel=r"Aircraft $N_d$ (cm$^{-3}$)",
        ylabel=r"Model $N_d$ (cm$^{-3}$)",
        title_text="Per-campaign scatter: aircraft vs model",
        log_axes=LOG_SCATTER_AXES,
    )
    plot_scatter_all(
        dfs,
        OUTPUT_DIR / "campaign_scatter_all.png",
        obs_col="Nd_obs",
        model_col="Nd_model",
        xlabel=r"Aircraft $N_d$ (cm$^{-3}$)",
        ylabel=r"Model $N_d$ (cm$^{-3}$)",
        title_text="All campaigns aggregated",
        log_axes=LOG_SCATTER_AXES,
    )

    # Sigma plots
    any_sigma = False
    for df in dfs:
        sub = subset_pair(df, "sigma_obs", "sigma_model", require_positive=False)
        if not sub.empty:
            any_sigma = True
            break

    if any_sigma:
        plot_timeseries_panel(
            dfs,
            OUTPUT_DIR / "campaign_sigma_timeseries_panel.png",
            obs_col="sigma_obs",
            model_col="sigma_model",
            ylabel=r"$\sigma(N_d)$ (cm$^{-3}$)",
            title_text="Segment-wise sigma time series: aircraft vs model",
        )
        plot_scatter_panel(
            dfs,
            OUTPUT_DIR / "campaign_sigma_scatter_panel.png",
            obs_col="sigma_obs",
            model_col="sigma_model",
            xlabel=r"Aircraft $\sigma(N_d)$ (cm$^{-3}$)",
            ylabel=r"Model $\sigma(N_d)$ (cm$^{-3}$)",
            title_text="Per-campaign sigma scatter: aircraft vs model",
            log_axes=LOG_SCATTER_AXES,
        )
        plot_scatter_all(
            dfs,
            OUTPUT_DIR / "campaign_sigma_scatter_all.png",
            obs_col="sigma_obs",
            model_col="sigma_model",
            xlabel=r"Aircraft $\sigma(N_d)$ (cm$^{-3}$)",
            ylabel=r"Model $\sigma(N_d)$ (cm$^{-3}$)",
            title_text="All campaigns aggregated: sigma",
            log_axes=LOG_SCATTER_AXES,
        )
    else:
        print("[INFO] No valid sigma points found; skipping sigma plots.")


if __name__ == "__main__":
    main()
