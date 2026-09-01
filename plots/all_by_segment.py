#!/usr/bin/env python3
from pathlib import Path
import math

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# User settings
# ============================================================
exp = "push8_sigma2"
SEGMENTS_DIR = Path(
    "/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/field_campaigns/data/"
)

MODEL_DIR = Path(
    "/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/"
    "inference/after_refinement/taper/NDnet_infer_daily/"
)
# MODEL_DIR = Path(
#     "/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/"
#     "inference/after_refinement/NDnet_infer_daily"
# )
# MODEL_DIR = Path(
#     "/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/"
#     "inference/NDnet_infer_daily/"
# )
print(MODEL_DIR)

MODIS_DIR = Path(
    "/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/"
    "dailies/MCD06COSP_D3_MODIS/on_Noutgrid"
)
USE_MODIS_RETRIEVAL_MASK = True
MODIS_COD_VAR = "Cloud_Optical_Thickness_Liquid_Mean"

INTERP_DIR = Path("interp_segments")
PLOT_DIR = Path("plots")
OUTPUT_DIR = Path(f"segment_compare_{exp}")

SEG_PATTERN = "*_segments_1deg_merraP.nc"
MODEL_VAR = "NDinc_cf01p"
# MODEL_VAR = "NDinc_cf01p_pred"
SIGMA_VAR = "sigmaNd"

LOG_SCATTER_AXES = True
MIN_ND_COMPARE = 1.0

# Optional plot limits. Set to None for automatic.
XMAX_SCATTER = 1000
YMAX_SCATTER = XMAX_SCATTER
XMIN_SCATTER = 10
YMIN_SCATTER = XMIN_SCATTER

MIN_COD_COMPARE = 0.1

# Journal plotting settings
FIG_WIDTH = 7.2
PANEL_HEIGHT_TIMESERIES = 1.75
PANEL_HEIGHT_SCATTER = 2.05
SINGLE_PANEL_WIDTH = 3.35
PLOT_DPI = 400
SAVE_PDF = True

OBS_COLOR = "black"
MODEL_COLOR = "#D55E00"
PRESSURE_CMAP = "viridis"


# ============================================================
# Journal style
# ============================================================
def set_journal_style():
    """Apply the compact journal style used by the other NDnet figures."""
    mpl.rcParams.update(
        {
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 7.3,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def panel_letter(index):
    """Convert 0,1,... into a,b,...,z,aa,ab,..."""
    result = ""
    i = int(index)
    while True:
        result = chr(ord("a") + (i % 26)) + result
        i = i // 26 - 1
        if i < 0:
            break
    return result


def style_panel_axis(ax):
    """Common low-ink panel styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linewidth=0.35, alpha=0.22, zorder=0)
    ax.tick_params(top=False, right=False)


def set_panel_titles(ax, index, campaign, r=None, bias=None, npts=None, empty=False):
    """Left campaign title and compact right-side statistics."""
    ax.set_title(
        f"({panel_letter(index)}) {campaign}",
        loc="left",
        fontsize=7.2,
        pad=2.5,
    )

    if empty:
        right = "no valid points"
    elif r is None or bias is None or npts is None:
        right = ""
    else:
        rtxt = "--" if not np.isfinite(r) else f"{r:.2f}"
        btxt = "--" if not np.isfinite(bias) else f"{bias:+.1f}"
        right = f"r={rtxt}  b={btxt}  n={npts}"

    if right:
        ax.set_title(
            right,
            loc="right",
            fontsize=6.1,
            pad=2.5,
        )


def save_figure(fig, out_path: Path):
    """Preserve the requested output and also write an editable PDF."""
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.03)
    print(f"[WROTE] {out_path}")

    if SAVE_PDF and out_path.suffix.lower() != ".pdf":
        pdf_path = out_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
        print(f"[WROTE] {pdf_path}")


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
    """Map actual dates to campaign-day index 1..N, ignoring calendar gaps."""
    day = pd.to_datetime(times).normalize()
    uniq = np.sort(day.unique())
    mapping = {d: i + 1 for i, d in enumerate(uniq)}
    return np.array([mapping[d] for d in day], dtype=int)


def add_within_day_offsets(df: pd.DataFrame, width=0.6):
    """Spread multiple same-day segments around the integer campaign day."""
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
            m &= np.isfinite(df["Nd_obs"].values) & (
                df["Nd_obs"].values >= min_nd_compare
            )
        if "Nd_model" in df.columns:
            m &= np.isfinite(df["Nd_model"].values) & (
                df["Nd_model"].values >= min_nd_compare
            )

    if USE_MODIS_RETRIEVAL_MASK and "modis_ok" in df.columns:
        m &= df["modis_ok"].values == 1

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
    bias = np.mean(y - x)  # model - obs
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

    hi = np.nanmax(vals)
    if XMAX_SCATTER is not None:
        hi = max(hi, XMAX_SCATTER)
    if YMAX_SCATTER is not None:
        hi = max(hi, YMAX_SCATTER)
    return 0.0, hi * 1.05


def scatter_limits(dfs, obs_col, model_col, log_axes=False):
    """Use configured limits where supplied, otherwise fall back to the data."""
    auto_lo, auto_hi = global_scatter_limits(
        dfs, obs_col, model_col, log_axes=log_axes
    )
    lo = XMIN_SCATTER if XMIN_SCATTER is not None else auto_lo
    hi = XMAX_SCATTER if XMAX_SCATTER is not None else auto_hi
    return float(lo), float(hi)


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
    Returns sampled COD and a COD > min_cod retrieval mask.
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

            lon_sel = _wrap_longitudes_to_grid(
                lon_box[idx_month], ds_modis["lon"].values
            )

            cod = (
                ds_modis[cod_var]
                .sel(
                    time=xr.DataArray(
                        day_vals[idx_month].to_numpy(), dims="segment"
                    ),
                    lat=xr.DataArray(lat_box[idx_month], dims="segment"),
                    lon=xr.DataArray(lon_sel, dims="segment"),
                    method="nearest",
                )
                .values
            )

            cod_modis[idx_month] = cod.astype(np.float32)
            modis_ok[idx_month] = (
                np.isfinite(cod_modis[idx_month])
                & (cod_modis[idx_month] > min_cod)
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
        lat_box = ds_seg["lat_box"].values.astype(float)
        lon_box = ds_seg["lon_box"].values.astype(float)
        p_merra = ds_seg["p_merra"].values.astype(float)

        nseg = ds_seg.sizes["segment"]

        nd_model = np.full(nseg, np.nan, dtype=np.float32)
        sigma_model = np.full(nseg, np.nan, dtype=np.float32)

        day_vals = pd.to_datetime(time_vals).normalize()
        unique_days = np.unique(day_vals)

        for day in unique_days:
            idx = np.where(day_vals == day)[0]
            if idx.size == 0:
                continue

            model_file = find_model_file(model_dir, day)
            if model_file is None:
                print(
                    f"[WARN] {campaign}: no model file for "
                    f"{pd.Timestamp(day).strftime('%Y-%m-%d')}"
                )
                continue

            with xr.open_dataset(model_file, decode_times=True) as ds_mod:
                if model_var not in ds_mod:
                    raise KeyError(f"{model_var!r} not found in {model_file}")

                lev = ds_mod["lev"].values.astype(float)
                lon_sel = _wrap_longitudes_to_grid(
                    lon_box[idx], ds_mod["lon"].values
                )

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

        out["Nd_model"].attrs.update(
            {
                "long_name": "Model Nd interpolated to segment pressure",
                "units": "cm-3",
                "source_variable": model_var,
                "interpolation": (
                    "same 1-degree cell, vertical linear interpolation in log-pressure"
                ),
            }
        )
        out["sigma_model"].attrs.update(
            {
                "long_name": "Model sigma interpolated to segment pressure",
                "units": "cm-3",
                "source_variable": sigma_var,
                "interpolation": (
                    "same 1-degree cell, vertical linear interpolation in log-pressure"
                ),
            }
        )
        out["campaign_day"].attrs.update(
            {"long_name": "Campaign day index (1..N) ignoring calendar gaps"}
        )
        out["cod_modis"].attrs.update(
            {
                "long_name": (
                    "MODIS Cloud Optical Thickness Liquid Mean sampled at "
                    "segment day and 1-degree box"
                ),
                "units": "none",
            }
        )
        out["modis_ok"].attrs.update(
            {
                "long_name": (
                    f"1 if MODIS retrieval exists (COD > {MIN_COD_COMPARE}), else 0"
                ),
                "units": "1",
            }
        )

        encoding = {
            "Nd_model": {
                "zlib": True,
                "complevel": 4,
                "_FillValue": np.float32(np.nan),
            },
            "sigma_model": {
                "zlib": True,
                "complevel": 4,
                "_FillValue": np.float32(np.nan),
            },
            "campaign_day": {
                "zlib": True,
                "complevel": 4,
                "_FillValue": np.int32(-1),
            },
            "cod_modis": {
                "zlib": True,
                "complevel": 4,
                "_FillValue": np.float32(np.nan),
            },
            "modis_ok": {
                "zlib": True,
                "complevel": 4,
                "_FillValue": np.int8(-1),
            },
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

        df = pd.DataFrame(
            {
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
            }
        )

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
def _pressure_range(dfs, obs_col, model_col, log_axes=False):
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
    return float(np.nanmin(p_all)), float(np.nanmax(p_all))


def _shared_pressure_colorbar(fig, mappable, axes, label="Pressure (hPa)"):
    cbar = fig.colorbar(
        mappable,
        ax=list(axes),
        orientation="vertical",
        fraction=0.024,
        pad=0.018,
        aspect=30,
    )
    cbar.set_label(label, fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.7, width=0.6, length=2.5)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.invert_yaxis()
    return cbar


def plot_timeseries_panel(
    dfs,
    out_png: Path,
    obs_col: str,
    model_col: str,
    ylabel: str,
    title_text: str,
):
    """Compact campaign time-series panels; title_text kept for API compatibility."""
    del title_text

    n = len(dfs)
    nrows, ncols = layout_for_n(n)
    figheight = PANEL_HEIGHT_TIMESERIES * nrows + 0.45

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FIG_WIDTH, figheight),
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    axes = axes.ravel()

    pmin, pmax = _pressure_range(dfs, obs_col, model_col, log_axes=False)
    last_sc = None

    for i, (ax, df) in enumerate(zip(axes, dfs)):
        campaign = df["campaign"].iloc[0]
        sub = subset_pair(
            df,
            obs_col,
            model_col,
            require_positive=False,
            min_nd_compare=MIN_ND_COMPARE if obs_col == "Nd_obs" else None,
        )
        sub = sub.sort_values("xplot").reset_index(drop=True)

        style_panel_axis(ax)

        if sub.empty:
            set_panel_titles(ax, i, campaign, empty=True)
            continue

        r, bias, npts = compute_stats(sub, obs_col, model_col)

        ax.plot(
            sub["xplot"].values,
            sub[obs_col].values,
            color=OBS_COLOR,
            linewidth=0.75,
            alpha=0.55,
            zorder=2,
        )
        ax.plot(
            sub["xplot"].values,
            sub[model_col].values,
            color=MODEL_COLOR,
            linewidth=0.8,
            alpha=0.6,
            zorder=2,
        )

        ax.scatter(
            sub["xplot"].values - 0.035,
            sub[obs_col].values,
            c=sub["p_merra"].values,
            s=14,
            cmap=PRESSURE_CMAP,
            vmin=pmin,
            vmax=pmax,
            marker="o",
            linewidths=0.25,
            edgecolors="black",
            alpha=0.92,
            zorder=3,
        )
        last_sc = ax.scatter(
            sub["xplot"].values + 0.035,
            sub[model_col].values,
            c=sub["p_merra"].values,
            s=19,
            cmap=PRESSURE_CMAP,
            vmin=pmin,
            vmax=pmax,
            marker="x",
            linewidths=0.65,
            alpha=0.92,
            zorder=4,
        )

        set_panel_titles(ax, i, campaign, r=r, bias=bias, npts=npts)

        xmax = int(np.nanmax(sub["campaign_day"].values))
        ax.set_xlim(0.5, xmax + 0.5)

        has_panel_below = i + ncols < n
        if has_panel_below:
            ax.tick_params(axis="x", labelbottom=False)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.supxlabel("Campaign day", fontsize=8.5, y=0.018)
    fig.supylabel(ylabel, fontsize=8.5, x=0.018)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=OBS_COLOR,
            lw=1.0,
            marker="o",
            markersize=3.2,
            markerfacecolor="0.75",
            markeredgecolor="black",
            markeredgewidth=0.4,
            label="Aircraft",
        ),
        Line2D(
            [0],
            [0],
            color=MODEL_COLOR,
            lw=1.0,
            marker="x",
            markersize=4.0,
            label="NDnet",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.6,
    )

    if last_sc is not None:
        _shared_pressure_colorbar(fig, last_sc, axes[:n])

    fig.subplots_adjust(
        left=0.075,
        right=0.90,
        bottom=0.075,
        top=0.94,
        wspace=0.20,
        hspace=0.28,
    )

    save_figure(fig, out_png)
    plt.close(fig)


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
    """Compact per-campaign scatter panels; title_text kept for compatibility."""
    del title_text

    n = len(dfs)
    nrows, ncols = layout_for_n(n)
    figheight = PANEL_HEIGHT_SCATTER * nrows + 0.35

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FIG_WIDTH, figheight),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()

    lo, hi = scatter_limits(dfs, obs_col, model_col, log_axes=log_axes)
    pmin, pmax = _pressure_range(dfs, obs_col, model_col, log_axes=log_axes)
    last_sc = None

    for i, (ax, df) in enumerate(zip(axes, dfs)):
        campaign = df["campaign"].iloc[0]
        sub = subset_pair(
            df,
            obs_col,
            model_col,
            require_positive=log_axes,
            min_nd_compare=MIN_ND_COMPARE if obs_col == "Nd_obs" else None,
        )

        style_panel_axis(ax)

        if log_axes:
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

        if sub.empty:
            set_panel_titles(ax, i, campaign, empty=True)
        else:
            r, bias, npts = compute_stats(sub, obs_col, model_col)

            last_sc = ax.scatter(
                sub[obs_col].values,
                sub[model_col].values,
                c=sub["p_merra"].values,
                s=15,
                cmap=PRESSURE_CMAP,
                vmin=pmin,
                vmax=pmax,
                alpha=0.78,
                linewidths=0,
                rasterized=True,
                zorder=3,
            )

            ax.plot(
                [lo, hi],
                [lo, hi],
                linestyle="--",
                linewidth=0.75,
                color="black",
                alpha=0.8,
                zorder=2,
            )
            set_panel_titles(ax, i, campaign, r=r, bias=bias, npts=npts)

        row = i // ncols
        col = i % ncols

        if col != 0:
            ax.tick_params(axis="y", labelleft=False)

        has_panel_below = i + ncols < n
        if has_panel_below:
            ax.tick_params(axis="x", labelbottom=False)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.supxlabel(xlabel, fontsize=8.5, y=0.015)
    fig.supylabel(ylabel, fontsize=8.5, x=0.015)

    if last_sc is not None:
        _shared_pressure_colorbar(fig, last_sc, axes[:n])

    fig.subplots_adjust(
        left=0.075,
        right=0.90,
        bottom=0.065,
        top=0.965,
        wspace=0.12,
        hspace=0.22,
    )

    save_figure(fig, out_png)
    plt.close(fig)


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
    lo, hi = scatter_limits(dfs, obs_col, model_col, log_axes=log_axes)

    pmin = np.nanmin(all_df["p_merra"].values)
    pmax = np.nanmax(all_df["p_merra"].values)

    fig, ax = plt.subplots(figsize=(SINGLE_PANEL_WIDTH, 3.15))

    sc = ax.scatter(
        all_df[obs_col].values,
        all_df[model_col].values,
        c=all_df["p_merra"].values,
        s=13,
        cmap=PRESSURE_CMAP,
        vmin=pmin,
        vmax=pmax,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )

    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=0.8,
        color="black",
        alpha=0.8,
        zorder=2,
    )

    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style_panel_axis(ax)

    ax.set_title(title_text, loc="left", fontsize=7.2, pad=2.5)
    rtxt = "--" if not np.isfinite(r) else f"{r:.2f}"
    btxt = "--" if not np.isfinite(bias) else f"{bias:+.1f}"
    ax.set_title(
        f"r={rtxt}  b={btxt}  n={npts}",
        loc="right",
        fontsize=6.2,
        pad=2.5,
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.048, pad=0.035, aspect=28)
    cbar.set_label("Pressure (hPa)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.7, width=0.6, length=2.5)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.invert_yaxis()

    fig.subplots_adjust(left=0.19, right=0.86, bottom=0.16, top=0.93)

    save_figure(fig, out_png)
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    set_journal_style()
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
        ylabel=r"NDnet $N_d$ (cm$^{-3}$)",
        title_text="Per-campaign scatter: aircraft vs model",
        log_axes=LOG_SCATTER_AXES,
    )
    plot_scatter_all(
        dfs,
        OUTPUT_DIR / "campaign_scatter_all.png",
        obs_col="Nd_obs",
        model_col="Nd_model",
        xlabel=r"Aircraft $N_d$ (cm$^{-3}$)",
        ylabel=r"NDnet $N_d$ (cm$^{-3}$)",
        title_text="All campaigns",
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
            ylabel=r"NDnet $\sigma(N_d)$ (cm$^{-3}$)",
            title_text="Per-campaign sigma scatter: aircraft vs model",
            log_axes=LOG_SCATTER_AXES,
        )
        plot_scatter_all(
            dfs,
            OUTPUT_DIR / "campaign_sigma_scatter_all.png",
            obs_col="sigma_obs",
            model_col="sigma_model",
            xlabel=r"Aircraft $\sigma(N_d)$ (cm$^{-3}$)",
            ylabel=r"NDnet $\sigma(N_d)$ (cm$^{-3}$)",
            title_text="All campaigns: sigma",
            log_axes=LOG_SCATTER_AXES,
        )
    else:
        print("[INFO] No valid sigma points found; skipping sigma plots.")


if __name__ == "__main__":
    main()
