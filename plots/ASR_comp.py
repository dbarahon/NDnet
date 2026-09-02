#!/usr/bin/env python3
"""
Compare fixed-site ASR daily droplet-number observations against NDnet 2-D
predictions stored as monthly NetCDF files, and make journal-style box plots.

Current NDnet monthly files are expected to look like, for example:
    NDnet_modis_compare_800_950hPa_201502.nc
or
    NDnet_modis_compare_800_950hPa_201502.nc4

Workflow
--------
1. Open each ASR site file once.
2. Read site coordinates and the filtered daily-mean obs time series.
3. Group required obs dates by month.
4. Open each needed NDnet monthly file once.
5. Extract nearest-grid-point NDnet values for all sites.
6. Align by site/date.
7. Write simplified statistics and make journal-style distribution figures.

Statistics reported
-------------------
* n_days: number of matched finite days
* mean_bias_cm3: mean(model - obs)
* mean_pct_bias: mean(100 * (model - obs) / obs) over days with obs != 0
* temporal_corr: Pearson correlation across matched daily pairs

Outputs
-------
* matched_daily_asr_ndnet.csv
* stats_by_site_and_ndnet_variable.csv
* stats_all_sites_by_ndnet_variable.csv
* site_metadata.csv
* site_to_ndnet_grid_mapping.csv
* missing_ndnet_months.csv                 (if needed)
* missing_ndnet_dates_inside_months.csv    (if needed)
* boxplot_ASR_vs_NDnet_allvars.png/.pdf
* boxplot_ASR_raw_filtered_by_site.png/.pdf
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


DEFAULT_OBS_DIR = Path(
    "/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/dailies/ASRsites/2std"
)
DEFAULT_NDNET_DIR = Path(
    "/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/inference/after_refinement/NDnet_2D_monthly"
)
DEFAULT_OUTDIR = Path("./NDnet_ASRsite_compare_boxplots")

DEFAULT_OBS_PATTERN = "*_daily_ndrop.nc"
DEFAULT_NDNET_MONTH_TEMPLATE = "NDnet_modis_compare_800_950hPa_{yyyymm}.nc"
DEFAULT_OBS_VAR = "daily_mean_filtered_cm3"

PREFERRED_ND_VARS = [
    "nd_no_mask",
    "nd_modis_track",
    "nd_modis_track_ctt",
    "nd_modis_track_ctt_cf",
    "nd_unweighted_no_mask",
    "nd_unweighted_modis_track",
    "nd_unweighted_modis_track_ctt",
    "nd_unweighted_modis_track_ctt_cf",
]

# Required for the compact single-panel site/campaign comparison.
SUMMARY_BOX_VARS = (
    "nd_unweighted_no_mask",
    "nd_modis_track_ctt_cf",
)

# Journal figure settings.
FIG_WIDTH = 7.2
PANEL_HEIGHT = 2.15
SINGLE_PANEL_HEIGHT = 3.35
PLOT_DPI = 400

OBS_COLOR = "#666666"
RAW_COLOR = "#D55E00"
FILTERED_COLOR = "#0072B2"
WEIGHTED_COLOR = "#4C72B0"
UNWEIGHTED_COLOR = "#DD8452"


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def parse_date_arg(x: Optional[str]) -> Optional[pd.Timestamp]:
    if x is None or str(x).strip() == "":
        return None
    return pd.Timestamp(x).normalize()


def normalize_dates(values) -> pd.DatetimeIndex:
    """Convert decoded xarray time values to midnight-normalized pandas dates."""
    try:
        out = pd.to_datetime(values)
    except Exception:
        out = pd.to_datetime([str(v) for v in np.asarray(values).ravel()])
    return pd.DatetimeIndex(out).normalize()


def attr_str(ds: xr.Dataset, key: str, default: str = "") -> str:
    val = ds.attrs.get(key, default)
    if isinstance(val, bytes):
        return val.decode(errors="replace")
    return str(val)


def site_names_from_attrs(ds: xr.Dataset, path: Path) -> Tuple[str, str, str, str, str]:
    datastream = attr_str(ds, "datastream", path.stem.replace("_daily_ndrop", ""))
    site = datastream.replace("_daily_ndrop", "")
    site_id = attr_str(ds, "site_id", "").upper()
    facility_id = attr_str(ds, "facility_id", "")

    if site_id and facility_id:
        label = f"{site_id}-{facility_id}"
    elif site_id:
        label = site_id
    else:
        label = site

    return site, label, site_id, facility_id, datastream


def convert_lon_to_grid(site_lon: float, grid_lons: np.ndarray) -> float:
    lon = float(site_lon)
    finite = np.asarray(grid_lons, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return lon

    lon_min = float(np.nanmin(finite))
    lon_max = float(np.nanmax(finite))

    if lon_min >= 0.0 and lon < 0.0:
        return lon % 360.0
    if lon_max <= 180.0 and lon > 180.0:
        return ((lon + 180.0) % 360.0) - 180.0
    return lon


def lon_for_distance(grid_lon: float, site_lon: float) -> float:
    glon = float(grid_lon)
    slon = float(site_lon)
    if slon < 0.0 and glon > 180.0:
        return glon - 360.0
    if slon > 180.0 and glon < 0.0:
        return glon + 360.0
    return glon


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * r * math.asin(math.sqrt(a))


def nearest_index_1d(coord: np.ndarray, target: float) -> int:
    coord = np.asarray(coord, dtype=float)
    return int(np.nanargmin(np.abs(coord - float(target))))


def split_csv_arg(x: Optional[str]) -> Optional[List[str]]:
    if x is None:
        return None
    out = [v.strip() for v in str(x).split(",") if v.strip()]
    return out or None


def format_ndnet_month_path(ndnet_dir: Path, template: str, yyyymm: str) -> Path:
    year = yyyymm[:4]
    month = yyyymm[4:6]
    return ndnet_dir / template.format(
        yyyymm=yyyymm,
        YYYYMM=yyyymm,
        month_key=yyyymm,
        year=year,
        yyyy=year,
        YYYY=year,
        mm=month,
        month=month,
        MM=month,
    )


# -----------------------------------------------------------------------------
# Read observations once
# -----------------------------------------------------------------------------

def read_obs_sites(
    obs_dir: Path,
    obs_pattern: str,
    obs_var: str,
    date_start: Optional[pd.Timestamp],
    date_end: Optional[pd.Timestamp],
    use_finite_obs_dates_only_for_metadata: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(Path(obs_dir).glob(obs_pattern))
    if not paths:
        raise FileNotFoundError(f"No obs files matching {obs_pattern!r} found in {obs_dir}")

    obs_frames: List[pd.DataFrame] = []
    site_rows: List[Dict] = []

    for path in paths:
        print(f"[OBS] {path.name}", flush=True)
        with xr.open_dataset(path, decode_times=True) as ds:
            if obs_var not in ds:
                raise KeyError(f"{path.name} does not contain obs variable {obs_var!r}")
            if "time" not in ds:
                raise KeyError(f"{path.name} has no time coordinate/variable")

            dates = normalize_dates(ds["time"].values)
            obs = np.asarray(ds[obs_var].values, dtype=float).reshape(-1)
            if len(dates) != obs.size:
                raise ValueError(
                    f"{path.name}: time length {len(dates)} does not match "
                    f"{obs_var} length {obs.size}"
                )

            site, label, site_id, facility_id, datastream = site_names_from_attrs(ds, path)
            lat = float(ds.attrs.get("lat", np.nan))
            lon = float(ds.attrs.get("lon", np.nan))
            if not np.isfinite(lat) or not np.isfinite(lon):
                raise ValueError(f"{path.name}: missing finite global attrs lat/lon")

            df = pd.DataFrame(
                {
                    "site": site,
                    "label": label,
                    "date": dates,
                    "obs_cm3": obs,
                    "obs_file": path.name,
                }
            )

            if date_start is not None:
                df = df[df["date"] >= date_start]
            if date_end is not None:
                df = df[df["date"] <= date_end]

            if df.empty:
                warnings.warn(f"{path.name}: no obs dates left after date filtering; skipping")
                continue

            range_df = (
                df[np.isfinite(df["obs_cm3"].to_numpy(float))]
                if use_finite_obs_dates_only_for_metadata
                else df
            )
            if range_df.empty:
                warnings.warn(f"{path.name}: no finite obs values in selected period; skipping")
                continue

            obs_frames.append(df)
            site_rows.append(
                {
                    "site": site,
                    "label": label,
                    "site_id": site_id,
                    "facility_id": facility_id,
                    "datastream": datastream,
                    "lat": lat,
                    "lon": lon,
                    "obs_file": path.name,
                    "location_description": attr_str(ds, "location_description", ""),
                    "obs_start": range_df["date"].min(),
                    "obs_end": range_df["date"].max(),
                    "n_obs_days_in_file": int(len(df)),
                    "n_finite_obs_days": int(
                        np.isfinite(df["obs_cm3"].to_numpy(float)).sum()
                    ),
                }
            )

    if not obs_frames:
        raise ValueError("No usable obs files found")

    obs_df = pd.concat(obs_frames, ignore_index=True)
    obs_df = (
        obs_df.groupby(["site", "label", "date", "obs_file"], as_index=False)[
            "obs_cm3"
        ]
        .mean()
        .sort_values(["site", "date"])
        .reset_index(drop=True)
    )

    site_df = (
        pd.DataFrame(site_rows)
        .drop_duplicates("site")
        .sort_values("label")
        .reset_index(drop=True)
    )
    return obs_df, site_df


# -----------------------------------------------------------------------------
# Monthly NDnet file selection
# -----------------------------------------------------------------------------

def build_needed_ndnet_months(
    obs_df: pd.DataFrame,
    ndnet_dir: Path,
    ndnet_month_template: str,
    only_finite_obs_dates: bool,
) -> Tuple[Dict[str, List[pd.Timestamp]], Dict[str, Path], List[str]]:
    df = obs_df.copy()
    if only_finite_obs_dates:
        df = df[np.isfinite(df["obs_cm3"].to_numpy(float))]

    if df.empty:
        raise ValueError("No obs dates available for NDnet extraction")

    dates = pd.DatetimeIndex(df["date"].drop_duplicates().sort_values()).normalize()

    dates_by_month: Dict[str, List[pd.Timestamp]] = {}
    for d in dates:
        key = pd.Timestamp(d).strftime("%Y%m")
        dates_by_month.setdefault(key, []).append(pd.Timestamp(d))

    file_by_month: Dict[str, Path] = {}
    missing_months: List[str] = []
    for key in sorted(dates_by_month):
        p = format_ndnet_month_path(ndnet_dir, ndnet_month_template, key)
        if p.exists():
            file_by_month[key] = p
        else:
            alt = None
            if str(p).endswith(".nc"):
                alt = Path(str(p) + "4")
            elif str(p).endswith(".nc4"):
                alt = Path(str(p)[:-1])
            if alt is not None and alt.exists():
                file_by_month[key] = alt
            else:
                missing_months.append(key)

    print(
        f"[NDnet] Need {len(dates)} obs dates in {len(dates_by_month)} months",
        flush=True,
    )
    print(
        f"[NDnet] Found {len(file_by_month)} expected monthly NDnet files",
        flush=True,
    )
    if missing_months:
        print(
            f"[NDnet] Missing {len(missing_months)} monthly NDnet files; "
            "first missing months:",
            flush=True,
        )
        print("        " + ", ".join(missing_months[:12]), flush=True)

    if not file_by_month:
        example_key = sorted(dates_by_month)[0]
        example = format_ndnet_month_path(ndnet_dir, ndnet_month_template, example_key)
        raise FileNotFoundError(
            "None of the expected monthly NDnet files exist. "
            f"Example expected path: {example}"
        )

    return dates_by_month, file_by_month, missing_months


# -----------------------------------------------------------------------------
# NDnet extraction
# -----------------------------------------------------------------------------

def infer_nd_vars(sample_file: Path, requested: Optional[Sequence[str]]) -> List[str]:
    with xr.open_dataset(sample_file, decode_times=False) as ds:
        if requested:
            missing = [v for v in requested if v not in ds]
            if missing:
                raise KeyError(
                    f"Requested NDnet variables missing from {sample_file.name}: {missing}"
                )
            return list(requested)

        preferred = [v for v in PREFERRED_ND_VARS if v in ds.data_vars]
        other = sorted(
            [v for v in ds.data_vars if v.startswith("nd_") and v not in preferred]
        )
        nd_vars = preferred + other
        if not nd_vars:
            raise KeyError(f"No NDnet variables starting with 'nd_' found in {sample_file}")
        return nd_vars


def ensure_summary_vars(
    sample_file: Path,
    nd_vars: Sequence[str],
) -> List[str]:
    """Ensure the raw/filtered variables needed by the grouped plot are extracted."""
    with xr.open_dataset(sample_file, decode_times=False) as ds:
        missing = [v for v in SUMMARY_BOX_VARS if v not in ds]
    if missing:
        raise KeyError(
            f"Required grouped-box variables missing from {sample_file.name}: {missing}"
        )

    out = list(nd_vars)
    for var in SUMMARY_BOX_VARS:
        if var not in out:
            out.append(var)
    return out


def compute_site_grid_mapping(
    site_df: pd.DataFrame,
    sample_file: Path,
    max_distance_km: float,
) -> pd.DataFrame:
    with xr.open_dataset(sample_file, decode_times=False) as ds:
        if "lat" not in ds or "lon" not in ds:
            raise KeyError(f"{sample_file.name} must contain lat and lon coordinates")
        grid_lats = np.asarray(ds["lat"].values, dtype=float)
        grid_lons = np.asarray(ds["lon"].values, dtype=float)

    rows = []
    for _, s in site_df.iterrows():
        site_lat = float(s["lat"])
        site_lon = float(s["lon"])
        target_lon = convert_lon_to_grid(site_lon, grid_lons)

        ilat = nearest_index_1d(grid_lats, site_lat)
        ilon = nearest_index_1d(grid_lons, target_lon)
        glat = float(grid_lats[ilat])
        glon = float(grid_lons[ilon])
        glon_dist = lon_for_distance(glon, site_lon)
        dist = haversine_km(site_lat, site_lon, glat, glon_dist)

        if np.isfinite(max_distance_km) and dist > max_distance_km:
            warnings.warn(
                f"Nearest grid cell for {s['site']} is {dist:.1f} km away "
                f"(>{max_distance_km:.1f} km). Check site coordinates/grid."
            )

        rows.append(
            {
                "site": s["site"],
                "label": s["label"],
                "site_lat": site_lat,
                "site_lon": site_lon,
                "target_lon_on_grid_convention": target_lon,
                "lat_index": ilat,
                "lon_index": ilon,
                "grid_lat": glat,
                "grid_lon": glon,
                "grid_lon_for_distance": glon_dist,
                "grid_distance_km": dist,
            }
        )

    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)


def extract_ndnet_monthly_for_sites(
    dates_by_month: Dict[str, List[pd.Timestamp]],
    file_by_month: Dict[str, Path],
    site_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    nd_vars: Sequence[str],
    progress_every: int = 12,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    site_order = site_df.sort_values("label")["site"].tolist()
    grid = grid_df.set_index("site").loc[site_order]

    lat_idx_da = xr.DataArray(
        grid["lat_index"].to_numpy(int), dims="site", coords={"site": site_order}
    )
    lon_idx_da = xr.DataArray(
        grid["lon_index"].to_numpy(int), dims="site", coords={"site": site_order}
    )

    months = sorted(file_by_month)
    records: List[Dict] = []
    missing_time_rows: List[Dict] = []

    for i, yyyymm in enumerate(months, start=1):
        path = file_by_month[yyyymm]
        want_dates = sorted(
            pd.Timestamp(d).normalize() for d in dates_by_month[yyyymm]
        )

        if i == 1 or i % progress_every == 0 or i == len(months):
            print(
                f"[NDnet] extracting month {i}/{len(months)}: {path.name} "
                f"({len(want_dates)} requested dates)",
                flush=True,
            )

        with xr.open_dataset(path, decode_times=True) as ds:
            if "time" not in ds:
                raise KeyError(f"{path.name} has no time coordinate")

            file_dates = normalize_dates(ds["time"].values)
            pos_by_date: Dict[pd.Timestamp, int] = {}
            for j, d in enumerate(file_dates):
                pos_by_date.setdefault(pd.Timestamp(d).normalize(), int(j))

            have_dates = [d for d in want_dates if d in pos_by_date]
            missing_dates = [d for d in want_dates if d not in pos_by_date]

            for d in missing_dates:
                missing_time_rows.append(
                    {
                        "month": yyyymm,
                        "missing_date_inside_month_file": d,
                        "file": path.name,
                    }
                )

            if not have_dates:
                warnings.warn(
                    f"{path.name}: none of the requested obs dates are present "
                    "in time coordinate"
                )
                continue

            time_idx_da = xr.DataArray(
                [pos_by_date[d] for d in have_dates],
                dims="date",
                coords={"date": have_dates},
            )

            vals_by_var: Dict[str, np.ndarray] = {}
            for var in nd_vars:
                if var not in ds:
                    vals_by_var[var] = np.full(
                        (len(have_dates), len(site_order)), np.nan, dtype=float
                    )
                    continue

                da = ds[var]
                if "lat" not in da.dims or "lon" not in da.dims:
                    warnings.warn(
                        f"{path.name}:{var} does not have lat/lon dims; filling NaN"
                    )
                    vals_by_var[var] = np.full(
                        (len(have_dates), len(site_order)), np.nan, dtype=float
                    )
                    continue

                if "time" in da.dims:
                    point = da.isel(
                        time=time_idx_da, lat=lat_idx_da, lon=lon_idx_da
                    )
                    if "date" in point.dims and "site" in point.dims:
                        point = point.transpose("date", "site")
                    arr = np.asarray(point.values, dtype=float)
                else:
                    point = da.isel(lat=lat_idx_da, lon=lon_idx_da)
                    if "site" in point.dims:
                        point = point.transpose("site")
                    one = np.asarray(point.values, dtype=float).reshape(1, -1)
                    arr = np.repeat(one, len(have_dates), axis=0)

                arr = np.asarray(arr, dtype=float)
                if arr.shape != (len(have_dates), len(site_order)):
                    arr = np.reshape(arr, (len(have_dates), len(site_order)))
                vals_by_var[var] = arr

        for t, date in enumerate(have_dates):
            for k, site in enumerate(site_order):
                row = {"site": site, "date": pd.Timestamp(date)}
                for var in nd_vars:
                    row[var] = float(vals_by_var[var][t, k])
                records.append(row)

    pred_df = (
        pd.DataFrame(records)
        .sort_values(["site", "date"])
        .reset_index(drop=True)
    )
    missing_time_df = pd.DataFrame(missing_time_rows)
    return pred_df, missing_time_df


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def compute_pair_stats(
    df: pd.DataFrame,
    var: str,
    min_corr_n: int,
) -> Dict[str, float]:
    sub = df[["obs_cm3", var]].replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(sub))
    if n == 0:
        return {
            "n_days": 0,
            "mean_bias_cm3": np.nan,
            "mean_pct_bias": np.nan,
            "temporal_corr": np.nan,
        }

    obs = sub["obs_cm3"].to_numpy(float)
    mod = sub[var].to_numpy(float)
    diff = mod - obs

    good_pct = np.isfinite(obs) & np.isfinite(mod) & (obs != 0.0)
    if np.any(good_pct):
        mean_pct_bias = float(
            np.nanmean(100.0 * (mod[good_pct] - obs[good_pct]) / obs[good_pct])
        )
    else:
        mean_pct_bias = np.nan

    corr = np.nan
    if n >= int(min_corr_n) and np.nanstd(obs) > 0.0 and np.nanstd(mod) > 0.0:
        corr = float(np.corrcoef(obs, mod)[0, 1])

    return {
        "n_days": n,
        "mean_bias_cm3": float(np.nanmean(diff)),
        "mean_pct_bias": mean_pct_bias,
        "temporal_corr": corr,
    }


def make_stats_tables(
    merged: pd.DataFrame,
    site_df: pd.DataFrame,
    nd_vars: Sequence[str],
    min_corr_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    meta = site_df.set_index("site").to_dict("index")

    for site, g in merged.groupby("site", sort=True):
        m = meta.get(site, {})
        for var in nd_vars:
            rows.append(
                {
                    "site": site,
                    "label": m.get("label", site),
                    "ndnet_variable": var,
                    **compute_pair_stats(g, var, min_corr_n),
                }
            )

    by_site = (
        pd.DataFrame(rows)
        .sort_values(["label", "ndnet_variable"])
        .reset_index(drop=True)
    )

    agg_rows = []
    for var in nd_vars:
        agg_rows.append(
            {
                "group": "all_sites",
                "ndnet_variable": var,
                **compute_pair_stats(merged, var, min_corr_n),
            }
        )
    aggregate = pd.DataFrame(agg_rows)

    return by_site, aggregate


# -----------------------------------------------------------------------------
# Journal plotting
# -----------------------------------------------------------------------------

def configure_matplotlib():
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": PLOT_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 7.5,
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
        }
    )


def panel_letter(index: int) -> str:
    """Convert 0,1,... into a,b,...,z,aa,ab,..."""
    result = ""
    i = int(index)
    while True:
        result = chr(ord("a") + (i % 26)) + result
        i = i // 26 - 1
        if i < 0:
            break
    return result


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="0.88", linewidth=0.45, zorder=0)
    ax.tick_params(top=False, right=False)


def save_figure_formats(
    fig,
    out_base: Path,
    formats: Sequence[str] = ("png", "pdf"),
):
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = out_base.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
        if fmt.lower() == "png":
            kwargs["dpi"] = PLOT_DPI
        fig.savefig(out, **kwargs)
        print(f"[WRITE] {out}", flush=True)


def var_label(var: str) -> str:
    mapping = {
        "obs_cm3": "Obs",
        "nd_no_mask": "W no mask",
        "nd_modis_track": "W MODIS track",
        "nd_modis_track_ctt": "W track + CTT",
        "nd_modis_track_ctt_cf": "W track + CTT + CF",
        "nd_unweighted_no_mask": "UW no mask",
        "nd_unweighted_modis_track": "UW MODIS track",
        "nd_unweighted_modis_track_ctt": "UW track + CTT",
        "nd_unweighted_modis_track_ctt_cf": "UW track + CTT + CF",
    }
    return mapping.get(var, var)


def box_color(var: str) -> str:
    if var == "obs_cm3":
        return OBS_COLOR
    if var.startswith("nd_unweighted"):
        return UNWEIGHTED_COLOR
    return WEIGHTED_COLOR


def finite_values(df: pd.DataFrame, var: str) -> np.ndarray:
    if var not in df.columns:
        return np.empty(0, dtype=float)
    vals = df[var].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    return vals[np.isfinite(vals)]


def make_boxplot_panels(
    merged: pd.DataFrame,
    site_df: pd.DataFrame,
    box_vars: Sequence[str],
    out_base: Path,
    ncols: int,
    ylim: Optional[Tuple[float, float]],
    formats: Sequence[str],
):
    """Journal-style version of the original per-site all-variable box plot."""
    configure_matplotlib()

    site_order = site_df.sort_values("label")["site"].tolist()
    label_lookup = site_df.set_index("site")["label"].to_dict()

    nsite = len(site_order)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(nsite / ncols))

    fig_h = PANEL_HEIGHT * nrows + 0.20
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FIG_WIDTH, fig_h),
        squeeze=False,
        sharey=True,
    )
    axes_flat = axes.ravel()

    for i, (ax, site) in enumerate(zip(axes_flat, site_order)):
        g = merged[merged["site"] == site].sort_values("date")

        plot_data = []
        labels = []
        colors = []
        for var in box_vars:
            vals = finite_values(g, var)
            if vals.size == 0:
                continue
            plot_data.append(vals)
            labels.append(var_label(var))
            colors.append(box_color(var))

        style_axis(ax)
        ax.set_title(
            f"({panel_letter(i)}) {label_lookup.get(site, site)}",
            loc="left",
            fontsize=7.5,
            pad=2.5,
        )

        if not plot_data:
            ax.text(
                0.5,
                0.5,
                "No valid data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=7.0,
            )
            continue

        bp = ax.boxplot(
            plot_data,
            patch_artist=True,
            widths=0.60,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 0.9},
            whiskerprops={"color": "0.25", "linewidth": 0.7},
            capprops={"color": "0.25", "linewidth": 0.7},
            boxprops={"color": "0.25", "linewidth": 0.7},
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.82)
            patch.set_edgecolor("0.25")

        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor")
        if ylim is not None:
            ax.set_ylim(*ylim)

        col = i % ncols
        if col != 0:
            ax.tick_params(axis="y", labelleft=False)

    for ax in axes_flat[nsite:]:
        ax.set_visible(False)

    fig.supylabel(r"$N_d$ (cm$^{-3}$)", fontsize=8.5, x=0.018)
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.115,
        top=0.965,
        wspace=0.12,
        hspace=0.30,
    )

    save_figure_formats(fig, out_base, formats=formats)
    plt.close(fig)


def make_grouped_site_boxplot(
    merged: pd.DataFrame,
    site_df: pd.DataFrame,
    out_base: Path,
    ylim: Optional[Tuple[float, float]],
    formats: Sequence[str] = ("png", "pdf"),
):
    """Single-panel site/campaign box plot: Obs, raw, and filtered NDnet."""
    configure_matplotlib()

    series = [
        ("obs_cm3", "Obs", OBS_COLOR),
        ("nd_unweighted_no_mask", "raw", RAW_COLOR),
        ("nd_modis_track_ctt_cf", "filtered", FILTERED_COLOR),
    ]

    missing = [var for var, _, _ in series if var not in merged.columns]
    if missing:
        raise KeyError(f"Grouped box plot is missing required columns: {missing}")

    ordered = site_df.sort_values("label").reset_index(drop=True)
    site_order = ordered["site"].tolist()
    labels = ordered["label"].tolist()
    centers = np.arange(len(site_order), dtype=float)

    box_width = 0.20
    offsets = (-0.24, 0.0, 0.24)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, SINGLE_PANEL_HEIGHT))
    style_axis(ax)

    for offset, (var, label, color) in zip(offsets, series):
        data = []
        for site in site_order:
            g = merged[merged["site"] == site]
            vals = finite_values(g, var)
            if vals.size == 0:
                vals = np.asarray([np.nan], dtype=float)
            data.append(vals)

        bp = ax.boxplot(
            data,
            positions=centers + offset,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            medianprops={"color": "black", "linewidth": 0.9},
            whiskerprops={"color": "0.25", "linewidth": 0.7},
            capprops={"color": "0.25", "linewidth": 0.7},
            boxprops={"color": "0.25", "linewidth": 0.7},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.84)
            patch.set_edgecolor("0.25")

    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.55, max(0.55, len(site_order) - 0.45))
    ax.set_xlabel("Site / campaign")
    ax.set_ylabel(r"$N_d$ (cm$^{-3}$)")

    if ylim is not None:
        ax.set_ylim(*ylim)

    handles = [
        Patch(facecolor=color, edgecolor="0.25", linewidth=0.7, label=label)
        for _, label, color in series
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.5,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.23,
        top=0.89,
    )

    save_figure_formats(fig, out_base, formats=formats)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare fixed-site ASR daily Nd against monthly NDnet 2-D files "
            "and make journal-style site box plots."
        )
    )
    p.add_argument("--obs-dir", type=Path, default=DEFAULT_OBS_DIR)
    p.add_argument("--ndnet-dir", type=Path, default=DEFAULT_NDNET_DIR)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--obs-pattern", default=DEFAULT_OBS_PATTERN)
    p.add_argument(
        "--ndnet-month-template",
        default=DEFAULT_NDNET_MONTH_TEMPLATE,
        help=(
            "Template relative to --ndnet-dir. Use {yyyymm} or {YYYYMM}. "
            f"Default: {DEFAULT_NDNET_MONTH_TEMPLATE}"
        ),
    )
    p.add_argument("--obs-var", default=DEFAULT_OBS_VAR)
    p.add_argument(
        "--nd-vars",
        default=None,
        help=(
            "Comma-separated NDnet variables for statistics/all-variable panels. "
            "The raw/filtered variables needed by the grouped plot are extracted "
            "automatically. Default: all nd_* vars, preferred first."
        ),
    )
    p.add_argument("--date-start", default=None, help="Optional YYYY-MM-DD lower bound")
    p.add_argument("--date-end", default=None, help="Optional YYYY-MM-DD upper bound")
    p.add_argument(
        "--use-all-obs-dates",
        action="store_true",
        help=(
            "By default NDnet extraction is based on finite obs dates only. "
            "Set this to include all obs time-coordinate dates."
        ),
    )
    p.add_argument("--ncols", type=int, default=2)
    p.add_argument("--ylim", default=None, help="Example: --ylim 0,300")
    p.add_argument("--min-corr-n", type=int, default=3)
    p.add_argument("--max-distance-km", type=float, default=150.0)
    p.add_argument("--progress-every", type=int, default=12)
    return p.parse_args()


def parse_ylim(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if s is None or str(s).strip() == "":
        return None
    parts = [float(x.strip()) for x in str(s).split(",")]
    if len(parts) != 2:
        raise ValueError("--ylim must be like 0,300")
    return parts[0], parts[1]


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    date_start = parse_date_arg(args.date_start)
    date_end = parse_date_arg(args.date_end)
    nd_vars_requested = split_csv_arg(args.nd_vars)
    ylim = parse_ylim(args.ylim)

    print("[CONFIG]")
    print(f"  obs_dir              = {args.obs_dir}")
    print(f"  ndnet_dir            = {args.ndnet_dir}")
    print(f"  ndnet_month_template = {args.ndnet_month_template}")
    print(f"  outdir               = {args.outdir}")
    print(f"  obs_var              = {args.obs_var}")

    obs_df, site_df = read_obs_sites(
        obs_dir=args.obs_dir,
        obs_pattern=args.obs_pattern,
        obs_var=args.obs_var,
        date_start=date_start,
        date_end=date_end,
        use_finite_obs_dates_only_for_metadata=not args.use_all_obs_dates,
    )

    site_df.to_csv(args.outdir / "site_metadata.csv", index=False)
    print(f"[WRITE] {args.outdir / 'site_metadata.csv'}")

    dates_by_month, file_by_month, missing_months = build_needed_ndnet_months(
        obs_df=obs_df,
        ndnet_dir=args.ndnet_dir,
        ndnet_month_template=args.ndnet_month_template,
        only_finite_obs_dates=not args.use_all_obs_dates,
    )

    sample_file = next(iter(file_by_month.values()))
    nd_vars = infer_nd_vars(sample_file, nd_vars_requested)
    extract_vars = ensure_summary_vars(sample_file, nd_vars)
    print(f"[NDnet] statistics variables: {', '.join(nd_vars)}")
    if extract_vars != list(nd_vars):
        added = [v for v in extract_vars if v not in nd_vars]
        print(f"[NDnet] added for grouped box plot: {', '.join(added)}")

    grid_df = compute_site_grid_mapping(site_df, sample_file, args.max_distance_km)
    grid_df.to_csv(args.outdir / "site_to_ndnet_grid_mapping.csv", index=False)
    print(f"[WRITE] {args.outdir / 'site_to_ndnet_grid_mapping.csv'}")

    pred_df, missing_time_df = extract_ndnet_monthly_for_sites(
        dates_by_month=dates_by_month,
        file_by_month=file_by_month,
        site_df=site_df,
        grid_df=grid_df,
        nd_vars=extract_vars,
        progress_every=args.progress_every,
    )

    merged = obs_df.merge(pred_df, on=["site", "date"], how="left")
    merged = merged.sort_values(["label", "date"]).reset_index(drop=True)
    merged.to_csv(args.outdir / "matched_daily_asr_ndnet.csv", index=False)
    print(f"[WRITE] {args.outdir / 'matched_daily_asr_ndnet.csv'}")

    stats_site, stats_all = make_stats_tables(
        merged, site_df, nd_vars, args.min_corr_n
    )
    stats_site.to_csv(
        args.outdir / "stats_by_site_and_ndnet_variable.csv", index=False
    )
    stats_all.to_csv(
        args.outdir / "stats_all_sites_by_ndnet_variable.csv", index=False
    )
    print(f"[WRITE] {args.outdir / 'stats_by_site_and_ndnet_variable.csv'}")
    print(f"[WRITE] {args.outdir / 'stats_all_sites_by_ndnet_variable.csv'}")

    if missing_months:
        miss_df = pd.DataFrame(
            {
                "missing_month": missing_months,
                "expected_file": [
                    str(
                        format_ndnet_month_path(
                            args.ndnet_dir, args.ndnet_month_template, m
                        )
                    )
                    for m in missing_months
                ],
            }
        )
        miss_df.to_csv(args.outdir / "missing_ndnet_months.csv", index=False)
        print(f"[WRITE] {args.outdir / 'missing_ndnet_months.csv'}")

    if not missing_time_df.empty:
        missing_time_df.to_csv(
            args.outdir / "missing_ndnet_dates_inside_months.csv", index=False
        )
        print(f"[WRITE] {args.outdir / 'missing_ndnet_dates_inside_months.csv'}")

    box_vars = ["obs_cm3"] + [v for v in nd_vars if v in merged.columns]
    make_boxplot_panels(
        merged=merged,
        site_df=site_df,
        box_vars=box_vars,
        out_base=args.outdir / "boxplot_ASR_vs_NDnet_allvars",
        ncols=args.ncols,
        ylim=ylim,
        formats=("png", "pdf"),
    )

    make_grouped_site_boxplot(
        merged=merged,
        site_df=site_df,
        out_base=args.outdir / "boxplot_ASR_raw_filtered_by_site",
        ylim=ylim,
        formats=("png", "pdf"),
    )

    print("[DONE]")
    print(f"  matched rows:               {len(merged)}")
    print(f"  sites:                      {len(site_df)}")
    print(f"  NDnet monthly files opened: {len(file_by_month)}")


if __name__ == "__main__":
    main()
