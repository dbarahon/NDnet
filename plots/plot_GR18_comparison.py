#!/usr/bin/env python3
"""
GR18 vs NDnet-prior vs NDnet comparison.

Figure layout
-------------
(a) NDnet-prior multi-year annual mean
(b) NDnet       multi-year annual mean
(c) GR18        multi-year annual mean
(d) NDnet-prior - GR18
(e) NDnet       - GR18
(f) Zonal means of the three annual-mean maps, with longitudinal IQR shading

Sampling / averaging
--------------------
* Only YYYYMM files available in ALL THREE archives are used.
* By default, only complete common years (all 12 months) are retained.
* Each file is first averaged over its internal ``time`` dimension.
* The resulting monthly means are averaged with equal weight. This avoids
  weighting NDnet by number of days while GR18 has a different internal time
  dimension.
* NDnet fields are interpolated to the GR18 180x360 grid before comparison.
* Difference maps are computed month-by-month on colocated finite pixels and
  then averaged, rather than subtracting two independently sampled means.

Outputs
-------
GR18_NDnet_comparison.png
GR18_NDnet_comparison.pdf
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point


# =============================================================================
# USER SETTINGS
# =============================================================================
GR18_DIR = Path(
    "/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/dailies/"
    "MODIS_ND_RETRIEVAL/Monthlies"
)
PRIOR_DIR = Path(
    "/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/inference/"
    "NDnet_2D_monthly"
)
NDNET_DIR = Path(
    "/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/inference/"
    "after_refinement/NDnet_2D_monthly"
)

GR18_GLOB = "Nd37_SZA_LT_65_CF_GT_80_CTH_LT_3.2km.*.nc"
NDNET_GLOB = "NDnet_modis_compare_800_950hPa_*.nc"

GR18_VAR = "Nd"
NDNET_VAR = "nd_modis_track_ctt_cf"

# None -> use full common period. Set, for example, 2009 and 2018 to restrict it.
START_YEAR = None
END_YEAR = None

# Recommended for a true annual climatology: keep only years for which all
# 12 months exist in all three datasets.
COMPLETE_YEARS_ONLY = True

# GR18 is 180x360 whereas NDnet is 181x360. Bilinear interpolation is the
# default. Change to "nearest" if you want to avoid any spatial smoothing.
REGRID_METHOD = "linear"

# Plot limits. These are deliberately fixed for comparability across figures.
ND_MIN = 0.0
ND_MAX = 300.0
ND_INTERVAL = 25.0
DIFF_MAX = 120.0
DIFF_INTERVAL = 20.0

ND_CMAP = "YlGnBu"
DIFF_CMAP = "RdBu_r"

# Zonal variability envelope. "iqr" = 25th-75th percentile across longitude;
# "std" = zonal mean +/- 1 standard deviation.
ZONAL_SPREAD = "iqr"

# Extra vertical whitespace between the two panel rows.  This is an explicit
# spacer in the GridSpec, so increasing it does not push the colorbars away
# from their associated panels.
VERTICAL_GAP_RATIO = 0.18

# Broken y axis for panel (f).  This keeps the ~100-300 cm-3 structure legible
# while retaining the high-latitude NDnet-prior values that approach ~700.
ZONAL_BROKEN_AXIS = True
ZONAL_LOW_LIM = (0.0, 350.0)
ZONAL_HIGH_LIM = (500.0, 750.0)
# Relative heights of the upper and lower pieces of the broken axis.
ZONAL_BREAK_HEIGHT_RATIOS = (1.0, 3.2)
ZONAL_BREAK_HSPACE = 0.06

OUTDIR = Path(".")
OUT_BASENAME = "GR18_NDnet_comparison"
DPI = 400


# =============================================================================
# JOURNAL STYLE
# =============================================================================
mpl.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "savefig.bbox": "tight",
    }
)

MAP_PROJ = ccrs.Robinson(central_longitude=0)
DATA_CRS = ccrs.PlateCarree()


# =============================================================================
# HELPERS
# =============================================================================
def discover_months(directory: Path, pattern: str) -> Dict[str, Path]:
    """Return {YYYYMM: path}; fail on duplicate months."""
    out: Dict[str, Path] = {}
    for path in sorted(directory.glob(pattern)):
        m = re.search(r"(\d{6})(?=\.nc$)", path.name)
        if m is None:
            continue
        yyyymm = m.group(1)
        if yyyymm in out:
            raise RuntimeError(
                f"Duplicate month {yyyymm} in {directory}:\n"
                f"  {out[yyyymm]}\n  {path}"
            )
        out[yyyymm] = path
    if not out:
        raise FileNotFoundError(f"No files matching {pattern!r} in {directory}")
    return out


def select_common_months(
    gr18: Dict[str, Path], prior: Dict[str, Path], ndnet: Dict[str, Path]
) -> list[str]:
    common = sorted(set(gr18) & set(prior) & set(ndnet))

    if START_YEAR is not None:
        common = [m for m in common if int(m[:4]) >= START_YEAR]
    if END_YEAR is not None:
        common = [m for m in common if int(m[:4]) <= END_YEAR]

    if COMPLETE_YEARS_ONLY:
        years = sorted({int(m[:4]) for m in common})
        complete = []
        for year in years:
            months = {int(m[4:6]) for m in common if int(m[:4]) == year}
            if months == set(range(1, 13)):
                complete.append(year)
        common = [m for m in common if int(m[:4]) in complete]

    if not common:
        raise RuntimeError(
            "No common months remain after applying the requested period / "
            "complete-year filtering."
        )
    return common


def canonicalize_latlon(da: xr.DataArray) -> xr.DataArray:
    """Return a 2-D field on increasing lat and [0, 360) increasing lon."""
    da = da.squeeze(drop=True)
    if set(da.dims) != {"lat", "lon"}:
        # Permit dimension order differences, but no extra dimensions here.
        extra = [d for d in da.dims if d not in ("lat", "lon")]
        if extra:
            raise ValueError(f"Unexpected dimensions after time mean: {da.dims}")
    da = da.transpose("lat", "lon")

    # Force obviously invalid fill values to NaN even if CF decoding did not.
    da = da.where(np.isfinite(da) & (np.abs(da) < 1.0e10))

    lon = np.mod(np.asarray(da["lon"].values, dtype=float), 360.0)
    da = da.assign_coords(lon=lon).sortby("lon").sortby("lat")

    # Remove a possible duplicate seam (e.g., both 0 and 360 degrees).
    rounded = np.round(np.asarray(da.lon.values), 8)
    _, unique_idx = np.unique(rounded, return_index=True)
    if unique_idx.size != da.sizes["lon"]:
        da = da.isel(lon=np.sort(unique_idx))

    return da


def load_monthly_mean(path: Path, varname: str) -> xr.DataArray:
    """Read one file, average over its internal time dimension, load to memory."""
    with xr.open_dataset(path, mask_and_scale=True, decode_times=False) as ds:
        if varname not in ds:
            raise KeyError(f"Variable {varname!r} not found in {path}")
        da = ds[varname]
        if "time" in da.dims:
            da = da.mean("time", skipna=True)
        da = da.load()
    return canonicalize_latlon(da)


def add_periodic_lon(da: xr.DataArray) -> xr.DataArray:
    """Add one grid cell at either side so interpolation is periodic in longitude."""
    left = da.isel(lon=[-1]).copy()
    left = left.assign_coords(lon=np.asarray(left.lon.values) - 360.0)
    right = da.isel(lon=[0]).copy()
    right = right.assign_coords(lon=np.asarray(right.lon.values) + 360.0)
    return xr.concat([left, da, right], dim="lon").sortby("lon")


def regrid_to(da: xr.DataArray, target_lat: np.ndarray, target_lon: np.ndarray) -> xr.DataArray:
    """Interpolate to target lat/lon, treating longitude as periodic."""
    src = add_periodic_lon(da)
    out = src.interp(
        lat=xr.DataArray(target_lat, dims="lat"),
        lon=xr.DataArray(target_lon, dims="lon"),
        method=REGRID_METHOD,
    )
    return out.transpose("lat", "lon")


class MeanAccumulator:
    """NaN-aware online mean for a fixed 2-D grid."""

    def __init__(self, shape: Tuple[int, int]):
        self.sum = np.zeros(shape, dtype=np.float64)
        self.count = np.zeros(shape, dtype=np.int32)

    def add(self, arr: np.ndarray) -> None:
        arr = np.asarray(arr, dtype=np.float64)
        valid = np.isfinite(arr)
        self.sum[valid] += arr[valid]
        self.count[valid] += 1

    def mean(self) -> np.ndarray:
        out = np.full(self.sum.shape, np.nan, dtype=np.float64)
        np.divide(self.sum, self.count, out=out, where=self.count > 0)
        return out


def area_weighted_mean(field: np.ndarray, lat: np.ndarray) -> float:
    """Cos(latitude)-weighted mean over all finite grid cells."""
    field = np.asarray(field, dtype=float)
    weights = np.cos(np.deg2rad(np.asarray(lat, dtype=float)))[:, None]
    valid = np.isfinite(field)
    den = np.sum(np.where(valid, weights, 0.0))
    if den == 0:
        return np.nan
    num = np.sum(np.where(valid, field * weights, 0.0))
    return float(num / den)


def zonal_stats(field: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zonal mean and requested longitudinal spread at each latitude."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(field, axis=1)
        if ZONAL_SPREAD.lower() == "iqr":
            lo = np.nanpercentile(field, 25.0, axis=1)
            hi = np.nanpercentile(field, 75.0, axis=1)
        elif ZONAL_SPREAD.lower() == "std":
            std = np.nanstd(field, axis=1)
            lo = mean - std
            hi = mean + std
        else:
            raise ValueError("ZONAL_SPREAD must be 'iqr' or 'std'")
    return mean, lo, hi


def map_field(
    ax,
    lon: np.ndarray,
    lat: np.ndarray,
    field: np.ndarray,
    *,
    cmap,
    norm,
    left_title: str,
    right_title: str,
):
    """Draw one global map with consistent journal formatting."""
    cyclic, cyclic_lon = add_cyclic_point(field, coord=lon, axis=1)
    mesh = ax.pcolormesh(
        cyclic_lon,
        lat,
        cyclic,
        transform=DATA_CRS,
        cmap=cmap,
        norm=norm,
        shading="auto",
        rasterized=True,
    )

    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="0.92", edgecolor="none", zorder=0)
    ax.coastlines(linewidth=0.45, color="0.25")

    gl = ax.gridlines(
        crs=DATA_CRS,
        draw_labels=True,
        xlocs=np.arange(-120, 181, 60),
        ylocs=np.arange(-60, 61, 30),
        linewidth=0.30,
        color="0.45",
        alpha=0.45,
        linestyle=":",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 6.5}
    gl.ylabel_style = {"size": 6.5}

    ax.set_title(left_title, loc="left", pad=3.0, fontweight="bold")
    ax.set_title(right_title, loc="right", pad=3.0, fontsize=7.5)
    return mesh



def add_broken_y_marks(ax_top, ax_bottom, size: float = 0.018) -> None:
    """Draw diagonal break marks between two vertically stacked axes."""
    kwargs_top = dict(
        transform=ax_top.transAxes,
        color="k",
        clip_on=False,
        linewidth=0.8,
    )
    kwargs_bottom = dict(
        transform=ax_bottom.transAxes,
        color="k",
        clip_on=False,
        linewidth=0.8,
    )

    # Bottom edge of upper axis.
    ax_top.plot((-size, +size), (-size, +size), **kwargs_top)
    ax_top.plot((1 - size, 1 + size), (-size, +size), **kwargs_top)

    # Top edge of lower axis.
    ax_bottom.plot((-size, +size), (1 - size, 1 + size), **kwargs_bottom)
    ax_bottom.plot((1 - size, 1 + size), (1 - size, 1 + size), **kwargs_bottom)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    gr18_files = discover_months(GR18_DIR, GR18_GLOB)
    prior_files = discover_months(PRIOR_DIR, NDNET_GLOB)
    ndnet_files = discover_months(NDNET_DIR, NDNET_GLOB)
    months = select_common_months(gr18_files, prior_files, ndnet_files)

    years = sorted({int(m[:4]) for m in months})
    print(f"Using {len(months)} common months")
    print(f"Years: {years[0]}-{years[-1]} ({len(years)} years)")
    if COMPLETE_YEARS_ONLY:
        print("Complete common years only: yes")

    # First GR18 month defines the common comparison grid.
    gr0 = load_monthly_mean(gr18_files[months[0]], GR18_VAR)
    target_lat = np.asarray(gr0.lat.values, dtype=float)
    target_lon = np.asarray(gr0.lon.values, dtype=float)
    shape = (target_lat.size, target_lon.size)

    acc_gr18 = MeanAccumulator(shape)
    acc_prior = MeanAccumulator(shape)
    acc_ndnet = MeanAccumulator(shape)
    acc_dprior = MeanAccumulator(shape)
    acc_dndnet = MeanAccumulator(shape)

    for i, yyyymm in enumerate(months, start=1):
        gr = load_monthly_mean(gr18_files[yyyymm], GR18_VAR)

        # Verify that the GR18 grid itself is stable across files.
        if (
            gr.sizes["lat"] != target_lat.size
            or gr.sizes["lon"] != target_lon.size
            or not np.allclose(gr.lat.values, target_lat)
            or not np.allclose(gr.lon.values, target_lon)
        ):
            gr = regrid_to(gr, target_lat, target_lon)

        prior = load_monthly_mean(prior_files[yyyymm], NDNET_VAR)
        ndnet = load_monthly_mean(ndnet_files[yyyymm], NDNET_VAR)
        prior = regrid_to(prior, target_lat, target_lon)
        ndnet = regrid_to(ndnet, target_lat, target_lon)

        g = np.asarray(gr.values, dtype=float)
        p = np.asarray(prior.values, dtype=float)
        n = np.asarray(ndnet.values, dtype=float)

        acc_gr18.add(g)
        acc_prior.add(p)
        acc_ndnet.add(n)

        # Colocated monthly differences. This is preferable to subtracting
        # two climatologies with different valid-pixel masks.
        vp = np.isfinite(p) & np.isfinite(g)
        vn = np.isfinite(n) & np.isfinite(g)
        acc_dprior.add(np.where(vp, p - g, np.nan))
        acc_dndnet.add(np.where(vn, n - g, np.nan))

        if i == 1 or i % 12 == 0 or i == len(months):
            print(f"  processed {i:4d}/{len(months)} months ({yyyymm})")

    mean_gr18 = acc_gr18.mean()
    mean_prior = acc_prior.mean()
    mean_ndnet = acc_ndnet.mean()
    diff_prior = acc_dprior.mean()
    diff_ndnet = acc_dndnet.mean()

    gm_prior = area_weighted_mean(mean_prior, target_lat)
    gm_ndnet = area_weighted_mean(mean_ndnet, target_lat)
    gm_gr18 = area_weighted_mean(mean_gr18, target_lat)
    gm_dprior = area_weighted_mean(diff_prior, target_lat)
    gm_dndnet = area_weighted_mean(diff_ndnet, target_lat)

    print("\nArea-weighted means on the GR18 grid:")
    print(f"  NDnet-prior : {gm_prior:8.3f} cm-3")
    print(f"  NDnet       : {gm_ndnet:8.3f} cm-3")
    print(f"  GR18        : {gm_gr18:8.3f} cm-3")
    print(f"  Prior-GR18  : {gm_dprior:8.3f} cm-3")
    print(f"  NDnet-GR18  : {gm_dndnet:8.3f} cm-3")

    nd_bounds = np.arange(ND_MIN, ND_MAX + 0.5 * ND_INTERVAL, ND_INTERVAL)
    d_bounds = np.arange(-DIFF_MAX, DIFF_MAX + 0.5 * DIFF_INTERVAL, DIFF_INTERVAL)
    nd_norm = mcolors.BoundaryNorm(nd_bounds, ncolors=mpl.colormaps[ND_CMAP].N, clip=False)
    d_norm = mcolors.BoundaryNorm(d_bounds, ncolors=mpl.colormaps[DIFF_CMAP].N, clip=False)

    fig = plt.figure(figsize=(11.8, 7.10))

    # Five outer rows: top maps, their colorbar, an explicit whitespace row,
    # bottom panels, and the difference colorbar.  VERTICAL_GAP_RATIO controls
    # only the separation between the two panel rows.
    outer = fig.add_gridspec(
        nrows=5,
        ncols=1,
        height_ratios=[1.0, 0.045, VERTICAL_GAP_RATIO, 1.0, 0.045],
        hspace=0.08,
        left=0.045,
        right=0.985,
        bottom=0.070,
        top=0.965,
    )

    gs_top = outer[0].subgridspec(1, 3, wspace=0.08)
    gs_bottom = outer[3].subgridspec(
        1, 3,
        width_ratios=[1.0, 1.0, 1.0],
        wspace=0.20,
    )
    gs_diff_cb = outer[4].subgridspec(1, 3, wspace=0.20)

    ax_a = fig.add_subplot(gs_top[0, 0], projection=MAP_PROJ)
    ax_b = fig.add_subplot(gs_top[0, 1], projection=MAP_PROJ)
    ax_c = fig.add_subplot(gs_top[0, 2], projection=MAP_PROJ)
    cax_top = fig.add_subplot(outer[1])

    ax_d = fig.add_subplot(gs_bottom[0, 0], projection=MAP_PROJ)
    ax_e = fig.add_subplot(gs_bottom[0, 1], projection=MAP_PROJ)

    # Panel (f) can use a broken y axis so the dominant 100-300 cm-3 range
    # remains expanded without discarding the ~700 cm-3 NDnet-prior tail.
    if ZONAL_BROKEN_AXIS:
        gs_zonal = gs_bottom[0, 2].subgridspec(
            2,
            1,
            height_ratios=ZONAL_BREAK_HEIGHT_RATIOS,
            hspace=ZONAL_BREAK_HSPACE,
        )
        ax_f_hi = fig.add_subplot(gs_zonal[0, 0])
        ax_f_lo = fig.add_subplot(gs_zonal[1, 0], sharex=ax_f_hi)
        zonal_axes = (ax_f_hi, ax_f_lo)
    else:
        ax_f_lo = fig.add_subplot(gs_bottom[0, 2])
        ax_f_hi = None
        zonal_axes = (ax_f_lo,)

    cax_diff = fig.add_subplot(gs_diff_cb[0, 0:2])
    ax_blank = fig.add_subplot(gs_diff_cb[0, 2])
    ax_blank.axis("off")

    m_nd = map_field(
        ax_a,
        target_lon,
        target_lat,
        mean_prior,
        cmap=ND_CMAP,
        norm=nd_norm,
        left_title="(a) NDnet-prior",
        right_title=rf"$\overline{{N_d}}_A={gm_prior:.1f}$",
    )
    map_field(
        ax_b,
        target_lon,
        target_lat,
        mean_ndnet,
        cmap=ND_CMAP,
        norm=nd_norm,
        left_title="(b) NDnet",
        right_title=rf"$\overline{{N_d}}_A={gm_ndnet:.1f}$",
    )
    map_field(
        ax_c,
        target_lon,
        target_lat,
        mean_gr18,
        cmap=ND_CMAP,
        norm=nd_norm,
        left_title="(c) GR18",
        right_title=rf"$\overline{{N_d}}_A={gm_gr18:.1f}$",
    )

    m_d = map_field(
        ax_d,
        target_lon,
        target_lat,
        diff_prior,
        cmap=DIFF_CMAP,
        norm=d_norm,
        left_title="(d) NDnet-prior - GR18",
        right_title=rf"$\overline{{\Delta N_d}}_A={gm_dprior:+.1f}$",
    )
    map_field(
        ax_e,
        target_lon,
        target_lat,
        diff_ndnet,
        cmap=DIFF_CMAP,
        norm=d_norm,
        left_title="(e) NDnet - GR18",
        right_title=rf"$\overline{{\Delta N_d}}_A={gm_dndnet:+.1f}$",
    )

    # Shared color bars: one for all absolute-Nd panels and one for both biases.
    cb1 = fig.colorbar(
        m_nd,
        cax=cax_top,
        orientation="horizontal",
        ticks=nd_bounds[::2],
        extend="max",
    )
    cb1.set_label(r"$N_d$ (cm$^{-3}$)", labelpad=2)
    cb1.ax.tick_params(length=2.5, pad=1.5)

    cb2 = fig.colorbar(
        m_d,
        cax=cax_diff,
        orientation="horizontal",
        ticks=d_bounds[::2],
        extend="both",
    )
    cb2.set_label(r"$\Delta N_d$ (cm$^{-3}$)", labelpad=2)
    cb2.ax.tick_params(length=2.5, pad=1.5)

    # Zonal means + longitudinal variability of the annual-mean maps.
    datasets = [
        ("NDnet-prior", mean_prior, "tab:blue", "--", 1.45),
        ("NDnet", mean_ndnet, "tab:orange", "-", 1.55),
        ("GR18", mean_gr18, "0.15", "-", 1.45),
    ]
    for label, field, color, ls, lw in datasets:
        zmean, zlo, zhi = zonal_stats(field)
        for ax in zonal_axes:
            ax.fill_between(
                target_lat, zlo, zhi, color=color, alpha=0.13, linewidth=0
            )
            ax.plot(target_lat, zmean, color=color, ls=ls, lw=lw, label=label)

    spread_label = (
        "IQR across longitude"
        if ZONAL_SPREAD.lower() == "iqr"
        else r"$\pm1\sigma$ across longitude"
    )

    # Common x formatting for either a normal or broken zonal axis.
    for ax in zonal_axes:
        ax.set_xlim(-90, 90)
        ax.set_xticks(np.arange(-90, 91, 30))
        ax.grid(True, which="major", linewidth=0.35, alpha=0.35, linestyle=":")
        ax.tick_params(direction="out", right=True)

    if ZONAL_BROKEN_AXIS:
        ax_f_lo.set_ylim(*ZONAL_LOW_LIM)
        ax_f_hi.set_ylim(*ZONAL_HIGH_LIM)

        # Hide the adjoining spines/ticks and add conventional diagonal marks.
        ax_f_hi.spines["bottom"].set_visible(False)
        ax_f_lo.spines["top"].set_visible(False)
        ax_f_hi.tick_params(
            axis="x", bottom=False, labelbottom=False, top=True, labeltop=False
        )
        ax_f_lo.tick_params(axis="x", top=False, bottom=True)
        add_broken_y_marks(ax_f_hi, ax_f_lo)

        ax_f_hi.set_title(
            "(f) Zonal mean", loc="left", pad=3.0, fontweight="bold"
        )
        ax_f_hi.set_title(spread_label, loc="right", pad=3.0, fontsize=7.5)
        ax_f_lo.set_xlabel("Latitude (deg)")
        ax_f_lo.set_ylabel(r"$N_d$ (cm$^{-3}$)")

        # Put the legend in the upper segment; it uses the same artists but
        # does not obscure the main 100-300 cm-3 structure in the lower axis.
        ax_f_hi.legend(
            frameon=False, loc="upper center", ncol=3, handlelength=2.0,
            columnspacing=0.9, borderaxespad=0.2
        )
    else:
        ax_f_lo.set_title(
            "(f) Zonal mean", loc="left", pad=3.0, fontweight="bold"
        )
        ax_f_lo.set_title(spread_label, loc="right", pad=3.0, fontsize=7.5)
        ax_f_lo.set_xlabel("Latitude (deg)")
        ax_f_lo.set_ylabel(r"$N_d$ (cm$^{-3}$)")
        ax_f_lo.set_ylim(bottom=0)
        ax_f_lo.legend(frameon=False, loc="upper right", handlelength=2.4)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    png = OUTDIR / f"{OUT_BASENAME}.png"
    pdf = OUTDIR / f"{OUT_BASENAME}.pdf"
    fig.savefig(png, dpi=DPI, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    print(f"\nSaved:\n  {png}\n  {pdf}")


if __name__ == "__main__":
    main()
