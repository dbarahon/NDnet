#!/usr/bin/env python3
"""
Journal-style comparison of NDnet with Grosvenor et al. (2018; GR18).

Figure layout
-------------
(a) NDnet multi-year annual mean
(b) GR18  multi-year annual mean
(c) NDnet - GR18
(d) Zonal means of the two annual-mean maps, with longitudinal IQR shading

Sampling / averaging
--------------------
* Only YYYYMM files available in BOTH archives are used.
* By default, only complete common years (all 12 months) are retained.
* Each file is first averaged over its internal ``time`` dimension.
* The resulting monthly means are averaged with equal weight. This avoids
  weighting NDnet by number of days while GR18 has a different internal time
  dimension.
* NDnet fields are interpolated to the GR18 180x360 grid before comparison.
  An optional NaN-surviving mode renormalizes linear interpolation over the
  finite source neighbors instead of allowing one NaN corner to mask the target.
* The difference map is computed month-by-month on colocated finite pixels and
  then averaged, rather than subtracting two independently sampled means.
* Panel (c) optionally hatches grid cells where the mean monthly difference
  exceeds a configurable multiple of its standard error (2 SE by default).

Outputs
-------
GR18_NDnet_comparison.png
GR18_NDnet_comparison.pdf
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Dict, Tuple

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
# 12 months exist in both datasets.
COMPLETE_YEARS_ONLY = True

# GR18 is 180x360 whereas NDnet is 181x360. Bilinear interpolation is the
# default. Change to "nearest" if you want to avoid any spatial smoothing.
REGRID_METHOD = "linear"

# If True, linear regridding ignores missing source neighbors and renormalizes
# the interpolation weights over the remaining finite values. If False, use
# xarray's standard interpolation, where a NaN source corner can yield NaN at
# the target point. This can also be overridden with --nan-surviving-regrid.
NAN_SURVIVING_REGRID = False

# Plot limits. Fixed values make the two absolute-Nd panels directly comparable.
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

# Hatching on panel (c). The monthly colocated NDnet-GR18 differences are used
# to estimate the sample standard error of the mean at every grid cell. Hatch
# where |mean difference| exceeds HATCH_SE_MULTIPLIER * SE and there are at
# least HATCH_MIN_MONTHS valid monthly pairs.
SHOW_HATCHING = True
HATCH_SE_MULTIPLIER = 2.0
HATCH_MIN_MONTHS = 12
HATCH_PATTERN = "/"
HATCH_LINEWIDTH = 0.35

# Figure spacing controls.
# ROW_GAP_RATIO inserts explicit whitespace between the two panel rows.
# ZONAL_TOP_PAD_RATIO lowers panel (d) slightly relative to panel (c).
ROW_GAP_RATIO = 0.12
ZONAL_TOP_PAD_RATIO = 0.08

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
    gr18: Dict[str, Path], ndnet: Dict[str, Path]
) -> list[str]:
    """Return common YYYYMM entries after optional period/year filtering."""
    common = sorted(set(gr18) & set(ndnet))

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


def regrid_to(
    da: xr.DataArray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    *,
    nan_surviving: bool = NAN_SURVIVING_REGRID,
) -> xr.DataArray:
    """Interpolate to target lat/lon, treating longitude as periodic.

    In NaN-surviving mode, interpolate the zero-filled field and a finite-data
    mask separately, then divide by the interpolated mask. This renormalizes
    the linear weights over finite source neighbors, so a single NaN corner
    does not automatically make the target grid cell NaN.
    """
    src = add_periodic_lon(da)
    target = {
        "lat": xr.DataArray(target_lat, dims="lat"),
        "lon": xr.DataArray(target_lon, dims="lon"),
    }

    if nan_surviving and REGRID_METHOD == "linear":
        finite = xr.where(np.isfinite(src), 1.0, 0.0)
        numerator = src.fillna(0.0).interp(method=REGRID_METHOD, **target)
        weight = finite.interp(method=REGRID_METHOD, **target)
        out = (numerator / weight).where(weight > 1.0e-12)
    else:
        out = src.interp(method=REGRID_METHOD, **target)

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


class MeanSEAccumulator:
    """NaN-aware online mean and standard error for a fixed 2-D grid."""

    def __init__(self, shape: Tuple[int, int]):
        self.sum = np.zeros(shape, dtype=np.float64)
        self.sumsq = np.zeros(shape, dtype=np.float64)
        self.count = np.zeros(shape, dtype=np.int32)

    def add(self, arr: np.ndarray) -> None:
        arr = np.asarray(arr, dtype=np.float64)
        valid = np.isfinite(arr)
        values = arr[valid]
        self.sum[valid] += values
        self.sumsq[valid] += values * values
        self.count[valid] += 1

    def mean(self) -> np.ndarray:
        out = np.full(self.sum.shape, np.nan, dtype=np.float64)
        np.divide(self.sum, self.count, out=out, where=self.count > 0)
        return out

    def standard_error(self) -> np.ndarray:
        """Return sample standard deviation / sqrt(N), requiring N >= 2."""
        out = np.full(self.sum.shape, np.nan, dtype=np.float64)
        valid = self.count >= 2
        if not np.any(valid):
            return out

        n = self.count[valid].astype(np.float64)
        # Numerically stable sample variance from sum and sum of squares.
        ss = self.sumsq[valid] - (self.sum[valid] ** 2) / n
        ss = np.maximum(ss, 0.0)
        sample_var = ss / (n - 1.0)
        out[valid] = np.sqrt(sample_var / n)
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


# =============================================================================
# MAIN
# =============================================================================
def main(nan_surviving_regrid: bool = NAN_SURVIVING_REGRID) -> None:
    gr18_files = discover_months(GR18_DIR, GR18_GLOB)
    ndnet_files = discover_months(NDNET_DIR, NDNET_GLOB)
    months = select_common_months(gr18_files, ndnet_files)

    years = sorted({int(m[:4]) for m in months})
    print(f"Using {len(months)} common months")
    print(f"Years: {years[0]}-{years[-1]} ({len(years)} years)")
    if COMPLETE_YEARS_ONLY:
        print("Complete common years only: yes")
    print(
        "NaN-surviving linear regridding: "
        f"{'yes' if nan_surviving_regrid else 'no'}"
    )

    # First GR18 month defines the common comparison grid.
    gr0 = load_monthly_mean(gr18_files[months[0]], GR18_VAR)
    target_lat = np.asarray(gr0.lat.values, dtype=float)
    target_lon = np.asarray(gr0.lon.values, dtype=float)
    shape = (target_lat.size, target_lon.size)

    acc_gr18 = MeanAccumulator(shape)
    acc_ndnet = MeanAccumulator(shape)
    acc_diff = MeanSEAccumulator(shape)

    for i, yyyymm in enumerate(months, start=1):
        gr = load_monthly_mean(gr18_files[yyyymm], GR18_VAR)

        # Verify that the GR18 grid itself is stable across files.
        if (
            gr.sizes["lat"] != target_lat.size
            or gr.sizes["lon"] != target_lon.size
            or not np.allclose(gr.lat.values, target_lat)
            or not np.allclose(gr.lon.values, target_lon)
        ):
            gr = regrid_to(
                gr,
                target_lat,
                target_lon,
                nan_surviving=nan_surviving_regrid,
            )

        ndnet = load_monthly_mean(ndnet_files[yyyymm], NDNET_VAR)
        ndnet = regrid_to(
            ndnet,
            target_lat,
            target_lon,
            nan_surviving=nan_surviving_regrid,
        )

        g = np.asarray(gr.values, dtype=float)
        n = np.asarray(ndnet.values, dtype=float)

        acc_gr18.add(g)
        acc_ndnet.add(n)

        # Colocated monthly differences avoid subtracting climatologies with
        # different valid-pixel masks. The same monthly differences are also
        # used to estimate the standard error for panel-(c) hatching.
        valid = np.isfinite(n) & np.isfinite(g)
        monthly_diff = np.where(valid, n - g, np.nan)
        acc_diff.add(monthly_diff)

        if i == 1 or i % 12 == 0 or i == len(months):
            print(f"  processed {i:4d}/{len(months)} months ({yyyymm})")

    mean_gr18 = acc_gr18.mean()
    mean_ndnet = acc_ndnet.mean()
    diff_ndnet = acc_diff.mean()
    diff_se = acc_diff.standard_error()
    diff_n = acc_diff.count.copy()

    gm_ndnet = area_weighted_mean(mean_ndnet, target_lat)
    gm_gr18 = area_weighted_mean(mean_gr18, target_lat)
    gm_diff = area_weighted_mean(diff_ndnet, target_lat)

    print("\nArea-weighted means on the GR18 grid:")
    print(f"  NDnet      : {gm_ndnet:8.3f} cm-3")
    print(f"  GR18       : {gm_gr18:8.3f} cm-3")
    print(f"  NDnet-GR18 : {gm_diff:8.3f} cm-3")
    if SHOW_HATCHING:
        hatch_valid = np.isfinite(diff_se) & (diff_n >= HATCH_MIN_MONTHS)
        hatch_sig = hatch_valid & (
            np.abs(diff_ndnet) > HATCH_SE_MULTIPLIER * diff_se
        )
        n_hatch = int(np.count_nonzero(hatch_sig))
        n_tested = int(np.count_nonzero(hatch_valid))
        print(
            "  Hatching    : "
            f"|mean difference| > {HATCH_SE_MULTIPLIER:g} SE, "
            f"N >= {HATCH_MIN_MONTHS}; "
            f"{n_hatch}/{n_tested} tested grid cells hatched"
        )
    else:
        hatch_sig = None

    nd_bounds = np.arange(ND_MIN, ND_MAX + 0.5 * ND_INTERVAL, ND_INTERVAL)
    diff_bounds = np.arange(
        -DIFF_MAX,
        DIFF_MAX + 0.5 * DIFF_INTERVAL,
        DIFF_INTERVAL,
    )
    nd_norm = mcolors.BoundaryNorm(
        nd_bounds,
        ncolors=mpl.colormaps[ND_CMAP].N,
        clip=False,
    )
    diff_norm = mcolors.BoundaryNorm(
        diff_bounds,
        ncolors=mpl.colormaps[DIFF_CMAP].N,
        clip=False,
    )

    # Compact two-by-two journal layout. Colorbars occupy narrow rows below
    # the panels and are not counted as figure panels. An explicit spacer row
    # controls the separation between the two panel rows.
    fig = plt.figure(figsize=(8.2, 6.45))
    gs = fig.add_gridspec(
        nrows=5,
        ncols=2,
        height_ratios=[1.0, 0.055, ROW_GAP_RATIO, 1.0, 0.055],
        hspace=0.14,
        wspace=0.16,
        left=0.065,
        right=0.985,
        bottom=0.075,
        top=0.965,
    )

    ax_a = fig.add_subplot(gs[0, 0], projection=MAP_PROJ)
    ax_b = fig.add_subplot(gs[0, 1], projection=MAP_PROJ)
    cax_nd = fig.add_subplot(gs[1, :])

    ax_c = fig.add_subplot(gs[3, 0], projection=MAP_PROJ)

    # Lower the zonal-mean panel slightly without moving panel (c).
    gs_zonal = gs[3, 1].subgridspec(
        2,
        1,
        height_ratios=[ZONAL_TOP_PAD_RATIO, 1.0],
        hspace=0.0,
    )
    ax_d = fig.add_subplot(gs_zonal[1, 0])

    cax_diff = fig.add_subplot(gs[4, 0])
    ax_cbar_blank = fig.add_subplot(gs[4, 1])
    ax_cbar_blank.axis("off")

    m_nd = map_field(
        ax_a,
        target_lon,
        target_lat,
        mean_ndnet,
        cmap=ND_CMAP,
        norm=nd_norm,
        left_title="(a) NDnet",
        right_title=rf"$\overline{{N_d}}_A={gm_ndnet:.1f}$",
    )
    map_field(
        ax_b,
        target_lon,
        target_lat,
        mean_gr18,
        cmap=ND_CMAP,
        norm=nd_norm,
        left_title="(b) Grosvenor et al. (2018)",
        right_title=rf"$\overline{{N_d}}_A={gm_gr18:.1f}$",
    )
    m_diff = map_field(
        ax_c,
        target_lon,
        target_lat,
        diff_ndnet,
        cmap=DIFF_CMAP,
        norm=diff_norm,
        left_title="(c) NDnet - GR18",
        right_title=rf"$\overline{{\Delta N_d}}_A={gm_diff:+.1f}$",
    )

    if SHOW_HATCHING and hatch_sig is not None:
        # Hatch cells where the mean monthly difference is large relative to
        # its sampling uncertainty. This is a 2-SE-style robustness screen,
        # not a formal autocorrelation-adjusted hypothesis test.
        mpl.rcParams["hatch.linewidth"] = HATCH_LINEWIDTH
        hatch_field = np.where(hatch_sig, 1.0, np.nan)
        hatch_cyclic, hatch_lon = add_cyclic_point(
            hatch_field,
            coord=target_lon,
            axis=1,
        )
        ax_c.contourf(
            hatch_lon,
            target_lat,
            hatch_cyclic,
            levels=[0.5, 1.5],
            colors="none",
            hatches=[HATCH_PATTERN],
            transform=DATA_CRS,
            zorder=3,
        )
    cb_nd = fig.colorbar(
        m_nd,
        cax=cax_nd,
        orientation="horizontal",
        ticks=nd_bounds[::2],
        extend="max",
    )
    cb_nd.set_label(r"$N_d$ (cm$^{-3}$)", labelpad=2)
    cb_nd.ax.tick_params(length=2.5, pad=1.5)

    cb_diff = fig.colorbar(
        m_diff,
        cax=cax_diff,
        orientation="horizontal",
        ticks=diff_bounds[::2],
        extend="both",
    )
    cb_diff.set_label(r"NDnet - GR18 (cm$^{-3}$)", labelpad=2)
    cb_diff.ax.tick_params(length=2.5, pad=1.5)

    # Zonal means and longitudinal variability of the climatological maps.
    zonal_cases = [
        ("NDnet", mean_ndnet, "tab:orange", "-", 1.55),
        ("GR18", mean_gr18, "0.15", "-", 1.45),
    ]
    for label, field, color, ls, lw in zonal_cases:
        zmean, zlo, zhi = zonal_stats(field)
        ax_d.fill_between(
            target_lat,
            zlo,
            zhi,
            color=color,
            alpha=0.13,
            linewidth=0,
        )
        ax_d.plot(
            target_lat,
            zmean,
            color=color,
            ls=ls,
            lw=lw,
            label=label,
        )

    spread_label = (
        "IQR across longitude"
        if ZONAL_SPREAD.lower() == "iqr"
        else r"$\pm1\sigma$ across longitude"
    )
    ax_d.set_xlim(-90, 90)
    ax_d.set_xticks(np.arange(-90, 91, 30))
    ax_d.set_ylim(bottom=0)
    ax_d.grid(True, which="major", linewidth=0.35, alpha=0.35, linestyle=":")
    ax_d.tick_params(direction="out", right=True)
    ax_d.set_xlabel("Latitude (deg)")
    ax_d.set_ylabel(r"$N_d$ (cm$^{-3}$)")
    ax_d.set_title("(d) Zonal mean", loc="left", pad=3.0, fontweight="bold")
    ax_d.set_title(spread_label, loc="right", pad=3.0, fontsize=7.5)
    ax_d.legend(frameon=False, loc="upper right", handlelength=2.4)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    png = OUTDIR / f"{OUT_BASENAME}.png"
    pdf = OUTDIR / f"{OUT_BASENAME}.pdf"
    fig.savefig(png, dpi=DPI, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    print(f"\nSaved:\n  {png}\n  {pdf}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare NDnet with Grosvenor et al. (2018)."
    )
    parser.add_argument(
        "--nan-surviving-regrid",
        action=argparse.BooleanOptionalAction,
        default=NAN_SURVIVING_REGRID,
        help=(
            "Renormalize linear interpolation over finite source neighbors "
            "(default: %(default)s). Use --no-nan-surviving-regrid for "
            "standard xarray NaN propagation."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(nan_surviving_regrid=args.nan_surviving_regrid)
