#!/usr/bin/env python3
"""
Journal-style comparison of Base, MODIS-corrected, and Refined Nd.

Rows 1-3: time-mean horizontal maps at three pressure levels
Row 4   : zonal mean (time mean, then longitude mean)

Columns: Base | MODIS corrected | Refined
"""

import os
import glob
import string
import argparse

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

BASE_PATH = "/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/dailies/stats_mkincloud/monthlies/GiOcean.NDINC*.nc4"
BASE_VAR = "NDinc_cf01p"

MODIS_CORRECTED_PATH = "/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/inference/monthlies/*.nc4"
MODIS_CORRECTED_VAR = "NDinc_cf01p_pred"

REFINED_PATH = f"/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/inference/after_refinement/push8_sigma2/monthlies/*.nc4"
REFINED_VAR = "NDinc_cf01p"


OUT_PNG = "Nd_base_modis_corrected_refined.png"

LEVELS_HPA = [900, 700, 500]

TIME_DIM = "time"
LEV_DIM = "lev"
LAT_DIM = "lat"
LON_DIM = "lon"

MIN_ND = 1.
COLOR_MIN = 5.0
COLOR_MAX = 500.0
COLOR_SCALE = "log"          # "log" or "linear"; CLI --color-scale overrides this
CMAP = "cividis" #"turbo"
DPI = 100

PRESSURE_TICKS = [1000, 850, 700, 500, 300, 200]
LON_TICKS = [-120, -60, 0, 60, 120]
LAT_TICKS = [-60, -30, 0, 30, 60]


# ============================================================
# I/O AND PREPARATION
# ============================================================

def _expand_files(path_pattern):
    path_pattern = os.path.expandvars(os.path.expanduser(path_pattern))
    files = sorted(glob.glob(path_pattern))

    if not files and os.path.isfile(path_pattern):
        files = [path_pattern]

    if not files:
        raise FileNotFoundError(f"No files found for:\n  {path_pattern}")

    return files


def _open_variable(path_pattern, variable):
    """Open one variable from one or more NetCDF files."""
    files = _expand_files(path_pattern)

    # Relative names avoid noisy getfattr messages on some systems.
    datasets = [xr.open_dataset(os.path.relpath(f)) for f in files]

    try:
        if len(datasets) == 1:
            ds = datasets[0]
        else:
            ds = xr.concat(
                datasets,
                dim=TIME_DIM,
                data_vars="minimal",
                coords="minimal",
                compat="override",
                join="outer",
            )

        if variable not in ds:
            available = ", ".join(ds.data_vars)
            raise KeyError(
                f"Variable '{variable}' not found in {path_pattern}.\n"
                f"Available data variables: {available}"
            )

        da = ds[variable]

        required = {LEV_DIM, LAT_DIM, LON_DIM}
        missing = required.difference(da.dims)
        if missing:
            raise ValueError(
                f"{variable} is missing required dimensions: {sorted(missing)}.\n"
                f"Found dimensions: {da.dims}"
            )

        lon = np.asarray(da[LON_DIM].values)
        if lon.size and np.nanmax(lon) > 180.0:
            new_lon = ((lon + 180.0) % 360.0) - 180.0
            da = da.assign_coords({LON_DIM: new_lon}).sortby(LON_DIM)

        if TIME_DIM in da.dims:
            da = da.sortby(TIME_DIM)

        return da, datasets

    except Exception:
        for dsi in datasets:
            dsi.close()
        raise


def _time_mean(da):
    da = da.where(np.isfinite(da))
    da = da.where(da >= MIN_ND)

    if TIME_DIM in da.dims:
        return da.mean(dim=TIME_DIM, skipna=True)

    return da


def _nearest_level_slice(da, requested_level):
    out = da.sel({LEV_DIM: requested_level}, method="nearest")
    actual_level = float(out[LEV_DIM].values)
    return out, actual_level


def _lat_weighted_global_mean(field):
    """Cos(latitude)-weighted mean over a regular lat-lon grid."""
    weights = np.cos(np.deg2rad(field[LAT_DIM]))
    gm = field.weighted(weights).mean(
        dim=(LAT_DIM, LON_DIM),
        skipna=True,
    )
    return float(gm.values)


def _make_color_norm(color_scale):
    """Build the requested shared color normalization."""
    color_scale = str(color_scale).strip().lower()

    if color_scale == "log":
        if COLOR_MIN <= 0:
            raise ValueError("COLOR_MIN must be > 0 when COLOR_SCALE='log'.")
        return mcolors.LogNorm(vmin=COLOR_MIN, vmax=COLOR_MAX)

    if color_scale == "linear":
        return mcolors.Normalize(vmin=COLOR_MIN, vmax=COLOR_MAX)

    raise ValueError(
        f"Unknown color scale '{color_scale}'. Use 'log' or 'linear'."
    )


def _configure_colorbar(cbar, color_scale):
    """Use readable ticks for either logarithmic or linear scaling."""
    color_scale = str(color_scale).strip().lower()

    if color_scale == "log":
        # Preserve the existing journal-style log ticks.
        log_ticks = [10, 20, 50, 100, 200, 500]
        ticks = [t for t in log_ticks if COLOR_MIN <= t <= COLOR_MAX]
        if ticks:
            cbar.set_ticks(ticks)
    else:
        # Evenly spaced values across the requested linear range.
        cbar.set_ticks(np.linspace(COLOR_MIN, COLOR_MAX, 6))

    # Plain-number labels rather than scientific/log notation.
    cbar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x:g}")
    )


# ============================================================
# PLOTTING
# ============================================================

def plot_comparison(cases, out_png, color_scale=COLOR_SCALE):
    if len(cases) != 3:
        raise ValueError("This layout requires exactly three cases.")
    if len(LEVELS_HPA) != 3:
        raise ValueError("LEVELS_HPA must contain exactly three levels.")

    proj = ccrs.PlateCarree()
    norm = _make_color_norm(color_scale)

    mean_cases = [(label, _time_mean(da)) for label, da in cases]

    # Compact journal-style figure; no global title.
    fig = plt.figure(figsize=(11.6, 8.2))
    gs = GridSpec(
        nrows=4,
        ncols=3,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.0, 0.9],
        hspace=0.15,
        wspace=0.12,
        left=0.065,
        right=0.91,
        bottom=0.075,
        top=0.94,
    )

    axes = np.empty((4, 3), dtype=object)

    for i in range(3):
        for j in range(3):
            axes[i, j] = fig.add_subplot(gs[i, j], projection=proj)

    for j in range(3):
        axes[3, j] = fig.add_subplot(gs[3, j])

    panel_letters = list(string.ascii_lowercase[:12])

    # Column headings only; there is intentionally no figure-wide title.
    for col, (label, _) in enumerate(mean_cases):
        axes[0, col].text(
            0.5,
            1.19,
            label,
            transform=axes[0, col].transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )

    # --------------------------------------------------------
    # Horizontal maps
    # --------------------------------------------------------
    for row, requested_level in enumerate(LEVELS_HPA):
        for col, (_, avg) in enumerate(mean_cases):
            ax = axes[row, col]
            panel_id = row * 3 + col
            letter = panel_letters[panel_id]

            field, actual_level = _nearest_level_slice(avg, requested_level)
            field = field.transpose(LAT_DIM, LON_DIM)

            ax.pcolormesh(
                field[LON_DIM].values,
                field[LAT_DIM].values,
                field.values,
                transform=proj,
                shading="auto",
                cmap=CMAP,
                norm=norm,
            )

            ax.coastlines(linewidth=0.55)
            ax.add_feature(cfeature.BORDERS, linewidth=0.25)
            ax.set_global()

            # Keep coordinate tick marks and labels on every map.
            ax.set_xticks(LON_TICKS, crs=proj)
            ax.set_yticks(LAT_TICKS, crs=proj)
            ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            ax.tick_params(
                axis="both",
                which="major",
                labelsize=7,
                length=2.5,
                width=0.6,
                pad=1.5,
            )

            # Left: panel label and pressure.
            ax.set_title(
                f"({letter})  {actual_level:g} hPa",
                loc="left",
                fontsize=9.5,
                pad=3,
            )

            # Right: latitude-weighted global mean.
            gmean = _lat_weighted_global_mean(field)
            ax.set_title(
                rf"$\langle N_d\rangle$ = {gmean:.1f}",
                loc="right",
                fontsize=9.5,
                pad=3,
            )

    # --------------------------------------------------------
    # Zonal means
    # --------------------------------------------------------
    for col, (_, avg) in enumerate(mean_cases):
        ax = axes[3, col]
        panel_id = 9 + col
        letter = panel_letters[panel_id]

        zonal = avg.mean(dim=LON_DIM, skipna=True).transpose(LEV_DIM, LAT_DIM)

        # Restrict every case to the same prescribed pressure range.
        # Pressure itself is used as the vertical coordinate, so the
        # zonal-mean y axis is LINEAR in hPa.
        p_top = min(PRESSURE_TICKS)
        p_bottom = max(PRESSURE_TICKS)

        zonal = zonal.where(
            (zonal[LEV_DIM] >= p_top) & (zonal[LEV_DIM] <= p_bottom),
            drop=True,
        ).sortby(LEV_DIM, ascending=False)

        lev = np.asarray(zonal[LEV_DIM].values, dtype=float)
        lat = np.asarray(zonal[LAT_DIM].values, dtype=float)

        ax.pcolormesh(
            lat,
            lev,
            zonal.values,
            shading="auto",
            cmap=CMAP,
            norm=norm,
        )

        # Linear pressure axis, identical in all three panels.
        # Higher pressure is at the bottom.
        ax.set_yscale("linear")
        ax.set_ylim(p_bottom, p_top)
        ax.set_yticks(PRESSURE_TICKS)
        ax.set_yticklabels([f"{p:g}" for p in PRESSURE_TICKS])

        ax.set_xlim(np.nanmin(lat), np.nanmax(lat))
        ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=7,
            length=2.5,
            width=0.6,
            pad=1.5,
            labelleft=(col == 0),
        )

        ax.set_xlabel("Latitude (deg)", fontsize=8.5, labelpad=2)

        if col == 0:
            ax.set_ylabel("Pressure (hPa)", fontsize=8.5, labelpad=2)
        else:
            ax.set_ylabel("")

        ax.set_title(
            f"({letter})  Zonal mean",
            loc="left",
            fontsize=9.5,
            pad=3,
        )

    cax = fig.add_axes([0.925, 0.12, 0.015, 0.76])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax)
    _configure_colorbar(cbar, color_scale)

    cbar.set_label(r"$N_d$ (cm$^{-3}$)", fontsize=9)
    cbar.ax.tick_params(labelsize=7, length=2.5, width=0.6)

    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[WROTE] {out_png}  color_scale={color_scale}")


# ============================================================
# MAIN
# ============================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Base/MODIS-corrected/Refined Nd comparison."
    )
    parser.add_argument(
        "--color-scale",
        choices=("log", "linear"),
        default=COLOR_SCALE,
        help=(
            "Color normalization for all map and zonal-mean panels "
            f"(default: {COLOR_SCALE})."
        ),
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    opened_datasets = []

    try:
        base, ds_base = _open_variable(BASE_PATH, BASE_VAR)
        opened_datasets.extend(ds_base)

        modis_corrected, ds_modis = _open_variable(
            MODIS_CORRECTED_PATH,
            MODIS_CORRECTED_VAR,
        )
        opened_datasets.extend(ds_modis)

        refined, ds_refined = _open_variable(
            REFINED_PATH,
            REFINED_VAR,
        )
        opened_datasets.extend(ds_refined)

        cases = [
            ("NDnet-prior", base),
            ("NDnet-prior (MODIS corrected)", modis_corrected),
            ("NDnet (refined)", refined),
        ]

        plot_comparison(cases, OUT_PNG, color_scale=args.color_scale)

    finally:
        for ds in opened_datasets:
            try:
                ds.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
