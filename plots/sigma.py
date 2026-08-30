#!/usr/bin/env python3
"""
Journal-style maps of absolute and relative Nd uncertainty.

Row 1: time-mean sigmaNd at three pressure levels
Row 2: time-mean 100 * sigmaNd / NDinc_cf01p at three pressure levels

Columns: pressure levels in LEVELS_HPA

For each panel:
  Left title  : panel letter and pressure
  Right title : latitude-weighted global mean

Two colorbars are placed on the right because the two rows have
different units.
"""

import os
import glob
import string

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

DATA_PATH = (
    "/panfs/ccds02/nobackup/people/dbarahon/NDnet/"
    "EOF/inference/after_refinement/push8_sigma2/monthlies/*.nc4"
)

SIGMA_VAR = "sigmaNd"
ND_VAR = "NDinc_cf01p"

OUT_PNG = "Nd_sigma_relative_sigma.png"

# One pressure level per column
LEVELS_HPA = [900, 700, 500]

TIME_DIM = "time"
LEV_DIM = "lev"
LAT_DIM = "lat"
LON_DIM = "lon"


# ------------------------------------------------------------
# DATA FILTERS
# ------------------------------------------------------------

# Do not calculate relative uncertainty where Nd is below this.
MIN_ND = 10.0

# sigmaNd is an uncertainty and should not be negative.
MIN_SIGMA = 00.0


# ------------------------------------------------------------
# COLOR SCALE: ROW 1, sigmaNd
# ------------------------------------------------------------

# Change these two numbers to change the sigmaNd color range.
SIGMA_COLOR_MIN = 5.0
SIGMA_COLOR_MAX = 500.0

# Keep the logarithmic scaling used in your Nd figure.
SIGMA_LOG_SCALE = True

SIGMA_COLOR_TICKS = [5, 10, 20, 50, 100, 200, 500]

SIGMA_CMAP = "turbo"


# ------------------------------------------------------------
# COLOR SCALE: ROW 2, relative uncertainty (%)
# ------------------------------------------------------------

# Change these two numbers to change the percentage color range.
REL_COLOR_MIN = 0.0
REL_COLOR_MAX = 100.0

REL_COLOR_TICKS = [0, 20, 40, 60, 80, 100]

REL_CMAP = "turbo"


# ------------------------------------------------------------
# FIGURE SETTINGS
# ------------------------------------------------------------

DPI = 100

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
        raise FileNotFoundError(
            f"No files found for:\n  {path_pattern}"
        )

    return files


def _open_dataset(path_pattern):
    """
    Open one or more NetCDF files.

    Relative names are used to avoid noisy getfattr messages on
    systems where extended attributes are queried for absolute paths.
    """
    files = _expand_files(path_pattern)

    datasets = [
        xr.open_dataset(os.path.relpath(f))
        for f in files
    ]

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

        # Check required variables.
        for variable in [SIGMA_VAR, ND_VAR]:
            if variable not in ds:
                available = ", ".join(ds.data_vars)

                raise KeyError(
                    f"Variable '{variable}' not found in:\n"
                    f"  {path_pattern}\n\n"
                    f"Available data variables:\n"
                    f"  {available}"
                )

        # Check dimensions.
        for variable in [SIGMA_VAR, ND_VAR]:
            da = ds[variable]

            required = {LEV_DIM, LAT_DIM, LON_DIM}
            missing = required.difference(da.dims)

            if missing:
                raise ValueError(
                    f"{variable} is missing required dimensions: "
                    f"{sorted(missing)}.\n"
                    f"Found dimensions: {da.dims}"
                )

        # Convert longitude from 0--360 to -180--180 if needed.
        lon = np.asarray(ds[LON_DIM].values)

        if lon.size and np.nanmax(lon) > 180.0:
            new_lon = ((lon + 180.0) % 360.0) - 180.0

            ds = (
                ds.assign_coords({LON_DIM: new_lon})
                .sortby(LON_DIM)
            )

        if TIME_DIM in ds.dims:
            ds = ds.sortby(TIME_DIM)

        return ds, datasets

    except Exception:
        for dsi in datasets:
            dsi.close()
        raise


def _prepare_fields(ds):
    """
    Prepare absolute and relative uncertainty.

    sigma_mean:
        time mean of sigmaNd

    relative_mean:
        time mean of
            100 * sigmaNd / NDinc_cf01p

    Thus the relative uncertainty is calculated at each time step
    first, and the resulting percentage is then averaged in time.
    """

    sigma = ds[SIGMA_VAR]
    nd = ds[ND_VAR]

    # Make sure coordinates overlap exactly.
    sigma, nd = xr.align(
        sigma,
        nd,
        join="inner",
    )

    # --------------------------------------------------------
    # Basic validity masks
    # --------------------------------------------------------

    sigma = sigma.where(np.isfinite(sigma))
    sigma = sigma.where(sigma >= MIN_SIGMA)

    nd = nd.where(np.isfinite(nd))
    nd = nd.where(nd >= MIN_ND)

    # --------------------------------------------------------
    # Relative uncertainty
    # --------------------------------------------------------

    relative = 100.0 * sigma / nd

    relative = relative.where(np.isfinite(relative))
    relative = relative.where(relative >= 0.0)

    # --------------------------------------------------------
    # Time means
    # --------------------------------------------------------

    if TIME_DIM in sigma.dims:
        sigma_mean = sigma.mean(
            dim=TIME_DIM,
            skipna=True,
        )
    else:
        sigma_mean = sigma

    if TIME_DIM in relative.dims:
        relative_mean = relative.mean(
            dim=TIME_DIM,
            skipna=True,
        )
    else:
        relative_mean = relative

    return sigma_mean, relative_mean


def _nearest_level_slice(da, requested_level):
    out = da.sel(
        {LEV_DIM: requested_level},
        method="nearest",
    )

    actual_level = float(out[LEV_DIM].values)

    return out, actual_level


def _lat_weighted_global_mean(field):
    """
    Cos(latitude)-weighted global mean over a regular lat-lon grid.
    """
    weights = np.cos(
        np.deg2rad(field[LAT_DIM])
    )

    gm = field.weighted(weights).mean(
        dim=(LAT_DIM, LON_DIM),
        skipna=True,
    )

    return float(gm.values)


def _setup_map_axis(ax, proj):
    """Apply the common map formatting."""

    ax.coastlines(linewidth=0.55)
    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.25,
    )

    ax.set_global()

    ax.set_xticks(
        LON_TICKS,
        crs=proj,
    )
    ax.set_yticks(
        LAT_TICKS,
        crs=proj,
    )

    ax.xaxis.set_major_formatter(
        LongitudeFormatter(
            zero_direction_label=True
        )
    )

    ax.yaxis.set_major_formatter(
        LatitudeFormatter()
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=7,
        length=2.5,
        width=0.6,
        pad=1.5,
    )


def _set_colorbar_ticks(
    cbar,
    ticks,
    vmin,
    vmax,
):
    """
    Apply only ticks that fall within the selected color range.
    """
    good_ticks = [
        tick for tick in ticks
        if vmin <= tick <= vmax
    ]

    if good_ticks:
        cbar.set_ticks(good_ticks)

    cbar.ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda x, pos: f"{x:g}"
        )
    )


# ============================================================
# PLOTTING
# ============================================================

def plot_uncertainty(
    sigma_mean,
    relative_mean,
    out_png,
):

    if len(LEVELS_HPA) != 3:
        raise ValueError(
            "LEVELS_HPA must contain exactly three levels."
        )

    proj = ccrs.PlateCarree()

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    if SIGMA_LOG_SCALE:
        if SIGMA_COLOR_MIN <= 0:
            raise ValueError(
                "SIGMA_COLOR_MIN must be > 0 when "
                "SIGMA_LOG_SCALE = True."
            )

        sigma_norm = mcolors.LogNorm(
            vmin=SIGMA_COLOR_MIN,
            vmax=SIGMA_COLOR_MAX,
        )

    else:
        sigma_norm = mcolors.Normalize(
            vmin=SIGMA_COLOR_MIN,
            vmax=SIGMA_COLOR_MAX,
        )

    rel_norm = mcolors.Normalize(
        vmin=REL_COLOR_MIN,
        vmax=REL_COLOR_MAX,
    )

    # --------------------------------------------------------
    # Figure layout
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(11.6, 5.2)
    )

    gs = GridSpec(
        nrows=2,
        ncols=3,
        figure=fig,
        height_ratios=[1.0, 1.0],
        hspace=0.16,
        wspace=0.12,
        left=0.065,
        right=0.90,
        bottom=0.075,
        top=0.94,
    )

    axes = np.empty(
        (2, 3),
        dtype=object,
    )

    for row in range(2):
        for col in range(3):
            axes[row, col] = fig.add_subplot(
                gs[row, col],
                projection=proj,
            )

    panel_letters = list(
        string.ascii_lowercase[:6]
    )

    # ========================================================
    # ROW 1: sigmaNd
    # ========================================================

    for col, requested_level in enumerate(LEVELS_HPA):

        ax = axes[0, col]
        letter = panel_letters[col]

        field, actual_level = _nearest_level_slice(
            sigma_mean,
            requested_level,
        )

        field = field.transpose(
            LAT_DIM,
            LON_DIM,
        )

        ax.pcolormesh(
            field[LON_DIM].values,
            field[LAT_DIM].values,
            field.values,
            transform=proj,
            shading="auto",
            cmap=SIGMA_CMAP,
            norm=sigma_norm,
        )

        _setup_map_axis(
            ax,
            proj,
        )

        # Left title: panel letter + pressure
        ax.set_title(
            f"({letter})  {actual_level:g} hPa",
            loc="left",
            fontsize=9.5,
            pad=3,
        )

        # Right title: area-weighted mean
        gmean = _lat_weighted_global_mean(
            field
        )

        ax.set_title(
            rf"$\langle\sigma_{{N_d}}\rangle$ = {gmean:.1f}",
            loc="right",
            fontsize=9.5,
            pad=3,
        )


    # ========================================================
    # ROW 2: 100 sigmaNd / Nd
    # ========================================================

    for col, requested_level in enumerate(LEVELS_HPA):

        ax = axes[1, col]
        letter = panel_letters[3 + col]

        field, actual_level = _nearest_level_slice(
            relative_mean,
            requested_level,
        )

        field = field.transpose(
            LAT_DIM,
            LON_DIM,
        )

        ax.pcolormesh(
            field[LON_DIM].values,
            field[LAT_DIM].values,
            field.values,
            transform=proj,
            shading="auto",
            cmap=REL_CMAP,
            norm=rel_norm,
        )

        _setup_map_axis(
            ax,
            proj,
        )

        # Left title: panel letter + pressure
        ax.set_title(
            f"({letter})  {actual_level:g} hPa",
            loc="left",
            fontsize=9.5,
            pad=3,
        )

        # Right title: area-weighted mean percentage
        gmean = _lat_weighted_global_mean(
            field
        )

        ax.set_title(
            rf"$\langle\sigma_{{N_d}}/N_d\rangle$ = {gmean:.1f}$\%$",
            loc="right",
            fontsize=9.5,
            pad=3,
        )


    # ========================================================
    # COLORBAR: ROW 1
    # ========================================================

    cax_sigma = fig.add_axes(
        [
            0.918,   # left
            0.555,   # bottom
            0.014,   # width
            0.335,   # height
        ]
    )

    sm_sigma = plt.cm.ScalarMappable(
        norm=sigma_norm,
        cmap=SIGMA_CMAP,
    )
    sm_sigma.set_array([])

    cbar_sigma = fig.colorbar(
        sm_sigma,
        cax=cax_sigma,
    )

    _set_colorbar_ticks(
        cbar_sigma,
        SIGMA_COLOR_TICKS,
        SIGMA_COLOR_MIN,
        SIGMA_COLOR_MAX,
    )

    cbar_sigma.set_label(
        r"$\sigma_{N_d}$ (cm$^{-3}$)",
        fontsize=9,
    )

    cbar_sigma.ax.tick_params(
        labelsize=7,
        length=2.5,
        width=0.6,
    )


    # ========================================================
    # COLORBAR: ROW 2
    # ========================================================

    cax_rel = fig.add_axes(
        [
            0.918,   # left
            0.125,   # bottom
            0.014,   # width
            0.335,   # height
        ]
    )

    sm_rel = plt.cm.ScalarMappable(
        norm=rel_norm,
        cmap=REL_CMAP,
    )
    sm_rel.set_array([])

    cbar_rel = fig.colorbar(
        sm_rel,
        cax=cax_rel,
    )

    _set_colorbar_ticks(
        cbar_rel,
        REL_COLOR_TICKS,
        REL_COLOR_MIN,
        REL_COLOR_MAX,
    )

    cbar_rel.set_label(
        r"$100\,\sigma_{N_d}/N_d$ (%)",
        fontsize=9,
    )

    cbar_rel.ax.tick_params(
        labelsize=7,
        length=2.5,
        width=0.6,
    )


    # ========================================================
    # SAVE
    # ========================================================

    fig.savefig(
        out_png,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[WROTE] {out_png}")


# ============================================================
# MAIN
# ============================================================

def main():

    opened_datasets = []
    combined_ds = None

    try:

        combined_ds, source_datasets = _open_dataset(
            DATA_PATH
        )

        opened_datasets.extend(
            source_datasets
        )

        sigma_mean, relative_mean = _prepare_fields(
            combined_ds
        )

        plot_uncertainty(
            sigma_mean,
            relative_mean,
            OUT_PNG,
        )

    finally:

        # Close concatenated dataset if it is distinct from the
        # individual source datasets.
        if combined_ds is not None:
            try:
                combined_ds.close()
            except Exception:
                pass

        for ds in opened_datasets:
            try:
                ds.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
