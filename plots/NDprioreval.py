#!/usr/bin/env python3
"""
Journal-style comparison of NDinc_cf01p prediction against reference.

Layout
------
Rows 1-3 : time-mean horizontal maps at three pressure levels
Row 4    : zonal mean (time mean, then longitude mean)

Columns
-------
1 : NDinc_cf01p          (reference)
2 : NDinc_cf01p_pred     (prediction)
3 : mean percent bias    100 * (mean(pred) - mean(ref)) / mean(ref)
4 : absolute-unit bias   mean(pred) - mean(ref), in cm^-3

The first two columns share a logarithmic colorbar. The percent-bias and
absolute-unit-bias columns use zero-centered diverging scales.

Edit the GLOBAL CONFIGURATION section for normal use.
"""

import os
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

INPUT_FILE = (
    "/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/"
    "in_cloudM2/NDnet_pred.nc"
)

REF_VAR = "NDinc_cf01p"
PRED_VAR = "NDinc_cf01p_pred"

REF_LABEL = "GiOcean"
PRED_LABEL = "NDnet-prior"
BIAS_LABEL = "Mean % bias"
ABS_BIAS_LABEL = r"Bias (cm$^{-3}$)"

OUT_PNG = "NDnetprior_eval.png"

LEVELS_HPA = [900, 700, 500]

TIME_DIM = "time"
LEV_DIM = "lev"
LAT_DIM = "lat"
LON_DIM = "lon"

MIN_ND = 0.1
BIAS_DENOM_MIN = 0.1

# Reference/prediction scale.
COLOR_MIN = 10.0
COLOR_MAX = 1000.0
ABS_COLOR_TICKS = [10, 20, 50, 100, 200, 500, 1000]
ABS_CMAP = "turbo"

# Percent-bias scale.
BIAS_MAX = 50.0
BIAS_TICKS = [-50, -25, 0, 25, 50]
BIAS_CMAP = "RdBu_r"

# Signed bias in absolute Nd units [cm^-3].  A symmetric-log normalization
# expands the near-zero structure while retaining the full +/-100 range.
ABS_BIAS_MAX = 100.0
ABS_BIAS_LINTHRESH = 1.0
ABS_BIAS_TICKS = [-100, -50, -20, -5, -1, 0, 1, 5, 20, 50, 100]
ABS_BIAS_CMAP = "RdBu_r"

PRESSURE_TICKS = [1000, 850, 700, 500, 300, 200]
LON_TICKS = [-120, -60, 0, 60, 120]
LAT_TICKS = [-60, -30, 0, 30, 60]

DPI = 100
FIELD_UNITS = r"cm$^{-3}$"


# ============================================================
# HELPERS
# ============================================================

def _open_dataset(path):
    path = os.path.expandvars(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found:\n  {path}")
    return xr.open_dataset(os.path.relpath(path))


def _prepare_pair(ref, pred):
    """Apply the same paired validity mask to reference and prediction."""
    ref = ref.where(np.isfinite(ref))
    pred = pred.where(np.isfinite(pred))
    ref = ref.where(ref >= MIN_ND)
    pred = pred.where(ref >= MIN_ND)
    pred = pred.where(pred >= MIN_ND)
    return ref, pred


def _time_mean(da):
    if TIME_DIM in da.dims:
        return da.mean(dim=TIME_DIM, skipna=True)
    return da


def _nearest_level_slice(da, requested_level):
    out = da.sel({LEV_DIM: requested_level}, method="nearest")
    return out, float(out[LEV_DIM].values)


def _lat_weighted_global_mean(field):
    weights = np.cos(np.deg2rad(field[LAT_DIM]))
    value = field.weighted(weights).mean(dim=(LAT_DIM, LON_DIM), skipna=True)
    return float(value.values)


def _plain_number(x, pos=None):
    return f"{x:g}"


# ============================================================
# PLOTTING
# ============================================================

def plot_comparison(ref, pred, out_png):
    """Make a 4 x 4 journal-style comparison figure."""
    ref, pred = _prepare_pair(ref, pred)

    avg_ref = _time_mean(ref)
    avg_pred = _time_mean(pred)
    delta = avg_pred - avg_ref

    denom = avg_ref.where(avg_ref >= BIAS_DENOM_MIN)
    avg_bias = 100.0 * delta / denom
    avg_abs_bias = delta

    abs_norm = mcolors.LogNorm(vmin=COLOR_MIN, vmax=COLOR_MAX)
    bias_norm = mcolors.TwoSlopeNorm(
        vmin=-BIAS_MAX, vcenter=0.0, vmax=BIAS_MAX
    )
    abs_bias_norm = mcolors.SymLogNorm(
        linthresh=ABS_BIAS_LINTHRESH,
        linscale=1.0,
        vmin=-ABS_BIAS_MAX,
        vmax=ABS_BIAS_MAX,
        base=10,
    )

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(15.6, 8.2))

    gs = GridSpec(
        nrows=4,
        ncols=4,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.0, 0.82],
        hspace=0.30,
        wspace=0.14,
        left=0.055,
        right=0.975,
        bottom=0.14,
        top=0.94,
    )

    axes = np.empty((4, 4), dtype=object)
    for row in range(3):
        for col in range(4):
            axes[row, col] = fig.add_subplot(gs[row, col], projection=proj)
    for col in range(4):
        axes[3, col] = fig.add_subplot(gs[3, col])

    panel_letters = list(string.ascii_lowercase[:16])
    column_headers = [REF_LABEL, PRED_LABEL, BIAS_LABEL, ABS_BIAS_LABEL]

    for col, title in enumerate(column_headers):
        axes[0, col].text(
            0.5, 1.19, title,
            transform=axes[0, col].transAxes,
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", clip_on=False,
        )

    # --------------------------------------------------------
    # Horizontal maps
    # --------------------------------------------------------
    for row, requested_level in enumerate(LEVELS_HPA):
        ref_slice, ref_level = _nearest_level_slice(avg_ref, requested_level)
        pred_slice, pred_level = _nearest_level_slice(avg_pred, requested_level)
        bias_slice, bias_level = _nearest_level_slice(avg_bias, requested_level)
        abs_bias_slice, abs_bias_level = _nearest_level_slice(
            avg_abs_bias, requested_level
        )

        panels = [
            (ref_slice.transpose(LAT_DIM, LON_DIM), ref_level,
             ABS_CMAP, abs_norm, "field"),
            (pred_slice.transpose(LAT_DIM, LON_DIM), pred_level,
             ABS_CMAP, abs_norm, "field"),
            (bias_slice.transpose(LAT_DIM, LON_DIM), bias_level,
             BIAS_CMAP, bias_norm, "bias"),
            (abs_bias_slice.transpose(LAT_DIM, LON_DIM), abs_bias_level,
             ABS_BIAS_CMAP, abs_bias_norm, "abs_bias"),
        ]

        for col, (field, actual_level, cmap, norm, kind) in enumerate(panels):
            ax = axes[row, col]
            letter = panel_letters[row * 4 + col]

            ax.pcolormesh(
                field[LON_DIM].values,
                field[LAT_DIM].values,
                field.values,
                transform=proj,
                shading="auto",
                cmap=cmap,
                norm=norm,
            )

            ax.coastlines(linewidth=0.55)
            ax.add_feature(cfeature.BORDERS, linewidth=0.25)
            ax.set_global()
            ax.set_xticks(LON_TICKS, crs=proj)
            ax.set_yticks(LAT_TICKS, crs=proj)
            ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            ax.tick_params(
                axis="both", which="major", labelsize=7,
                length=2.5, width=0.6, pad=1.5,
                labelleft=(col == 0),
                labelbottom=(row == 2),
            )

            ax.set_title(
                f"({letter})  {actual_level:g} hPa",
                loc="left", fontsize=9.5, pad=3,
            )

            gmean = _lat_weighted_global_mean(field)
            if kind == "bias":
                right_title = rf"$\langle bias\rangle$ = {gmean:.1f}%"
            elif kind == "abs_bias":
                right_title = rf"$\langle \Delta N_d\rangle$ = {gmean:.1f}"
            else:
                right_title = rf"$\langle N_d\rangle$ = {gmean:.1f}"

            ax.set_title(right_title, loc="right", fontsize=9.5, pad=3)

    # --------------------------------------------------------
    # Zonal means
    # --------------------------------------------------------
    zonal_ref = avg_ref.mean(dim=LON_DIM, skipna=True)
    zonal_pred = avg_pred.mean(dim=LON_DIM, skipna=True)
    zonal_denom = zonal_ref.where(zonal_ref >= BIAS_DENOM_MIN)
    zonal_bias = 100.0 * (zonal_pred - zonal_ref) / zonal_denom
    zonal_abs_bias = zonal_pred - zonal_ref

    zonal_fields = [
        (zonal_ref, ABS_CMAP, abs_norm),
        (zonal_pred, ABS_CMAP, abs_norm),
        (zonal_bias, BIAS_CMAP, bias_norm),
        (zonal_abs_bias, ABS_BIAS_CMAP, abs_bias_norm),
    ]

    p_top = min(PRESSURE_TICKS)
    p_bottom = max(PRESSURE_TICKS)

    for col, (zonal, cmap, norm) in enumerate(zonal_fields):
        ax = axes[3, col]
        letter = panel_letters[12 + col]
        zonal = zonal.transpose(LEV_DIM, LAT_DIM)
        zonal = zonal.where(
            (zonal[LEV_DIM] >= p_top) & (zonal[LEV_DIM] <= p_bottom),
            drop=True,
        ).sortby(LEV_DIM, ascending=False)

        lev = np.asarray(zonal[LEV_DIM].values, dtype=float)
        lat = np.asarray(zonal[LAT_DIM].values, dtype=float)

        ax.pcolormesh(lat, lev, zonal.values, shading="auto", cmap=cmap, norm=norm)
        ax.set_yscale("linear")
        ax.set_ylim(p_bottom, p_top)
        ax.set_yticks(PRESSURE_TICKS)
        ax.set_yticklabels([f"{p:g}" for p in PRESSURE_TICKS])
        ax.set_xlim(np.nanmin(lat), np.nanmax(lat))
        ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
        ax.tick_params(
            axis="both", which="major", labelsize=7,
            length=2.5, width=0.6, pad=1.5,
            labelleft=(col == 0),
        )

        if col == 0:
            ax.set_ylabel("Pressure (hPa)", fontsize=8.5, labelpad=2)
        else:
            ax.set_ylabel("")

        # Center the common x label across the four-column zonal row.
        ax.set_xlabel("Latitude (deg)" if col == 1 else "", fontsize=8.5, labelpad=2)
        ax.set_title(f"({letter})  Zonal mean", loc="left", fontsize=9.5, pad=3)

    # --------------------------------------------------------
    # Colorbars: compact, column-aligned journal layout
    # --------------------------------------------------------
    sm_abs = plt.cm.ScalarMappable(norm=abs_norm, cmap=ABS_CMAP)
    sm_abs.set_array([])
    cax_abs = fig.add_axes([0.075, 0.055, 0.405, 0.018])
    cbar_abs = fig.colorbar(sm_abs, cax=cax_abs, orientation="horizontal")
    abs_ticks = [t for t in ABS_COLOR_TICKS if COLOR_MIN <= t <= COLOR_MAX]
    cbar_abs.set_ticks(abs_ticks)
    cbar_abs.ax.xaxis.set_major_formatter(FuncFormatter(_plain_number))
    cbar_abs.set_label(rf"$N_d$ ({FIELD_UNITS})", fontsize=9, labelpad=3)
    cbar_abs.ax.tick_params(labelsize=7, length=2.5, width=0.6, pad=2)

    sm_bias = plt.cm.ScalarMappable(norm=bias_norm, cmap=BIAS_CMAP)
    sm_bias.set_array([])
    cax_bias = fig.add_axes([0.535, 0.055, 0.185, 0.018])
    cbar_bias = fig.colorbar(sm_bias, cax=cax_bias, orientation="horizontal")
    cbar_bias.set_ticks([t for t in BIAS_TICKS if -BIAS_MAX <= t <= BIAS_MAX])
    cbar_bias.ax.xaxis.set_major_formatter(FuncFormatter(_plain_number))
    cbar_bias.set_label("Mean bias (%)", fontsize=9, labelpad=3)
    cbar_bias.ax.tick_params(labelsize=7, length=2.5, width=0.6, pad=2)

    sm_abs_bias = plt.cm.ScalarMappable(norm=abs_bias_norm, cmap=ABS_BIAS_CMAP)
    sm_abs_bias.set_array([])
    cax_abs_bias = fig.add_axes([0.775, 0.055, 0.185, 0.018])
    cbar_abs_bias = fig.colorbar(
        sm_abs_bias, cax=cax_abs_bias, orientation="horizontal"
    )
    cbar_abs_bias.set_ticks(ABS_BIAS_TICKS)
    cbar_abs_bias.ax.xaxis.set_major_formatter(FuncFormatter(_plain_number))
    cbar_abs_bias.set_label(rf"Bias ({FIELD_UNITS})", fontsize=9, labelpad=3)
    cbar_abs_bias.ax.tick_params(labelsize=7, length=2.5, width=0.6, pad=2)

    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[WROTE] {out_png}")


# ============================================================
# MAIN
# ============================================================

def main():
    ds = _open_dataset(INPUT_FILE)
    try:
        if REF_VAR not in ds:
            raise KeyError(f"'{REF_VAR}' not found. Available variables: {list(ds.data_vars)}")
        if PRED_VAR not in ds:
            raise KeyError(f"'{PRED_VAR}' not found. Available variables: {list(ds.data_vars)}")

        ref = ds[REF_VAR]
        pred = ds[PRED_VAR]
        required_dims = {TIME_DIM, LEV_DIM, LAT_DIM, LON_DIM}

        for name, da in [(REF_VAR, ref), (PRED_VAR, pred)]:
            missing = required_dims.difference(da.dims)
            if missing:
                raise ValueError(
                    f"{name} is missing dimensions {sorted(missing)}; found {da.dims}"
                )

        lon = np.asarray(ref[LON_DIM].values)
        if lon.size and np.nanmax(lon) > 180.0:
            new_lon = ((lon + 180.0) % 360.0) - 180.0
            ref = ref.assign_coords({LON_DIM: new_lon}).sortby(LON_DIM)
            pred = pred.assign_coords({LON_DIM: new_lon}).sortby(LON_DIM)

        plot_comparison(ref, pred, OUT_PNG)
    finally:
        ds.close()


if __name__ == "__main__":
    main()
