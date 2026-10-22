#!/usr/bin/env python3
"""
Journal-style differences between the NDnet Base, MODIS-corrected, and Refined products.

Rows 1-3: time-mean horizontal differences at 900, 700, and 500 hPa.
Row 4   : zonal-mean differences (time mean, then longitude mean).

Columns:
    MODIS-corrected - Base
    Refined - MODIS-corrected

The plotting choices intentionally follow plots/NDcomparison.py.
"""

import os
import glob
import string
import argparse

import numpy as np
import xarray as xr
import matplotlib as mpl
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

REFINED_PATH = "/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/inference/after_refinement/monthlies/*.nc4"
REFINED_VAR = "NDinc_cf01p"

OUT_PNG = "Nd_product_differences.png"
OUT_PDF = "Nd_product_differences.pdf"

LEVELS_HPA = [900, 700, 500]

TIME_DIM = "time"
LEV_DIM = "lev"
LAT_DIM = "lat"
LON_DIM = "lon"

MIN_ND = 1.0
DIFF_MAX = None          # cm-3; None -> shared robust 98th percentile
ROBUST_PERCENTILE = 98.0
CMAP = "RdBu_r"
DPI = 400

PRESSURE_TICKS = [1000, 850, 700, 500, 300, 200]
LON_TICKS = [-120, -60, 0, 60, 120]
LAT_TICKS = [-60, -30, 0, 30, 60]

mpl.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


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


def _difference_of_means(upper, lower):
    """
    Return climatological upper-lower difference on common lev/lat/lon support.

    Time averaging is performed before alignment, matching NDcomparison.py and
    avoiding dependence on small differences in time-coordinate conventions
    among the monthly products.
    """
    upper_mean = _time_mean(upper)
    lower_mean = _time_mean(lower)
    upper_mean, lower_mean = xr.align(
        upper_mean,
        lower_mean,
        join="inner",
        copy=False,
    )
    return upper_mean - lower_mean


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


def _resolve_diff_max(differences, requested):
    if requested is not None:
        if requested <= 0:
            raise ValueError("--diff-max must be > 0.")
        return float(requested)

    samples = []
    for _, da in differences:
        # Use exactly the fields represented in the figure: the three maps
        # plus the zonal-mean panel.
        for level in LEVELS_HPA:
            field, _ = _nearest_level_slice(da, level)
            values = np.asarray(field.values, dtype=float)
            values = np.abs(values[np.isfinite(values)])
            if values.size:
                samples.append(values)

        zonal = da.mean(dim=LON_DIM, skipna=True)
        values = np.asarray(zonal.values, dtype=float)
        values = np.abs(values[np.isfinite(values)])
        if values.size:
            samples.append(values)

    if not samples:
        raise ValueError("No finite differences available to determine color limits.")

    vmax = float(np.nanpercentile(np.concatenate(samples), ROBUST_PERCENTILE))
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("Could not determine a positive difference color limit.")

    return vmax


def _plain_colorbar(cbar):
    cbar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x:g}")
    )


# ============================================================
# PLOTTING
# ============================================================

def plot_differences(differences, out_png, out_pdf, diff_max=None):
    if len(differences) != 2:
        raise ValueError("This layout requires exactly two difference fields.")
    if len(LEVELS_HPA) != 3:
        raise ValueError("LEVELS_HPA must contain exactly three levels.")

    proj = ccrs.PlateCarree()
    vmax = _resolve_diff_max(differences, diff_max)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig = plt.figure(figsize=(8.3, 8.2))
    gs = GridSpec(
        nrows=4,
        ncols=2,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.0, 0.9],
        hspace=0.15,
        wspace=0.12,
        left=0.075,
        right=0.885,
        bottom=0.075,
        top=0.94,
    )

    axes = np.empty((4, 2), dtype=object)

    for row in range(3):
        for col in range(2):
            axes[row, col] = fig.add_subplot(gs[row, col], projection=proj)

    for col in range(2):
        axes[3, col] = fig.add_subplot(gs[3, col])

    panel_letters = list(string.ascii_lowercase[:8])

    # Column headings only; no global title.
    for col, (label, _) in enumerate(differences):
        axes[0, col].text(
            0.5,
            1.19,
            label,
            transform=axes[0, col].transAxes,
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            clip_on=False,
        )

    # --------------------------------------------------------
    # Horizontal maps
    # --------------------------------------------------------
    for row, requested_level in enumerate(LEVELS_HPA):
        for col, (_, diff) in enumerate(differences):
            ax = axes[row, col]
            panel_id = row * 2 + col
            letter = panel_letters[panel_id]

            field, actual_level = _nearest_level_slice(diff, requested_level)
            field = field.transpose(LAT_DIM, LON_DIM)

            ax.pcolormesh(
                field[LON_DIM].values,
                field[LAT_DIM].values,
                field.values,
                transform=proj,
                shading="auto",
                cmap=CMAP,
                norm=norm,
                rasterized=True,
            )

            ax.coastlines(linewidth=0.55)
            ax.add_feature(cfeature.BORDERS, linewidth=0.25)
            ax.set_global()

            ax.set_xticks(LON_TICKS, crs=proj)
            ax.set_yticks(LAT_TICKS, crs=proj)
            ax.xaxis.set_major_formatter(
                LongitudeFormatter(zero_direction_label=True)
            )
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            ax.tick_params(
                axis="both",
                which="major",
                labelsize=7,
                length=2.5,
                width=0.6,
                pad=1.5,
            )

            ax.set_title(
                f"({letter})  {actual_level:g} hPa",
                loc="left",
                fontsize=9.5,
                pad=3,
            )

            gmean = _lat_weighted_global_mean(field)
            ax.set_title(
                rf"$\langle\Delta N_d\rangle$ = {gmean:+.1f}",
                loc="right",
                fontsize=9.5,
                pad=3,
            )

    # --------------------------------------------------------
    # Zonal means
    # --------------------------------------------------------
    for col, (_, diff) in enumerate(differences):
        ax = axes[3, col]
        panel_id = 6 + col
        letter = panel_letters[panel_id]

        zonal = diff.mean(dim=LON_DIM, skipna=True).transpose(LEV_DIM, LAT_DIM)

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
            rasterized=True,
        )

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

    # One shared colorbar because both columns are the same signed quantity.
    cax = fig.add_axes([0.91, 0.12, 0.017, 0.76])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax, extend="both")
    _plain_colorbar(cbar)
    cbar.set_label(r"$\Delta N_d$ (cm$^{-3}$)", fontsize=9)
    cbar.ax.tick_params(labelsize=7, length=2.5, width=0.6)

    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print(f"[WROTE] {out_png}")
    print(f"[WROTE] {out_pdf}")
    print(f"[SCALE] shared signed range = {-vmax:.2f} to {vmax:.2f} cm-3")


# ============================================================
# MAIN
# ============================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot MODIS-corrected minus Base and Refined minus "
            "MODIS-corrected Nd differences."
        )
    )
    parser.add_argument(
        "--diff-max",
        type=float,
        default=DIFF_MAX,
        help=(
            "Symmetric color limit in cm^-3. Default: robust shared "
            f"{ROBUST_PERCENTILE:g}th percentile of absolute differences."
        ),
    )
    parser.add_argument("--out-png", default=OUT_PNG)
    parser.add_argument("--out-pdf", default=OUT_PDF)
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

        modis_minus_base = _difference_of_means(modis_corrected, base)
        refined_minus_modis = _difference_of_means(refined, modis_corrected)

        differences = [
            ("MODIS-corrected - Base", modis_minus_base),
            ("Refined - MODIS-corrected", refined_minus_modis),
        ]

        plot_differences(
            differences,
            args.out_png,
            args.out_pdf,
            diff_max=args.diff_max,
        )

    finally:
        for ds in opened_datasets:
            try:
                ds.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
