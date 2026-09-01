#!/usr/bin/env python3

import os
import re
import glob
import argparse

import numpy as np
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# =====================================================================
# Defaults
# =====================================================================

DEFAULT_DIR = (
    "/panfs/ccds02/nobackup/people/dbarahon/"
    "NDnet/EOF/refinement/push6"
)

REQUIRED_VARS = {
    "Nd_true",
    "Nd_base",
    "Nd_rec",
    "site_mask",
    "lev",
    "site",
}


# Journal-friendly / colorblind-safe styles
SERIES = {
    "obs": {
        "label": "OBS",
        "color": "black",
        "ls": "-",
        "lw": 1.7,
        "alpha": 0.10,
    },
    "prior": {
        "label": "NDnet-prior",
        "color": "#0072B2",
        "ls": "--",
        "lw": 1.55,
        "alpha": 0.13,
    },
    "rec": {
        "label": "NDnet",
        "color": "#D55E00",
        "ls": "-",
        "lw": 1.55,
        "alpha": 0.13,
    },
}


# =====================================================================
# Matplotlib journal settings
# =====================================================================

def set_journal_style():
    """
    Matplotlib settings suitable for journal figures.

    PDF/PS fonts are stored as TrueType fonts so text remains editable
    in most journal production workflows.
    """

    mpl.rcParams.update(
        {
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 8.0,

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


# =====================================================================
# Input-file discovery
# =====================================================================

def is_valid_input_file(path):
    """
    Return True if path looks like the requested NDnet_site_test file.
    """
    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            available = set(ds.variables)
            return REQUIRED_VARS.issubset(available)
    except Exception:
        return False


def find_input_file(indir, infile=None):
    """
    Find the input NetCDF.

    Priority:
      1. Explicit --infile
      2. <indir>/NDnet_site_test.nc
      3. NDnet_site_test*.nc
      4. Other *.nc files containing the required variables
    """

    if infile is not None:
        infile = os.path.abspath(infile)

        if not os.path.exists(infile):
            raise FileNotFoundError(
                f"Input file does not exist:\n{infile}"
            )

        if not is_valid_input_file(infile):
            raise ValueError(
                f"{infile}\n"
                "does not contain all required variables:\n"
                f"{sorted(REQUIRED_VARS)}"
            )

        return infile

    indir = os.path.abspath(indir)

    if not os.path.isdir(indir):
        raise NotADirectoryError(
            f"Input directory does not exist:\n{indir}"
        )

    # First try the expected exact filename
    exact = os.path.join(indir, "NDnet_site_test.nc")

    if os.path.exists(exact) and is_valid_input_file(exact):
        return exact

    # Then preferred filename pattern
    candidates = sorted(
        glob.glob(os.path.join(indir, "NDnet_site_test*.nc"))
    )

    good = [
        f for f in candidates
        if is_valid_input_file(f)
    ]

    if len(good) == 1:
        return good[0]

    if len(good) > 1:
        raise RuntimeError(
            "More than one NDnet_site_test-style file was found.\n"
            "Specify the desired one with --infile:\n\n"
            + "\n".join(good)
        )

    # Last resort: inspect all NetCDFs in the directory
    candidates = sorted(
        glob.glob(os.path.join(indir, "*.nc"))
    )

    good = [
        f for f in candidates
        if is_valid_input_file(f)
    ]

    if len(good) == 1:
        return good[0]

    if len(good) > 1:
        raise RuntimeError(
            "More than one compatible NetCDF was found.\n"
            "Specify the desired one with --infile:\n\n"
            + "\n".join(good)
        )

    raise FileNotFoundError(
        "Could not find an NDnet_site_test-style NetCDF in:\n"
        f"{indir}\n\n"
        "Required variables are:\n"
        + ", ".join(sorted(REQUIRED_VARS))
    )


# =====================================================================
# Helpers
# =====================================================================

def to_string(value):
    """
    Convert xarray/netCDF string or bytes values to normal Python strings.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()

    return str(value).strip()


def panel_letter(index):
    """
    Convert 0,1,... into a,b,...,z,aa,ab,...
    """
    result = ""
    i = index

    while True:
        result = chr(ord("a") + (i % 26)) + result
        i = i // 26 - 1

        if i < 0:
            break

    return result


def pressure_to_hpa(lev):
    """
    Return pressure coordinate in hPa.

    If pressure appears to be in Pa, convert automatically.
    Otherwise leave it unchanged.
    """

    lev = np.asarray(lev, dtype=float)

    finite = lev[np.isfinite(lev)]

    if finite.size == 0:
        return lev

    # Typical pressure in Pa is O(100000), while hPa is O(1000).
    if np.nanmedian(np.abs(finite)) > 2000.0:
        print("Pressure coordinate appears to be in Pa; converting to hPa.")
        lev = lev / 100.0

    return lev


def paired_statistics(obs, model):
    """Return Pearson correlation, mean bias (model - OBS), and sample count."""
    x = np.asarray(obs, dtype=float).ravel()
    y = np.asarray(model, dtype=float).ravel()

    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]

    n = int(x.size)
    if n == 0:
        return np.nan, np.nan, 0

    bias = float(np.mean(y - x))

    r = np.nan
    if n >= 2 and np.std(x) > 0.0 and np.std(y) > 0.0:
        r = float(np.corrcoef(x, y)[0, 1])

    return r, bias, n


def format_panel_statistics(result):
    """Compact Base/NN statistics for a small multi-panel annotation."""
    def fmt_r(value):
        return f"{value:.2f}" if np.isfinite(value) else "--"

    def fmt_b(value):
        return f"{value:+.1f}" if np.isfinite(value) else "--"

    return (
        f"B: r={fmt_r(result['prior_r'])}  b={fmt_b(result['prior_bias'])}\n"
        f"NN: r={fmt_r(result['rec_r'])}  b={fmt_b(result['rec_bias'])}"
    )


def calculate_common_xmax(results, pooled=None):
    """
    Determine a common positive x-axis maximum for all panels.

    Includes mean + one standard deviation so the shaded envelopes are
    generally contained inside the axes.
    """

    values = []

    if pooled is not None:
        for prefix in ("obs", "prior", "rec"):
            m = np.asarray(pooled[f"{prefix}_mean"])
            s = np.asarray(pooled[f"{prefix}_std"])

            v = m + s
            v = v[np.isfinite(v) & (v >= 0)]

            if v.size:
                values.append(v)

    for r in results:
        for prefix in ("obs", "prior", "rec"):
            m = np.asarray(r[f"{prefix}_mean"])
            s = np.asarray(r[f"{prefix}_std"])

            v = m + s
            v = v[np.isfinite(v) & (v >= 0)]

            if v.size:
                values.append(v)

    if not values:
        return None

    vmax = np.nanmax(
        np.concatenate(values)
    )

    if not np.isfinite(vmax) or vmax <= 0:
        return None

    return 1.05 * vmax


# =====================================================================
# Read and prepare data
# =====================================================================

def read_input(path):
    """
    Read required variables into memory and close the source NetCDF.
    """

    print(f"Reading:\n  {path}")

    with xr.open_dataset(path, decode_times=False) as ds:
        missing = REQUIRED_VARS.difference(ds.variables)

        if missing:
            raise KeyError(
                "Input file is missing required variables:\n"
                + ", ".join(sorted(missing))
            )

        ds = ds[
            [
                "Nd_true",
                "Nd_base",
                "Nd_rec",
                "site_mask",
            ]
        ].load()

    return ds


def select_sites(ds, include=None, exclude=None):
    """
    Optionally select site/experiment labels using regex filters.

    Since the new file contains no separate experiment dimension,
    each entry of the `site` coordinate is treated as one
    experiment/campaign for the multi-panel plot.
    """

    names = [
        to_string(v)
        for v in ds["site"].values
    ]

    rx_include = re.compile(include) if include else None
    rx_exclude = re.compile(exclude) if exclude else None

    keep = []

    for i, name in enumerate(names):

        if rx_include and not rx_include.search(name):
            continue

        if rx_exclude and rx_exclude.search(name):
            continue

        keep.append(i)

    if not keep:
        raise ValueError(
            "No sites/experiments remain after include/exclude filtering."
        )

    ds = ds.isel(site=keep)

    names = [
        to_string(v)
        for v in ds["site"].values
    ]

    return ds, names


def prepare_masked_arrays(
    ds,
    minN=0.1,
    use_site_mask=True,
):
    """
    Prepare OBS, ND-prior and NDnet arrays.

    Validity follows the original script:
      OBS must be finite and OBS > minN.

    If use_site_mask=True, site_mask must also be non-zero.

    The OBS-based mask is then applied to ND-prior and NDnet.
    Missing prediction values remain NaN and are omitted by skipna=True.
    """

    obs = ds["Nd_true"].transpose(
        "time", "lev", "site"
    )

    prior = ds["Nd_base"].transpose(
        "time", "lev", "site"
    )

    rec = ds["Nd_rec"].transpose(
        "time", "lev", "site"
    )

    # Same basic validity criterion as original script
    valid = (
        np.isfinite(obs)
        & (obs > minN)
    )

    if use_site_mask:
        smask = (
            ds["site_mask"]
            .transpose("time", "site")
            > 0
        )

        # xarray broadcasts (time,site) over lev
        valid = valid & smask

    obs_valid = obs.where(valid)

    prior_valid = prior.where(
        valid & np.isfinite(prior)
    )

    rec_valid = rec.where(
        valid & np.isfinite(rec)
    )

    return obs_valid, prior_valid, rec_valid


# =====================================================================
# Statistics
# =====================================================================

def build_statistics(
    ds,
    site_names,
    minN=0.1,
    min_samples=10,
    use_site_mask=True,
):
    """
    Calculate:

      1. A true pooled profile over all valid (time, site) samples.
      2. One time-mean profile for each site/experiment.
      3. Per-site Base/NDnet correlation and mean bias against OBS.

    Levels with fewer than min_samples valid OBS values are masked.

    OBS, ND-prior and NDnet are evaluated over the same OBS-defined
    sampling so that the comparison remains collocated. Per-site scalar
    statistics use all collocated (time, pressure) samples at the pressure
    levels that satisfy the same min_samples criterion used by the plot.
    """

    obs, prior, rec = prepare_masked_arrays(
        ds,
        minN=minN,
        use_site_mask=use_site_mask,
    )

    lev = pressure_to_hpa(
        ds["lev"].values
    )

    # =============================================================
    # Pooled profiles
    # =============================================================

    nobs_pooled = obs.count(
        dim=("time", "site")
    )

    enough_pooled = (
        nobs_pooled >= min_samples
    )

    obs_mean = obs.mean(
        dim=("time", "site"),
        skipna=True,
    ).where(enough_pooled)

    obs_std = obs.std(
        dim=("time", "site"),
        skipna=True,
    ).where(enough_pooled)

    prior_mean = prior.mean(
        dim=("time", "site"),
        skipna=True,
    ).where(enough_pooled)

    prior_std = prior.std(
        dim=("time", "site"),
        skipna=True,
    ).where(enough_pooled)

    rec_mean = rec.mean(
        dim=("time", "site"),
        skipna=True,
    ).where(enough_pooled)

    rec_std = rec.std(
        dim=("time", "site"),
        skipna=True,
    ).where(enough_pooled)

    pooled = {
        "lev": lev,

        "obs_mean": obs_mean.values,
        "obs_std": obs_std.values,

        "prior_mean": prior_mean.values,
        "prior_std": prior_std.values,

        "rec_mean": rec_mean.values,
        "rec_std": rec_std.values,

        "nobs": nobs_pooled.values,
    }

    # =============================================================
    # Individual experiment/site profiles
    # =============================================================

    results = []

    for isite, name in enumerate(site_names):

        obs_s = obs.isel(site=isite)
        prior_s = prior.isel(site=isite)
        rec_s = rec.isel(site=isite)

        # Number of valid observations at every pressure level
        nobs = obs_s.count("time")

        enough = (
            nobs >= min_samples
        )

        # Skip experiment if no level meets threshold
        if not bool(enough.any()):
            print(
                f"[{name}] skipped: no pressure levels have "
                f">= {min_samples} observations."
            )
            continue

        # Scalar statistics are based on the same retained pressure levels as
        # the plotted profile, but use all collocated daily samples rather than
        # the time-mean profile itself.
        obs_stats = obs_s.where(enough)
        prior_stats = prior_s.where(enough)
        rec_stats = rec_s.where(enough)

        prior_r, prior_bias, prior_n = paired_statistics(
            obs_stats.values,
            prior_stats.values,
        )
        rec_r, rec_bias, rec_n = paired_statistics(
            obs_stats.values,
            rec_stats.values,
        )

        results.append(
            {
                "site": name,
                "lev": lev,

                "obs_mean":
                    obs_s.mean(
                        "time",
                        skipna=True,
                    ).where(enough).values,

                "obs_std":
                    obs_s.std(
                        "time",
                        skipna=True,
                    ).where(enough).values,

                "prior_mean":
                    prior_s.mean(
                        "time",
                        skipna=True,
                    ).where(enough).values,

                "prior_std":
                    prior_s.std(
                        "time",
                        skipna=True,
                    ).where(enough).values,

                "rec_mean":
                    rec_s.mean(
                        "time",
                        skipna=True,
                    ).where(enough).values,

                "rec_std":
                    rec_s.std(
                        "time",
                        skipna=True,
                    ).where(enough).values,

                "nobs": nobs.values,
                "prior_r": prior_r,
                "prior_bias": prior_bias,
                "prior_n": prior_n,
                "rec_r": rec_r,
                "rec_bias": rec_bias,
                "rec_n": rec_n,
            }
        )

    if not results:
        raise ValueError(
            "No valid site/experiment profiles were found."
        )

    return pooled, results


# =====================================================================
# Plotting helpers
# =====================================================================

def plot_three_profiles(
    ax,
    lev,
    obs_mean,
    obs_std,
    prior_mean,
    prior_std,
    rec_mean,
    rec_std,
):
    """
    Plot OBS, ND-prior and NDnet on a single axis.
    """

    data = [
        (
            "obs",
            np.asarray(obs_mean),
            np.asarray(obs_std),
        ),
        (
            "prior",
            np.asarray(prior_mean),
            np.asarray(prior_std),
        ),
        (
            "rec",
            np.asarray(rec_mean),
            np.asarray(rec_std),
        ),
    ]

    for key, mean, std in data:

        style = SERIES[key]

        ax.plot(
            mean,
            lev,
            color=style["color"],
            linestyle=style["ls"],
            linewidth=style["lw"],
            label=style["label"],
            zorder=3,
        )

        good = (
            np.isfinite(mean)
            & np.isfinite(std)
            & np.isfinite(lev)
        )

        if np.any(good):
            ax.fill_betweenx(
                lev[good],
                mean[good] - std[good],
                mean[good] + std[good],
                color=style["color"],
                alpha=style["alpha"],
                linewidth=0,
                zorder=1,
            )

    # Explicitly linear pressure coordinate
    ax.set_yscale("linear")

    # Journal-style axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        True,
        which="major",
        linewidth=0.35,
        alpha=0.25,
        zorder=0,
    )

    ax.tick_params(
        top=False,
        right=False,
    )

    ax.xaxis.set_major_locator(
        MaxNLocator(
            nbins=4,
            min_n_ticks=3,
        )
    )


# =====================================================================
# Figure 1: all experiments pooled
# =====================================================================

def plot_pooled_profile(
    pooled,
    outfile,
    pbottom=1000.0,
    ptop=250.0,
    xmax=None,
    dpi=400,
):
    """
    Plot the pooled profile over all valid (time,site) samples.

    No global/title text is added.
    """

    fig, ax = plt.subplots(
        figsize=(3.35, 4.25)
    )

    plot_three_profiles(
        ax=ax,
        lev=pooled["lev"],

        obs_mean=pooled["obs_mean"],
        obs_std=pooled["obs_std"],

        prior_mean=pooled["prior_mean"],
        prior_std=pooled["prior_std"],

        rec_mean=pooled["rec_mean"],
        rec_std=pooled["rec_std"],
    )

    ax.set_xlabel(
        r"$N_{\rm d}$ (cm$^{-3}$)"
    )

    ax.set_ylabel(
        "Pressure (hPa)"
    )

    ax.set_ylim(
        pbottom,
        ptop,
    )

    ax.set_xlim(
        left=0.0
    )

    if xmax is not None:
        ax.set_xlim(
            0.0,
            xmax,
        )

    ax.legend(
        loc="upper right",
        frameon=False,
        handlelength=2.6,
    )

    fig.savefig(
        outfile,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    plt.close(fig)

    print(f"Wrote:\n  {outfile}")


# =====================================================================
# Figure 2: one panel per experiment/site
# =====================================================================

def plot_profiles_by_experiment(
    results,
    outfile,
    ncol=4,
    figwidth=7.2,
    panel_height=1.75,
    pbottom=1000.0,
    ptop=250.0,
    xmax=None,
    dpi=400,
):
    """
    Create one panel for each experiment/site.

    Journal formatting:
      - no global title
      - shared legend
      - shared x/y labels
      - panel labels (a), (b), ...
      - compact Base/NN correlation and bias in every panel
      - no redundant interior tick labels
      - common x and y limits
    """

    nsite = len(results)

    ncol = min(
        ncol,
        nsite,
    )

    nrow = int(
        np.ceil(nsite / ncol)
    )

    figheight = (
        panel_height * nrow
        + 0.45
    )

    fig, axs = plt.subplots(
        nrow,
        ncol,
        figsize=(
            figwidth,
            figheight,
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    axs = axs.ravel()

    # -------------------------------------------------------------
    # Draw panels
    # -------------------------------------------------------------

    for i, r in enumerate(results):

        ax = axs[i]

        plot_three_profiles(
            ax=ax,
            lev=r["lev"],

            obs_mean=r["obs_mean"],
            obs_std=r["obs_std"],

            prior_mean=r["prior_mean"],
            prior_std=r["prior_std"],

            rec_mean=r["rec_mean"],
            rec_std=r["rec_std"],
        )

        ax.set_ylim(
            pbottom,
            ptop,
        )

        ax.set_xlim(
            left=0.0
        )

        if xmax is not None:
            ax.set_xlim(
                0.0,
                xmax,
            )

        # Panel label + experiment/site name
        label = panel_letter(i)

        ax.set_title(
            f"({label}) {r['site']}",
            loc="left",
            fontsize=7.2,
            pad=2.5,
        )

        # Compact collocated statistics. Bias is model - OBS in cm^-3.
        ax.text(
            0.98,
            0.98,
            format_panel_statistics(r),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.4,
            linespacing=1.05,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "pad": 0.8,
            },
            zorder=10,
        )

        row = i // ncol
        col = i % ncol

        # Y tick labels only in first column
        if col != 0:
            ax.tick_params(
                axis="y",
                labelleft=False,
            )

        # X tick labels only if this panel has no panel below it
        has_panel_below = (
            i + ncol < nsite
        )

        if has_panel_below:
            ax.tick_params(
                axis="x",
                labelbottom=False,
            )

    # Hide unused axes
    for ax in axs[nsite:]:
        ax.set_visible(False)

    # -------------------------------------------------------------
    # Shared labels
    # -------------------------------------------------------------

    fig.supxlabel(
        r"$N_{\rm d}$ (cm$^{-3}$)",
        fontsize=8.5,
        y=0.012,
    )

    fig.supylabel(
        "Pressure (hPa)",
        fontsize=8.5,
        x=0.015,
    )

    # -------------------------------------------------------------
    # Shared legend
    # -------------------------------------------------------------

    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.997),
        ncol=3,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.6,
    )

    # Compact journal spacing
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.045,
        top=0.955,
        wspace=0.18,
        hspace=0.28,
    )

    fig.savefig(
        outfile,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    plt.close(fig)

    print(f"Wrote:\n  {outfile}")


# =====================================================================
# Main
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Plot OBS, ND-prior and NDnet vertical profiles from "
            "an NDnet_site_test-style NetCDF."
        )
    )

    # -------------------------------------------------------------
    # Input
    # -------------------------------------------------------------

    parser.add_argument(
        "--min_samples",
        type=int,
        default=10,
        help=(
            "Minimum number of valid OBS samples required at each "
            "pressure level. Default: 10"
        ),
    )

    parser.add_argument(
        "--indir",
        default=DEFAULT_DIR,
        help=(
            "Directory containing the NDnet_site_test-style NetCDF. "
            f"Default: {DEFAULT_DIR}"
        ),
    )

    parser.add_argument(
        "--infile",
        default=None,
        help=(
            "Explicit input NetCDF. If omitted, the script searches "
            "--indir automatically."
        ),
    )

    # -------------------------------------------------------------
    # Selection / masking
    # -------------------------------------------------------------

    parser.add_argument(
        "--minN",
        type=float,
        default=0.1,
        help=(
            "Minimum valid OBS Nd in cm-3. "
            "OBS <= minN is masked. Default: 0.1"
        ),
    )

    parser.add_argument(
        "--ignore_site_mask",
        action="store_true",
        help=(
            "Ignore site_mask. By default site_mask > 0 is required."
        ),
    )

    parser.add_argument(
        "--include",
        default=None,
        help=(
            "Regex selecting site/experiment names to include."
        ),
    )

    parser.add_argument(
        "--exclude",
        default=None,
        help=(
            "Regex selecting site/experiment names to exclude."
        ),
    )

    # -------------------------------------------------------------
    # Figure geometry
    # -------------------------------------------------------------

    parser.add_argument(
        "--ncol",
        type=int,
        default=6,
        help=(
            "Number of columns in the experiment-panel figure. "
            "Default: 6"
        ),
    )

    parser.add_argument(
        "--figwidth",
        type=float,
        default=7.2,
        help=(
            "Total width of multi-panel figure in inches. "
            "Default: 7.2"
        ),
    )

    parser.add_argument(
        "--panel_height",
        type=float,
        default=1.75,
        help=(
            "Height per row in the multi-panel figure. "
            "Default: 1.75 inches"
        ),
    )

    parser.add_argument(
        "--pbottom",
        type=float,
        default=1000.0,
        help=(
            "Bottom pressure limit in hPa. Default: 1000"
        ),
    )

    parser.add_argument(
        "--ptop",
        type=float,
        default=250.0,
        help=(
            "Top pressure limit in hPa. Default: 250"
        ),
    )

    parser.add_argument(
        "--xmax",
        type=float,
        default=None,
        help=(
            "Explicit common maximum Nd for x axis. "
            "If omitted it is calculated automatically."
        ),
    )

    # -------------------------------------------------------------
    # Output
    # -------------------------------------------------------------

    parser.add_argument(
        "--outdir",
        default="./",
        help=(
            "Output directory. Default: directory containing input file."
        ),
    )

    parser.add_argument(
        "--format",
        choices=("pdf", "png"),
        default="pdf",
        help=(
            "Output format. Default: pdf"
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help=(
            "Output DPI for raster output. Default: 400"
        ),
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Style
    # -------------------------------------------------------------

    set_journal_style()

    # -------------------------------------------------------------
    # Locate/read data
    # -------------------------------------------------------------

    infile = find_input_file(
        indir=args.indir,
        infile=args.infile,
    )

    ds = read_input(
        infile
    )

    ds, site_names = select_sites(
        ds,
        include=args.include,
        exclude=args.exclude,
    )

    print(
        f"Selected site/experiment entries: {len(site_names)}"
    )

    # -------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------

    pooled, results = build_statistics(
        ds,
        site_names=site_names,
        minN=args.minN,
        min_samples=args.min_samples,
        use_site_mask=not args.ignore_site_mask,
    )

    print(
        f"Valid site/experiment panels: {len(results)}"
    )

    # -------------------------------------------------------------
    # Common x range
    # -------------------------------------------------------------

    if args.xmax is None:
        xmax = calculate_common_xmax(
            results,
            pooled=pooled,
        )
    else:
        xmax = args.xmax

    if xmax is not None:
        print(
            f"Common Nd x-axis range: 0 to {xmax:.1f} cm-3"
        )

    # -------------------------------------------------------------
    # Output paths
    # -------------------------------------------------------------

    if args.outdir is None:
        outdir = os.path.dirname(
            os.path.abspath(infile)
        )
    else:
        outdir = os.path.abspath(
            args.outdir
        )

    os.makedirs(
        outdir,
        exist_ok=True,
    )

    pooled_out = os.path.join(
        outdir,
        f"NDnet_profiles_all_experiments.{args.format}",
    )

    panels_out = os.path.join(
        outdir,
        f"NDnet_profiles_by_experiment.{args.format}",
    )

    # -------------------------------------------------------------
    # Plot pooled figure
    # -------------------------------------------------------------

    plot_pooled_profile(
        pooled=pooled,
        outfile=pooled_out,
        pbottom=args.pbottom,
        ptop=args.ptop,
        xmax=xmax,
        dpi=args.dpi,
    )

    # -------------------------------------------------------------
    # Plot experiment/site panels
    # -------------------------------------------------------------

    plot_profiles_by_experiment(
        results=results,
        outfile=panels_out,
        ncol=args.ncol,
        figwidth=args.figwidth,
        panel_height=args.panel_height,
        pbottom=args.pbottom,
        ptop=args.ptop,
        xmax=xmax,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
