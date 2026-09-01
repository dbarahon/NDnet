#!/usr/bin/env python3

import os
import re
import glob
import csv
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
    """Matplotlib settings suitable for journal figures."""
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
    """Return True if path looks like the requested NDnet_site_test file."""
    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            return REQUIRED_VARS.issubset(set(ds.variables))
    except Exception:
        return False


def find_input_file(indir, infile=None):
    """Find the input NetCDF, preferring NDnet_site_test.nc."""
    if infile is not None:
        infile = os.path.abspath(infile)
        if not os.path.exists(infile):
            raise FileNotFoundError(f"Input file does not exist:\n{infile}")
        if not is_valid_input_file(infile):
            raise ValueError(
                f"{infile}\ndoes not contain all required variables:\n"
                f"{sorted(REQUIRED_VARS)}"
            )
        return infile

    indir = os.path.abspath(indir)
    if not os.path.isdir(indir):
        raise NotADirectoryError(f"Input directory does not exist:\n{indir}")

    exact = os.path.join(indir, "NDnet_site_test.nc")
    if os.path.exists(exact) and is_valid_input_file(exact):
        return exact

    candidates = sorted(glob.glob(os.path.join(indir, "NDnet_site_test*.nc")))
    good = [f for f in candidates if is_valid_input_file(f)]
    if len(good) == 1:
        return good[0]
    if len(good) > 1:
        raise RuntimeError(
            "More than one NDnet_site_test-style file was found.\n"
            "Specify the desired one with --infile:\n\n" + "\n".join(good)
        )

    candidates = sorted(glob.glob(os.path.join(indir, "*.nc")))
    good = [f for f in candidates if is_valid_input_file(f)]
    if len(good) == 1:
        return good[0]
    if len(good) > 1:
        raise RuntimeError(
            "More than one compatible NetCDF was found.\n"
            "Specify the desired one with --infile:\n\n" + "\n".join(good)
        )

    raise FileNotFoundError(
        "Could not find an NDnet_site_test-style NetCDF in:\n"
        f"{indir}\n\nRequired variables are:\n"
        + ", ".join(sorted(REQUIRED_VARS))
    )


# =====================================================================
# Helpers
# =====================================================================

def to_string(value):
    """Convert xarray/netCDF string or bytes values to normal strings."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def panel_letter(index):
    """Convert 0,1,... into a,b,...,z,aa,ab,..."""
    result = ""
    i = index
    while True:
        result = chr(ord("a") + (i % 26)) + result
        i = i // 26 - 1
        if i < 0:
            break
    return result


def pressure_to_hpa(lev):
    """Return pressure coordinate in hPa, converting automatically from Pa."""
    lev = np.asarray(lev, dtype=float)
    finite = lev[np.isfinite(lev)]
    if finite.size and np.nanmedian(np.abs(finite)) > 2000.0:
        print("Pressure coordinate appears to be in Pa; converting to hPa.")
        lev = lev / 100.0
    return lev


def _trapz(y, x):
    """Compatibility wrapper for NumPy 1.x/2.x trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def profile_wasserstein_logp(pressure_hpa, obs_profile, model_profile):
    """1-D Earth Mover/Wasserstein distance for profile shape in ln(p).

    Both profiles are normalized to unit integral in dln(p), so W measures
    vertical shape/location rather than total Nd amplitude. Nonphysical
    negative Nd values are clipped to zero before normalization. The returned
    distance is in log-pressure coordinate units.
    """
    p = np.asarray(pressure_hpa, dtype=float).ravel()
    obs = np.asarray(obs_profile, dtype=float).ravel()
    model = np.asarray(model_profile, dtype=float).ravel()

    good = (
        np.isfinite(p)
        & (p > 0.0)
        & np.isfinite(obs)
        & np.isfinite(model)
    )
    if np.count_nonzero(good) < 2:
        return np.nan

    x = np.log(p[good])
    obs = np.maximum(obs[good], 0.0)
    model = np.maximum(model[good], 0.0)

    order = np.argsort(x)
    x = x[order]
    obs = obs[order]
    model = model[order]

    # Remove duplicate pressure coordinates if any slipped into the input.
    unique = np.concatenate(([True], np.diff(x) > 0.0))
    x = x[unique]
    obs = obs[unique]
    model = model[unique]
    if x.size < 2:
        return np.nan

    obs_area = float(_trapz(obs, x))
    model_area = float(_trapz(model, x))
    if obs_area <= 0.0 or model_area <= 0.0:
        return np.nan

    obs_density = obs / obs_area
    model_density = model / model_area
    dx = np.diff(x)

    obs_cdf = np.concatenate(
        ([0.0], np.cumsum(0.5 * (obs_density[:-1] + obs_density[1:]) * dx))
    )
    model_cdf = np.concatenate(
        ([0.0], np.cumsum(0.5 * (model_density[:-1] + model_density[1:]) * dx))
    )

    # The 1-Wasserstein distance in one dimension is the integral of the
    # absolute CDF difference.
    return float(_trapz(np.abs(obs_cdf - model_cdf), x))


def integrated_bias_pct_logp(pressure_hpa, obs_profile, model_profile):
    """Relative bias (%) of profile-integrated Nd in dln(p)."""
    p = np.asarray(pressure_hpa, dtype=float).ravel()
    obs = np.asarray(obs_profile, dtype=float).ravel()
    model = np.asarray(model_profile, dtype=float).ravel()

    good = (
        np.isfinite(p)
        & (p > 0.0)
        & np.isfinite(obs)
        & np.isfinite(model)
    )
    if np.count_nonzero(good) < 2:
        return np.nan

    x = np.log(p[good])
    obs = np.maximum(obs[good], 0.0)
    model = np.maximum(model[good], 0.0)
    order = np.argsort(x)
    x = x[order]
    obs = obs[order]
    model = model[order]

    obs_int = float(_trapz(obs, x))
    model_int = float(_trapz(model, x))
    if obs_int <= 0.0:
        return np.nan

    return 100.0 * (model_int - obs_int) / obs_int


def daily_wasserstein_pair(lev, obs, prior, rec, retained_levels):
    """Daily W for Base and NN using identical collocated vertical support."""
    lev = np.asarray(lev, dtype=float)
    obs = np.asarray(obs, dtype=float)
    prior = np.asarray(prior, dtype=float)
    rec = np.asarray(rec, dtype=float)
    retained_levels = np.asarray(retained_levels, dtype=bool)

    if obs.ndim != 2:
        raise ValueError(f"Expected daily site profiles as (time, lev), got {obs.shape}")

    w_prior = []
    w_rec = []

    for it in range(obs.shape[0]):
        common = (
            retained_levels
            & np.isfinite(lev)
            & (lev > 0.0)
            & np.isfinite(obs[it])
            & np.isfinite(prior[it])
            & np.isfinite(rec[it])
        )
        if np.count_nonzero(common) < 2:
            continue

        wb = profile_wasserstein_logp(
            lev[common], obs[it, common], prior[it, common]
        )
        wn = profile_wasserstein_logp(
            lev[common], obs[it, common], rec[it, common]
        )

        # Keep only days for which both comparisons are defined, preserving a
        # strictly identical daily sample pool for Base vs NN.
        if np.isfinite(wb) and np.isfinite(wn):
            w_prior.append(wb)
            w_rec.append(wn)

    return np.asarray(w_prior, dtype=float), np.asarray(w_rec, dtype=float)


def summarize_distribution(values):
    """Return median, q25, q75, IQR width and finite sample count."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "iqr": np.nan,
            "n": 0,
        }

    q25, median, q75 = np.percentile(values, [25.0, 50.0, 75.0])
    return {
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "n": int(values.size),
    }


def format_panel_statistics(result):
    """Compact Base/NN W and integrated-bias annotation."""
    def fmt_w(value):
        return f"{value:.3f}" if np.isfinite(value) else "--"

    def fmt_ib(value):
        return f"{value:+.0f}%" if np.isfinite(value) else "--"

    return (
        f"B: W={fmt_w(result['prior_W_daily_median'])}  Ib={fmt_ib(result['prior_Ib_pct'])}\n"
        f"NN: W={fmt_w(result['rec_W_daily_median'])}  Ib={fmt_ib(result['rec_Ib_pct'])}"
    )


def calculate_common_xmax(results, pooled=None):
    """Determine a common positive x-axis maximum for all panels."""
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

    vmax = np.nanmax(np.concatenate(values))
    if not np.isfinite(vmax) or vmax <= 0:
        return None
    return 1.05 * vmax


# =====================================================================
# Read and prepare data
# =====================================================================

def read_input(path):
    """Read required variables into memory and close the source NetCDF."""
    print(f"Reading:\n  {path}")
    with xr.open_dataset(path, decode_times=False) as ds:
        missing = REQUIRED_VARS.difference(ds.variables)
        if missing:
            raise KeyError(
                "Input file is missing required variables:\n"
                + ", ".join(sorted(missing))
            )
        ds = ds[["Nd_true", "Nd_base", "Nd_rec", "site_mask"]].load()
    return ds


def select_sites(ds, include=None, exclude=None):
    """Optionally select site/experiment labels using regex filters."""
    names = [to_string(v) for v in ds["site"].values]
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
        raise ValueError("No sites/experiments remain after include/exclude filtering.")

    ds = ds.isel(site=keep)
    names = [to_string(v) for v in ds["site"].values]
    return ds, names


def prepare_masked_arrays(ds, minN=0.1, use_site_mask=True):
    """Prepare collocated OBS, ND-prior and NDnet arrays.

    The three fields are placed on one common finite sample pool so every Base
    and NN diagnostic uses exactly the same (time, pressure, site) points.
    """
    obs = ds["Nd_true"].transpose("time", "lev", "site")
    prior = ds["Nd_base"].transpose("time", "lev", "site")
    rec = ds["Nd_rec"].transpose("time", "lev", "site")

    valid = np.isfinite(obs) & (obs > minN)
    if use_site_mask:
        smask = ds["site_mask"].transpose("time", "site") > 0
        valid = valid & smask

    common = valid & np.isfinite(prior) & np.isfinite(rec)
    return obs.where(common), prior.where(common), rec.where(common)


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
    """Build plotted profiles and per-site vertical-profile diagnostics.

    W is a shape/location metric: each daily profile is normalized to unit
    integral and the 1-D Wasserstein distance is evaluated in ln(p). The panel
    reports the median daily W. W of the temporal-mean profile and the daily W
    IQR are retained for the supplementary CSV.

    Ib is the relative bias of the temporal-mean profile integral in dln(p),
    expressed as percent. Thus W and Ib separate vertical shape/location from
    overall profile amplitude.
    """
    obs, prior, rec = prepare_masked_arrays(
        ds,
        minN=minN,
        use_site_mask=use_site_mask,
    )
    lev = pressure_to_hpa(ds["lev"].values)

    # Pooled profiles
    nobs_pooled = obs.count(dim=("time", "site"))
    enough_pooled = nobs_pooled >= min_samples

    pooled = {
        "lev": lev,
        "obs_mean": obs.mean(("time", "site"), skipna=True).where(enough_pooled).values,
        "obs_std": obs.std(("time", "site"), skipna=True).where(enough_pooled).values,
        "prior_mean": prior.mean(("time", "site"), skipna=True).where(enough_pooled).values,
        "prior_std": prior.std(("time", "site"), skipna=True).where(enough_pooled).values,
        "rec_mean": rec.mean(("time", "site"), skipna=True).where(enough_pooled).values,
        "rec_std": rec.std(("time", "site"), skipna=True).where(enough_pooled).values,
        "nobs": nobs_pooled.values,
    }

    results = []
    for isite, name in enumerate(site_names):
        obs_s = obs.isel(site=isite)
        prior_s = prior.isel(site=isite)
        rec_s = rec.isel(site=isite)

        nobs = obs_s.count("time")
        enough = nobs >= min_samples
        if not bool(enough.any()):
            print(
                f"[{name}] skipped: no pressure levels have "
                f">= {min_samples} observations."
            )
            continue

        obs_mean = obs_s.mean("time", skipna=True).where(enough).values
        obs_std = obs_s.std("time", skipna=True).where(enough).values
        prior_mean = prior_s.mean("time", skipna=True).where(enough).values
        prior_std = prior_s.std("time", skipna=True).where(enough).values
        rec_mean = rec_s.mean("time", skipna=True).where(enough).values
        rec_std = rec_s.std("time", skipna=True).where(enough).values

        # Daily Wasserstein distributions use the same retained pressure levels
        # and require both Base and NN to exist on each accepted day.
        prior_w_daily, rec_w_daily = daily_wasserstein_pair(
            lev=lev,
            obs=obs_s.values,
            prior=prior_s.values,
            rec=rec_s.values,
            retained_levels=enough.values,
        )
        prior_w = summarize_distribution(prior_w_daily)
        rec_w = summarize_distribution(rec_w_daily)

        # Mean-profile diagnostics use a common vertical support for OBS/Base/NN.
        mean_common = (
            enough.values
            & np.isfinite(lev)
            & (lev > 0.0)
            & np.isfinite(obs_mean)
            & np.isfinite(prior_mean)
            & np.isfinite(rec_mean)
        )

        if np.count_nonzero(mean_common) >= 2:
            prior_w_mean = profile_wasserstein_logp(
                lev[mean_common], obs_mean[mean_common], prior_mean[mean_common]
            )
            rec_w_mean = profile_wasserstein_logp(
                lev[mean_common], obs_mean[mean_common], rec_mean[mean_common]
            )
            prior_ib = integrated_bias_pct_logp(
                lev[mean_common], obs_mean[mean_common], prior_mean[mean_common]
            )
            rec_ib = integrated_bias_pct_logp(
                lev[mean_common], obs_mean[mean_common], rec_mean[mean_common]
            )
        else:
            prior_w_mean = np.nan
            rec_w_mean = np.nan
            prior_ib = np.nan
            rec_ib = np.nan

        results.append(
            {
                "site": name,
                "lev": lev,
                "obs_mean": obs_mean,
                "obs_std": obs_std,
                "prior_mean": prior_mean,
                "prior_std": prior_std,
                "rec_mean": rec_mean,
                "rec_std": rec_std,
                "nobs": nobs.values,
                "prior_W_mean_profile": prior_w_mean,
                "prior_W_daily_median": prior_w["median"],
                "prior_W_daily_q25": prior_w["q25"],
                "prior_W_daily_q75": prior_w["q75"],
                "prior_W_daily_iqr": prior_w["iqr"],
                "prior_W_daily_n": prior_w["n"],
                "prior_Ib_pct": prior_ib,
                "rec_W_mean_profile": rec_w_mean,
                "rec_W_daily_median": rec_w["median"],
                "rec_W_daily_q25": rec_w["q25"],
                "rec_W_daily_q75": rec_w["q75"],
                "rec_W_daily_iqr": rec_w["iqr"],
                "rec_W_daily_n": rec_w["n"],
                "rec_Ib_pct": rec_ib,
            }
        )

    if not results:
        raise ValueError("No valid site/experiment profiles were found.")

    return pooled, results


def write_supplementary_statistics(results, outfile):
    """Write site-level W/Ib statistics for supplementary material."""
    fields = [
        "site",
        "base_W_mean_profile_logp",
        "base_W_daily_median_logp",
        "base_W_daily_q25_logp",
        "base_W_daily_q75_logp",
        "base_W_daily_IQR_logp",
        "base_Ib_pct",
        "base_n_daily_profiles",
        "NN_W_mean_profile_logp",
        "NN_W_daily_median_logp",
        "NN_W_daily_q25_logp",
        "NN_W_daily_q75_logp",
        "NN_W_daily_IQR_logp",
        "NN_Ib_pct",
        "NN_n_daily_profiles",
    ]

    def fnum(value, digits=6):
        return "" if not np.isfinite(value) else f"{float(value):.{digits}f}"

    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "site": r["site"],
                    "base_W_mean_profile_logp": fnum(r["prior_W_mean_profile"]),
                    "base_W_daily_median_logp": fnum(r["prior_W_daily_median"]),
                    "base_W_daily_q25_logp": fnum(r["prior_W_daily_q25"]),
                    "base_W_daily_q75_logp": fnum(r["prior_W_daily_q75"]),
                    "base_W_daily_IQR_logp": fnum(r["prior_W_daily_iqr"]),
                    "base_Ib_pct": fnum(r["prior_Ib_pct"], digits=2),
                    "base_n_daily_profiles": int(r["prior_W_daily_n"]),
                    "NN_W_mean_profile_logp": fnum(r["rec_W_mean_profile"]),
                    "NN_W_daily_median_logp": fnum(r["rec_W_daily_median"]),
                    "NN_W_daily_q25_logp": fnum(r["rec_W_daily_q25"]),
                    "NN_W_daily_q75_logp": fnum(r["rec_W_daily_q75"]),
                    "NN_W_daily_IQR_logp": fnum(r["rec_W_daily_iqr"]),
                    "NN_Ib_pct": fnum(r["rec_Ib_pct"], digits=2),
                    "NN_n_daily_profiles": int(r["rec_W_daily_n"]),
                }
            )

    print(f"Wrote:\n  {outfile}")


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
    """Plot OBS, ND-prior and NDnet on a single axis."""
    data = [
        ("obs", np.asarray(obs_mean), np.asarray(obs_std)),
        ("prior", np.asarray(prior_mean), np.asarray(prior_std)),
        ("rec", np.asarray(rec_mean), np.asarray(rec_std)),
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

        good = np.isfinite(mean) & np.isfinite(std) & np.isfinite(lev)
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

    ax.set_yscale("linear")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linewidth=0.35, alpha=0.25, zorder=0)
    ax.tick_params(top=False, right=False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))


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
    """Plot the pooled profile over all valid (time, site) samples."""
    fig, ax = plt.subplots(figsize=(3.35, 4.25))
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

    ax.set_xlabel(r"$N_{\rm d}$ (cm$^{-3}$)")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_ylim(pbottom, ptop)
    ax.set_xlim(left=0.0)
    if xmax is not None:
        ax.set_xlim(0.0, xmax)

    ax.legend(loc="upper right", frameon=False, handlelength=2.6)
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
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
    """Create one journal-style panel for each site/experiment."""
    nsite = len(results)
    ncol = min(ncol, nsite)
    nrow = int(np.ceil(nsite / ncol))
    figheight = panel_height * nrow + 0.45

    fig, axs = plt.subplots(
        nrow,
        ncol,
        figsize=(figwidth, figheight),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axs = axs.ravel()

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

        ax.set_ylim(pbottom, ptop)
        ax.set_xlim(left=0.0)
        if xmax is not None:
            ax.set_xlim(0.0, xmax)

        label = panel_letter(i)
        ax.set_title(
            f"({label}) {r['site']}",
            loc="left",
            fontsize=7.2,
            pad=2.5,
        )

        # W is median daily profile Wasserstein distance in ln(p); Ib is the
        # mean-profile integral bias. Both are compact enough for six columns.
        ax.text(
            0.98,
            0.98,
            format_panel_statistics(r),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.2,
            linespacing=1.05,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "pad": 0.8,
            },
            zorder=10,
        )

        col = i % ncol
        if col != 0:
            ax.tick_params(axis="y", labelleft=False)

        if i + ncol < nsite:
            ax.tick_params(axis="x", labelbottom=False)

    for ax in axs[nsite:]:
        ax.set_visible(False)

    fig.supxlabel(r"$N_{\rm d}$ (cm$^{-3}$)", fontsize=8.5, y=0.012)
    fig.supylabel("Pressure (hPa)", fontsize=8.5, x=0.015)

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

    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.045,
        top=0.955,
        wspace=0.18,
        hspace=0.28,
    )
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
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
    parser.add_argument(
        "--minN",
        type=float,
        default=0.1,
        help=(
            "Minimum valid OBS Nd in cm-3. OBS <= minN is masked. "
            "Default: 0.1"
        ),
    )
    parser.add_argument(
        "--ignore_site_mask",
        action="store_true",
        help="Ignore site_mask. By default site_mask > 0 is required.",
    )
    parser.add_argument(
        "--include",
        default=None,
        help="Regex selecting site/experiment names to include.",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="Regex selecting site/experiment names to exclude.",
    )
    parser.add_argument(
        "--ncol",
        type=int,
        default=6,
        help="Number of columns in the experiment-panel figure. Default: 6",
    )
    parser.add_argument(
        "--figwidth",
        type=float,
        default=7.2,
        help="Total width of multi-panel figure in inches. Default: 7.2",
    )
    parser.add_argument(
        "--panel_height",
        type=float,
        default=1.75,
        help="Height per row in the multi-panel figure. Default: 1.75 inches",
    )
    parser.add_argument(
        "--pbottom",
        type=float,
        default=1000.0,
        help="Bottom pressure limit in hPa. Default: 1000",
    )
    parser.add_argument(
        "--ptop",
        type=float,
        default=250.0,
        help="Top pressure limit in hPa. Default: 250",
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
    parser.add_argument(
        "--outdir",
        default="./",
        help="Output directory. Default: current directory.",
    )
    parser.add_argument(
        "--format",
        choices=("pdf", "png"),
        default="pdf",
        help="Output format. Default: pdf",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="Output DPI for raster output. Default: 400",
    )

    args = parser.parse_args()
    set_journal_style()

    infile = find_input_file(indir=args.indir, infile=args.infile)
    ds = read_input(infile)
    ds, site_names = select_sites(ds, include=args.include, exclude=args.exclude)
    print(f"Selected site/experiment entries: {len(site_names)}")

    pooled, results = build_statistics(
        ds,
        site_names=site_names,
        minN=args.minN,
        min_samples=args.min_samples,
        use_site_mask=not args.ignore_site_mask,
    )
    print(f"Valid site/experiment panels: {len(results)}")

    if args.xmax is None:
        xmax = calculate_common_xmax(results, pooled=pooled)
    else:
        xmax = args.xmax
    if xmax is not None:
        print(f"Common Nd x-axis range: 0 to {xmax:.1f} cm-3")

    if args.outdir is None:
        outdir = os.path.dirname(os.path.abspath(infile))
    else:
        outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    pooled_out = os.path.join(
        outdir, f"NDnet_profiles_all_experiments.{args.format}"
    )
    panels_out = os.path.join(
        outdir, f"NDnet_profiles_by_experiment.{args.format}"
    )
    stats_out = os.path.join(outdir, "NDnet_site_profile_statistics.csv")

    # Write the supplementary table before plotting so the numerical
    # diagnostics are available even if a graphics backend later fails.
    write_supplementary_statistics(results, stats_out)

    plot_pooled_profile(
        pooled=pooled,
        outfile=pooled_out,
        pbottom=args.pbottom,
        ptop=args.ptop,
        xmax=xmax,
        dpi=args.dpi,
    )
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
