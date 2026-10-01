#!/usr/bin/env python3
"""Evaluate NDnet site profiles against the original, unfilled CloudNet files.

Companion to plots/sites_eval.py.  The existing diagnostic reads Nd_true from
NDnet_site_test.nc, but that field is the conditioned/fill-completed site field
used by the refinement pipeline.  This version reads the original daily
CloudNet files directly and uses their unfilled NDinc_cf01p values as OBS.

OBS remains on its native pressure grid. NDnet-prior and NDnet are interpolated
to the native OBS pressure levels, dates are aligned by calendar day, and all
statistics use one common finite OBS/prior/NDnet sample pool.

Soverato is excluded by default because it is intentionally omitted from the
analysis for insufficient sampling. Pass --exclude "" to disable that default.
"""

import argparse
import csv
import glob
import os
import re

import numpy as np
import xarray as xr

import sites_eval as base


DEFAULT_DIR = (
    "/panfs/ccds02/nobackup/people/dbarahon/"
    "NDnet/EOF/refinement/push6"
)

DEFAULT_OBS_DIR = (
    "/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/dailies/"
    "Cloudnet_sites_multiyear"
)

OBS_VAR = "NDinc_cf01p"
DEFAULT_EXCLUDE = r"^soverato$"


def normalize_daily_time(da):
    """Normalize a DataArray time coordinate to midnight calendar days."""
    if "time" not in da.coords:
        raise ValueError("DataArray has no time coordinate.")

    values = np.asarray(da["time"].values)
    try:
        days = values.astype("datetime64[D]").astype("datetime64[ns]")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Could not convert time coordinate to numpy datetime64 days. "
            "Open both input files with decoded CF times."
        ) from exc

    if np.unique(days).size != days.size:
        raise ValueError(
            "Duplicate calendar days found after normalizing the time coordinate."
        )

    return da.assign_coords(time=("time", days))


def read_predictions(path):
    """Read only the prior/refined predictions needed for this diagnostic."""
    print(f"Reading predictions:\n  {path}")
    with xr.open_dataset(path, decode_times=True) as ds:
        required = {"Nd_base", "Nd_rec", "site", "lev"}
        missing = required.difference(ds.variables)
        if missing:
            raise KeyError(
                "Input file is missing required variables:\n"
                + ", ".join(sorted(missing))
            )

        pred = ds[["Nd_base", "Nd_rec"]].load()
        pred = pred.assign_coords(
            lev=("lev", base.pressure_to_hpa(ds["lev"].values))
        )

    for name in ("Nd_base", "Nd_rec"):
        pred[name] = normalize_daily_time(
            pred[name].transpose("time", "lev", "site")
        )

    return pred


def select_sites(ds, include=None, exclude=None):
    """Apply optional regex site selection."""
    names = [base.to_string(v) for v in ds["site"].values]
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
        raise ValueError("No sites remain after include/exclude filtering.")

    ds = ds.isel(site=keep)
    names = [base.to_string(v) for v in ds["site"].values]
    return ds, names


def resolve_obs_file(obs_dir, site, obs_var):
    """Resolve exactly one original daily CloudNet file for a site."""
    pattern = os.path.join(
        obs_dir,
        f"Cloudnet.{site}.{obs_var}.tavg1D.*.nc",
    )
    hits = sorted(glob.glob(pattern))

    if not hits:
        raise FileNotFoundError(
            f"No original CloudNet file found for site '{site}':\n{pattern}"
        )
    if len(hits) > 1:
        raise RuntimeError(
            f"More than one original CloudNet file found for '{site}':\n"
            + "\n".join(hits)
        )

    return hits[0]


def read_raw_obs(path, obs_var):
    """Read one unfilled CloudNet Nd field on its native pressure grid."""
    with xr.open_dataset(path, decode_times=True) as ds:
        if obs_var not in ds:
            raise KeyError(f"{path} does not contain variable '{obs_var}'")
        obs = ds[obs_var].load()

    try:
        obs = obs.transpose("time", "lev")
    except Exception as exc:
        raise ValueError(
            f"{path}: expected {obs_var} to have time and lev dimensions; "
            f"got {obs.dims}"
        ) from exc

    obs = obs.assign_coords(
        lev=("lev", base.pressure_to_hpa(obs["lev"].values))
    )
    obs = obs.sortby("lev")
    return normalize_daily_time(obs)


def collocate_site(obs, prior, rec, minN):
    """Collocate native-grid OBS with prior/refined model profiles.

    The model profiles are interpolated to the original observation pressure
    levels. The observations themselves are never vertically interpolated.
    """
    prior = prior.sortby("lev").interp(lev=obs["lev"])
    rec = rec.sortby("lev").interp(lev=obs["lev"])

    # Exact daily alignment to the dates in the original observation file.
    prior = prior.reindex(time=obs["time"])
    rec = rec.reindex(time=obs["time"])

    common = (
        np.isfinite(obs)
        & (obs > float(minN))
        & np.isfinite(prior)
        & np.isfinite(rec)
    )

    return (
        obs.where(common),
        prior.where(common),
        rec.where(common),
        common,
    )


def build_statistics(
    pred,
    site_names,
    obs_dir,
    obs_var=OBS_VAR,
    minN=0.1,
    min_samples=10,
):
    """Build site profiles/statistics from the original unfilled observations."""
    results = []

    for isite, name in enumerate(site_names):
        obs_file = resolve_obs_file(obs_dir, name, obs_var)
        obs_raw = read_raw_obs(obs_file, obs_var)

        raw_valid = np.isfinite(obs_raw) & (obs_raw > float(minN))
        raw_valid_days = int(raw_valid.any("lev").sum().item())

        obs, prior, rec, common = collocate_site(
            obs=obs_raw,
            prior=pred["Nd_base"].isel(site=isite),
            rec=pred["Nd_rec"].isel(site=isite),
            minN=minN,
        )

        nobs = obs.count("time")
        enough = nobs >= int(min_samples)

        collocated_days = int(common.any("lev").sum().item())
        max_collocated = int(nobs.max().item()) if nobs.size else 0
        retained_levels = int(enough.sum().item())

        if not bool(enough.any()):
            print(
                f"[{name}] skipped: raw_valid_days={raw_valid_days}, "
                f"collocated_days={collocated_days}, "
                f"max_collocated_per_level={max_collocated}; "
                f"no level has >= {min_samples} samples."
            )
            continue

        lev = obs["lev"].values.astype(float)

        obs_mean = obs.mean("time", skipna=True).where(enough).values
        obs_std = obs.std("time", skipna=True).where(enough).values
        prior_mean = prior.mean("time", skipna=True).where(enough).values
        prior_std = prior.std("time", skipna=True).where(enough).values
        rec_mean = rec.mean("time", skipna=True).where(enough).values
        rec_std = rec.std("time", skipna=True).where(enough).values

        prior_w_daily, rec_w_daily = base.daily_wasserstein_pair(
            lev=lev,
            obs=obs.values,
            prior=prior.values,
            rec=rec.values,
            retained_levels=enough.values,
        )
        prior_w = base.summarize_distribution(prior_w_daily)
        rec_w = base.summarize_distribution(rec_w_daily)

        mean_common = (
            enough.values
            & np.isfinite(lev)
            & (lev > 0.0)
            & np.isfinite(obs_mean)
            & np.isfinite(prior_mean)
            & np.isfinite(rec_mean)
        )

        if np.count_nonzero(mean_common) >= 2:
            prior_w_mean = base.profile_wasserstein_logp(
                lev[mean_common],
                obs_mean[mean_common],
                prior_mean[mean_common],
            )
            rec_w_mean = base.profile_wasserstein_logp(
                lev[mean_common],
                obs_mean[mean_common],
                rec_mean[mean_common],
            )
            prior_ib = base.integrated_bias_pct_logp(
                lev[mean_common],
                obs_mean[mean_common],
                prior_mean[mean_common],
            )
            rec_ib = base.integrated_bias_pct_logp(
                lev[mean_common],
                obs_mean[mean_common],
                rec_mean[mean_common],
            )
        else:
            prior_w_mean = np.nan
            rec_w_mean = np.nan
            prior_ib = np.nan
            rec_ib = np.nan

        print(
            f"[{name}] KEEP: raw_valid_days={raw_valid_days}, "
            f"collocated_days={collocated_days}, "
            f"retained_levels={retained_levels}"
        )

        results.append(
            {
                "site": name,
                "obs_file": obs_file,
                "lev": lev,
                "obs_mean": obs_mean,
                "obs_std": obs_std,
                "prior_mean": prior_mean,
                "prior_std": prior_std,
                "rec_mean": rec_mean,
                "rec_std": rec_std,
                "nobs": nobs.values,
                "raw_valid_days": raw_valid_days,
                "collocated_days": collocated_days,
                "retained_levels": retained_levels,
                "max_collocated_per_level": max_collocated,
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
        raise ValueError("No valid raw-observation site profiles were found.")

    return results


def write_statistics(results, outfile):
    """Write raw-observation W/Ib and sampling diagnostics."""
    fields = [
        "site",
        "raw_valid_days",
        "collocated_days",
        "retained_pressure_levels",
        "max_collocated_samples_per_level",
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
        "obs_file",
    ]

    def fnum(value, digits=6):
        return "" if not np.isfinite(value) else f"{float(value):.{digits}f}"

    with open(outfile, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for r in results:
            writer.writerow(
                {
                    "site": r["site"],
                    "raw_valid_days": r["raw_valid_days"],
                    "collocated_days": r["collocated_days"],
                    "retained_pressure_levels": r["retained_levels"],
                    "max_collocated_samples_per_level":
                        r["max_collocated_per_level"],
                    "base_W_mean_profile_logp":
                        fnum(r["prior_W_mean_profile"]),
                    "base_W_daily_median_logp":
                        fnum(r["prior_W_daily_median"]),
                    "base_W_daily_q25_logp":
                        fnum(r["prior_W_daily_q25"]),
                    "base_W_daily_q75_logp":
                        fnum(r["prior_W_daily_q75"]),
                    "base_W_daily_IQR_logp":
                        fnum(r["prior_W_daily_iqr"]),
                    "base_Ib_pct":
                        fnum(r["prior_Ib_pct"], digits=2),
                    "base_n_daily_profiles":
                        int(r["prior_W_daily_n"]),
                    "NN_W_mean_profile_logp":
                        fnum(r["rec_W_mean_profile"]),
                    "NN_W_daily_median_logp":
                        fnum(r["rec_W_daily_median"]),
                    "NN_W_daily_q25_logp":
                        fnum(r["rec_W_daily_q25"]),
                    "NN_W_daily_q75_logp":
                        fnum(r["rec_W_daily_q75"]),
                    "NN_W_daily_IQR_logp":
                        fnum(r["rec_W_daily_iqr"]),
                    "NN_Ib_pct":
                        fnum(r["rec_Ib_pct"], digits=2),
                    "NN_n_daily_profiles":
                        int(r["rec_W_daily_n"]),
                    "obs_file": r["obs_file"],
                }
            )

    print(f"Wrote:\n  {outfile}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare NDnet-prior and NDnet profiles against the original, "
            "unfilled CloudNet site files."
        )
    )
    parser.add_argument(
        "--indir",
        default=DEFAULT_DIR,
        help=f"Directory containing NDnet_site_test.nc. Default: {DEFAULT_DIR}",
    )
    parser.add_argument(
        "--infile",
        default=None,
        help="Explicit NDnet_site_test-style prediction NetCDF.",
    )
    parser.add_argument(
        "--obs-dir",
        default=DEFAULT_OBS_DIR,
        help=f"Directory containing original CloudNet site files. Default: {DEFAULT_OBS_DIR}",
    )
    parser.add_argument(
        "--obs-var",
        default=OBS_VAR,
        help=f"Observation variable. Default: {OBS_VAR}",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=10,
        help="Minimum collocated daily samples required at a pressure level. Default: 10",
    )
    parser.add_argument(
        "--minN",
        type=float,
        default=0.1,
        help="Minimum valid observed Nd in cm-3. Default: 0.1",
    )
    parser.add_argument(
        "--include",
        default=None,
        help="Regex selecting site names to include.",
    )
    parser.add_argument(
        "--exclude",
        default=DEFAULT_EXCLUDE,
        help=(
            "Regex selecting site names to exclude. "
            f"Default: {DEFAULT_EXCLUDE!r}; pass an empty string to disable."
        ),
    )
    parser.add_argument(
        "--ncol",
        type=int,
        default=6,
        help="Number of columns in the site-panel figure. Default: 6",
    )
    parser.add_argument(
        "--figwidth",
        type=float,
        default=7.2,
        help="Total multi-panel figure width in inches. Default: 7.2",
    )
    parser.add_argument(
        "--panel_height",
        type=float,
        default=1.75,
        help="Height per row in inches. Default: 1.75",
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
        help="Explicit common maximum Nd for x axis; otherwise automatic.",
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
    base.set_journal_style()

    obs_dir = os.path.abspath(
        os.path.expanduser(os.path.expandvars(args.obs_dir))
    )
    if not os.path.isdir(obs_dir):
        raise NotADirectoryError(
            f"Original CloudNet observation directory does not exist:\n{obs_dir}"
        )

    infile = base.find_input_file(indir=args.indir, infile=args.infile)
    pred = read_predictions(infile)
    pred, site_names = select_sites(
        pred,
        include=args.include,
        exclude=args.exclude,
    )

    print(f"Selected site entries before raw-data checks: {len(site_names)}")
    if args.exclude:
        print(f"Exclusion regex: {args.exclude}")

    results = build_statistics(
        pred=pred,
        site_names=site_names,
        obs_dir=obs_dir,
        obs_var=args.obs_var,
        minN=args.minN,
        min_samples=args.min_samples,
    )
    print(f"Valid raw-observation site panels: {len(results)}")

    if args.xmax is None:
        xmax = base.calculate_common_xmax(results)
    else:
        xmax = args.xmax

    if xmax is not None:
        print(f"Common Nd x-axis range: 0 to {xmax:.1f} cm-3")

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    panels_out = os.path.join(
        outdir,
        f"NDnet_profiles_by_experiment_raw_obs.{args.format}",
    )
    stats_out = os.path.join(
        outdir,
        "NDnet_site_profile_statistics_raw_obs.csv",
    )

    write_statistics(results, stats_out)

    base.plot_profiles_by_experiment(
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
