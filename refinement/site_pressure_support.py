#!/usr/bin/env python3
"""
Summarize raw vertical pressure support from fixed CloudNet refinement sites.

This utility follows the fixed-site data path used by refinement training:

- site list from SITE_JSON;
- per-site CloudNet files containing NDinc_cf01p(time, lev);
- pressures below minP are excluded;
- if the site lev grid differs from the EOF grid, NDinc_cf01p is interpolated
  onto the EOF lev grid before support is counted, matching data_handling.py.

Crucially, support is computed BEFORE the training-time missing-value filling.
This therefore measures direct observational support, not the effective filled
training target.

Outputs
-------
site_pressure_support.csv
    Per-site distribution of raw finite observations over EOF pressure levels.
    Each <SITE>_pct column sums to 100% for that site. Includes pooled
    sample-weighted support, equal-site support, cumulative support aloft,
    number of sites with support, and effective number of contributing sites.

site_pressure_availability.csv
    For each site and EOF level, percentage of usable raw site-days that have a
    finite NDinc_cf01p value at that level. Usable raw site-days require at
    least --min-valid-lev finite pressure levels, matching the refinement gate.
    Includes site-balanced and day-weighted aggregate availability.

site_pressure_summary.csv
    Per-site counts and pressure-range diagnostics.

Notes
-----
The equal-site aggregate is the preferred quantity for comparison with the
campaign-balanced support diagnostic because it prevents long-running sites
from dominating the vertical-support estimate merely through record length.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_SITE_DIR = Path(
    "/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/dailies/Cloudnet_sites_multiyear"
)
DEFAULT_SITE_JSON = DEFAULT_SITE_DIR / "site_timeprofile_diagnostics.jsonl"
DEFAULT_EOF_FILE = Path(
    "/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/in_cloudM2/best/Nd_EOF.nc"
)
DEFAULT_VAR = "NDinc_cf01p"
DEFAULT_SITE_PREFIX = "*20[1-2]*"
DEFAULT_MIN_P_HPA = 250.0
DEFAULT_MIN_VALID_LEV = 2
DEFAULT_THRESHOLDS_HPA = (800.0, 750.0, 700.0, 650.0, 600.0, 550.0, 500.0, 450.0, 400.0, 350.0)


def _as_hpa(values: np.ndarray, name: str) -> np.ndarray:
    """Return pressure values in hPa; auto-convert Pa when clearly needed."""
    out = np.asarray(values, dtype=float)
    finite = out[np.isfinite(out) & (out > 0)]
    if finite.size == 0:
        return out
    med = float(np.nanmedian(finite))
    if med > 2000.0:
        print(f"[INFO] {name}: median={med:.1f}; assuming Pa and converting to hPa")
        out = out / 100.0
    return out


def load_model_levels(eof_file: Path, levels_text: str | None) -> np.ndarray:
    if levels_text:
        levels = np.array([float(x) for x in levels_text.split(",")], dtype=float)
    else:
        with xr.open_dataset(eof_file) as ds:
            if "lev" not in ds.coords and "lev" not in ds:
                raise KeyError(f"{eof_file} has no 'lev' coordinate")
            levels = np.asarray(ds["lev"].values, dtype=float)

    levels = _as_hpa(levels, "EOF levels")
    levels = levels[np.isfinite(levels) & (levels > 0)]
    levels = np.unique(levels)
    if levels.size < 2:
        raise ValueError("Need at least two valid EOF pressure levels")
    return np.sort(levels)[::-1]


def load_site_catalog(path: Path) -> list[str]:
    """Load site names with the same JSON/JSONL conventions as data_handling.py."""
    names: list[str] = []
    if str(path).endswith(".jsonl"):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if "site" in d:
                    names.append(str(d["site"]))
    else:
        with open(path, "r") as f:
            obj = json.load(f)
        names = [str(k) for k in obj.keys()]

    return list(dict.fromkeys(names))


def resolve_site_nc(var_dir: Path, site: str, varname: str, prefix: str) -> Path:
    """Mirror the resolver used in refinement/data_handling.py."""
    var_dir = Path(var_dir)

    if prefix and (("*" in prefix) or ("?" in prefix) or str(prefix).endswith(".nc")):
        hits = sorted(glob.glob(str(var_dir / prefix)))
        if hits:
            for h in hits:
                bn = Path(h).name
                if f".{site}." in bn and f".{varname}." in bn:
                    return Path(h)
            for h in hits:
                if f".{site}." in Path(h).name:
                    return Path(h)
            return Path(hits[0])

    if prefix:
        fn = var_dir / f"Cloudnet.{site}.{varname}.tavg1D.{prefix}.nc"
        if fn.exists():
            return fn

    hits = sorted(var_dir.glob(f"Cloudnet.{site}.{varname}.tavg1D.*.nc"))
    if not hits:
        raise FileNotFoundError(
            f"No per-site file for site={site!r} var={varname!r} under {var_dir}"
        )
    return hits[0]


def _prepare_site_da(
    path: Path,
    varname: str,
    eof_levels_hpa: np.ndarray,
    min_p_hpa: float,
) -> xr.DataArray:
    """Load raw site Nd and put it on the EOF pressure grid before any filling."""
    with xr.open_dataset(path, decode_times=False) as ds:
        if varname not in ds:
            raise KeyError(f"{path} missing variable {varname!r}")
        if "lev" not in ds.coords:
            raise KeyError(f"{path} missing pressure coordinate 'lev'")

        da = ds[varname].load()
        if "time" not in da.dims or "lev" not in da.dims:
            raise ValueError(
                f"{path}: {varname} must contain dims ('time','lev'); got {da.dims}"
            )

        extra = [d for d in da.dims if d not in ("time", "lev")]
        for d in extra:
            if da.sizes[d] != 1:
                raise ValueError(
                    f"{path}: unexpected non-singleton dimension {d!r}={da.sizes[d]}"
                )
            da = da.isel({d: 0}, drop=True)

        lev_native = _as_hpa(np.asarray(da["lev"].values, dtype=float), f"{path.name}: lev")
        da = da.assign_coords(lev=("lev", lev_native)).sortby("lev")
        da = da.sel(lev=da["lev"].where(da["lev"] >= float(min_p_hpa), drop=True))

        eof_asc = np.sort(np.asarray(eof_levels_hpa, dtype=float))
        native = np.asarray(da["lev"].values, dtype=float)
        if not np.array_equal(native, eof_asc):
            da = da.interp(lev=xr.DataArray(eof_asc, dims="lev"))

        da = da.reindex(lev=eof_asc)
        return da.transpose("time", "lev")


def _percent(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = float(np.nansum(values))
    if total <= 0:
        return np.full(values.shape, np.nan, dtype=float)
    return 100.0 * values / total


def _effective_number(weights_by_site_level: np.ndarray) -> np.ndarray:
    """Effective number of sites at each level from nonnegative site weights."""
    w = np.asarray(weights_by_site_level, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    num = np.sum(w, axis=0) ** 2
    den = np.sum(w * w, axis=0)
    out = np.zeros(w.shape[1], dtype=float)
    good = den > 0
    out[good] = num[good] / den[good]
    return out


def build_tables(
    site_dir: Path,
    site_json: Path,
    site_prefix: str,
    varname: str,
    eof_levels_hpa: np.ndarray,
    min_p_hpa: float,
    min_valid_lev: int,
    thresholds_hpa: list[float],
):
    sites = load_site_catalog(site_json)
    if not sites:
        raise RuntimeError(f"No sites found in {site_json}")

    levels_desc = np.sort(np.asarray(eof_levels_hpa, dtype=float))[::-1]
    levels_asc = levels_desc[::-1]

    counts_by_site: dict[str, np.ndarray] = {}
    avail_by_site: dict[str, np.ndarray] = {}
    valid_days_by_site: dict[str, int] = {}
    summary_rows = []
    used_sites: list[str] = []

    for site in sites:
        try:
            path = resolve_site_nc(site_dir, site, varname, site_prefix)
            da = _prepare_site_da(path, varname, levels_desc, min_p_hpa)
        except FileNotFoundError as exc:
            print(f"[SKIP] {site}: {exc}")
            continue

        finite = np.isfinite(np.asarray(da.values, dtype=float))
        ntime = int(finite.shape[0])
        nlev_each_day = finite.sum(axis=1)
        valid_day = nlev_each_day >= int(min_valid_lev)
        n_valid_days = int(valid_day.sum())

        finite_use = finite & valid_day[:, None]
        counts_asc = finite_use.sum(axis=0).astype(float)

        if n_valid_days > 0:
            avail_asc = 100.0 * counts_asc / float(n_valid_days)
        else:
            avail_asc = np.full(counts_asc.shape, np.nan, dtype=float)

        counts = counts_asc[::-1]
        avail = avail_asc[::-1]

        counts_by_site[site] = counts
        avail_by_site[site] = avail
        valid_days_by_site[site] = n_valid_days
        used_sites.append(site)

        supported = levels_desc[counts > 0]
        row = {
            "site": site,
            "file": str(path),
            "n_time": ntime,
            "n_valid_days": n_valid_days,
            "n_raw_level_observations": int(np.sum(counts)),
            "p_min_hpa": float(np.min(supported)) if supported.size else np.nan,
            "p_max_hpa": float(np.max(supported)) if supported.size else np.nan,
        }
        pct_dist = _percent(counts)
        for threshold in thresholds_hpa:
            mask = levels_desc <= float(threshold)
            row[f"support_pct_p_le_{threshold:g}hPa"] = (
                float(np.nansum(pct_dist[mask])) if np.isfinite(pct_dist).any() else np.nan
            )
            if n_valid_days > 0:
                raw_asc = finite_use[:, levels_asc <= float(threshold)]
                row[f"days_pct_any_p_le_{threshold:g}hPa"] = (
                    100.0 * float(np.any(raw_asc, axis=1).sum()) / n_valid_days
                    if raw_asc.shape[1] > 0 else 0.0
                )
            else:
                row[f"days_pct_any_p_le_{threshold:g}hPa"] = np.nan

        summary_rows.append(row)
        print(
            f"[READ] {site:24s}: valid_days={n_valid_days:5d} "
            f"raw_level_obs={int(np.sum(counts)):7d} file={path.name}"
        )

    if not used_sites:
        raise RuntimeError("No usable site files found")

    support_df = pd.DataFrame({"pressure_hpa": levels_desc})
    site_pct_stack = []
    raw_count_stack = []
    for site in used_sites:
        pct = _percent(counts_by_site[site])
        support_df[f"{site}_pct"] = pct
        site_pct_stack.append(pct)
        raw_count_stack.append(counts_by_site[site])

    site_pct_arr = np.stack(site_pct_stack, axis=0)
    raw_count_arr = np.stack(raw_count_stack, axis=0)

    pooled_counts = np.sum(raw_count_arr, axis=0)
    pooled_pct = _percent(pooled_counts)
    balanced_pct = np.nanmean(site_pct_arr, axis=0)
    balanced_pct = _percent(balanced_pct)

    support_df["ALL_sample_weighted_pct"] = pooled_pct
    support_df["ALL_site_balanced_pct"] = balanced_pct
    support_df["ALL_sample_weighted_cum_aloft_pct"] = np.cumsum(pooled_pct[::-1])[::-1]
    support_df["ALL_site_balanced_cum_aloft_pct"] = np.cumsum(balanced_pct[::-1])[::-1]
    support_df["n_sites_supported"] = np.sum(raw_count_arr > 0, axis=0).astype(int)
    support_df["N_eff_sites"] = _effective_number(site_pct_arr)

    avail_df = pd.DataFrame({"pressure_hpa": levels_desc})
    avail_stack = []
    for site in used_sites:
        avail_df[f"{site}_pct_days"] = avail_by_site[site]
        avail_stack.append(avail_by_site[site])

    avail_arr = np.stack(avail_stack, axis=0)
    avail_df["ALL_site_balanced_pct_days"] = np.nanmean(avail_arr, axis=0)

    day_weights = np.asarray([valid_days_by_site[s] for s in used_sites], dtype=float)
    den = np.sum(day_weights)
    if den > 0:
        avail_df["ALL_day_weighted_pct_days"] = 100.0 * pooled_counts / den
    else:
        avail_df["ALL_day_weighted_pct_days"] = np.nan

    avail_df["n_sites_supported"] = np.sum(raw_count_arr > 0, axis=0).astype(int)
    avail_df["N_eff_sites"] = _effective_number(np.nan_to_num(avail_arr, nan=0.0))

    summary_df = pd.DataFrame(summary_rows)
    return used_sites, support_df, avail_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, default=DEFAULT_SITE_DIR)
    parser.add_argument("--site-json", type=Path, default=DEFAULT_SITE_JSON)
    parser.add_argument("--site-prefix", default=DEFAULT_SITE_PREFIX)
    parser.add_argument("--var", default=DEFAULT_VAR)
    parser.add_argument("--eof-file", type=Path, default=DEFAULT_EOF_FILE)
    parser.add_argument(
        "--levels",
        default=None,
        help="Comma-separated EOF pressure levels in hPa; overrides --eof-file.",
    )
    parser.add_argument("--min-p", type=float, default=DEFAULT_MIN_P_HPA)
    parser.add_argument("--min-valid-lev", type=int, default=DEFAULT_MIN_VALID_LEV)
    parser.add_argument(
        "--thresholds",
        default=",".join(str(x) for x in DEFAULT_THRESHOLDS_HPA),
        help="Comma-separated hPa thresholds for cumulative support diagnostics.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    levels_hpa = load_model_levels(args.eof_file, args.levels)
    levels_hpa = levels_hpa[levels_hpa >= float(args.min_p)]
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    used_sites, support_df, avail_df, summary_df = build_tables(
        site_dir=args.site_dir,
        site_json=args.site_json,
        site_prefix=args.site_prefix,
        varname=args.var,
        eof_levels_hpa=levels_hpa,
        min_p_hpa=args.min_p,
        min_valid_lev=args.min_valid_lev,
        thresholds_hpa=thresholds,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    support_path = args.out_dir / "site_pressure_support.csv"
    avail_path = args.out_dir / "site_pressure_availability.csv"
    summary_path = args.out_dir / "site_pressure_summary.csv"

    support_df.to_csv(support_path, index=False, float_format="%.4f")
    avail_df.to_csv(avail_path, index=False, float_format="%.4f")
    summary_df.to_csv(summary_path, index=False, float_format="%.3f")

    print("\nEqual-site raw support by EOF level")
    display_cols = [
        "pressure_hpa",
        "ALL_sample_weighted_pct",
        "ALL_site_balanced_pct",
        "ALL_site_balanced_cum_aloft_pct",
        "n_sites_supported",
        "N_eff_sites",
    ]
    print(
        support_df[display_cols].to_string(
            index=False, float_format=lambda x: f"{x:7.2f}"
        )
    )

    print(f"\n[INFO] usable sites: {len(used_sites)}")
    print("[WROTE]")
    print(f"  {support_path}")
    print(f"  {avail_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
