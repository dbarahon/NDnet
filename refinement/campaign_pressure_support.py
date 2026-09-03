#!/usr/bin/env python3
"""
Summarize vertical pressure support in field-campaign training files.

The campaign refinement loss samples the reconstructed Nd profile at each
segment pressure (p_merra) using linear interpolation in log-pressure. This
utility reports support on the model/EOF pressure levels in two ways:

1. nearest_pct: each valid segment is assigned to its nearest model level
   in log-pressure;
2. interp_pct: each valid segment is split between the two bracketing model
   levels using the same log-pressure linear interpolation weights as the
   campaign loss. This is the most relevant measure of which model levels
   are directly constrained by campaign observations.

Outputs
-------
campaign_pressure_support.csv
    One row per model pressure level. Campaign columns contain interpolation-
    weighted percentages that sum to 100% for each campaign. Also includes
    pooled segment-weighted and campaign-balanced support.

campaign_pressure_support_nearest.csv
    Same layout using nearest-level assignment.

campaign_pressure_summary.csv
    Per-campaign pressure quantiles and percentages of observations aloft of
    selected pressure thresholds.

The script also prints a compact support table and two heuristic taper ranges
based on pressure quantiles. These are diagnostics, not automatic choices.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_TRAIN_DIR = Path(
    "/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/field_campaigns/training"
)
DEFAULT_EOF_FILE = Path(
    "/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/in_cloudM2/best/Nd_EOF.nc"
)
DEFAULT_PATTERN = "*_segments_1deg_merraP_training.nc"
DEFAULT_THRESHOLDS_HPA = (700.0, 600.0, 500.0, 400.0, 300.0)


def campaign_name_from_path(path: Path) -> str:
    stem = path.stem
    for suffix in (
        "_segments_1deg_merraP_training",
        "_segments_1deg_merraP",
        "_training",
    ):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


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

    levels = _as_hpa(levels, "model levels")
    levels = levels[np.isfinite(levels) & (levels > 0)]
    levels = np.unique(levels)
    if levels.size < 2:
        raise ValueError("Need at least two valid model pressure levels")
    return np.sort(levels)[::-1]


def read_valid_pressures(path: Path) -> np.ndarray:
    """Read pressures for the observations that actually enter campaign training."""
    with xr.open_dataset(path, decode_times=False) as ds:
        if "p_merra" not in ds:
            raise KeyError(f"{path} missing required variable 'p_merra'")
        p = np.asarray(ds["p_merra"].values, dtype=float).reshape(-1)

        valid = np.isfinite(p) & (p > 0)
        if "valid_obs" in ds:
            valid &= np.asarray(ds["valid_obs"].values).reshape(-1).astype(bool)
        if "Nd_mean" in ds:
            valid &= np.isfinite(np.asarray(ds["Nd_mean"].values, dtype=float).reshape(-1))

    p = _as_hpa(p[valid], f"{path.name}: p_merra")
    return p[np.isfinite(p) & (p > 0)]


def nearest_support(pressures: np.ndarray, levels_hpa: np.ndarray) -> np.ndarray:
    """Counts from nearest model level in log-pressure."""
    p = np.asarray(pressures, dtype=float)
    lev = np.asarray(levels_hpa, dtype=float)
    out = np.zeros(lev.size, dtype=float)
    if p.size == 0:
        return out

    in_range = (p > np.min(lev)) & (p <= np.max(lev))
    p = p[in_range]
    if p.size == 0:
        return out

    distance = np.abs(np.log(p)[:, None] - np.log(lev)[None, :])
    idx = np.argmin(distance, axis=1)
    np.add.at(out, idx, 1.0)
    return out


def interpolation_support(pressures: np.ndarray, levels_hpa: np.ndarray) -> np.ndarray:
    """
    Effective level support using the exact two-level linear interpolation
    weights in log-pressure.

    Each in-range segment contributes total weight 1.0. Out-of-range pressures
    contribute zero, matching the campaign interpolation behavior.
    """
    p = np.asarray(pressures, dtype=float)
    lev_out = np.asarray(levels_hpa, dtype=float)
    support_out = np.zeros(lev_out.size, dtype=float)
    if p.size == 0:
        return support_out

    order = np.argsort(np.log(lev_out))
    lev = lev_out[order]
    xlev = np.log(lev)
    xp = np.log(p)

    # Match SparseSiteCampaignNLLRefineLoss._interp_logp_profiles.
    in_range = (xp > xlev[0]) & (xp <= xlev[-1])
    xp = xp[in_range]
    if xp.size == 0:
        return support_out

    j = np.searchsorted(xlev, xp, side="right")
    j = np.clip(j, 1, xlev.size - 1)
    i = j - 1

    dx = xlev[j] - xlev[i]
    wj = np.divide(xp - xlev[i], dx, out=np.zeros_like(xp), where=dx > 0)
    wi = 1.0 - wj

    support_sorted = np.zeros(xlev.size, dtype=float)
    np.add.at(support_sorted, i, wi)
    np.add.at(support_sorted, j, wj)
    support_out[order] = support_sorted
    return support_out


def weighted_quantile(values: np.ndarray, quantiles: Iterable[float], weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    q = np.asarray(list(quantiles), dtype=float)
    weights = np.asarray(weights, dtype=float)

    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[good]
    weights = weights[good]
    if values.size == 0:
        return np.full(q.shape, np.nan, dtype=float)

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) - 0.5 * weights
    cdf /= np.sum(weights)
    return np.interp(q, cdf, values, left=values[0], right=values[-1])


def percent(support: np.ndarray) -> np.ndarray:
    total = float(np.sum(support))
    if total <= 0:
        return np.full_like(support, np.nan, dtype=float)
    return 100.0 * support / total


def build_tables(
    train_dir: Path,
    pattern: str,
    levels_hpa: np.ndarray,
    thresholds_hpa: Iterable[float],
):
    files = sorted(train_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {train_dir}")

    campaigns: dict[str, np.ndarray] = {}
    for path in files:
        name = campaign_name_from_path(path)
        p = read_valid_pressures(path)
        campaigns[name] = p
        print(f"[READ] {name:12s}: {p.size:6d} valid segments")

    interp = {name: interpolation_support(p, levels_hpa) for name, p in campaigns.items()}
    nearest = {name: nearest_support(p, levels_hpa) for name, p in campaigns.items()}

    def support_frame(raw: dict[str, np.ndarray]) -> pd.DataFrame:
        df = pd.DataFrame({"pressure_hpa": levels_hpa})
        pct_cols = []
        for name, support in raw.items():
            col = f"{name}_pct"
            df[col] = percent(support)
            pct_cols.append(col)

        pooled = np.sum(np.stack(list(raw.values()), axis=0), axis=0)
        df["ALL_segment_weighted_pct"] = percent(pooled)
        df["ALL_campaign_balanced_pct"] = df[pct_cols].mean(axis=1, skipna=True)
        df["n_campaigns_supported"] = np.sum(
            np.stack([np.asarray(raw[name]) > 0 for name in raw], axis=0), axis=0
        )
        return df

    interp_df = support_frame(interp)
    nearest_df = support_frame(nearest)

    rows = []
    for name, p in campaigns.items():
        row = {
            "campaign": name,
            "n_valid": int(p.size),
            "n_in_model_range": int(np.sum((p > np.min(levels_hpa)) & (p <= np.max(levels_hpa)))) if p.size else 0,
            "p_min_hpa": np.nanmin(p) if p.size else np.nan,
            "p01_hpa": np.nanquantile(p, 0.01) if p.size else np.nan,
            "p05_hpa": np.nanquantile(p, 0.05) if p.size else np.nan,
            "p10_hpa": np.nanquantile(p, 0.10) if p.size else np.nan,
            "p20_hpa": np.nanquantile(p, 0.20) if p.size else np.nan,
            "p50_hpa": np.nanquantile(p, 0.50) if p.size else np.nan,
            "p80_hpa": np.nanquantile(p, 0.80) if p.size else np.nan,
            "p95_hpa": np.nanquantile(p, 0.95) if p.size else np.nan,
            "p_max_hpa": np.nanmax(p) if p.size else np.nan,
        }
        for threshold in thresholds_hpa:
            row[f"pct_p_le_{threshold:g}hPa"] = (
                100.0 * np.mean(p <= threshold) if p.size else np.nan
            )
        rows.append(row)
    summary_df = pd.DataFrame(rows)

    return campaigns, interp_df, nearest_df, summary_df


def print_taper_diagnostics(campaigns: dict[str, np.ndarray]) -> None:
    nonempty = {k: v for k, v in campaigns.items() if v.size > 0}
    if not nonempty:
        return

    pooled_p = np.concatenate(list(nonempty.values()))
    pooled_w = np.ones_like(pooled_p, dtype=float)

    # Give each campaign the same total weight so a long campaign cannot
    # dominate the vertical support diagnostic merely because it has more segments.
    balanced_p = []
    balanced_w = []
    for p in nonempty.values():
        balanced_p.append(p)
        balanced_w.append(np.full(p.size, 1.0 / p.size, dtype=float))
    balanced_p = np.concatenate(balanced_p)
    balanced_w = np.concatenate(balanced_w)

    q = [0.01, 0.05, 0.10, 0.20]
    qp = weighted_quantile(pooled_p, q, pooled_w)
    qb = weighted_quantile(balanced_p, q, balanced_w)

    print("\nUpper-tail pressure quantiles (lower pressure = higher altitude)")
    print("            p01     p05     p10     p20  [hPa]")
    print("pooled    " + " ".join(f"{x:7.1f}" for x in qp))
    print("balanced  " + " ".join(f"{x:7.1f}" for x in qb))

    print(
        "\nHeuristic data-support taper (campaign-balanced): "
        f"start ~{qb[3]:.0f} hPa, top ~{qb[1]:.0f} hPa "
        "(p20 -> p05)."
    )
    print(
        "Use this as a sensitivity-test anchor, not as a fitted optimum; "
        "the final choice should also preserve the physically defensible prior aloft."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--eof-file", type=Path, default=DEFAULT_EOF_FILE)
    parser.add_argument(
        "--levels",
        default=None,
        help="Comma-separated pressure levels in hPa. Overrides --eof-file.",
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(str(x) for x in DEFAULT_THRESHOLDS_HPA),
        help="Comma-separated hPa thresholds for %% of observations aloft (p <= threshold).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    levels_hpa = load_model_levels(args.eof_file, args.levels)
    thresholds_hpa = [float(x) for x in args.thresholds.split(",") if x.strip()]

    campaigns, interp_df, nearest_df, summary_df = build_tables(
        train_dir=args.train_dir,
        pattern=args.pattern,
        levels_hpa=levels_hpa,
        thresholds_hpa=thresholds_hpa,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    interp_path = args.out_dir / "campaign_pressure_support.csv"
    nearest_path = args.out_dir / "campaign_pressure_support_nearest.csv"
    summary_path = args.out_dir / "campaign_pressure_summary.csv"

    interp_df.to_csv(interp_path, index=False, float_format="%.4f")
    nearest_df.to_csv(nearest_path, index=False, float_format="%.4f")
    summary_df.to_csv(summary_path, index=False, float_format="%.3f")

    print("\nInterpolation-weighted support by model level (%)")
    display_cols = [
        "pressure_hpa",
        "ALL_segment_weighted_pct",
        "ALL_campaign_balanced_pct",
        "n_campaigns_supported",
    ]
    print(interp_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:7.2f}"))

    print_taper_diagnostics(campaigns)

    print("\n[WROTE]")
    print(f"  {interp_path}")
    print(f"  {nearest_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
