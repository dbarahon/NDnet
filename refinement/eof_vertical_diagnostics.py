#!/usr/bin/env python3
"""
Diagnose whether EOF vertical structure, mode-energy normalization, or low-Nd
clipping can contribute to pressure-level banding in NDnet refinement.

The utility is intentionally read-only.  It mirrors the definitions used by
refinement/config.py and NN_construct_v2.py:

    mode_energy[m] = sum_z EOF[m,z]^2
    mode_weight[m] = mode_energy[m] / mean_m(mode_energy)
    sigma_mode_var_rel[m] = max(mode_weight[m], sigma_mode_var_floor)

These quantities are global per EOF mode; they do NOT vary with pressure.
Pressure-dependent diagnostics are computed separately from the EOF column
vectors EOF[:,z].

Optional --global-test input adds diagnostics for an existing
*_global_test.nc file, including the fraction of values near zero / below
chosen thresholds and vertical derivatives of the global-mean profile.

Outputs
-------
eof_mode_summary.csv
    Per-mode energy, normalized weight used by the refinement loss, inverse
    sigma-penalty scale, extrema, and log-pressure roughness metrics.

eof_pressure_summary.csv
    Per-pressure Nd_mean, local EOF energy, effective number of modes,
    dominant-mode fraction, adjacent-level basis similarity/jump, and
    derivative/curvature energy.

eof_pressure_modes.csv
    Nd_mean and every EOF value / local fractional energy by pressure.

eof_vertical_structure.png
    Compact diagnostic figure for Nd_mean, local EOF scale, EOF profiles,
    and fractional local energy by mode.

global_test_vertical_summary.csv              [only with --global-test]
global_test_floor_diagnostics.png             [only with --global-test]
    Low-value fractions and vertical-profile comparison for Base vs NN.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_EOF_FILE = Path(
    "/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/in_cloudM2/best/Nd_EOF.nc"
)
DEFAULT_MODE_VAR_FLOOR = 0.05
DEFAULT_THRESHOLDS = (0.1, 1.0, 10.0)


def _as_hpa(values: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    finite = out[np.isfinite(out) & (out > 0)]
    if finite.size and float(np.nanmedian(finite)) > 2000.0:
        print(f"[INFO] {name}: assuming Pa; converting to hPa")
        out = out / 100.0
    return out


def _safe_gradient(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """np.gradient with finite-value protection for small diagnostic arrays."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.full_like(y, np.nan, dtype=float)
    return np.gradient(y, x, axis=axis, edge_order=1)


def load_eof(eof_file: Path):
    with xr.open_dataset(eof_file) as ds:
        if "EOFs" not in ds:
            raise KeyError(f"{eof_file} missing 'EOFs'")
        if "Nd_mean" not in ds:
            raise KeyError(f"{eof_file} missing 'Nd_mean'")
        eof_da = ds["EOFs"].load()
        mean_da = ds["Nd_mean"].load()

    if set(("mode", "lev")) - set(eof_da.dims):
        raise ValueError(f"EOFs must contain dims ('mode','lev'); got {eof_da.dims}")
    eof_da = eof_da.transpose("mode", "lev")
    mean_da = mean_da.transpose("lev")

    lev = _as_hpa(eof_da["lev"].values, "EOF lev")
    eof_da = eof_da.assign_coords(lev=("lev", lev))
    mean_da = mean_da.assign_coords(lev=("lev", lev))
    return eof_da, mean_da


def eof_diagnostics(eof_da: xr.DataArray, mean_da: xr.DataArray, mode_var_floor: float):
    eof = np.asarray(eof_da.values, dtype=float)
    mean = np.asarray(mean_da.values, dtype=float)
    lev = np.asarray(eof_da["lev"].values, dtype=float)
    mode_coord = np.asarray(eof_da["mode"].values)

    if np.any(~np.isfinite(lev)) or np.any(lev <= 0):
        raise ValueError("EOF pressure levels must be finite and positive")
    if eof.shape[1] != lev.size or mean.size != lev.size:
        raise ValueError("EOF / Nd_mean pressure dimensions do not agree")

    mode_energy = np.nansum(eof * eof, axis=1)
    mode_energy_mean = float(np.nanmean(mode_energy))
    if not np.isfinite(mode_energy_mean) or mode_energy_mean <= 0:
        raise ValueError("Invalid EOF mode energies")
    mode_weight = mode_energy / mode_energy_mean
    mode_var_rel = np.maximum(mode_weight, float(mode_var_floor))
    inverse_sigma_penalty_scale = 1.0 / mode_var_rel

    order = np.argsort(np.log(lev))
    inv_order = np.argsort(order)
    logp_s = np.log(lev[order])
    eof_s = eof[:, order]
    d1_s = _safe_gradient(eof_s, logp_s, axis=1)
    d2_s = _safe_gradient(d1_s, logp_s, axis=1)
    d1 = d1_s[:, inv_order]
    d2 = d2_s[:, inv_order]

    mode_rows = []
    for m in range(eof.shape[0]):
        ab = np.abs(eof[m])
        imax = int(np.nanargmax(ab)) if np.isfinite(ab).any() else 0
        mode_rows.append(
            {
                "mode": mode_coord[m],
                "mode_energy_sum_eof2": mode_energy[m],
                "mode_weight_rel": mode_weight[m],
                "sigma_mode_var_rel_after_floor": mode_var_rel[m],
                "inverse_sigma_penalty_scale": inverse_sigma_penalty_scale[m],
                "hit_mode_var_floor": int(mode_weight[m] < float(mode_var_floor)),
                "max_abs_eof": ab[imax],
                "pressure_of_max_abs_eof_hpa": lev[imax],
                "rms_dEOF_dlnp": float(np.sqrt(np.nanmean(d1[m] ** 2))),
                "rms_d2EOF_dlnp2": float(np.sqrt(np.nanmean(d2[m] ** 2))),
            }
        )
    mode_df = pd.DataFrame(mode_rows)

    local_energy = np.nansum(eof * eof, axis=0)
    local_scale = np.sqrt(np.maximum(local_energy, 0.0))
    frac = np.divide(
        eof * eof,
        local_energy[None, :],
        out=np.zeros_like(eof, dtype=float),
        where=local_energy[None, :] > 0,
    )
    frac2sum = np.nansum(frac * frac, axis=0)
    neff = np.divide(
        1.0,
        frac2sum,
        out=np.zeros(lev.size, dtype=float),
        where=frac2sum > 0,
    )
    dominant_idx = np.nanargmax(frac, axis=0)
    dominant_frac = frac[dominant_idx, np.arange(lev.size)]
    top3_frac = np.sort(frac, axis=0)[-min(3, eof.shape[0]):, :].sum(axis=0)

    d1_energy = np.sqrt(np.nansum(d1 * d1, axis=0))
    d2_energy = np.sqrt(np.nansum(d2 * d2, axis=0))

    desc = np.argsort(lev)[::-1]
    e_desc = eof[:, desc]
    p_desc = lev[desc]
    cos_desc = np.full(p_desc.size, np.nan)
    jump_desc = np.full(p_desc.size, np.nan)
    jump_per_dlnp_desc = np.full(p_desc.size, np.nan)
    for j in range(p_desc.size - 1):
        a = e_desc[:, j]
        b = e_desc[:, j + 1]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na > 0 and nb > 0:
            cos_desc[j] = float(np.dot(a, b) / (na * nb))
            denom = 0.5 * (na + nb)
            jump_desc[j] = float(np.linalg.norm(b - a) / denom) if denom > 0 else np.nan
            dlnp = abs(np.log(p_desc[j + 1]) - np.log(p_desc[j]))
            jump_per_dlnp_desc[j] = jump_desc[j] / dlnp if dlnp > 0 else np.nan

    cos_next = np.full(lev.size, np.nan)
    jump_next = np.full(lev.size, np.nan)
    jump_dlnp_next = np.full(lev.size, np.nan)
    cos_next[desc] = cos_desc
    jump_next[desc] = jump_desc
    jump_dlnp_next[desc] = jump_per_dlnp_desc

    mean_abs_floor = np.maximum(np.abs(mean), 1e-6)
    local_scale_to_mean = local_scale / mean_abs_floor

    pressure_df = pd.DataFrame(
        {
            "pressure_hpa": lev,
            "Nd_mean": mean,
            "local_eof_energy_sum_modes_eof2": local_energy,
            "local_eof_scale_sqrt_energy": local_scale,
            "local_eof_scale_over_abs_Nd_mean": local_scale_to_mean,
            "effective_number_modes_local": neff,
            "dominant_mode": mode_coord[dominant_idx],
            "dominant_mode_fraction_local_energy": dominant_frac,
            "top3_mode_fraction_local_energy": top3_frac,
            "basis_dEOF_dlnp_energy": d1_energy,
            "basis_d2EOF_dlnp2_energy": d2_energy,
            "basis_cosine_to_next_lower_pressure_level": cos_next,
            "basis_jump_to_next_lower_pressure_level": jump_next,
            "basis_jump_per_dlnp_to_next_lower_pressure_level": jump_dlnp_next,
        }
    ).sort_values("pressure_hpa", ascending=False)

    modes_out = {"pressure_hpa": lev, "Nd_mean": mean}
    for m in range(eof.shape[0]):
        label = str(mode_coord[m])
        modes_out[f"EOF_mode_{label}"] = eof[m]
        modes_out[f"frac_energy_mode_{label}"] = frac[m]
        modes_out[f"dEOF_dlnp_mode_{label}"] = d1[m]
        modes_out[f"d2EOF_dlnp2_mode_{label}"] = d2[m]
    modes_df = pd.DataFrame(modes_out).sort_values("pressure_hpa", ascending=False)

    return mode_df, pressure_df, modes_df, frac, d1, d2


def _weighted_global_mean(da: xr.DataArray) -> xr.DataArray:
    dims = [d for d in ("time", "lat", "lon") if d in da.dims]
    if "lat" in da.dims:
        w = np.cos(np.deg2rad(da["lat"]))
        return da.weighted(w).mean(dims, skipna=True)
    return da.mean(dims, skipna=True)


def _weighted_fraction(da: xr.DataArray, condition: xr.DataArray) -> xr.DataArray:
    valid = np.isfinite(da)
    x = xr.where(valid, condition.astype(float), np.nan)
    return _weighted_global_mean(x)


def global_test_diagnostics(path: Path, thresholds: list[float]) -> pd.DataFrame:
    with xr.open_dataset(path) as ds:
        if "Nd_base" not in ds or "Nd_nn" not in ds:
            raise KeyError(f"{path} must contain Nd_base and Nd_nn")
        base = ds["Nd_base"].load()
        nn = ds["Nd_nn"].load()
        raw = ds["Nd_nn_raw"].load() if "Nd_nn_raw" in ds else None

    lev = _as_hpa(base["lev"].values, f"{path.name}: lev")
    base = base.assign_coords(lev=("lev", lev))
    nn = nn.assign_coords(lev=("lev", lev))
    if raw is not None:
        raw = raw.assign_coords(lev=("lev", lev))

    gbase = _weighted_global_mean(base)
    gnn = _weighted_global_mean(nn)
    gdelta = _weighted_global_mean(nn - base)

    out = {
        "pressure_hpa": lev,
        "global_mean_base": np.asarray(gbase.values, dtype=float),
        "global_mean_nn": np.asarray(gnn.values, dtype=float),
        "global_mean_nn_minus_base": np.asarray(gdelta.values, dtype=float),
    }
    if raw is not None:
        out["global_mean_nn_raw"] = np.asarray(_weighted_global_mean(raw).values, dtype=float)

    for th in thresholds:
        tag = str(th).replace(".", "p")
        out[f"base_frac_le_{tag}"] = np.asarray(_weighted_fraction(base, base <= th).values, dtype=float)
        out[f"nn_frac_le_{tag}"] = np.asarray(_weighted_fraction(nn, nn <= th).values, dtype=float)
        if raw is not None:
            out[f"nn_raw_frac_le_{tag}"] = np.asarray(_weighted_fraction(raw, raw <= th).values, dtype=float)

    order = np.argsort(np.log(lev))
    inv = np.argsort(order)
    logp = np.log(lev[order])
    for key in [k for k in out if k.startswith("global_mean_")]:
        vals = np.asarray(out[key], dtype=float)
        der = _safe_gradient(vals[order], logp)[inv]
        out[f"d_dlnp__{key}"] = der

    return pd.DataFrame(out).sort_values("pressure_hpa", ascending=False)


def make_eof_plot(
    eof_da: xr.DataArray,
    mean_da: xr.DataArray,
    pressure_df: pd.DataFrame,
    frac: np.ndarray,
    out_path: Path,
):
    import matplotlib.pyplot as plt

    eof = np.asarray(eof_da.values, dtype=float)
    lev = np.asarray(eof_da["lev"].values, dtype=float)
    mean = np.asarray(mean_da.values, dtype=float)

    fig, axes = plt.subplots(1, 4, figsize=(14, 6), sharey=False)

    axes[0].plot(mean, lev, label="Nd_mean")
    pidx = pressure_df.set_index("pressure_hpa")
    axes[0].plot(
        pidx.loc[lev, "local_eof_scale_sqrt_energy"].values,
        lev,
        label="sqrt(sum EOF^2)",
    )
    axes[0].set_xlabel("Nd / EOF scale")
    axes[0].set_ylabel("Pressure (hPa)")
    axes[0].legend(fontsize=8)

    for m in range(eof.shape[0]):
        axes[1].plot(eof[m], lev, linewidth=1)
    axes[1].axvline(0.0, linewidth=0.8)
    axes[1].set_xlabel("EOF value")
    axes[1].set_ylabel("Pressure (hPa)")

    im = axes[2].imshow(
        frac,
        aspect="auto",
        origin="lower",
        extent=[float(np.nanmin(lev)), float(np.nanmax(lev)), -0.5, eof.shape[0] - 0.5],
    )
    axes[2].set_xlabel("Pressure (hPa)")
    axes[2].set_ylabel("Mode")
    axes[2].set_title("Local fractional EOF energy", fontsize=9)
    fig.colorbar(im, ax=axes[2], orientation="horizontal", pad=0.12, fraction=0.05)

    p = pressure_df["pressure_hpa"].values
    axes[3].plot(pressure_df["basis_jump_per_dlnp_to_next_lower_pressure_level"], p, label="basis jump / dlnp")
    axes[3].plot(pressure_df["basis_d2EOF_dlnp2_energy"], p, label="curvature energy")
    axes[3].set_xlabel("Vertical roughness")
    axes[3].set_ylabel("Pressure (hPa)")
    axes[3].legend(fontsize=8)

    for ax in (axes[0], axes[1], axes[3]):
        ax.set_ylim(float(np.nanmax(lev)), float(np.nanmin(lev)))
        ax.grid(True, alpha=0.25)
    axes[2].set_xlim(float(np.nanmax(lev)), float(np.nanmin(lev)))

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_global_plot(df: pd.DataFrame, thresholds: list[float], out_path: Path):
    import matplotlib.pyplot as plt

    p = df["pressure_hpa"].values
    fig, axes = plt.subplots(1, 3, figsize=(11, 6), sharey=True)

    axes[0].plot(df["global_mean_base"], p, label="Base")
    axes[0].plot(df["global_mean_nn"], p, label="NN")
    if "global_mean_nn_raw" in df:
        axes[0].plot(df["global_mean_nn_raw"], p, label="NN raw")
    axes[0].set_xlabel("Global mean Nd")
    axes[0].set_ylabel("Pressure (hPa)")
    axes[0].legend(fontsize=8)

    for th in thresholds:
        tag = str(th).replace(".", "p")
        axes[1].plot(100.0 * df[f"base_frac_le_{tag}"], p, linestyle="--", label=f"Base <= {th:g}")
        axes[1].plot(100.0 * df[f"nn_frac_le_{tag}"], p, label=f"NN <= {th:g}")
    axes[1].set_xlabel("Area-time fraction (%)")
    axes[1].legend(fontsize=7)

    axes[2].plot(df["d_dlnp__global_mean_nn"], p, label="d NN / dlnp")
    axes[2].plot(df["d_dlnp__global_mean_nn_minus_base"], p, label="d (NN-Base) / dlnp")
    axes[2].axvline(0.0, linewidth=0.8)
    axes[2].set_xlabel("Vertical derivative")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.set_ylim(float(np.nanmax(p)), float(np.nanmin(p)))
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eof-file", type=Path, default=DEFAULT_EOF_FILE)
    ap.add_argument("--mode-var-floor", type=float, default=DEFAULT_MODE_VAR_FLOOR)
    ap.add_argument(
        "--global-test",
        type=Path,
        default=None,
        help="Optional *_global_test.nc with Nd_base and Nd_nn.",
    )
    ap.add_argument(
        "--thresholds",
        default=",".join(str(x) for x in DEFAULT_THRESHOLDS),
        help="Nd thresholds for optional low-value fractions, e.g. 0.1,1,10.",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("eof_vertical_diagnostics"))
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    eof_da, mean_da = load_eof(args.eof_file)
    mode_df, pressure_df, modes_df, frac, _, _ = eof_diagnostics(
        eof_da, mean_da, args.mode_var_floor
    )

    mode_path = args.out_dir / "eof_mode_summary.csv"
    pressure_path = args.out_dir / "eof_pressure_summary.csv"
    modes_path = args.out_dir / "eof_pressure_modes.csv"
    plot_path = args.out_dir / "eof_vertical_structure.png"

    mode_df.to_csv(mode_path, index=False, float_format="%.8g")
    pressure_df.to_csv(pressure_path, index=False, float_format="%.8g")
    modes_df.to_csv(modes_path, index=False, float_format="%.8g")
    make_eof_plot(eof_da, mean_da, pressure_df, frac, plot_path)

    print("\n[MODE SUMMARY]")
    print(
        mode_df[
            [
                "mode",
                "mode_energy_sum_eof2",
                "mode_weight_rel",
                "sigma_mode_var_rel_after_floor",
                "inverse_sigma_penalty_scale",
                "hit_mode_var_floor",
            ]
        ].to_string(index=False)
    )

    print("\n[PRESSURE SUMMARY: 300-750 hPa]")
    mid = pressure_df[
        (pressure_df["pressure_hpa"] >= 300.0)
        & (pressure_df["pressure_hpa"] <= 750.0)
    ]
    print(
        mid[
            [
                "pressure_hpa",
                "Nd_mean",
                "local_eof_scale_sqrt_energy",
                "local_eof_scale_over_abs_Nd_mean",
                "effective_number_modes_local",
                "dominant_mode",
                "dominant_mode_fraction_local_energy",
                "basis_cosine_to_next_lower_pressure_level",
                "basis_jump_per_dlnp_to_next_lower_pressure_level",
            ]
        ].to_string(index=False)
    )

    print("\n[WROTE]")
    for p in (mode_path, pressure_path, modes_path, plot_path):
        print(f"  {p}")

    if args.global_test is not None:
        gdf = global_test_diagnostics(args.global_test, thresholds)
        gpath = args.out_dir / "global_test_vertical_summary.csv"
        gplot = args.out_dir / "global_test_floor_diagnostics.png"
        gdf.to_csv(gpath, index=False, float_format="%.8g")
        make_global_plot(gdf, thresholds, gplot)
        print(f"  {gpath}")
        print(f"  {gplot}")

        print("\n[GLOBAL TEST: 300-750 hPa]")
        gmid = gdf[(gdf["pressure_hpa"] >= 300.0) & (gdf["pressure_hpa"] <= 750.0)]
        cols = ["pressure_hpa", "global_mean_base", "global_mean_nn", "global_mean_nn_minus_base"]
        for th in thresholds:
            tag = str(th).replace(".", "p")
            cols.extend([f"base_frac_le_{tag}", f"nn_frac_le_{tag}"])
        print(gmid[cols].to_string(index=False))


if __name__ == "__main__":
    main()
