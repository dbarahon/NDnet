#!/usr/bin/env python3
"""4x3 comparison of NDnet-prior COSP inputs and refinement MODIS retrievals.

Rows: liquid optical thickness, liquid particle size, liquid retrieval fraction,
cloud-top pressure. Columns: GiOcean/COSP, MODIS retrieval, retrieval-COSP.
The default 2000-2019 window matches the prior training selection. Only common
dates and colocated finite values are used. COSP is bilinearly regridded to the
refinement/Nd grid. Values are plotted in common physical (unstandardized)
units. No standardization information is added to the figure; instead, a CSV
and stdout table quantify the COSP-to-MODIS domain shift.

Examples
--------
python MODIS_input_comparison.py
python MODIS_input_comparison.py --difference percentage
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
DATA_ROOT = Path("/panfs/ccds02/nobackup/people/dbarahon/NDnet/data")
ROOT = DATA_ROOT / "dailies"
STATS_DIR = DATA_ROOT / "stats"
PRIOR_DIR = ROOT / "mod_inst_6hr_glo_L720x361_sfc"
RETR_DIR = ROOT / "MCD06COSP_D3_MODIS" / "on_Noutgrid"
RETR_GLOB = "MCD06COSP_D3_MODIS.*.nc"
WEIGHTS = Path("cache/weights_cosp_to_refinement_grid.nc")

START_YEAR = 2000
END_YEAR = 2019
DIFF_MODE = "absolute"  # absolute | percentage
OUT_BASENAME = "MODIS_input_comparison"
DPI = 400

# key: prior variable, retrieval variable, plot label, long label, common units
VARS = {
    "tau": (
        "MDSOPTHCKWTR", "Cloud_Optical_Thickness_Liquid_Mean",
        r"$\tau_{\rm liq}$", "Liquid optical thickness", "1",
    ),
    "reff": (
        "MDSCLDSZWTR", "Cloud_Particle_Size_Liquid_Mean",
        r"$r_{e,{\rm liq}}$", "Liquid particle size", r"$\mu$m",
    ),
    "frac": (
        "MDSCLDFRCWTR", "Cloud_Retrieval_Fraction_Liquid_Mean",
        r"$f_{\rm liq}$", "Liquid retrieval fraction", "1",
    ),
    "ctp": (
        "SWCLDPRS", "Cloud_Top_Pressure_Mean",
        r"$p_{\rm c}$", "Cloud-top pressure", "hPa",
    ),
}

# MODIS normalization actually used by NDinfer_base/refinement/production inference.
MODIS_NORM = {
    "tau": (13.945, 9.665),
    "reff": (12.08, 3.126),
    "frac": (0.268, 0.205),
    "ctp": (551.241, 177.277),
}

# Common absolute scale for columns 1-2. None = robust 2-98 percentile.
PLOT_LIMITS = {
    "tau": (0.0, None),
    "reff": (None, None),
    "frac": (0.0, 1.0),
    "ctp": (None, None),
}
# Symmetric third-column limits. None = robust 98th percentile of |difference|.
ABS_DIFF_LIMITS = {k: None for k in VARS}
PCT_DIFF_LIMITS = {"tau": 100.0, "reff": 50.0, "frac": 100.0, "ctp": 50.0}
# Percentage is 100*(retrieval-COSP)/COSP; mask tiny denominators.
PCT_FLOOR = {"tau": 0.10, "reff": 0.50, "frac": 0.01, "ctp": 10.0}

ABS_CMAP = "YlGnBu"
DIFF_CMAP = "RdBu_r"
MAP_PROJ = ccrs.Robinson(central_longitude=0)
DATA_CRS = ccrs.PlateCarree()

mpl.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def files(directory: Path, pattern="*.nc*") -> list[str]:
    out = sorted(glob.glob(str(directory / pattern)))
    if not out:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    return out


def open_only(paths: list[str], names: list[str]) -> xr.Dataset:
    def keep(ds):
        missing = [v for v in names if v not in ds]
        if missing:
            raise KeyError(f"Missing {missing} in {ds.encoding.get('source', 'dataset')}")
        return ds[names]

    return xr.open_mfdataset(
        paths, combine="by_coords", parallel=True, preprocess=keep,
        chunks={"time": 31}, data_vars="minimal", coords="minimal",
        compat="override",
    )


def prep(ds: xr.Dataset, y0: int, y1: int) -> xr.Dataset:
    """Daily timestamps, increasing lat, unique [0,360) lon, requested years."""
    t = pd.to_datetime(ds.time.values).normalize().values.astype("datetime64[ns]")
    ds = ds.assign_coords(time=t).sortby("time")
    if np.unique(t).size != t.size:
        ds = ds.groupby("time").mean("time", skipna=True)

    lon = np.mod(np.asarray(ds.lon.values, float), 360.0)
    ds = ds.assign_coords(lon=lon).sortby("lon").sortby("lat")
    _, idx = np.unique(np.round(ds.lon.values, 8), return_index=True)
    if idx.size != ds.sizes["lon"]:
        ds = ds.isel(lon=np.sort(idx))

    return ds.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))


def _unit_factor(key: str, units: str, representative: float) -> float:
    """Multiplicative factor to the common units used by refinement/MODIS."""
    u = str(units or "").lower().replace(" ", "")
    rep = abs(float(representative)) if np.isfinite(representative) else np.nan

    if key == "reff":
        if u in {"m", "meter", "meters", "metre", "metres"}:
            return 1.0e6
        if u in {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}:
            return 1.0e3
        if u in {"um", "micron", "microns", "micrometer", "micrometers", "micrometre", "micrometres", "µm", "μm"}:
            return 1.0
        if np.isfinite(rep) and rep < 1.0e-3:
            return 1.0e6

    if key == "ctp":
        if u in {"pa", "pascal", "pascals"}:
            return 1.0e-2
        if np.isfinite(rep) and rep > 2000.0:
            return 1.0e-2

    if key == "frac":
        if "%" in u or "percent" in u:
            return 1.0e-2
        if np.isfinite(rep) and rep > 1.5:
            return 1.0e-2

    return 1.0


def convert(key: str, da: xr.DataArray) -> xr.DataArray:
    """Convert fields to common physical units before plotting/differencing."""
    out = da.astype("float32")
    sample = np.asarray(out.isel(time=0).values, float)
    finite = sample[np.isfinite(sample)]
    representative = float(np.nanmedian(finite)) if finite.size else np.nan
    factor = _unit_factor(key, da.attrs.get("units", ""), representative)
    return out * factor


def canonical(ds: xr.Dataset, source: int) -> xr.Dataset:
    return xr.Dataset({k: convert(k, ds[meta[source]]).rename(k) for k, meta in VARS.items()})


def regrid_prior(prior: xr.Dataset, retr: xr.Dataset) -> xr.Dataset:
    same = (
        prior.sizes["lat"] == retr.sizes["lat"]
        and prior.sizes["lon"] == retr.sizes["lon"]
        and np.allclose(prior.lat, retr.lat)
        and np.allclose(prior.lon, retr.lon)
    )
    if same:
        return prior

    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    target = xr.Dataset(coords={"lat": retr.lat, "lon": retr.lon})
    r = xe.Regridder(
        prior, target, "bilinear", periodic=True,
        filename=str(WEIGHTS), reuse_weights=WEIGHTS.exists(),
    )
    return r(prior).assign_coords(lat=retr.lat, lon=retr.lon)


def _normalization_stats() -> pd.DataFrame:
    """Read the exact COSP prior normalization and compare with MODIS constants."""
    rows = []
    for key, meta in VARS.items():
        pvar, rvar, _, long_name, units = meta
        path = STATS_DIR / f"{pvar}_stats.nc4"
        if not path.exists():
            raise FileNotFoundError(f"Missing prior normalization file: {path}")

        with xr.open_dataset(path) as ds:
            mname = f"{pvar}_mean"
            sname = f"{pvar}_std"
            if mname not in ds or sname not in ds:
                raise KeyError(f"Expected {mname} and {sname} in {path}")
            mu_raw = float(ds[mname].mean().values)
            sd_raw = float(ds[sname].mean().values)
            stat_units = ds[mname].attrs.get("units", ds[sname].attrs.get("units", ""))

        factor = _unit_factor(key, stat_units, mu_raw)
        mu_cosp = mu_raw * factor
        sd_cosp = sd_raw * abs(factor)
        mu_modis, sd_modis = MODIS_NORM[key]

        rows.append({
            "key": key,
            "quantity": long_name,
            "common_units": units,
            "cosp_variable": pvar,
            "modis_variable": rvar,
            "cosp_norm_mean": mu_cosp,
            "cosp_norm_std": sd_cosp,
            "modis_norm_mean": mu_modis,
            "modis_norm_std": sd_modis,
            "norm_mean_delta_modis_minus_cosp": mu_modis - mu_cosp,
            "norm_mean_shift_in_cosp_sigma": (mu_modis - mu_cosp) / sd_cosp,
            "norm_std_ratio_modis_over_cosp": sd_modis / sd_cosp,
        })
    return pd.DataFrame(rows)


def paired_products(prior: xr.Dataset, retr: xr.Dataset):
    """Paired climatology maps, support counts, and common-period moments."""
    prior, retr = xr.align(prior, retr, join="inner")
    if prior.sizes["time"] == 0:
        raise RuntimeError("No common dates")
    print(
        f"Common dates: {prior.sizes['time']} "
        f"({str(prior.time.values[0])[:10]} to {str(prior.time.values[-1])[:10]})"
    )

    fields = {}
    wlat = xr.DataArray(
        np.cos(np.deg2rad(np.asarray(prior.lat.values, dtype=float))),
        dims=("lat",), coords={"lat": prior.lat}, name="area_weight",
    )

    for key in VARS:
        p = prior[key]
        r = retr[key]
        valid = np.isfinite(p) & np.isfinite(r)
        fields[f"p_{key}"] = p.where(valid).mean("time", skipna=True)
        fields[f"r_{key}"] = r.where(valid).mean("time", skipna=True)
        fields[f"n_{key}"] = valid.sum("time")

        w = wlat.broadcast_like(p).where(valid)
        fields[f"sw_{key}"] = w.sum(("time", "lat", "lon"), skipna=True)
        fields[f"sp_{key}"] = (p.where(valid) * w).sum(("time", "lat", "lon"), skipna=True)
        fields[f"sp2_{key}"] = ((p.where(valid) ** 2) * w).sum(("time", "lat", "lon"), skipna=True)
        fields[f"sr_{key}"] = (r.where(valid) * w).sum(("time", "lat", "lon"), skipna=True)
        fields[f"sr2_{key}"] = ((r.where(valid) ** 2) * w).sum(("time", "lat", "lon"), skipna=True)

    allout = xr.Dataset(fields).compute()
    means = allout[[v for v in allout.data_vars if v.startswith(("p_", "r_"))]]
    counts = xr.Dataset({k: allout[f"n_{k}"] for k in VARS})

    rows = []
    for key, meta in VARS.items():
        _, _, _, long_name, units = meta
        sw = float(allout[f"sw_{key}"].values)
        mu_p = float(allout[f"sp_{key}"].values) / sw
        mu_r = float(allout[f"sr_{key}"].values) / sw
        e2_p = float(allout[f"sp2_{key}"].values) / sw
        e2_r = float(allout[f"sr2_{key}"].values) / sw
        sd_p = np.sqrt(max(e2_p - mu_p * mu_p, 0.0))
        sd_r = np.sqrt(max(e2_r - mu_r * mu_r, 0.0))
        rows.append({
            "key": key,
            "quantity": long_name,
            "common_units": units,
            "paired_cosp_mean": mu_p,
            "paired_cosp_std": sd_p,
            "paired_modis_mean": mu_r,
            "paired_modis_std": sd_r,
            "paired_mean_delta_modis_minus_cosp": mu_r - mu_p,
            "paired_mean_shift_in_cosp_sigma": (mu_r - mu_p) / sd_p if sd_p > 0 else np.nan,
            "paired_std_ratio_modis_over_cosp": sd_r / sd_p if sd_p > 0 else np.nan,
        })

    return means, counts, pd.DataFrame(rows)


def write_domain_shift_stats(paired_stats: pd.DataFrame, outdir: Path, basename: str) -> Path:
    """Write/print normalization and paired-distribution diagnostics in common units."""
    norm = _normalization_stats()
    table = norm.merge(
        paired_stats.drop(columns=["quantity", "common_units"]),
        on="key", how="left", validate="one_to_one",
    )

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{basename}_standardization_domain_shift.csv"
    table.to_csv(path, index=False, float_format="%.8g")

    show = table[[
        "key", "common_units",
        "cosp_norm_mean", "cosp_norm_std",
        "modis_norm_mean", "modis_norm_std",
        "norm_mean_shift_in_cosp_sigma", "norm_std_ratio_modis_over_cosp",
        "paired_cosp_mean", "paired_cosp_std",
        "paired_modis_mean", "paired_modis_std",
        "paired_mean_shift_in_cosp_sigma", "paired_std_ratio_modis_over_cosp",
    ]].copy()
    print("\nCOSP -> MODIS input-domain statistics (common physical units)")
    print("Normalization columns reproduce the coordinate transforms seen by the networks.")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"Saved {path}")
    return path


def awmean(da: xr.DataArray) -> float:
    w = np.cos(np.deg2rad(da.lat))
    return float(da.weighted(w).mean(("lat", "lon"), skipna=True).values)


def limits(p: np.ndarray, r: np.ndarray, key: str) -> tuple[float, float]:
    x = np.concatenate((p.ravel(), r.ravel()))
    x = x[np.isfinite(x)]
    auto = np.nanpercentile(x, [2, 98])
    lo, hi = PLOT_LIMITS[key]
    return (auto[0] if lo is None else lo, auto[1] if hi is None else hi)


def dlimit(d: np.ndarray, key: str, mode: str) -> float:
    fixed = ABS_DIFF_LIMITS[key] if mode == "absolute" else PCT_DIFF_LIMITS[key]
    if fixed is not None:
        return float(fixed)
    x = np.abs(d[np.isfinite(d)])
    return float(np.nanpercentile(x, 98)) if x.size else 1.0


def draw_map(ax, lon, lat, field, cmap, norm, left, right, row, col):
    z, x = add_cyclic_point(field, coord=lon, axis=1)
    m = ax.pcolormesh(
        x, lat, z, transform=DATA_CRS, cmap=cmap, norm=norm,
        shading="auto", rasterized=True,
    )
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="0.92", edgecolor="none", zorder=0)
    ax.coastlines(linewidth=0.45, color="0.25")
    gl = ax.gridlines(
        crs=DATA_CRS, draw_labels=True,
        xlocs=np.arange(-120, 181, 60), ylocs=np.arange(-60, 61, 30),
        linewidth=0.30, color="0.45", alpha=0.45, linestyle=":",
    )
    gl.top_labels = gl.right_labels = False
    gl.left_labels = col == 0
    gl.bottom_labels = row == 3
    gl.xlabel_style = gl.ylabel_style = {"size": 6.3}
    ax.set_title(left, loc="left", pad=2.5, fontweight="bold")
    ax.set_title(right, loc="right", pad=2.5, fontsize=7.2)
    return m


def plot(means: xr.Dataset, counts: xr.Dataset, mode: str, outdir: Path, basename: str):
    lat = np.asarray(means.lat, float)
    lon = np.asarray(means.lon, float)
    fig = plt.figure(figsize=(11.8, 11.0))
    outer = fig.add_gridspec(
        8, 1, height_ratios=[1, .04, 1, .04, 1, .04, 1, .04],
        hspace=.11, left=.045, right=.985, bottom=.055, top=.965,
    )
    mapgs = [outer[2*i].subgridspec(1, 3, wspace=.055) for i in range(4)]
    cbgs = [outer[2*i+1].subgridspec(1, 3, wspace=.15) for i in range(4)]
    headers = ["GiOcean/COSP", "MODIS retrieval", "Retrieval - COSP"]
    letters = iter("abcdefghijkl")

    for row, (key, meta) in enumerate(VARS.items()):
        _, _, label, long_name, units = meta
        pda, rda = means[f"p_{key}"], means[f"r_{key}"]
        p, r = np.asarray(pda, float), np.asarray(rda, float)

        if mode == "absolute":
            d = r - p
            dunits = units
        else:
            d = np.full_like(p, np.nan)
            ok = np.isfinite(p) & np.isfinite(r) & (np.abs(p) >= PCT_FLOOR[key])
            d[ok] = 100.0 * (r[ok] - p[ok]) / p[ok]
            dunits = "%"

        vmin, vmax = limits(p, r, key)
        anorm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        dl = dlimit(d, key, mode)
        dnorm = mcolors.TwoSlopeNorm(vmin=-dl, vcenter=0.0, vmax=dl)
        dda = xr.DataArray(d, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
        g = [awmean(pda), awmean(rda), awmean(dda)]

        meshes = []
        for col, (field, cmap, norm) in enumerate([
            (p, ABS_CMAP, anorm), (r, ABS_CMAP, anorm), (d, DIFF_CMAP, dnorm)
        ]):
            ax = fig.add_subplot(mapgs[row][0, col], projection=MAP_PROJ)
            letter = next(letters)
            right = rf"$\overline{{x}}_A={g[col]:.2f}$" if col < 2 else rf"$\overline{{\Delta}}_A={g[col]:+.2f}$"
            meshes.append(draw_map(ax, lon, lat, field, cmap, norm, f"({letter}) {label}", right, row, col))
            if row == 0:
                ax.text(.5, 1.105, headers[col], transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

        c1 = fig.add_subplot(cbgs[row][0, 0:2])
        cb1 = fig.colorbar(meshes[0], cax=c1, orientation="horizontal")
        cb1.ax.xaxis.set_label_position("top")
        cb1.ax.xaxis.set_ticks_position("bottom")
        cb1.set_label(
            long_name if units == "1" else f"{long_name} ({units})",
            labelpad=3.0,
        )
        cb1.ax.tick_params(length=2.5, pad=1.3)

        c2 = fig.add_subplot(cbgs[row][0, 2])
        cb2 = fig.colorbar(meshes[2], cax=c2, orientation="horizontal", extend="both")
        cb2.ax.xaxis.set_label_position("top")
        cb2.ax.xaxis.set_ticks_position("bottom")
        if mode == "absolute":
            cb2.set_label(
                "Retrieval - COSP" if dunits == "1" else f"Retrieval - COSP ({dunits})",
                labelpad=3.0,
            )
        else:
            cb2.set_label("Relative difference (%)", labelpad=3.0)
        cb2.ax.tick_params(length=2.5, pad=1.3)

        n = np.asarray(counts[key], float)
        n = n[np.isfinite(n) & (n > 0)]
        print(
            f"{key:4s}: COSP={g[0]:.3f}, retrieval={g[1]:.3f}, "
            f"delta={g[2]:+.3f} {dunits}; median paired days={np.nanmedian(n):.0f}"
        )

    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / f"{basename}_{mode}"
    fig.savefig(str(stem) + ".png", dpi=DPI, bbox_inches="tight", pad_inches=.03)
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight", pad_inches=.03)
    plt.close(fig)
    print(f"Saved {stem}.png and {stem}.pdf")


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--difference", choices=("absolute", "percentage"), default=DIFF_MODE)
    p.add_argument("--start-year", type=int, default=START_YEAR)
    p.add_argument("--end-year", type=int, default=END_YEAR)
    p.add_argument("--outdir", type=Path, default=Path("."))
    p.add_argument("--basename", default=OUT_BASENAME)
    return p.parse_args()


def main():
    a = args()
    pnames = [v[0] for v in VARS.values()]
    rnames = [v[1] for v in VARS.values()]
    p = prep(open_only(files(PRIOR_DIR), pnames), a.start_year, a.end_year)
    r = prep(open_only(files(RETR_DIR, RETR_GLOB), rnames), a.start_year, a.end_year)

    common = np.intersect1d(p.time.values, r.time.values)
    if common.size == 0:
        raise RuntimeError(f"No common dates in {a.start_year}-{a.end_year}")
    p, r = p.sel(time=common), r.sel(time=common)
    p, r = canonical(p, 0), canonical(r, 1)
    p = regrid_prior(p, r)

    means, counts, paired_stats = paired_products(p, r)
    write_domain_shift_stats(paired_stats, a.outdir, a.basename)
    plot(means, counts, a.difference, a.outdir, a.basename)


if __name__ == "__main__":
    main()
