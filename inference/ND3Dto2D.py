#!/usr/bin/env python3
"""
Collapse daily 3-D NDnet fields to 2-D monthly time-series files for comparison
against 1-degree daily MODIS Nd retrieval products.

For each NDnet daily file, this script computes two vertical reductions of
NDinc_cf01p_pred over an optional pressure range, default 800--950 hPa:

  1) Liquid-mass-weighted mean:

         weight = rho_air * QL * dz

     where rho_air is computed from GiOcean temperature and pressure using the
     ideal gas law, and dz is approximated from GiOcean H after interpolation
     to the NDnet vertical grid.

  2) Unweighted mean:

         arithmetic mean over finite NDnet levels in the selected pressure range

Masks are applied progressively to both ND reductions:
  1) no_mask
       No scene mask at all.
  2) modis_track
       MODIS liquid optical-thickness valid-retrieval / track mask only.
  3) modis_track_ctt
       MODIS track mask AND GiOcean cloud-top temperature > 268 K.
  4) modis_track_ctt_cf
       MODIS track mask AND GiOcean cloud-top temperature > 268 K AND
       MODIS Cloud_Retrieval_Fraction_Liquid_Mean > 0.8.

If sigmaNd is present in the NDnet file, the script also writes the original
liquid-mass-weighted sigma diagnostics:
  - sigma_no_mask, sigma_modis_track, sigma_modis_track_ctt,
    sigma_modis_track_ctt_cf
      Liquid-mass-weighted vertical means of sigmaNd.
  - sigmaW_no_mask, sigmaW_modis_track, sigmaW_modis_track_ctt,
    sigmaW_modis_track_ctt_cf
      Propagated integrated uncertainty of the weighted-column value, computed
      as sqrt(sum((w_i * sigma_i)^2)) / sum(w_i).

This version writes one NetCDF file per month, with an unlimited time dimension
inside each monthly file. It appends one day at a time to the current month file,
so it avoids both thousands of daily outputs and one very large slow full-period
file.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

try:
    import netCDF4
except ImportError as exc:
    raise ImportError(
        "This monthly writer requires netCDF4. Install it with, for example, "
        "`conda install -c conda-forge netcdf4`."
    ) from exc


ND_VAR = "NDinc_cf01p"
SIGMA_VAR = "sigmaNd"
MODIS_MASK_VAR = "Cloud_Optical_Thickness_Liquid_Mean"
MODIS_CF_VAR = "Cloud_Retrieval_Fraction_Liquid_Mean"
CTT_VAR = "SWCLDTMP"
QL_VAR = "QL"
H_VAR = "H"
T_VAR = "T"
RD = 287.05  # J kg-1 K-1
FILL_FLOAT = np.float32(np.nan)


# -----------------------------------------------------------------------------
# User defaults
# -----------------------------------------------------------------------------
# With these defaults, the standard run is simply:
#
#     python build_NDnet_2D_period.py
#
# It will read the standard NDnet, GiOcean, and MODIS directories; average over
# 800--950 hPa; write both QL-weighted and unweighted Nd fields; and produce
# one output file per month:
#
#     NDnet_2D_monthly/NDnet_modis_compare_800_950hPa_YYYYMM.nc
#
# Use command-line flags only when overriding these defaults.

DEFAULT_NDNET_DIR = "NDnet_infer_daily"
DEFAULT_GIO_ATM_DIR = "/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/dailies/atm_inst_6hr_glo_L720x361_p49"
DEFAULT_GIO_SFC_DIR = "/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/dailies/mod_inst_6hr_glo_L720x361_sfc"
DEFAULT_MODIS_MASK_DIR = "/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/dailies/MCD06COSP_D3_MODIS/on_Noutgrid/"
DEFAULT_OUT_DIR = "NDnet_2D_monthly"
DEFAULT_OUT_TEMPLATE = "NDnet_modis_compare_800_950hPa_{yyyymm}.nc"
DEFAULT_PATTERN = "NDnet_infer_*.nc*"
# Minimum NDnet Nd retained before vertical averaging.
# Set to None to disable, or e.g. 10.0 to discard Nd < 10 cm-3.
DEFAULT_MIN_ND = 10.
DEFAULT_MIN_QL = 1.0e-5
DEFAULT_MIN_TOTAL_WEIGHT = 1.0e-4
DEFAULT_MIN_COD = 1
DEFAULT_PLEV_MIN = 500.0
DEFAULT_PLEV_MAX = 1000.
DEFAULT_ALL_LEVELS = False
DEFAULT_START_MONTH = None  # e.g., "200301"
DEFAULT_END_MONTH = None    # e.g., "202512"
# Month-level parallelism. Each worker writes one independent monthly output file.
# For shared filesystems, 4 is usually a safer default than using all cores.
DEFAULT_WORKERS = 24
DEFAULT_VERBOSE_DAYS = False


@dataclass
class MonthCacheEntry:
    atm: xr.Dataset
    sfc: xr.Dataset
    modis: xr.Dataset


class MonthCache:
    def __init__(self, gio_atm_dir: str, gio_sfc_dir: str, modis_mask_dir: str):
        self.gio_atm_dir = gio_atm_dir
        self.gio_sfc_dir = gio_sfc_dir
        self.modis_mask_dir = modis_mask_dir
        self._cache: Dict[str, MonthCacheEntry] = {}

    def get(self, yyyymm: str) -> MonthCacheEntry:
        if yyyymm not in self._cache:
            atm_path = find_month_file(
                self.gio_atm_dir,
                [
                    f"GiOCEAN_e1.atm_inst_6hr_glo_L720x361_p49.tavg1D.{yyyymm}.nc",
                    f"*atm_inst_6hr_glo_L720x361_p49*tavg1D.{yyyymm}.nc",
                ],
            )
            sfc_path = find_month_file(
                self.gio_sfc_dir,
                [
                    f"GiOCEAN_e1.mod_inst_6hr_glo_L720x361_sfc.tavg1D.{yyyymm}.nc",
                    f"*mod_inst_6hr_glo_L720x361_sfc*tavg1D.{yyyymm}.nc",
                ],
            )
            modis_path = find_month_file(
                self.modis_mask_dir,
                [
                    f"MCD06COSP_D3_MODIS.{yyyymm}.Noutgrid.nc",
                    f"MCD06COSP_D3_MODIS.{yyyymm}.nc",
                    f"*MCD06COSP_D3_MODIS*{yyyymm}*.nc",
                ],
            )
            self._cache[yyyymm] = MonthCacheEntry(
                atm=xr.open_dataset(atm_path, decode_times=True),
                sfc=xr.open_dataset(sfc_path, decode_times=True),
                modis=xr.open_dataset(modis_path, decode_times=True),
            )
        return self._cache[yyyymm]

    def close(self) -> None:
        for entry in self._cache.values():
            entry.atm.close()
            entry.sfc.close()
            entry.modis.close()
        self._cache.clear()


_DATE_RE = re.compile(r"(\d{8})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build monthly 2-D NDnet comparison files with weighted and unweighted pressure-layer means."
    )
    p.add_argument("--ndnet-dir", default=DEFAULT_NDNET_DIR)
    p.add_argument("--gio-atm-dir", default=DEFAULT_GIO_ATM_DIR)
    p.add_argument("--gio-sfc-dir", default=DEFAULT_GIO_SFC_DIR)
    p.add_argument("--modis-mask-dir", default=DEFAULT_MODIS_MASK_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--out-template",
        default=DEFAULT_OUT_TEMPLATE,
        help=(
            "Monthly output filename template. Must contain {yyyymm}. "
            "Relative paths are placed under --out-dir."
        ),
    )
    p.add_argument("--pattern", default=DEFAULT_PATTERN)
    p.add_argument(
        "--min-nd",
        type=float,
        default=DEFAULT_MIN_ND,
        help="Minimum NDnet Nd retained before vertical averaging, in cm-3. Use --min-nd nan to disable.",
    )
    p.add_argument("--min-ql", type=float, default=DEFAULT_MIN_QL, help="Minimum QL used in liquid-mass weights")
    p.add_argument(
        "--min-total-weight",
        type=float,
        default=DEFAULT_MIN_TOTAL_WEIGHT,
        help="Minimum vertically integrated liquid-mass weight required to keep a weighted column",
    )
    p.add_argument(
        "--min-cod",
        type=float,
        default=DEFAULT_MIN_COD,
        help="Optional minimum Cloud_Optical_Thickness_Liquid_Mean for MODIS track mask. Use --min-cod nan for finite-only.",
    )
    p.add_argument(
        "--plev-min",
        type=float,
        default=DEFAULT_PLEV_MIN,
        help=f"Lower pressure bound in hPa for vertical means. Default: {DEFAULT_PLEV_MIN:g}.",
    )
    p.add_argument(
        "--plev-max",
        type=float,
        default=DEFAULT_PLEV_MAX,
        help=f"Upper pressure bound in hPa for vertical means. Default: {DEFAULT_PLEV_MAX:g}.",
    )
    p.add_argument(
        "--all-levels",
        action="store_true",
        default=DEFAULT_ALL_LEVELS,
        help="Ignore --plev-min/--plev-max and average over all available NDnet levels.",
    )
    p.add_argument(
        "--start-month",
        default=DEFAULT_START_MONTH,
        help="Optional first month to process, in YYYYMM format.",
    )
    p.add_argument(
        "--end-month",
        default=DEFAULT_END_MONTH,
        help="Optional last month to process, in YYYYMM format.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Number of months to process in parallel. Each worker handles one month "
            "and writes one independent monthly NetCDF. Use 1 for serial execution."
        ),
    )
    p.add_argument(
        "--verbose-days",
        action="store_true",
        default=DEFAULT_VERBOSE_DAYS,
        help="Print one progress line per appended day. By default, only monthly summaries are printed.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def find_month_file(base_dir: str, patterns: list[str]) -> str:
    for pat in patterns:
        matches = sorted(glob.glob(os.path.join(base_dir, pat)))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No monthly file found in {base_dir} for patterns: {patterns}")


def infer_date_from_path(path: str) -> pd.Timestamp:
    m = _DATE_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Could not extract YYYYMMDD from filename: {path}")
    return pd.to_datetime(m.group(1), format="%Y%m%d")


def resolve_month_out_file(args: argparse.Namespace, yyyymm: str) -> str:
    if "{yyyymm}" not in args.out_template:
        raise ValueError("--out-template must contain the token {yyyymm}")
    fname = args.out_template.format(yyyymm=yyyymm)
    if os.path.isabs(fname):
        return fname
    return os.path.join(args.out_dir, fname)


def month_is_in_requested_range(yyyymm: str, args: argparse.Namespace) -> bool:
    if args.start_month is not None and yyyymm < str(args.start_month):
        return False
    if args.end_month is not None and yyyymm > str(args.end_month):
        return False
    return True


def normalize_longitudes(ds: xr.Dataset | xr.DataArray, target_lons: xr.DataArray) -> xr.Dataset | xr.DataArray:
    if "lon" not in ds.coords:
        return ds
    out = ds
    src = out["lon"].values
    tgt = target_lons.values
    src_min = np.nanmin(src)
    tgt_min = np.nanmin(tgt)
    if tgt_min < 0 and src_min >= 0:
        out = out.assign_coords(lon=((out["lon"] + 180.0) % 360.0) - 180.0).sortby("lon")
    elif tgt_min >= 0 and src_min < 0:
        out = out.assign_coords(lon=(out["lon"] % 360.0)).sortby("lon")
    _, unique_idx = np.unique(out["lon"].values, return_index=True)
    if len(unique_idx) != out.sizes["lon"]:
        out = out.isel(lon=np.sort(unique_idx))
    return out


def sel_day(ds: xr.Dataset, day: pd.Timestamp) -> xr.Dataset:
    try:
        out = ds.sel(time=day)
    except Exception:
        idx = pd.DatetimeIndex(pd.to_datetime(ds["time"].values)).normalize() == day.normalize()
        if not np.any(idx):
            raise KeyError(f"Date {day.date()} not found in dataset")
        out = ds.isel(time=np.where(idx)[0][0])
    if "time" in out.dims:
        out = out.squeeze("time", drop=True)
    return out


def interp_to_ndnet_grid(
    da: xr.DataArray,
    nd_lev: Optional[xr.DataArray],
    nd_lat: xr.DataArray,
    nd_lon: xr.DataArray,
) -> xr.DataArray:
    out = normalize_longitudes(da, nd_lon)
    interp_kwargs = {}
    if "lev" in out.dims and nd_lev is not None:
        interp_kwargs["lev"] = nd_lev
    if "lat" in out.dims:
        interp_kwargs["lat"] = nd_lat
    if "lon" in out.dims:
        interp_kwargs["lon"] = nd_lon
    if interp_kwargs:
        out = out.interp(**interp_kwargs)
    return out


def restrict_to_pressure_range(
    da: xr.DataArray,
    plev_min: Optional[float],
    plev_max: Optional[float],
    all_levels: bool = False,
) -> xr.DataArray:
    """Keep pressure levels in [min(plev_min, plev_max), max(...)] in hPa."""
    if all_levels or "lev" not in da.dims:
        return da
    if plev_min is None or plev_max is None:
        return da
    p0, p1 = sorted((float(plev_min), float(plev_max)))
    lev_mask = (da["lev"] >= p0) & (da["lev"] <= p1)
    out = da.where(lev_mask, drop=True)
    if out.sizes.get("lev", 0) == 0:
        raise ValueError(f"No levels found in requested pressure range [{p0}, {p1}] hPa")
    return out


def pressure_range_attr(args: argparse.Namespace) -> str:
    if args.all_levels:
        return "all available NDnet levels"
    p0, p1 = sorted((float(args.plev_min), float(args.plev_max)))
    return f"{p0:g}-{p1:g} hPa inclusive"



def apply_min_nd_filter(nd: xr.DataArray, min_nd: Optional[float]) -> xr.DataArray:
    """Mask NDnet values below min_nd before any vertical averaging."""
    if min_nd is None or not np.isfinite(min_nd):
        return nd
    out = xr.where(np.isfinite(nd) & (nd >= float(min_nd)), nd, np.nan)
    out.attrs.update(nd.attrs)
    return out

def approximate_layer_thickness(height_da: xr.DataArray) -> xr.DataArray:
    if "lev" not in height_da.dims:
        raise ValueError("height_da must contain a 'lev' dimension")
    h = height_da.astype("float64")
    vals = h.values
    lev_axis = h.get_axis_num("lev")
    dz = np.full_like(vals, np.nan, dtype=np.float64)

    sl0 = [slice(None)] * vals.ndim
    sl1 = [slice(None)] * vals.ndim
    sl0[lev_axis] = 0
    sl1[lev_axis] = 1
    dz[tuple(sl0)] = np.abs(vals[tuple(sl1)] - vals[tuple(sl0)])

    slm1 = [slice(None)] * vals.ndim
    slm2 = [slice(None)] * vals.ndim
    slm1[lev_axis] = -1
    slm2[lev_axis] = -2
    dz[tuple(slm1)] = np.abs(vals[tuple(slm1)] - vals[tuple(slm2)])

    if vals.shape[lev_axis] > 2:
        slmid = [slice(None)] * vals.ndim
        slp = [slice(None)] * vals.ndim
        sln = [slice(None)] * vals.ndim
        slmid[lev_axis] = slice(1, -1)
        slp[lev_axis] = slice(2, None)
        sln[lev_axis] = slice(None, -2)
        dz[tuple(slmid)] = 0.5 * np.abs(vals[tuple(slp)] - vals[tuple(sln)])

    return xr.DataArray(
        dz,
        coords=h.coords,
        dims=h.dims,
        attrs={"units": "m", "long_name": "approximate_layer_thickness_from_H"},
    )


def air_density_from_temperature(temperature: xr.DataArray, lev_hpa: xr.DataArray) -> xr.DataArray:
    if "lev" not in temperature.dims:
        raise ValueError("temperature must contain a 'lev' dimension")
    p_pa = xr.DataArray(
        lev_hpa.astype("float64").values * 100.0,
        coords={"lev": lev_hpa},
        dims=("lev",),
        attrs={"units": "Pa"},
    )
    t_ok = xr.where(np.isfinite(temperature) & (temperature > 0.0), temperature.astype("float64"), np.nan)
    rho = p_pa / (RD * t_ok)
    rho.attrs.update({"units": "kg m-3", "long_name": "air_density_from_pressure_and_temperature"})
    return rho


def build_liquid_mass_weight(
    ql: xr.DataArray,
    rho: xr.DataArray,
    dz: xr.DataArray,
    min_ql: float,
) -> xr.DataArray:
    ql_ok = xr.where(np.isfinite(ql) & (ql > min_ql), ql.astype("float64"), 0.0)
    rho_ok = xr.where(np.isfinite(rho) & (rho > 0.0), rho.astype("float64"), 0.0)
    dz_ok = xr.where(np.isfinite(dz) & (dz > 0.0), dz.astype("float64"), 0.0)
    w = ql_ok * rho_ok * dz_ok
    w.attrs.update({"units": "kg m-2", "long_name": "liquid_mass_weight_per_layer"})
    return w


def collapse_field_weighted_mean(
    field: xr.DataArray,
    weight: xr.DataArray,
    min_total_weight: float,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    field_ok = xr.where(np.isfinite(field), field.astype("float64"), np.nan)
    valid_weight = xr.where(np.isfinite(field_ok), weight, 0.0)
    weighted_sum = (field_ok * valid_weight).sum("lev", skipna=True)
    total_weight = valid_weight.sum("lev", skipna=True)
    nlev_pos = ((valid_weight > 0.0) & np.isfinite(field_ok)).sum("lev")
    col = xr.where(total_weight > min_total_weight, weighted_sum / total_weight, np.nan)
    return col.astype("float32"), total_weight.astype("float32"), nlev_pos.astype("int16")


def collapse_field_unweighted_mean(field: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    field_ok = xr.where(np.isfinite(field), field.astype("float64"), np.nan)
    nlev_valid = np.isfinite(field_ok).sum("lev")
    col = field_ok.mean("lev", skipna=True)
    col = xr.where(nlev_valid > 0, col, np.nan)
    return col.astype("float32"), nlev_valid.astype("int16")


def collapse_sigma_propagated(
    sigma: xr.DataArray,
    weight: xr.DataArray,
    min_total_weight: float,
) -> xr.DataArray:
    sigma_ok = xr.where(np.isfinite(sigma) & (sigma >= 0.0), sigma.astype("float64"), np.nan)
    valid_weight = xr.where(np.isfinite(sigma_ok), weight, 0.0)
    total_weight = valid_weight.sum("lev", skipna=True)
    numer = np.sqrt(((sigma_ok * valid_weight) ** 2).sum("lev", skipna=True))
    sigma_w = xr.where(total_weight > min_total_weight, numer / total_weight, np.nan)
    sigma_w.attrs.update({
        "long_name": "propagated integrated uncertainty of liquid-mass-weighted column value",
        "units": getattr(sigma, "units", getattr(sigma, "unit", "same_as_sigmaNd")),
    })
    return sigma_w.astype("float32")


def build_modis_track_mask(modis_cod: xr.DataArray, min_cod: Optional[float]) -> xr.DataArray:
    mask = np.isfinite(modis_cod)
    if min_cod is not None and np.isfinite(min_cod):
        mask = mask & (modis_cod > min_cod)
    return mask.astype(bool)


def apply_mask(field: xr.DataArray, mask: xr.DataArray, name: str, long_name: str) -> xr.DataArray:
    out = xr.where(mask, field, np.nan).astype("float32")
    out.name = name
    out.attrs.update({"long_name": long_name, "units": field.attrs.get("units", "")})
    return out


def process_one_day(ndnet_path: str, cache: MonthCache, args: argparse.Namespace) -> xr.Dataset:
    day = infer_date_from_path(ndnet_path)
    yyyymm = day.strftime("%Y%m")

    with xr.open_dataset(ndnet_path, decode_times=True) as nd_ds:
        nd_day = nd_ds.squeeze("time", drop=True) if "time" in nd_ds.dims else nd_ds
        nd_full = nd_day[ND_VAR].astype("float32")
        nd = restrict_to_pressure_range(nd_full, args.plev_min, args.plev_max, args.all_levels)
        nd = apply_min_nd_filter(nd, args.min_nd)
        nd_lat = nd_day["lat"]
        nd_lon = nd_day["lon"]
        nd_lev = nd["lev"]

        has_sigma = SIGMA_VAR in nd_day.data_vars
        sigma = None
        if has_sigma:
            sigma = restrict_to_pressure_range(
                nd_day[SIGMA_VAR].astype("float32"), args.plev_min, args.plev_max, args.all_levels
            )
            # Keep sigma diagnostics consistent with any Nd filtering, including --min-nd.
            sigma = sigma.where(np.isfinite(nd))

        entry = cache.get(yyyymm)
        gio_atm_day = sel_day(entry.atm, day)
        gio_sfc_day = sel_day(entry.sfc, day)
        modis_day = sel_day(entry.modis, day)

        ql = interp_to_ndnet_grid(gio_atm_day[QL_VAR], nd_lev, nd_lat, nd_lon)
        h = interp_to_ndnet_grid(gio_atm_day[H_VAR], nd_lev, nd_lat, nd_lon)
        temp = interp_to_ndnet_grid(gio_atm_day[T_VAR], nd_lev, nd_lat, nd_lon)
        ctt = interp_to_ndnet_grid(gio_sfc_day[CTT_VAR], None, nd_lat, nd_lon)
        modis_cod = interp_to_ndnet_grid(modis_day[MODIS_MASK_VAR], None, nd_lat, nd_lon)
        modis_cf = interp_to_ndnet_grid(modis_day[MODIS_CF_VAR], None, nd_lat, nd_lon)

        dz = approximate_layer_thickness(h)
        rho = air_density_from_temperature(temp, nd_lev)
        liquid_mass_weight = build_liquid_mass_weight(ql, rho, dz, args.min_ql)

        nd_col, total_weight, nlev_liq = collapse_field_weighted_mean(
            nd, liquid_mass_weight, args.min_total_weight
        )
        nd_col.attrs.update({
            "long_name": "liquid-mass-weighted pressure-layer NDnet Nd",
            "units": getattr(nd, "units", "cm-3"),
        })
        total_weight.attrs.update({
            "long_name": "pressure-layer integrated liquid mass weight used for weighted mean",
            "units": "kg m-2",
        })
        nlev_liq.attrs.update({"long_name": "number of selected pressure levels with positive liquid mass weight and finite Nd"})

        nd_unweighted_col, nlev_unweighted = collapse_field_unweighted_mean(nd)
        nd_unweighted_col.attrs.update({
            "long_name": "unweighted arithmetic pressure-layer NDnet Nd",
            "units": getattr(nd, "units", "cm-3"),
        })
        nlev_unweighted.attrs.update({"long_name": "number of selected pressure levels with finite Nd"})

        if has_sigma and sigma is not None:
            sigma_col, _, _ = collapse_field_weighted_mean(
                sigma, liquid_mass_weight, args.min_total_weight
            )
            sigma_col.attrs.update({
                "long_name": "liquid-mass-weighted pressure-layer sigmaNd",
                "units": getattr(sigma, "units", "cm-3"),
            })
            sigma_w = collapse_sigma_propagated(sigma, liquid_mass_weight, args.min_total_weight)
        else:
            sigma_col = None
            sigma_w = None

        modis_track = build_modis_track_mask(modis_cod, args.min_cod)
        ctt_gt_268 = np.isfinite(ctt) & (ctt > 268.0)
        cf_gt_0p8 = np.isfinite(modis_cf) & (modis_cf > 0.8)

        mask_no = xr.ones_like(modis_track, dtype=bool)
        mask_track = modis_track
        mask_track_ctt = mask_track & ctt_gt_268
        mask_track_ctt_cf = mask_track_ctt & cf_gt_0p8

        data_vars = {
            # Liquid-mass-weighted Nd: original variable names retained for backward compatibility.
            "nd_no_mask": apply_mask(
                nd_col, mask_no, "nd_no_mask", "QL-liquid-mass-weighted pressure-layer NDnet Nd with no scene mask"
            ),
            "nd_modis_track": apply_mask(
                nd_col, mask_track, "nd_modis_track", "QL-liquid-mass-weighted pressure-layer NDnet Nd with MODIS track mask"
            ),
            "nd_modis_track_ctt": apply_mask(
                nd_col,
                mask_track_ctt,
                "nd_modis_track_ctt",
                "QL-liquid-mass-weighted pressure-layer NDnet Nd with MODIS track mask and cloud-top temperature > 268 K",
            ),
            "nd_modis_track_ctt_cf": apply_mask(
                nd_col,
                mask_track_ctt_cf,
                "nd_modis_track_ctt_cf",
                "QL-liquid-mass-weighted pressure-layer NDnet Nd with MODIS track mask, cloud-top temperature > 268 K, and MODIS cloud fraction > 0.8",
            ),
            # New unweighted Nd set.
            "nd_unweighted_no_mask": apply_mask(
                nd_unweighted_col,
                mask_no,
                "nd_unweighted_no_mask",
                "unweighted arithmetic pressure-layer NDnet Nd with no scene mask",
            ),
            "nd_unweighted_modis_track": apply_mask(
                nd_unweighted_col,
                mask_track,
                "nd_unweighted_modis_track",
                "unweighted arithmetic pressure-layer NDnet Nd with MODIS track mask",
            ),
            "nd_unweighted_modis_track_ctt": apply_mask(
                nd_unweighted_col,
                mask_track_ctt,
                "nd_unweighted_modis_track_ctt",
                "unweighted arithmetic pressure-layer NDnet Nd with MODIS track mask and cloud-top temperature > 268 K",
            ),
            "nd_unweighted_modis_track_ctt_cf": apply_mask(
                nd_unweighted_col,
                mask_track_ctt_cf,
                "nd_unweighted_modis_track_ctt_cf",
                "unweighted arithmetic pressure-layer NDnet Nd with MODIS track mask, cloud-top temperature > 268 K, and MODIS cloud fraction > 0.8",
            ),
            "mask_no_mask": mask_no.astype("int8"),
            "mask_modis_track": mask_track.astype("int8"),
            "mask_modis_track_ctt": mask_track_ctt.astype("int8"),
            "mask_modis_track_ctt_cf": mask_track_ctt_cf.astype("int8"),
            "cloud_top_temperature": ctt.astype("float32"),
            "modis_cloud_optical_thickness_liquid_mean": modis_cod.astype("float32"),
            "modis_cloud_fraction_liquid_mean": modis_cf.astype("float32"),
            "liquid_mass_weight_sum": total_weight,
            "nlev_liquid": nlev_liq,
            "nlev_unweighted": nlev_unweighted,
            "air_density_sampled_mean": xr.where(liquid_mass_weight > 0.0, rho, np.nan).mean("lev", skipna=True).astype("float32"),
        }

        if has_sigma and sigma_col is not None and sigma_w is not None:
            data_vars.update({
                "sigma_no_mask": apply_mask(
                    sigma_col, mask_no, "sigma_no_mask", "liquid-mass-weighted pressure-layer sigmaNd with no scene mask"
                ),
                "sigma_modis_track": apply_mask(
                    sigma_col, mask_track, "sigma_modis_track", "liquid-mass-weighted pressure-layer sigmaNd with MODIS track mask"
                ),
                "sigma_modis_track_ctt": apply_mask(
                    sigma_col,
                    mask_track_ctt,
                    "sigma_modis_track_ctt",
                    "liquid-mass-weighted pressure-layer sigmaNd with MODIS track mask and cloud-top temperature > 268 K",
                ),
                "sigma_modis_track_ctt_cf": apply_mask(
                    sigma_col,
                    mask_track_ctt_cf,
                    "sigma_modis_track_ctt_cf",
                    "liquid-mass-weighted pressure-layer sigmaNd with MODIS track mask, cloud-top temperature > 268 K, and MODIS cloud fraction > 0.8",
                ),
                "sigmaW_no_mask": apply_mask(
                    sigma_w,
                    mask_no,
                    "sigmaW_no_mask",
                    "propagated integrated uncertainty with no scene mask",
                ),
                "sigmaW_modis_track": apply_mask(
                    sigma_w,
                    mask_track,
                    "sigmaW_modis_track",
                    "propagated integrated uncertainty with MODIS track mask",
                ),
                "sigmaW_modis_track_ctt": apply_mask(
                    sigma_w,
                    mask_track_ctt,
                    "sigmaW_modis_track_ctt",
                    "propagated integrated uncertainty with MODIS track mask and cloud-top temperature > 268 K",
                ),
                "sigmaW_modis_track_ctt_cf": apply_mask(
                    sigma_w,
                    mask_track_ctt_cf,
                    "sigmaW_modis_track_ctt_cf",
                    "propagated integrated uncertainty with MODIS track mask, cloud-top temperature > 268 K, and MODIS cloud fraction > 0.8",
                ),
            })

        out = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": xr.DataArray([np.datetime64(day.date())], dims=("time",)),
                "lat": nd_lat,
                "lon": nd_lon,
            },
            attrs={
                "title": "Monthly NDnet pressure-layer fields collapsed for comparison to daily 1-degree MODIS Nd",
                "source_ndnet_dir": args.ndnet_dir,
                "source_ndnet_pattern": args.pattern,
                "source_gio_atm_dir": args.gio_atm_dir,
                "source_gio_sfc_dir": args.gio_sfc_dir,
                "source_modis_dir": args.modis_mask_dir,
                "vertical_pressure_range": pressure_range_attr(args),
                "minimum_nd_cm3": ("disabled" if args.min_nd is None or not np.isfinite(args.min_nd) else float(args.min_nd)),
                "weighted_vertical_collapse": "liquid-mass-weighted mean using rho_air * QL * dz over selected pressure range",
                "unweighted_vertical_collapse": "arithmetic mean over finite NDnet levels in selected pressure range",
                "rho_definition": "rho_air = p / (Rd * T), with p from NDnet/GiOcean lev in hPa and T from GiOcean T",
                "dz_definition": "approximate absolute layer thickness derived from GiOcean H on selected NDnet levels",
                "modis_track_definition": (
                    f"finite {MODIS_MASK_VAR}"
                    if args.min_cod is None or not np.isfinite(args.min_cod)
                    else f"finite {MODIS_MASK_VAR} and {MODIS_MASK_VAR} > {args.min_cod}"
                ),
                "cloud_top_temperature_threshold_K": 268.0,
                "cloud_fraction_threshold": 0.8,
                "cloud_fraction_variable": MODIS_CF_VAR,
                "sigmaW_definition": "sqrt(sum((w_i*sigma_i)^2))/sum(w_i) with w_i = rho*QL*dz over selected pressure range",
                "history": f"Created by {os.path.basename(__file__)}",
            },
        )

        for name in list(out.data_vars):
            if "time" not in out[name].dims:
                out[name] = out[name].expand_dims(time=out["time"])

        for mask_name in ["mask_no_mask", "mask_modis_track", "mask_modis_track_ctt", "mask_modis_track_ctt_cf"]:
            out[mask_name].attrs.update({"units": "1"})
        out["cloud_top_temperature"].attrs.update({"units": "K"})
        out["modis_cloud_optical_thickness_liquid_mean"].attrs.update({"units": "1"})
        out["modis_cloud_fraction_liquid_mean"].attrs.update({"units": "1"})
        out["air_density_sampled_mean"].attrs.update({"units": "kg m-3", "long_name": "mean air density over liquid-weighted valid levels"})

        return out


def _clean_attrs(attrs: Dict) -> Dict:
    clean = {}
    for key, val in attrs.items():
        if val is None:
            continue
        if isinstance(val, (str, int, float, np.integer, np.floating)):
            clean[key] = val.item() if hasattr(val, "item") else val
        elif isinstance(val, (list, tuple)) and all(isinstance(x, (str, int, float, np.integer, np.floating)) for x in val):
            clean[key] = [x.item() if hasattr(x, "item") else x for x in val]
        else:
            clean[key] = str(val)
    return clean


class NetcdfTimeSeriesWriter:
    """Append one xarray Dataset time slice at a time to one monthly NetCDF4 file."""

    def __init__(self, path: str, first_ds: xr.Dataset):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.nc = netCDF4.Dataset(path, "w", format="NETCDF4")
        self.i = 0
        self._init_file(first_ds)

    def _init_file(self, ds: xr.Dataset) -> None:
        self.nc.createDimension("time", None)
        self.nc.createDimension("lat", ds.sizes["lat"])
        self.nc.createDimension("lon", ds.sizes["lon"])

        tvar = self.nc.createVariable("time", "i8", ("time",))
        tvar.units = "days since 1970-01-01 00:00:00"
        tvar.calendar = "proleptic_gregorian"
        tvar.long_name = "time"

        lat_dtype = ds["lat"].dtype if ds["lat"].dtype.kind in "fiu" else np.float32
        lon_dtype = ds["lon"].dtype if ds["lon"].dtype.kind in "fiu" else np.float32
        latvar = self.nc.createVariable("lat", lat_dtype, ("lat",))
        lonvar = self.nc.createVariable("lon", lon_dtype, ("lon",))
        latvar[:] = ds["lat"].values
        lonvar[:] = ds["lon"].values
        latvar.setncatts(_clean_attrs(ds["lat"].attrs))
        lonvar.setncatts(_clean_attrs(ds["lon"].attrs))

        self.nc.setncatts(_clean_attrs(ds.attrs))
        self.nc.Conventions = "CF-1.8"

        for name in ds.data_vars:
            self._ensure_variable(name, ds[name])

    def _ensure_variable(self, name: str, da: xr.DataArray) -> None:
        if name in self.nc.variables:
            return
        dims = tuple(da.dims)
        for dim in dims:
            if dim not in self.nc.dimensions:
                raise ValueError(f"Variable {name} uses unsupported dimension {dim}; expected time/lat/lon")

        dtype = da.dtype
        kwargs = {"zlib": True, "complevel": 4}
        if dtype.kind == "f":
            fill_value = FILL_FLOAT
        elif dtype.kind in "iu":
            fill_value = None
        else:
            raise TypeError(f"Unsupported dtype for variable {name}: {dtype}")

        if fill_value is None:
            var = self.nc.createVariable(name, dtype, dims, **kwargs)
        else:
            var = self.nc.createVariable(name, dtype, dims, fill_value=fill_value, **kwargs)
        var.setncatts(_clean_attrs(da.attrs))

    @staticmethod
    def _time_to_days_since_1970(time_value) -> np.int64:
        day64 = np.asarray(time_value).astype("datetime64[D]")
        return (day64 - np.datetime64("1970-01-01", "D")).astype("timedelta64[D]").astype(np.int64)

    def append(self, ds: xr.Dataset) -> None:
        for name in ds.data_vars:
            self._ensure_variable(name, ds[name])

        tval = ds["time"].values[0]
        self.nc.variables["time"][self.i] = self._time_to_days_since_1970(tval)

        for name in ds.data_vars:
            var = self.nc.variables[name]
            da = ds[name]
            da_t = da.transpose(*var.dimensions)
            values = da_t.values
            if "time" in var.dimensions:
                slicer = [slice(None)] * values.ndim
                t_axis = var.dimensions.index("time")
                slicer[t_axis] = 0
                values_one_time = values[tuple(slicer)]
                out_slicer = [slice(None)] * len(var.dimensions)
                out_slicer[t_axis] = self.i
                var[tuple(out_slicer)] = values_one_time
            else:
                var[:] = values

        self.i += 1
        self.nc.sync()

    def close(self) -> None:
        self.nc.n_time_written = self.i
        self.nc.close()


def existing_month_is_complete(path: str, expected_days: int) -> tuple[bool, int]:
    """Return (complete, n_time) for an existing monthly output file."""
    if not os.path.exists(path):
        return False, 0
    try:
        with netCDF4.Dataset(path, "r") as nc:
            if "time" not in nc.dimensions:
                return False, 0
            n_time = len(nc.dimensions["time"])
            return n_time >= expected_days, n_time
    except Exception:
        return False, 0


def group_files_by_month(paths: list[str]) -> Dict[str, list[str]]:
    grouped: Dict[str, list[str]] = {}
    for fp in paths:
        day = infer_date_from_path(fp)
        yyyymm = day.strftime("%Y%m")
        grouped.setdefault(yyyymm, []).append(fp)
    return {k: sorted(v) for k, v in sorted(grouped.items())}


def process_one_month(
    yyyymm: str,
    month_files: list[str],
    args: argparse.Namespace,
    imon: int = 1,
    nmonths: int = 1,
) -> str:
    """Process one YYYYMM group and write exactly one monthly NetCDF file."""
    cache = MonthCache(args.gio_atm_dir, args.gio_sfc_dir, args.modis_mask_dir)
    out_path = resolve_month_out_file(args, yyyymm)
    expected_days = len(month_files)

    try:
        if os.path.exists(out_path):
            if args.overwrite:
                os.remove(out_path)
            else:
                complete, n_time = existing_month_is_complete(out_path, expected_days)
                if complete:
                    return f"[SKIP] month {imon:03d}/{nmonths:03d} {yyyymm} {out_path} has {n_time}/{expected_days} days"
                print(
                    f"[REBUILD] month {imon:03d}/{nmonths:03d} {yyyymm} "
                    f"{out_path} has only {n_time}/{expected_days} days",
                    flush=True,
                )
                os.remove(out_path)

        writer: Optional[NetcdfTimeSeriesWriter] = None
        n_written = 0
        try:
            for n, fp in enumerate(month_files, start=1):
                ds_day = process_one_day(fp, cache, args)
                try:
                    if writer is None:
                        writer = NetcdfTimeSeriesWriter(out_path, ds_day)
                    writer.append(ds_day)
                    n_written += 1
                    if args.verbose_days:
                        day_string = pd.to_datetime(ds_day["time"].values[0]).strftime("%Y-%m-%d")
                        print(
                            f"[OK] month {imon:03d}/{nmonths:03d} "
                            f"{yyyymm} day {n:02d}/{expected_days:02d} appended {day_string}",
                            flush=True,
                        )
                finally:
                    ds_day.close()
        finally:
            if writer is not None:
                writer.close()

        return f"[DONE] month {imon:03d}/{nmonths:03d} {yyyymm} wrote {n_written}/{expected_days} days -> {out_path}"
    finally:
        cache.close()


def _process_one_month_worker(payload: tuple[str, list[str], dict, int, int]) -> str:
    """Pickle-friendly multiprocessing wrapper."""
    yyyymm, month_files, args_dict, imon, nmonths = payload
    args = argparse.Namespace(**args_dict)
    return process_one_month(yyyymm, month_files, args, imon=imon, nmonths=nmonths)


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    ndnet_files = sorted(glob.glob(os.path.join(args.ndnet_dir, args.pattern)))
    if not ndnet_files:
        raise FileNotFoundError(f"No NDnet files found in {args.ndnet_dir} with pattern {args.pattern}")

    month_to_files = group_files_by_month(ndnet_files)
    month_to_files = {
        yyyymm: files
        for yyyymm, files in month_to_files.items()
        if month_is_in_requested_range(yyyymm, args)
    }
    if not month_to_files:
        raise FileNotFoundError("No NDnet files remain after applying --start-month/--end-month filters")

    month_items = list(month_to_files.items())
    nmonths = len(month_items)
    workers = min(args.workers, nmonths)

    print(
        f"[START] processing {nmonths} month(s) with {workers} worker(s); "
        f"output directory: {args.out_dir}",
        flush=True,
    )

    if workers == 1:
        for imon, (yyyymm, month_files) in enumerate(month_items, start=1):
            print(process_one_month(yyyymm, month_files, args, imon=imon, nmonths=nmonths), flush=True)
        return

    # Each worker receives only paths and simple scalar args, opens its own inputs,
    # and writes a different monthly NetCDF file. Do not parallelize by day because
    # that would require multiple processes writing/appending to the same file.
    args_dict = vars(args).copy()
    payloads = [
        (yyyymm, files, args_dict, imon, nmonths)
        for imon, (yyyymm, files) in enumerate(month_items, start=1)
    ]

    with cf.ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_month = {
            executor.submit(_process_one_month_worker, payload): payload[0]
            for payload in payloads
        }
        for fut in cf.as_completed(future_to_month):
            yyyymm = future_to_month[fut]
            try:
                print(fut.result(), flush=True)
            except Exception as exc:
                for pending in future_to_month:
                    pending.cancel()
                raise RuntimeError(f"Failed while processing month {yyyymm}") from exc


if __name__ == "__main__":
    main()
