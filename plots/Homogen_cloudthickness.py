#!/usr/bin/env python3
"""
Daily cloud-thickness Nd heterogeneity using variable pressure bounds.

For each daily NDnet grid column, define
    Pbottom = lowermost pressure level with Nd > ND_THRESHOLD_CM3
and use two alternative cloud-top pressures:
    row 1: Ptop = uppermost pressure level with Nd > ND_THRESHOLD_CM3
    row 2: Ptop = MODIS Cloud_Top_Pressure_Mean

Within [Ptop, Pbottom], all moments are integrated in ln(p), requiring at
least three pressure nodes per integral:
    <Nd> = integral Nd dlnp / Delta lnp
    V = integral (Nd - <Nd>)^2 dlnp / Delta lnp
    H = sqrt(V) / <Nd>
    R = sqrt(V) / sqrt(<sigmaNd^2>)

The 2 x 3 journal figure shows, for each Ptop definition:
    median H (%), percent of valid days with H > H_THRESHOLD,
    and median R.

Daily H/R are written to annual intermediates so exact temporal medians can
be calculated in spatial tiles without loading the full multi-year cube.

MODIS source follows inference/config.py and inference/data_handling.py:
    /panfs/ccds02/nobackup/people/dbarahon/NDnet/data/dailies/
      MCD06COSP_D3_MODIS/on_Noutgrid/MCD06COSP_D3_MODIS.*YYYYMM*.nc
Variable:
    Cloud_Top_Pressure_Mean

Examples
--------
    python Homogen_cloudthickness.py compute
    python Homogen_cloudthickness.py plot
    python Homogen_cloudthickness.py all

Optional:
    python Homogen_cloudthickness.py all --nd-threshold 1 --h-threshold 0.3
"""

import argparse
import glob
import os
import re
import string
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date

import numpy as np

try:
    from netCDF4 import Dataset, num2date
except ImportError as exc:
    raise ImportError(
        "This script requires netCDF4. Install/load it in the environment, "
        "for example: conda install netcdf4"
    ) from exc

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


# ============================================================
# CONFIGURATION
# ============================================================

ND_INPUT_GLOB = (
    "/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/inference/"
    "after_refinement/taper/NDnet_infer_daily/NDnet_infer_*.nc4"
)

MODIS_DIR = (
    "/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/dailies/"
    "MCD06COSP_D3_MODIS/on_Noutgrid"
)
MODIS_SOURCE = "MCD06COSP_D3_MODIS"
MODIS_VAR = "Cloud_Top_Pressure_Mean"

ND_VAR = "NDinc_cf01p"
SIGMA_VAR = "sigmaNd"

TIME_DIM = "time"
LEV_DIM = "lev"
LAT_DIM = "lat"
LON_DIM = "lon"

ND_THRESHOLD_CM3 = 1.0
MIN_LAYER_MEAN_ND = 1.0
MIN_SIGMA_RMS = 1.0e-6
H_THRESHOLD = 0.30
MIN_INTEGRATION_LEVELS = 3

ANNUAL_OUT_DIR = "H_cloudthickness_annual"
ANNUAL_PREFIX = "Nd_cloudthickness_HR"
OVERWRITE_ANNUAL = False

N_WORKERS = 4
NETCDF_COMPLEVEL = 4
NC_CHUNK_LAT = 45
NC_CHUNK_LON = 90
TILE_LAT = 45
TILE_LON = 90
MIN_VALID_DAYS = 30

OUT_PNG = "Nd_cloudthickness_heterogeneity.png"

CMAP_H = "turbo"
CMAP_FREQ = "turbo"
CMAP_R = "turbo"

H_COLOR_MIN = 0.0
H_COLOR_MAX = 50.0
FREQ_COLOR_MIN = 0.0
FREQ_COLOR_MAX = 100.0
R_COLOR_MIN = 0.0
R_COLOR_MAX = 2.0

H_CBAR_TICKS = [0, 10, 20, 30, 40, 50]
FREQ_CBAR_TICKS = [0, 20, 40, 60, 80, 100]
R_CBAR_TICKS = [0, 0.5, 1, 1.5, 2]

DPI = 150
LON_TICKS = [-120, -60, 0, 60, 120]
LAT_TICKS = [-60, -30, 0, 30, 60]

DATE_RE = re.compile(r"(\d{8})")
EPOCH = date(1970, 1, 1)

TOP_METHODS = ("nd", "modis")
TOP_METHOD_LABELS = (
    r"$P_{\rm top}$ from $N_{\rm d}$",
    r"$P_{\rm top}$ from MODIS",
)

ALGORITHM_VERSION = 2


# ============================================================
# SMALL UTILITIES
# ============================================================

def _quiet_path(path):
    try:
        return os.path.relpath(path)
    except Exception:
        return path


def _expand_files(pattern):
    pattern = os.path.expandvars(os.path.expanduser(pattern))
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for:\n  {pattern}")
    return files


def _date_from_filename(path):
    matches = DATE_RE.findall(os.path.basename(path))
    if not matches:
        raise ValueError(f"Could not find YYYYMMDD date in filename: {path}")
    stamp = matches[-1]
    return date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))


def _group_files_by_year(files):
    grouped = {}
    for path in files:
        day = _date_from_filename(path)
        grouped.setdefault(day.year, []).append(path)
    for year in grouped:
        grouped[year] = sorted(grouped[year], key=_date_from_filename)
    return grouped


def _as_float_array(data):
    if np.ma.isMaskedArray(data):
        data = np.ma.filled(data, np.nan)
    return np.asarray(data, dtype=np.float32)


def _filled_float32(data):
    return _as_float_array(data)


def _pressure_to_hpa(p):
    p = np.asarray(p, dtype=np.float64)
    finite = p[np.isfinite(p)]
    if finite.size == 0:
        raise ValueError("Pressure coordinate contains no finite values.")
    if np.nanmedian(finite) > 2000.0:
        p = p / 100.0
    return p


def _read_lev_lat_lon_variable(nc, var_name):
    if var_name not in nc.variables:
        raise KeyError(
            f"Variable '{var_name}' not found in {nc.filepath()}; "
            f"available={list(nc.variables)}"
        )

    var = nc.variables[var_name]
    dims = list(var.dimensions)
    required = {LEV_DIM, LAT_DIM, LON_DIM}
    if not required.issubset(dims):
        raise ValueError(
            f"{var_name} must contain {sorted(required)}; found dimensions {dims}"
        )

    index = []
    remaining = []
    for dim in dims:
        if dim == TIME_DIM:
            index.append(0)
        else:
            index.append(slice(None))
            remaining.append(dim)

    arr = _as_float_array(var[tuple(index)])
    perm = [remaining.index(dim) for dim in (LEV_DIM, LAT_DIM, LON_DIM)]
    return np.transpose(arr, axes=perm)


def _read_2d_time_slice(nc, var_name, time_index):
    if var_name not in nc.variables:
        raise KeyError(
            f"Variable '{var_name}' not found in {nc.filepath()}; "
            f"available={list(nc.variables)}"
        )

    var = nc.variables[var_name]
    dims = list(var.dimensions)

    index = []
    remaining = []
    for dim in dims:
        if dim == TIME_DIM:
            index.append(time_index)
        else:
            index.append(slice(None))
            remaining.append(dim)

    arr = _as_float_array(var[tuple(index)])
    target = [LAT_DIM, LON_DIM]
    if not set(target).issubset(remaining):
        raise ValueError(
            f"{var_name} must contain lat/lon after time indexing; found {dims}"
        )

    arr = np.transpose(arr, axes=[remaining.index(d) for d in target])
    return arr


def _normalize_modis_ctp_hpa(ctp):
    ctp = np.asarray(ctp, dtype=np.float32)
    finite = ctp[np.isfinite(ctp)]
    if finite.size and np.nanmedian(finite) > 2000.0:
        ctp = ctp / 100.0
    ctp[(ctp <= 0.0) | (ctp > 1100.0)] = np.nan
    return ctp


def _modis_month_file(day):
    ym = day.strftime("%Y%m")
    pattern = os.path.join(
        MODIS_DIR, f"{MODIS_SOURCE}.*{ym}*.nc"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No MODIS monthly file found for {day.isoformat()} with:\n  {pattern}"
        )
    if len(files) > 1:
        exact = [f for f in files if ym in os.path.basename(f)]
        if len(exact) == 1:
            return exact[0]
        raise RuntimeError(
            f"Ambiguous MODIS files for {ym}: " + ", ".join(files)
        )
    return files[0]


def _time_index_for_day(nc, day):
    if TIME_DIM not in nc.variables:
        raise KeyError(f"No time variable in MODIS file {nc.filepath()}")

    tvar = nc.variables[TIME_DIM]
    units = getattr(tvar, "units", None)
    calendar = getattr(tvar, "calendar", "standard")

    if units:
        vals = np.asarray(tvar[:])
        dates = num2date(
            vals,
            units=units,
            calendar=calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        for i, dt in enumerate(np.atleast_1d(dates)):
            if (dt.year, dt.month, dt.day) == (day.year, day.month, day.day):
                return i

    nt = len(nc.dimensions[TIME_DIM])
    if nt == 1:
        return 0
    if nt in (28, 29, 30, 31) and 1 <= day.day <= nt:
        return day.day - 1

    raise ValueError(
        f"Could not resolve {day.isoformat()} in MODIS time coordinate "
        f"of {nc.filepath()}"
    )


def _open_modis_month(day, expected_lat, expected_lon):
    """Open one MODIS monthly file and validate its grid once."""
    path = _modis_month_file(day)
    nc = Dataset(_quiet_path(path), "r")

    try:
        lat = np.asarray(nc.variables[LAT_DIM][:], dtype=np.float64)
        lon = np.asarray(nc.variables[LON_DIM][:], dtype=np.float64)

        if lat.shape != np.shape(expected_lat) or not np.allclose(
            lat, expected_lat, rtol=0.0, atol=1.0e-6
        ):
            raise ValueError(
                f"MODIS and NDnet latitude coordinates differ in {path}"
            )

        if lon.shape != np.shape(expected_lon) or not np.allclose(
            lon, expected_lon, rtol=0.0, atol=1.0e-6
        ):
            raise ValueError(
                f"MODIS and NDnet longitude coordinates differ in {path}"
            )

        return nc
    except Exception:
        nc.close()
        raise


def _read_modis_ctp_from_open(nc, day, expected_shape):
    """Read one daily CTP field from an already-open monthly MODIS file."""
    it = _time_index_for_day(nc, day)
    ctp = _read_2d_time_slice(nc, MODIS_VAR, it)

    if ctp.shape != expected_shape:
        raise ValueError(
            f"MODIS {MODIS_VAR} shape {ctp.shape} does not match "
            f"NDnet grid {expected_shape} in {nc.filepath()}"
        )

    return _normalize_modis_ctp_hpa(ctp)


def _read_modis_ctp(day, expected_lat, expected_lon):
    """Convenience one-shot reader; year processing keeps each month open."""
    nc = _open_modis_month(day, expected_lat, expected_lon)
    try:
        return _read_modis_ctp_from_open(
            nc, day, (len(expected_lat), len(expected_lon))
        )
    finally:
        nc.close()


# ============================================================
# DAILY VARIABLE-BOUND DIAGNOSTICS
# ============================================================

def _cloud_bounds_from_nd(nd, p_hpa, nd_threshold):
    """
    Return Ptop/Pbottom from the uppermost/lowermost model levels satisfying
    Nd > threshold. Since pressure increases downward, uppermost=min(p) and
    lowermost=max(p).
    """
    p = np.asarray(p_hpa, dtype=np.float64)
    order = np.argsort(p)
    p_sorted = p[order]
    nd_sorted = np.asarray(nd, dtype=np.float64)[order]

    cloudy = np.isfinite(nd_sorted) & (nd_sorted > nd_threshold)
    any_cloud = cloudy.any(axis=0)

    first = np.argmax(cloudy, axis=0)
    last = cloudy.shape[0] - 1 - np.argmax(cloudy[::-1], axis=0)

    ptop = np.full(any_cloud.shape, np.nan, dtype=np.float32)
    pbottom = np.full(any_cloud.shape, np.nan, dtype=np.float32)
    ptop[any_cloud] = p_sorted[first[any_cloud]].astype(np.float32)
    pbottom[any_cloud] = p_sorted[last[any_cloud]].astype(np.float32)
    return ptop, pbottom


def _compute_variable_bound_H_R(
    nd,
    sigma,
    p_hpa,
    ptop,
    pbottom,
    nd_threshold,
):
    """
    Vectorized H and R for per-column [Ptop, Pbottom].

    This is mathematically equivalent to constructing the pressure nodes
    [Ptop, interior model levels, Pbottom] separately for every grid column,
    but avoids the extremely expensive Python loop over columns.

    Boundary Nd and sigmaNd are linearly interpolated in ln(p). Trapezoidal
    moments are then assembled from:
      1) the partial top segment,
      2) cumulative full model-level segments, and
      3) the partial bottom segment.

    At least MIN_INTEGRATION_LEVELS pressure nodes are required. All selected
    Nd nodes must be finite and > nd_threshold. R additionally requires finite
    nonnegative sigmaNd at the same nodes.
    """
    p = np.asarray(p_hpa, dtype=np.float64)
    order = np.argsort(p)
    p = p[order]
    x = np.log(p)

    nd = np.asarray(nd, dtype=np.float64)[order]
    sigma = np.asarray(sigma, dtype=np.float64)[order]

    nz, ny, nx = nd.shape
    ncol = ny * nx
    cols = np.arange(ncol)

    nd2d = nd.reshape(nz, ncol)
    sig2d = sigma.reshape(nz, ncol)
    top = np.asarray(ptop, dtype=np.float64).reshape(ncol)
    bottom = np.asarray(pbottom, dtype=np.float64).reshape(ncol)

    H = np.full(ncol, np.nan, dtype=np.float32)
    R = np.full(ncol, np.nan, dtype=np.float32)

    base_valid = (
        np.isfinite(top)
        & np.isfinite(bottom)
        & (top < bottom)
        & (top >= p[0])
        & (bottom <= p[-1])
    )
    if not np.any(base_valid):
        return H.reshape(ny, nx), R.reshape(ny, nx)

    # Safe placeholders keep all vectorized index operations in bounds.
    top_safe = np.where(base_valid, top, p[0])
    bottom_safe = np.where(base_valid, bottom, p[-1])
    x_top = np.log(top_safe)
    x_bottom = np.log(bottom_safe)

    def _interp_at(field, target, x_target):
        lo = np.searchsorted(p, target, side="right") - 1
        lo = np.clip(lo, 0, nz - 2)
        hi = lo + 1
        alpha = (x_target - x[lo]) / (x[hi] - x[lo])
        value = (
            (1.0 - alpha) * field[lo, cols]
            + alpha * field[hi, cols]
        )
        return value

    nd_top = _interp_at(nd2d, top_safe, x_top)
    nd_bottom = _interp_at(nd2d, bottom_safe, x_bottom)
    sig_top = _interp_at(sig2d, top_safe, x_top)
    sig_bottom = _interp_at(sig2d, bottom_safe, x_bottom)

    # Strictly interior model levels. The conceptual integration nodes are:
    # [Ptop, p[i0], ..., p[i1-1], Pbottom].
    i0 = np.searchsorted(p, top_safe, side="right")
    i1 = np.searchsorted(p, bottom_safe, side="left")
    n_nodes = 2 + (i1 - i0)

    structural_valid = (
        base_valid
        & (n_nodes >= MIN_INTEGRATION_LEVELS)
        & (i0 < i1)
        & (i0 < nz)
        & (i1 > 0)
    )

    # Safe gather indices for columns later masked as invalid.
    i0_safe = np.clip(i0, 0, nz - 1)
    ilast_safe = np.clip(i1 - 1, 0, nz - 1)

    # Require every interior Nd node to be finite and above threshold.
    bad_nd_node = (~np.isfinite(nd2d)) | (nd2d <= nd_threshold)
    bad_nd_prefix = np.vstack(
        [
            np.zeros((1, ncol), dtype=np.int16),
            np.cumsum(bad_nd_node, axis=0, dtype=np.int16),
        ]
    )
    interior_bad_nd = (
        bad_nd_prefix[np.clip(i1, 0, nz), cols]
        - bad_nd_prefix[np.clip(i0, 0, nz), cols]
    )

    valid_nd = (
        structural_valid
        & np.isfinite(nd_top)
        & (nd_top > nd_threshold)
        & np.isfinite(nd_bottom)
        & (nd_bottom > nd_threshold)
        & (interior_bad_nd == 0)
    )

    # R uses the same pressure nodes and additionally requires valid sigmaNd.
    bad_sig_node = (~np.isfinite(sig2d)) | (sig2d < 0.0)
    bad_sig_prefix = np.vstack(
        [
            np.zeros((1, ncol), dtype=np.int16),
            np.cumsum(bad_sig_node, axis=0, dtype=np.int16),
        ]
    )
    interior_bad_sig = (
        bad_sig_prefix[np.clip(i1, 0, nz), cols]
        - bad_sig_prefix[np.clip(i0, 0, nz), cols]
    )
    valid_sigma = (
        structural_valid
        & np.isfinite(sig_top)
        & (sig_top >= 0.0)
        & np.isfinite(sig_bottom)
        & (sig_bottom >= 0.0)
        & (interior_bad_sig == 0)
    )

    dx = np.diff(x)[:, None]

    def _node_prefix(q):
        """Integral from p[0] to every model node using full trapezoids."""
        seg = 0.5 * (q[:-1] + q[1:]) * dx
        # Invalid segments are harmless because any interval containing them
        # is rejected by the strict node-validity masks above.
        seg = np.where(np.isfinite(seg), seg, 0.0)
        return np.vstack(
            [
                np.zeros((1, ncol), dtype=np.float64),
                np.cumsum(seg, axis=0, dtype=np.float64),
            ]
        )

    nd_prefix = _node_prefix(nd2d)
    nd2_prefix = _node_prefix(nd2d * nd2d)
    sig2_prefix = _node_prefix(sig2d * sig2d)

    def _interval_integral(prefix, q, q_top, q_bottom):
        """
        Exact trapezoidal integral on the conceptual variable node set.

        Splitting the two boundary segments explicitly is important: for the
        second moments the interpolated boundary value must be squared *after*
        interpolation, matching the original per-column implementation.
        """
        first = (
            0.5
            * (q_top + q[i0_safe, cols])
            * (x[i0_safe] - x_top)
        )
        last = (
            0.5
            * (q[ilast_safe, cols] + q_bottom)
            * (x_bottom - x[ilast_safe])
        )
        middle = (
            prefix[ilast_safe, cols]
            - prefix[i0_safe, cols]
        )
        return first + middle + last

    int_nd = _interval_integral(
        nd_prefix, nd2d, nd_top, nd_bottom
    )
    int_nd2 = _interval_integral(
        nd2_prefix, nd2d * nd2d, nd_top * nd_top, nd_bottom * nd_bottom
    )
    int_sig2 = _interval_integral(
        sig2_prefix, sig2d * sig2d, sig_top * sig_top, sig_bottom * sig_bottom
    )

    span = x_bottom - x_top

    nd_mean = int_nd / span
    nd_second = int_nd2 / span
    variance = np.maximum(nd_second - nd_mean * nd_mean, 0.0)
    vertical_rms = np.sqrt(variance)

    valid_H = (
        valid_nd
        & np.isfinite(nd_mean)
        & (nd_mean >= MIN_LAYER_MEAN_ND)
        & np.isfinite(vertical_rms)
    )
    H[valid_H] = (
        vertical_rms[valid_H] / nd_mean[valid_H]
    ).astype(np.float32)

    sigma_second = int_sig2 / span
    sigma_rms = np.sqrt(np.maximum(sigma_second, 0.0))

    valid_R = (
        valid_H
        & valid_sigma
        & np.isfinite(sigma_rms)
        & (sigma_rms > MIN_SIGMA_RMS)
    )
    R[valid_R] = (
        vertical_rms[valid_R] / sigma_rms[valid_R]
    ).astype(np.float32)

    return H.reshape(ny, nx), R.reshape(ny, nx)


# ============================================================
# ANNUAL OUTPUT
# ============================================================

def _create_annual_output(path, year, lat, lon, nd_threshold, h_threshold):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    nc = Dataset(_quiet_path(path), "w", format="NETCDF4")

    nlat = len(lat)
    nlon = len(lon)

    nc.createDimension(TIME_DIM, None)
    nc.createDimension("top_method", 2)
    nc.createDimension(LAT_DIM, nlat)
    nc.createDimension(LON_DIM, nlon)

    tvar = nc.createVariable(TIME_DIM, "i4", (TIME_DIM,))
    tvar.units = "days since 1970-01-01 00:00:00"
    tvar.calendar = "proleptic_gregorian"

    mvar = nc.createVariable("top_method", "i1", ("top_method",))
    mvar[:] = np.arange(2, dtype=np.int8)
    mvar.long_name = "cloud-top pressure definition"
    mvar.method_0 = "uppermost NDnet pressure level with Nd > threshold"
    mvar.method_1 = "MODIS Cloud_Top_Pressure_Mean"

    latvar = nc.createVariable(LAT_DIM, "f8", (LAT_DIM,))
    lonvar = nc.createVariable(LON_DIM, "f8", (LON_DIM,))
    latvar[:] = lat
    lonvar[:] = lon
    latvar.units = "degrees_north"
    lonvar.units = "degrees_east"

    chunks = (
        1,
        1,
        min(NC_CHUNK_LAT, nlat),
        min(NC_CHUNK_LON, nlon),
    )
    kwargs = dict(
        zlib=True,
        complevel=NETCDF_COMPLEVEL,
        shuffle=True,
        chunksizes=chunks,
        fill_value=np.float32(np.nan),
    )

    hvar = nc.createVariable(
        "H", "f4", (TIME_DIM, "top_method", LAT_DIM, LON_DIM), **kwargs
    )
    hvar.units = "1"
    hvar.long_name = "relative vertical RMS heterogeneity of Nd over daily cloud thickness"
    hvar.definition = "sqrt(<(Nd-<Nd>)^2>_lnp) / <Nd>_lnp"

    rvar = nc.createVariable(
        "R", "f4", (TIME_DIM, "top_method", LAT_DIM, LON_DIM), **kwargs
    )
    rvar.units = "1"
    rvar.long_name = "daily cloud-thickness Nd heterogeneity relative to sigmaNd RMS"
    rvar.definition = "sqrt(<(Nd-<Nd>)^2>_lnp) / sqrt(<sigmaNd^2>_lnp)"

    pbvar = nc.createVariable(
        "Pbottom_nd", "f4", (TIME_DIM, LAT_DIM, LON_DIM),
        zlib=True,
        complevel=NETCDF_COMPLEVEL,
        shuffle=True,
        chunksizes=(1, min(NC_CHUNK_LAT, nlat), min(NC_CHUNK_LON, nlon)),
        fill_value=np.float32(np.nan),
    )
    pbvar.units = "hPa"
    pbvar.long_name = "lowermost pressure level with Nd above threshold"

    nc.setncattr("year", int(year))
    nc.setncattr("integration_coordinate", "ln(pressure)")
    nc.setncattr("boundary_interpolation", "linear in ln(pressure)")
    nc.setncattr("nd_threshold_cm-3", float(nd_threshold))
    nc.setncattr("h_exceedance_threshold", float(h_threshold))
    nc.setncattr("minimum_layer_mean_Nd_cm-3", float(MIN_LAYER_MEAN_ND))
    nc.setncattr("minimum_integration_levels", int(MIN_INTEGRATION_LEVELS))
    nc.setncattr("modis_variable", MODIS_VAR)
    nc.setncattr("modis_source", MODIS_SOURCE)
    nc.setncattr("algorithm_version", int(ALGORITHM_VERSION))

    return nc


def _annual_file_is_current(path, nd_threshold, h_threshold):
    if not os.path.exists(path):
        return False
    try:
        with Dataset(_quiet_path(path), "r") as nc:
            if "H" not in nc.variables or "R" not in nc.variables:
                return False
            if int(getattr(nc, "algorithm_version", -1)) != ALGORITHM_VERSION:
                return False
            if int(getattr(nc, "minimum_integration_levels", -1)) != MIN_INTEGRATION_LEVELS:
                return False
            if not np.isclose(
                float(getattr(nc, "nd_threshold_cm-3", np.nan)),
                nd_threshold,
                rtol=0.0,
                atol=1.0e-8,
            ):
                return False
            if not np.isclose(
                float(getattr(nc, "h_exceedance_threshold", np.nan)),
                h_threshold,
                rtol=0.0,
                atol=1.0e-8,
            ):
                return False
        return True
    except Exception:
        return False


def _process_one_year(year, files, nd_threshold, h_threshold):
    out_path = os.path.join(ANNUAL_OUT_DIR, f"{ANNUAL_PREFIX}_{year}.nc4")

    with Dataset(_quiet_path(files[0]), "r") as nc0:
        lat = np.asarray(nc0.variables[LAT_DIM][:], dtype=np.float64)
        lon = np.asarray(nc0.variables[LON_DIM][:], dtype=np.float64)
        p_hpa = _pressure_to_hpa(nc0.variables[LEV_DIM][:])

    if os.path.exists(out_path) and not OVERWRITE_ANNUAL:
        if _annual_file_is_current(out_path, nd_threshold, h_threshold):
            return f"[SKIP] {out_path} already current"

    tmp_path = out_path + f".tmp.{os.getpid()}"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    out = _create_annual_output(
        tmp_path, year, lat, lon, nd_threshold, h_threshold
    )

    modis_nc = None
    modis_ym = None

    try:
        for it, infile in enumerate(files):
            day = _date_from_filename(infile)

            with Dataset(_quiet_path(infile), "r") as nc:
                nd = _read_lev_lat_lon_variable(nc, ND_VAR)
                sigma = _read_lev_lat_lon_variable(nc, SIGMA_VAR)

            if nd.shape != (len(p_hpa), len(lat), len(lon)):
                raise ValueError(f"Unexpected {ND_VAR} shape in {infile}: {nd.shape}")
            if sigma.shape != nd.shape:
                raise ValueError(
                    f"{SIGMA_VAR} shape {sigma.shape} does not match {ND_VAR} "
                    f"shape {nd.shape} in {infile}"
                )

            ptop_nd, pbottom = _cloud_bounds_from_nd(
                nd, p_hpa, nd_threshold
            )

            ym = day.strftime("%Y%m")
            if ym != modis_ym:
                if modis_nc is not None:
                    modis_nc.close()
                modis_nc = _open_modis_month(day, lat, lon)
                modis_ym = ym

            ptop_modis = _read_modis_ctp_from_open(
                modis_nc, day, (len(lat), len(lon))
            )

            H_nd, R_nd = _compute_variable_bound_H_R(
                nd, sigma, p_hpa, ptop_nd, pbottom, nd_threshold
            )
            H_modis, R_modis = _compute_variable_bound_H_R(
                nd, sigma, p_hpa, ptop_modis, pbottom, nd_threshold
            )

            out.variables[TIME_DIM][it] = (day - EPOCH).days
            out.variables["H"][it, 0, :, :] = H_nd
            out.variables["H"][it, 1, :, :] = H_modis
            out.variables["R"][it, 0, :, :] = R_nd
            out.variables["R"][it, 1, :, :] = R_modis
            out.variables["Pbottom_nd"][it, :, :] = pbottom

            if (it + 1) % 30 == 0 or (it + 1) == len(files):
                print(f"[{year}] {it + 1:4d}/{len(files):4d} days", flush=True)

        out.sync()
    finally:
        if modis_nc is not None:
            modis_nc.close()
        out.close()

    os.replace(tmp_path, out_path)
    return f"[WROTE] {out_path}"


def compute_all_years(nd_threshold, h_threshold, workers=N_WORKERS):
    files = _expand_files(ND_INPUT_GLOB)
    grouped = _group_files_by_year(files)

    print(f"Found {len(files)} NDnet daily files spanning {len(grouped)} years.")
    print(
        f"Pbottom/Ptop(Nd): Nd > {nd_threshold:g} cm^-3; "
        f"H exceedance threshold={h_threshold:g} ({100*h_threshold:g}%); "
        f"minimum integration levels={MIN_INTEGRATION_LEVELS}."
    )

    years = sorted(grouped)
    workers = max(1, min(int(workers), len(years)))

    if workers == 1:
        for year in years:
            print(
                _process_one_year(
                    year, grouped[year], nd_threshold, h_threshold
                ),
                flush=True,
            )
        return

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _process_one_year,
                year,
                grouped[year],
                nd_threshold,
                h_threshold,
            ): year
            for year in years
        }
        for fut in as_completed(futures):
            year = futures[fut]
            try:
                print(fut.result(), flush=True)
            except Exception as exc:
                raise RuntimeError(f"Failed while processing year {year}") from exc


# ============================================================
# TILED CLIMATOLOGY
# ============================================================

def _annual_files():
    return _expand_files(
        os.path.join(ANNUAL_OUT_DIR, f"{ANNUAL_PREFIX}_*.nc4")
    )


def _load_time_tile(datasets, var_name, method, yslice, xslice, ntime):
    ny = yslice.stop - yslice.start
    nx = xslice.stop - xslice.start
    block = np.empty((ntime, ny, nx), dtype=np.float32)

    t0 = 0
    for nc in datasets:
        nt = len(nc.dimensions[TIME_DIM])
        block[t0:t0 + nt] = _filled_float32(
            nc.variables[var_name][:, method, yslice, xslice]
        )
        t0 += nt
    return block


def calculate_climatology(h_threshold):
    files = _annual_files()
    datasets = [Dataset(_quiet_path(f), "r") for f in files]

    try:
        first = datasets[0]
        lat = np.asarray(first.variables[LAT_DIM][:], dtype=np.float64)
        lon = np.asarray(first.variables[LON_DIM][:], dtype=np.float64)

        ntime = sum(len(nc.dimensions[TIME_DIM]) for nc in datasets)
        nlat = len(lat)
        nlon = len(lon)

        med_H = np.full((2, nlat, nlon), np.nan, dtype=np.float32)
        pct_H_gt = np.full_like(med_H, np.nan)
        med_R = np.full_like(med_H, np.nan)

        for method in range(2):
            print(f"Climatology: {TOP_METHODS[method]}")
            for y0 in range(0, nlat, TILE_LAT):
                y1 = min(y0 + TILE_LAT, nlat)
                ys = slice(y0, y1)

                for x0 in range(0, nlon, TILE_LON):
                    x1 = min(x0 + TILE_LON, nlon)
                    xs = slice(x0, x1)

                    hblock = _load_time_tile(
                        datasets, "H", method, ys, xs, ntime
                    )
                    valid = np.isfinite(hblock)
                    nvalid = valid.sum(axis=0)
                    nexceed = ((hblock > h_threshold) & valid).sum(axis=0)
                    good = nvalid >= MIN_VALID_DAYS

                    pct = np.full(nvalid.shape, np.nan, dtype=np.float32)
                    pct[good] = (
                        100.0 * nexceed[good] / nvalid[good]
                    ).astype(np.float32)
                    pct_H_gt[method, ys, xs] = pct

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        hmed = np.nanmedian(
                            hblock, axis=0, overwrite_input=True
                        ).astype(np.float32)
                    hmed[~good] = np.nan
                    med_H[method, ys, xs] = 100.0 * hmed

                    rblock = _load_time_tile(
                        datasets, "R", method, ys, xs, ntime
                    )
                    rvalid = np.isfinite(rblock)
                    nrvalid = rvalid.sum(axis=0)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        rmed = np.nanmedian(
                            rblock, axis=0, overwrite_input=True
                        ).astype(np.float32)
                    rmed[nrvalid < MIN_VALID_DAYS] = np.nan
                    med_R[method, ys, xs] = rmed

                    del hblock, valid, nvalid, nexceed, pct, hmed
                    del rblock, rvalid, nrvalid, rmed

        return lat, lon, med_H, pct_H_gt, med_R
    finally:
        for nc in datasets:
            nc.close()


# ============================================================
# JOURNAL-STYLE PLOTTING
# ============================================================

def _to_minus180_180(lon, *fields):
    lon = np.asarray(lon, dtype=np.float64)
    if lon.size and np.nanmax(lon) > 180.0:
        new_lon = ((lon + 180.0) % 360.0) - 180.0
        order = np.argsort(new_lon)
        return (new_lon[order], *[f[..., order] for f in fields])
    return (lon, *fields)


def _lat_weighted_global_mean(field, lat):
    field = np.asarray(field, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    weights = np.cos(np.deg2rad(lat))[:, None]
    valid = np.isfinite(field)
    denom = np.sum(np.where(valid, weights, 0.0))
    if denom <= 0.0:
        return np.nan
    return float(
        np.sum(np.where(valid, field * weights, 0.0)) / denom
    )


def plot_climatology(h_threshold, out_png=OUT_PNG):
    lat, lon, med_H, pct_H_gt, med_R = calculate_climatology(h_threshold)

    lon, med_H, pct_H_gt, med_R = _to_minus180_180(
        lon, med_H, pct_H_gt, med_R
    )

    proj = ccrs.PlateCarree()
    norms = [
        mcolors.Normalize(vmin=H_COLOR_MIN, vmax=H_COLOR_MAX),
        mcolors.Normalize(vmin=FREQ_COLOR_MIN, vmax=FREQ_COLOR_MAX),
        mcolors.Normalize(vmin=R_COLOR_MIN, vmax=R_COLOR_MAX),
    ]
    cmaps = [CMAP_H, CMAP_FREQ, CMAP_R]
    columns = [med_H, pct_H_gt, med_R]

    column_titles = [
        r"Median $H$",
        rf"Days with $H>{100.0*h_threshold:g}$%",
        r"Median $R$",
    ]
    right_formats = [
        lambda x: rf"$\langle H\rangle$ = {x:.1f}$\%$",
        lambda x: rf"$\langle f\rangle$ = {x:.1f}$\%$",
        lambda x: rf"$\langle R\rangle$ = {x:.2f}",
    ]

    fig = plt.figure(figsize=(11.6, 5.3))
    gs = GridSpec(
        nrows=3,
        ncols=3,
        figure=fig,
        height_ratios=[1.0, 1.0, 0.10],
        hspace=0.18,
        wspace=0.12,
        left=0.065,
        right=0.985,
        bottom=0.09,
        top=0.91,
    )

    axes = np.empty((2, 3), dtype=object)
    for row in range(2):
        for col in range(3):
            axes[row, col] = fig.add_subplot(gs[row, col], projection=proj)

    for col, title in enumerate(column_titles):
        axes[0, col].text(
            0.5,
            1.19,
            title,
            transform=axes[0, col].transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )

    letters = list(string.ascii_lowercase[:6])
    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            field = columns[col][row]

            ax.pcolormesh(
                lon,
                lat,
                field,
                transform=proj,
                shading="auto",
                cmap=cmaps[col],
                norm=norms[col],
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
                f"({letters[row * 3 + col]})  {TOP_METHOD_LABELS[row]}",
                loc="left",
                fontsize=9.5,
                pad=3,
            )

            gmean = _lat_weighted_global_mean(field, lat)
            ax.set_title(
                right_formats[col](gmean),
                loc="right",
                fontsize=9.5,
                pad=3,
            )

    cbar_ticks = [H_CBAR_TICKS, FREQ_CBAR_TICKS, R_CBAR_TICKS]
    cbar_labels = [
        r"$H$ (%)",
        rf"Valid days with $H>{100.0*h_threshold:g}$% (%)",
        r"$R$",
    ]

    for col in range(3):
        cax = fig.add_subplot(gs[2, col])
        sm = plt.cm.ScalarMappable(norm=norms[col], cmap=cmaps[col])
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_ticks(cbar_ticks[col])
        cbar.set_label(cbar_labels[col], fontsize=9, labelpad=2)
        cbar.ax.tick_params(labelsize=7, length=2.5, width=0.6, pad=1.5)

    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    print(f"[WROTE] {out_png}")

    out_pdf = os.path.splitext(out_png)[0] + ".pdf"
    if os.path.abspath(out_pdf) != os.path.abspath(out_png):
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"[WROTE] {out_pdf}")

    plt.close(fig)


# ============================================================
# COMMAND LINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute and plot daily cloud-thickness Nd heterogeneity using "
            "Nd-derived or MODIS cloud-top pressure."
        )
    )
    parser.add_argument(
        "mode",
        choices=("compute", "plot", "all"),
        help="compute annual daily H/R, plot climatology, or both",
    )
    parser.add_argument(
        "--nd-threshold",
        type=float,
        default=ND_THRESHOLD_CM3,
        help=(
            "Nd threshold (cm^-3) defining lowermost/uppermost cloudy model "
            f"levels (default: {ND_THRESHOLD_CM3:g})"
        ),
    )
    parser.add_argument(
        "--h-threshold",
        type=float,
        default=H_THRESHOLD,
        help=(
            "dimensionless H threshold used for exceedance fraction "
            f"(default: {H_THRESHOLD:g}, i.e. {100*H_THRESHOLD:g}%)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=N_WORKERS,
        help=f"number of year-level compute workers (default: {N_WORKERS})",
    )
    parser.add_argument(
        "--out",
        default=OUT_PNG,
        help=f"output PNG for plot mode (default: {OUT_PNG})",
    )
    args = parser.parse_args()

    if args.nd_threshold < 0.0:
        parser.error("--nd-threshold must be nonnegative")
    if args.h_threshold < 0.0:
        parser.error("--h-threshold must be nonnegative")

    if args.mode in ("compute", "all"):
        compute_all_years(
            nd_threshold=args.nd_threshold,
            h_threshold=args.h_threshold,
            workers=args.workers,
        )

    if args.mode in ("plot", "all"):
        plot_climatology(
            h_threshold=args.h_threshold,
            out_png=args.out,
        )


if __name__ == "__main__":
    main()
