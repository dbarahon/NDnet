#!/usr/bin/env python3
"""
Vertical Nd heterogeneity in ln(p), followed by journal-style climatology plots.

Input
-----
Daily NDnet inference files containing
    NDinc_cf01p(time, lev, lat, lon)
    sigmaNd(time, lev, lat, lon)
with one time step per file.

Diagnostics
-----------
For each day/grid column and for each configured map layer [p_top, p_bottom]:

    <Nd> = (1 / Delta ln p) integral Nd d ln p

    V = (1 / Delta ln p) integral (Nd - <Nd>)^2 d ln p

    H = 100 * sqrt(V) / <Nd>                         [%]

    R = sqrt(V) / sqrt(<sigmaNd^2>_lnp)              [-]

The three map rows use TOP_LEVELS_HPA / BOTTOM_LEVELS_HPA. The additional
zonal-mean row evaluates the same diagnostics cumulatively from 1000 hPa to a
variable layer-top pressure p. Its plotted p coordinate runs from 950 hPa to
200 hPa by default and is linear in pressure.

For the zonal row, each daily 3-D diagnostic is first averaged across longitude.
The plotted H and R are temporal medians of those daily zonal means. The plotted
f is the temporal mean of the daily zonal fraction of valid longitudes for which
H exceeds --threshold.

All vertical integrations use trapezoidal integration in ln(p). If an exact
boundary does not coincide with a model pressure level, Nd and sigmaNd are
linearly interpolated in ln(p) to that boundary.

Workflow
--------
1. Compute/rebuild annual intermediates:
       python Homogen.py compute --threshold 30

2. Plot climatology:
       python Homogen.py plot --threshold 30

3. Or do both:
       python Homogen.py all --threshold 30

The map climatology is evaluated in spatial tiles, allowing exact temporal
medians without loading the complete multi-year 4-D array into memory. Zonal
products are stored compactly in the same annual files as (time, pressure, lat).
Because zonal f depends on the H threshold, changing --threshold requires the
annual intermediates to be recomputed; compute/all detects and rebuilds stale
files automatically.
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
    from netCDF4 import Dataset
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
# GLOBAL CONFIGURATION
# ============================================================

# Daily refined NDnet files. Edit if your path differs.
INPUT_GLOB = (
    "/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/inference/"
    "after_refinement/taper/NDnet_infer_daily/NDnet_infer_2005*.nc4"
)

ND_VAR = "NDinc_cf01p"
SIGMA_VAR = "sigmaNd"

TIME_DIM = "time"
LEV_DIM = "lev"
LAT_DIM = "lat"
LON_DIM = "lon"

# Three fixed layers shown as the map rows. Entries are paired by index.
TOP_LEVELS_HPA = [800.0, 700.0, 600.0]
BOTTOM_LEVELS_HPA = [975.0, 975.0, 975.0]

# Zonal row: each plotted pressure p is the TOP of an integrated [p, 1000 hPa]
# layer. The y axis itself begins at 950 hPa, as requested, and is linear.
ZONAL_INTEGRATION_BOTTOM_HPA = 1000.0
ZONAL_PLOT_BOTTOM_HPA = 950.0
ZONAL_PLOT_TOP_HPA = 200.0
ZONAL_PRESSURE_TICKS = [950, 850, 700, 500, 300, 200]
ZONAL_LAT_TICKS = [-90, -60, -30, 0, 30, 60, 90]

# A daily layer is accepted only when Nd is finite at every integration node.
# This avoids silently integrating across missing vertical portions.
MIN_ND = 1.0                 # cm^-3 at every integration node
MIN_LAYER_MEAN_ND = 1.0      # cm^-3
MIN_SIGMA_RMS = 1.0e-6       # avoid division by zero in R

# Intermediate annual output. Fixed-layer map H/R plus compact zonal products.
ANNUAL_OUT_DIR = "H_lnp_annual"
ANNUAL_PREFIX = "Nd_vertical_H_lnp"
OVERWRITE_ANNUAL = False

# Parallelism is across years, so workers never write the same file.
# 4 is intentionally conservative for a shared parallel filesystem.
N_WORKERS = 4

# NetCDF compression/chunking. These chunks also align with plotting tiles.
NETCDF_COMPLEVEL = 4
NC_CHUNK_LAT = 45
NC_CHUNK_LON = 90

# Plotting tiles: exact map medians are calculated one spatial tile at a time.
TILE_LAT = 45
TILE_LON = 90
MIN_VALID_DAYS = 30

OUT_PNG = "Nd_vertical_heterogeneity_lnp.png"

# Journal-style plotting settings based on the project template.
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

ZONAL_LEVEL_DIM = "zonal_level"
ZONAL_LEVEL_VAR = "zonal_top_hpa"
ZONAL_H_VAR = "zonal_H"
ZONAL_F_VAR = "zonal_f"
ZONAL_R_VAR = "zonal_R"
ZONAL_THRESHOLD_ATTR = "zonal_exceedance_threshold_percent"


# ============================================================
# SMALL UTILITIES
# ============================================================

def _quiet_path(path):
    """Use a relative path where practical; this can suppress getfattr noise."""
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
    for f in files:
        d = _date_from_filename(f)
        grouped.setdefault(d.year, []).append(f)
    for year in grouped:
        grouped[year] = sorted(grouped[year], key=_date_from_filename)
    return grouped


def _as_float_array(data):
    """Convert netCDF4 masked arrays/fill values to float32 NaN arrays."""
    if np.ma.isMaskedArray(data):
        data = np.ma.filled(data, np.nan)
    return np.asarray(data, dtype=np.float32)


def _read_lev_lat_lon_variable(nc, var_name):
    """Read one time slice and return array ordered as (lev, lat, lon)."""
    if var_name not in nc.variables:
        available = ", ".join(nc.variables.keys())
        raise KeyError(
            f"Variable '{var_name}' not found in {nc.filepath()}.\n"
            f"Available variables: {available}"
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
    target = [LEV_DIM, LAT_DIM, LON_DIM]
    perm = [remaining.index(d) for d in target]
    return np.transpose(arr, axes=perm)


def _pressure_to_hpa(p):
    """Accept hPa or Pa pressure coordinates; return hPa."""
    p = np.asarray(p, dtype=np.float64)
    finite = p[np.isfinite(p)]
    if finite.size == 0:
        raise ValueError("Pressure coordinate contains no finite values.")
    if np.nanmedian(finite) > 2000.0:
        return p / 100.0
    return p


def _configured_layers():
    """Return validated (top, bottom) pressure pairs from the configuration."""
    tops = np.asarray(TOP_LEVELS_HPA, dtype=np.float64)
    bottoms = np.asarray(BOTTOM_LEVELS_HPA, dtype=np.float64)

    if tops.ndim != 1 or bottoms.ndim != 1 or tops.size == 0:
        raise ValueError(
            "TOP_LEVELS_HPA and BOTTOM_LEVELS_HPA must be non-empty 1-D arrays."
        )
    if tops.shape != bottoms.shape:
        raise ValueError(
            "TOP_LEVELS_HPA and BOTTOM_LEVELS_HPA must have the same length; "
            f"got {tops.size} tops and {bottoms.size} bottoms."
        )
    if np.any(~np.isfinite(tops)) or np.any(~np.isfinite(bottoms)):
        raise ValueError("All configured top and bottom pressure levels must be finite.")
    if np.any(tops >= bottoms):
        bad = np.where(tops >= bottoms)[0]
        details = ", ".join(
            f"row {i}: top={tops[i]:g}, bottom={bottoms[i]:g}" for i in bad
        )
        raise ValueError(f"Each layer must satisfy top < bottom; invalid {details}.")

    return list(zip(tops.tolist(), bottoms.tolist()))


def _nanmean_last_axis(field):
    """NaN-aware mean over the last axis without RuntimeWarning noise."""
    field = np.asarray(field)
    valid = np.isfinite(field)
    count = valid.sum(axis=-1)
    total = np.sum(np.where(valid, field, 0.0), axis=-1, dtype=np.float64)
    out = np.full(count.shape, np.nan, dtype=np.float32)
    good = count > 0
    out[good] = (total[good] / count[good]).astype(np.float32)
    return out, count


# ============================================================
# ln(p) LAYER GEOMETRY: FIXED MAP LAYERS
# ============================================================

def _trapezoid_mean_weights(x):
    """Normalized trapezoidal integration weights for arbitrary 1-D x."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2 or not np.all(np.diff(x) > 0.0):
        raise ValueError("Integration coordinate must be strictly increasing.")

    dx = np.diff(x)
    w = np.zeros_like(x)
    w[0] = 0.5 * dx[0]
    w[-1] = 0.5 * dx[-1]
    if x.size > 2:
        w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    w /= (x[-1] - x[0])
    return w


def _build_layer_spec(p_source_hpa, p_top_hpa, p_bottom_hpa):
    """Build exact fixed-layer nodes and interpolation information in ln(p)."""
    p_source_hpa = np.asarray(p_source_hpa, dtype=np.float64)

    if p_top_hpa >= p_bottom_hpa:
        raise ValueError("p_top_hpa must be less than p_bottom_hpa.")
    if np.any(~np.isfinite(p_source_hpa)) or np.any(p_source_hpa <= 0.0):
        raise ValueError("Pressure coordinate must be finite and positive.")

    order = np.argsort(p_source_hpa)
    p_sorted = p_source_hpa[order]

    if np.any(np.diff(p_sorted) <= 0.0):
        raise ValueError("Pressure coordinate contains duplicate levels.")

    if p_top_hpa < p_sorted[0] or p_bottom_hpa > p_sorted[-1]:
        raise ValueError(
            f"Requested layer {p_bottom_hpa:g}-{p_top_hpa:g} hPa lies outside "
            f"available pressure range {p_sorted[0]:g}-{p_sorted[-1]:g} hPa."
        )

    interior = p_sorted[(p_sorted > p_top_hpa) & (p_sorted < p_bottom_hpa)]
    p_nodes = np.concatenate(([p_top_hpa], interior, [p_bottom_hpa]))
    p_nodes = np.unique(p_nodes)
    p_nodes.sort()

    x_source = np.log(p_sorted)
    x_nodes = np.log(p_nodes)

    lo = np.empty(x_nodes.size, dtype=np.int64)
    hi = np.empty(x_nodes.size, dtype=np.int64)
    alpha = np.empty(x_nodes.size, dtype=np.float64)

    for k, x in enumerate(x_nodes):
        j = int(np.searchsorted(x_source, x, side="left"))
        if j < x_source.size and np.isclose(x_source[j], x, rtol=0.0, atol=1e-12):
            lo[k] = j
            hi[k] = j
            alpha[k] = 0.0
        else:
            if j == 0 or j == x_source.size:
                raise ValueError("Boundary interpolation would require extrapolation.")
            lo[k] = j - 1
            hi[k] = j
            alpha[k] = (x - x_source[j - 1]) / (x_source[j] - x_source[j - 1])

    weights = _trapezoid_mean_weights(x_nodes)

    return {
        "p_top": float(p_top_hpa),
        "p_bottom": float(p_bottom_hpa),
        "order": order,
        "p_nodes": p_nodes,
        "lo": lo,
        "hi": hi,
        "alpha": alpha,
        "weights": weights,
    }


def _interp_to_nodes(field_lev_lat_lon, spec):
    """Interpolate a 3-D field from source pressure levels to spec nodes."""
    field = field_lev_lat_lon[spec["order"], :, :]
    lo = spec["lo"]
    hi = spec["hi"]
    a = spec["alpha"][:, None, None]

    left = field[lo, :, :]
    right = field[hi, :, :]
    return (1.0 - a) * left + a * right


def _compute_H_R(nd, sigma, spec):
    """Compute daily H (%) and R for one fixed pressure layer."""
    nd_nodes = _interp_to_nodes(nd, spec)
    sigma_nodes = _interp_to_nodes(sigma, spec)

    w = spec["weights"][:, None, None]

    valid_nd = np.all(np.isfinite(nd_nodes), axis=0)
    valid_nd &= np.all(nd_nodes >= MIN_ND, axis=0)

    nd_mean = np.sum(w * nd_nodes, axis=0)
    valid_nd &= np.isfinite(nd_mean) & (nd_mean >= MIN_LAYER_MEAN_ND)

    dev = nd_nodes - nd_mean[None, :, :]
    variance = np.sum(w * dev * dev, axis=0)
    variance = np.maximum(variance, 0.0)
    nd_vertical_rms = np.sqrt(variance)

    H = np.full(nd_mean.shape, np.nan, dtype=np.float32)
    H[valid_nd] = (
        100.0 * nd_vertical_rms[valid_nd] / nd_mean[valid_nd]
    ).astype(np.float32)

    valid_sigma = np.all(np.isfinite(sigma_nodes), axis=0)
    valid_sigma &= np.all(sigma_nodes >= 0.0, axis=0)

    sigma_rms = np.sqrt(np.sum(w * sigma_nodes * sigma_nodes, axis=0))
    valid_R = valid_nd & valid_sigma & np.isfinite(sigma_rms)
    valid_R &= sigma_rms > MIN_SIGMA_RMS

    R = np.full(nd_mean.shape, np.nan, dtype=np.float32)
    R[valid_R] = (
        nd_vertical_rms[valid_R] / sigma_rms[valid_R]
    ).astype(np.float32)

    return H, R


# ============================================================
# ln(p) CUMULATIVE GEOMETRY: ZONAL ROW
# ============================================================

def _build_zonal_profile_spec(p_source_hpa):
    """
    Build a common node grid for all cumulative [p, 1000 hPa] integrations.

    Source levels between 200 and 1000 hPa are retained. Exact 200, 950 and
    1000 hPa nodes are inserted when needed by interpolation in ln(p). The
    profile output consists of every node from 200 through 950 hPa; 1000 hPa
    and any source levels between 950 and 1000 are integration-only nodes.
    """
    p_source_hpa = np.asarray(p_source_hpa, dtype=np.float64)
    if np.any(~np.isfinite(p_source_hpa)) or np.any(p_source_hpa <= 0.0):
        raise ValueError("Pressure coordinate must be finite and positive.")

    order = np.argsort(p_source_hpa)
    p_sorted = p_source_hpa[order]
    if np.any(np.diff(p_sorted) <= 0.0):
        raise ValueError("Pressure coordinate contains duplicate levels.")

    p_top = float(ZONAL_PLOT_TOP_HPA)
    p_plot_bottom = float(ZONAL_PLOT_BOTTOM_HPA)
    p_integral_bottom = float(ZONAL_INTEGRATION_BOTTOM_HPA)

    if not (p_top < p_plot_bottom < p_integral_bottom):
        raise ValueError(
            "Require ZONAL_PLOT_TOP_HPA < ZONAL_PLOT_BOTTOM_HPA < "
            "ZONAL_INTEGRATION_BOTTOM_HPA."
        )
    if p_top < p_sorted[0] or p_integral_bottom > p_sorted[-1]:
        raise ValueError(
            f"Requested zonal integration range {p_integral_bottom:g}-{p_top:g} "
            f"hPa lies outside available pressure range "
            f"{p_sorted[0]:g}-{p_sorted[-1]:g} hPa."
        )

    interior = p_sorted[(p_sorted > p_top) & (p_sorted < p_integral_bottom)]
    p_nodes = np.concatenate(
        ([p_top, p_plot_bottom, p_integral_bottom], interior)
    )
    p_nodes = np.unique(p_nodes)
    p_nodes.sort()

    x_source = np.log(p_sorted)
    x_nodes = np.log(p_nodes)
    lo = np.empty(x_nodes.size, dtype=np.int64)
    hi = np.empty(x_nodes.size, dtype=np.int64)
    alpha = np.empty(x_nodes.size, dtype=np.float64)

    for k, x in enumerate(x_nodes):
        j = int(np.searchsorted(x_source, x, side="left"))
        if j < x_source.size and np.isclose(x_source[j], x, rtol=0.0, atol=1e-12):
            lo[k] = j
            hi[k] = j
            alpha[k] = 0.0
        else:
            if j == 0 or j == x_source.size:
                raise ValueError("Zonal boundary interpolation would require extrapolation.")
            lo[k] = j - 1
            hi[k] = j
            alpha[k] = (x - x_source[j - 1]) / (x_source[j] - x_source[j - 1])

    top_indices = np.where(
        (p_nodes >= p_top - 1.0e-8)
        & (p_nodes <= p_plot_bottom + 1.0e-8)
    )[0]
    if top_indices.size < 2:
        raise ValueError("Too few zonal pressure nodes in the requested plotting range.")

    if not np.isclose(p_nodes[-1], p_integral_bottom):
        raise ValueError("Internal error: 1000 hPa is not the final zonal node.")

    return {
        "order": order,
        "p_nodes": p_nodes,
        "x_nodes": x_nodes,
        "lo": lo,
        "hi": hi,
        "alpha": alpha,
        "top_indices": top_indices,
        "top_pressures": p_nodes[top_indices].copy(),
    }


def _reverse_segment_integral(values, x_nodes):
    """Trapezoidal integrals from every node i to the final node."""
    dx = np.diff(x_nodes)[:, None, None]
    segments = 0.5 * (values[:-1] + values[1:]) * dx
    return np.cumsum(segments[::-1, :, :], axis=0)[::-1, :, :]


def _compute_zonal_profile_H_R(nd, sigma, spec):
    """
    Compute H and R for all [p, 1000 hPa] layers in one cumulative pass.

    Returns arrays shaped (zonal_level, lat, lon).
    """
    nd_nodes = _interp_to_nodes(nd, spec)
    sigma_nodes = _interp_to_nodes(sigma, spec)
    x_nodes = spec["x_nodes"]
    idx = spec["top_indices"]

    int_nd = _reverse_segment_integral(nd_nodes, x_nodes)
    int_nd2 = _reverse_segment_integral(nd_nodes * nd_nodes, x_nodes)
    int_sigma2 = _reverse_segment_integral(sigma_nodes * sigma_nodes, x_nodes)

    span = (x_nodes[-1] - x_nodes[idx])[:, None, None]
    nd_mean = int_nd[idx] / span
    nd_second = int_nd2[idx] / span
    variance = np.maximum(nd_second - nd_mean * nd_mean, 0.0)
    nd_vertical_rms = np.sqrt(variance)

    valid_nd_nodes = np.isfinite(nd_nodes) & (nd_nodes >= MIN_ND)
    valid_nd_from_node = np.logical_and.accumulate(
        valid_nd_nodes[::-1, :, :], axis=0
    )[::-1, :, :]
    valid_nd = valid_nd_from_node[idx]
    valid_nd &= np.isfinite(nd_mean) & (nd_mean >= MIN_LAYER_MEAN_ND)

    H = np.full(nd_mean.shape, np.nan, dtype=np.float32)
    H[valid_nd] = (
        100.0 * nd_vertical_rms[valid_nd] / nd_mean[valid_nd]
    ).astype(np.float32)

    sigma_rms = np.sqrt(int_sigma2[idx] / span)
    valid_sigma_nodes = np.isfinite(sigma_nodes) & (sigma_nodes >= 0.0)
    valid_sigma_from_node = np.logical_and.accumulate(
        valid_sigma_nodes[::-1, :, :], axis=0
    )[::-1, :, :]
    valid_R = valid_nd & valid_sigma_from_node[idx]
    valid_R &= np.isfinite(sigma_rms) & (sigma_rms > MIN_SIGMA_RMS)

    R = np.full(nd_mean.shape, np.nan, dtype=np.float32)
    R[valid_R] = (
        nd_vertical_rms[valid_R] / sigma_rms[valid_R]
    ).astype(np.float32)

    return H, R


def _daily_zonal_diagnostics(H_profile, R_profile, threshold):
    """Return daily zonal H mean, exceedance fraction f, and R mean."""
    H_zonal, hcount = _nanmean_last_axis(H_profile)
    R_zonal, _ = _nanmean_last_axis(R_profile)

    hvalid = np.isfinite(H_profile)
    nexceed = ((H_profile > threshold) & hvalid).sum(axis=-1)
    f_zonal = np.full(hcount.shape, np.nan, dtype=np.float32)
    good = hcount > 0
    f_zonal[good] = (
        100.0 * nexceed[good] / hcount[good]
    ).astype(np.float32)

    return H_zonal, f_zonal, R_zonal


# ============================================================
# ANNUAL NETCDF WRITING
# ============================================================

def _create_annual_output(path, year, lat, lon, zonal_pressures, threshold):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    nc = Dataset(_quiet_path(path), "w", format="NETCDF4")

    layers = _configured_layers()
    nlat = len(lat)
    nlon = len(lon)
    nlayer = len(layers)
    nzonal = len(zonal_pressures)

    nc.createDimension(TIME_DIM, None)
    nc.createDimension("layer", nlayer)
    nc.createDimension(ZONAL_LEVEL_DIM, nzonal)
    nc.createDimension(LAT_DIM, nlat)
    nc.createDimension(LON_DIM, nlon)

    tvar = nc.createVariable(TIME_DIM, "i4", (TIME_DIM,))
    tvar.units = "days since 1970-01-01 00:00:00"
    tvar.calendar = "proleptic_gregorian"

    topvar = nc.createVariable("layer_top_hpa", "f4", ("layer",))
    topvar[:] = np.asarray([top for top, _ in layers], dtype=np.float32)
    topvar.units = "hPa"
    topvar.long_name = "top pressure of vertical heterogeneity layer"

    bottomvar = nc.createVariable("layer_bottom_hpa", "f4", ("layer",))
    bottomvar[:] = np.asarray([bottom for _, bottom in layers], dtype=np.float32)
    bottomvar.units = "hPa"
    bottomvar.long_name = "bottom pressure of vertical heterogeneity layer"

    zlevvar = nc.createVariable(ZONAL_LEVEL_VAR, "f4", (ZONAL_LEVEL_DIM,))
    zlevvar[:] = np.asarray(zonal_pressures, dtype=np.float32)
    zlevvar.units = "hPa"
    zlevvar.long_name = "top pressure p of cumulative [p, 1000 hPa] zonal layer"

    latvar = nc.createVariable(LAT_DIM, "f8", (LAT_DIM,))
    lonvar = nc.createVariable(LON_DIM, "f8", (LON_DIM,))
    latvar[:] = lat
    lonvar[:] = lon
    latvar.units = "degrees_north"
    lonvar.units = "degrees_east"

    map_chunks = (
        1,
        1,
        min(NC_CHUNK_LAT, nlat),
        min(NC_CHUNK_LON, nlon),
    )
    map_kwargs = dict(
        zlib=True,
        complevel=NETCDF_COMPLEVEL,
        shuffle=True,
        chunksizes=map_chunks,
        fill_value=np.float32(np.nan),
    )

    hvar = nc.createVariable(
        "H", "f4", (TIME_DIM, "layer", LAT_DIM, LON_DIM), **map_kwargs
    )
    hvar.units = "%"
    hvar.long_name = "relative vertical RMS heterogeneity of Nd in ln pressure"
    hvar.definition = (
        "100 * sqrt(<(Nd-<Nd>_lnp)^2>_lnp) / <Nd>_lnp; "
        "trapezoidal integration in ln(p)"
    )

    rvar = nc.createVariable(
        "R", "f4", (TIME_DIM, "layer", LAT_DIM, LON_DIM), **map_kwargs
    )
    rvar.units = "1"
    rvar.long_name = "vertical Nd heterogeneity relative to sigmaNd RMS"
    rvar.definition = (
        "sqrt(<(Nd-<Nd>_lnp)^2>_lnp) / sqrt(<sigmaNd^2>_lnp)"
    )

    zonal_chunks = (
        1,
        max(1, min(nzonal, 32)),
        min(NC_CHUNK_LAT, nlat),
    )
    zonal_kwargs = dict(
        zlib=True,
        complevel=NETCDF_COMPLEVEL,
        shuffle=True,
        chunksizes=zonal_chunks,
        fill_value=np.float32(np.nan),
    )

    zh = nc.createVariable(
        ZONAL_H_VAR,
        "f4",
        (TIME_DIM, ZONAL_LEVEL_DIM, LAT_DIM),
        **zonal_kwargs,
    )
    zh.units = "%"
    zh.long_name = "daily zonal mean H for cumulative [p, 1000 hPa] layers"

    zf = nc.createVariable(
        ZONAL_F_VAR,
        "f4",
        (TIME_DIM, ZONAL_LEVEL_DIM, LAT_DIM),
        **zonal_kwargs,
    )
    zf.units = "%"
    zf.long_name = "daily zonal fraction of valid longitudes exceeding H threshold"
    zf.threshold_percent = float(threshold)

    zr = nc.createVariable(
        ZONAL_R_VAR,
        "f4",
        (TIME_DIM, ZONAL_LEVEL_DIM, LAT_DIM),
        **zonal_kwargs,
    )
    zr.units = "1"
    zr.long_name = "daily zonal mean R for cumulative [p, 1000 hPa] layers"

    nc.setncattr("year", int(year))
    nc.setncattr("integration_coordinate", "ln(pressure)")
    nc.setncattr("boundary_interpolation", "linear in ln(pressure)")
    nc.setncattr("minimum_Nd_cm-3", float(MIN_ND))
    nc.setncattr("minimum_layer_mean_Nd_cm-3", float(MIN_LAYER_MEAN_ND))
    nc.setncattr("zonal_integration_bottom_hpa", float(ZONAL_INTEGRATION_BOTTOM_HPA))
    nc.setncattr(ZONAL_THRESHOLD_ATTR, float(threshold))
    nc.setncattr(
        "zonal_aggregation",
        "daily longitude mean for H/R; daily longitude fraction for f; "
        "plot uses temporal median H/R and temporal mean f",
    )
    nc.setncattr(
        "validity_requirement",
        "Nd must be finite and >= minimum Nd at every integration node; "
        "R additionally requires finite nonnegative sigmaNd at every node",
    )

    return nc


def _annual_file_is_current(path, zonal_pressures, threshold):
    """Return True only if an existing annual file contains the new zonal data."""
    if not os.path.exists(path):
        return False
    try:
        with Dataset(_quiet_path(path), "r") as nc:
            required = {ZONAL_LEVEL_VAR, ZONAL_H_VAR, ZONAL_F_VAR, ZONAL_R_VAR}
            if not required.issubset(nc.variables):
                return False
            if not hasattr(nc, ZONAL_THRESHOLD_ATTR):
                return False
            stored_threshold = float(getattr(nc, ZONAL_THRESHOLD_ATTR))
            if not np.isclose(stored_threshold, threshold, rtol=0.0, atol=1.0e-6):
                return False
            stored_p = np.asarray(nc.variables[ZONAL_LEVEL_VAR][:], dtype=np.float64)
            return stored_p.shape == np.shape(zonal_pressures) and np.allclose(
                stored_p, zonal_pressures, rtol=0.0, atol=1.0e-5
            )
    except Exception:
        return False


def _process_one_year(year, files, threshold):
    out_path = os.path.join(ANNUAL_OUT_DIR, f"{ANNUAL_PREFIX}_{year}.nc4")

    first = files[0]
    with Dataset(_quiet_path(first), "r") as nc0:
        lat = np.asarray(nc0.variables[LAT_DIM][:], dtype=np.float64)
        lon = np.asarray(nc0.variables[LON_DIM][:], dtype=np.float64)
        p_hpa = _pressure_to_hpa(nc0.variables[LEV_DIM][:])

    layers = _configured_layers()
    specs = [
        _build_layer_spec(p_hpa, ptop, pbottom)
        for ptop, pbottom in layers
    ]
    zonal_spec = _build_zonal_profile_spec(p_hpa)
    zonal_pressures = zonal_spec["top_pressures"]

    if os.path.exists(out_path) and not OVERWRITE_ANNUAL:
        if _annual_file_is_current(out_path, zonal_pressures, threshold):
            return f"[SKIP] {out_path} already contains current zonal diagnostics"
        print(
            f"[REBUILD] {out_path} is missing/stale for zonal diagnostics "
            f"at threshold={threshold:g}%",
            flush=True,
        )

    tmp_path = out_path + f".tmp.{os.getpid()}"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    os.makedirs(ANNUAL_OUT_DIR, exist_ok=True)
    out = _create_annual_output(
        tmp_path, year, lat, lon, zonal_pressures, threshold
    )

    try:
        for it, infile in enumerate(files):
            day = _date_from_filename(infile)

            with Dataset(_quiet_path(infile), "r") as nc:
                nd = _read_lev_lat_lon_variable(nc, ND_VAR)
                sigma = _read_lev_lat_lon_variable(nc, SIGMA_VAR)

                if nd.shape != (len(p_hpa), len(lat), len(lon)):
                    raise ValueError(
                        f"Unexpected {ND_VAR} shape in {infile}: {nd.shape}"
                    )
                if sigma.shape != nd.shape:
                    raise ValueError(
                        f"{SIGMA_VAR} shape {sigma.shape} does not match {ND_VAR} "
                        f"shape {nd.shape} in {infile}"
                    )

            H_day = np.empty(
                (len(layers), len(lat), len(lon)), dtype=np.float32
            )
            R_day = np.empty_like(H_day)
            for ilayer, spec in enumerate(specs):
                H_day[ilayer], R_day[ilayer] = _compute_H_R(nd, sigma, spec)

            H_profile, R_profile = _compute_zonal_profile_H_R(
                nd, sigma, zonal_spec
            )
            H_zonal, f_zonal, R_zonal = _daily_zonal_diagnostics(
                H_profile, R_profile, threshold
            )

            out.variables[TIME_DIM][it] = (day - EPOCH).days
            out.variables["H"][it, :, :, :] = H_day
            out.variables["R"][it, :, :, :] = R_day
            out.variables[ZONAL_H_VAR][it, :, :] = H_zonal
            out.variables[ZONAL_F_VAR][it, :, :] = f_zonal
            out.variables[ZONAL_R_VAR][it, :, :] = R_zonal

            if (it + 1) % 30 == 0 or (it + 1) == len(files):
                print(
                    f"[{year}] {it + 1:4d}/{len(files):4d} days",
                    flush=True,
                )

        out.sync()
    finally:
        out.close()

    os.replace(tmp_path, out_path)
    return f"[WROTE] {out_path}"


def compute_all_years(threshold, workers=N_WORKERS):
    files = _expand_files(INPUT_GLOB)
    grouped = _group_files_by_year(files)
    layers = _configured_layers()

    print(f"Found {len(files)} daily files spanning {len(grouped)} years.")
    print(
        "Map layers: "
        + ", ".join(
            f"{bottom:g}-{top:g} hPa" for top, bottom in layers
        )
    )
    print(
        f"Zonal layers: {ZONAL_INTEGRATION_BOTTOM_HPA:g} hPa to variable p; "
        f"plot p={ZONAL_PLOT_BOTTOM_HPA:g}..{ZONAL_PLOT_TOP_HPA:g} hPa; "
        f"f threshold={threshold:g}%"
    )

    years = sorted(grouped)
    workers = max(1, min(int(workers), len(years)))

    if workers == 1:
        for year in years:
            print(_process_one_year(year, grouped[year], threshold), flush=True)
        return

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_one_year, year, grouped[year], threshold): year
            for year in years
        }
        for fut in as_completed(futures):
            year = futures[fut]
            try:
                print(fut.result(), flush=True)
            except Exception as exc:
                raise RuntimeError(f"Failed while processing year {year}") from exc


# ============================================================
# EXACT TILED MAP CLIMATOLOGY
# ============================================================

def _annual_files():
    pattern = os.path.join(ANNUAL_OUT_DIR, f"{ANNUAL_PREFIX}_*.nc4")
    return _expand_files(pattern)


def _filled_float32(data):
    if np.ma.isMaskedArray(data):
        data = np.ma.filled(data, np.nan)
    return np.asarray(data, dtype=np.float32)


def _load_time_tile(datasets, var_name, ilayer, yslice, xslice, ntime):
    ny = yslice.stop - yslice.start
    nx = xslice.stop - xslice.start
    block = np.empty((ntime, ny, nx), dtype=np.float32)

    t0 = 0
    for nc in datasets:
        nt = len(nc.dimensions[TIME_DIM])
        block[t0:t0 + nt, :, :] = _filled_float32(
            nc.variables[var_name][:, ilayer, yslice, xslice]
        )
        t0 += nt

    return block


def calculate_climatology(threshold):
    """Calculate the three fixed-layer map climatologies."""
    files = _annual_files()
    datasets = [Dataset(_quiet_path(f), "r") for f in files]

    try:
        first = datasets[0]
        lat = np.asarray(first.variables[LAT_DIM][:], dtype=np.float64)
        lon = np.asarray(first.variables[LON_DIM][:], dtype=np.float64)
        tops = np.asarray(first.variables["layer_top_hpa"][:], dtype=np.float64)

        if "layer_bottom_hpa" in first.variables:
            bottoms = np.asarray(
                first.variables["layer_bottom_hpa"][:], dtype=np.float64
            )
        elif hasattr(first, "bottom_level_hpa"):
            bottoms = np.full(tops.shape, float(getattr(first, "bottom_level_hpa")))
        else:
            raise ValueError(
                "Annual files contain neither layer_bottom_hpa nor the legacy "
                "bottom_level_hpa attribute. Recompute the annual files."
            )

        configured = _configured_layers()
        expected_tops = np.asarray([top for top, _ in configured], dtype=np.float64)
        expected_bottoms = np.asarray(
            [bottom for _, bottom in configured], dtype=np.float64
        )

        if tops.shape != expected_tops.shape or not np.allclose(tops, expected_tops):
            raise ValueError(
                f"Annual files contain layer tops {tops}, but script expects "
                f"{expected_tops}."
            )
        if (
            bottoms.shape != expected_bottoms.shape
            or not np.allclose(bottoms, expected_bottoms)
        ):
            raise ValueError(
                f"Annual files contain layer bottoms {bottoms}, but script expects "
                f"{expected_bottoms}. Recompute annual files for the configured layers."
            )

        ntime = sum(len(nc.dimensions[TIME_DIM]) for nc in datasets)
        nlayer = len(tops)
        nlat = len(lat)
        nlon = len(lon)

        med_H = np.full((nlayer, nlat, nlon), np.nan, dtype=np.float32)
        pct_H_gt = np.full_like(med_H, np.nan)
        med_R = np.full_like(med_H, np.nan)

        print(
            f"Map climatology from {ntime} daily records in {len(files)} annual files."
        )
        print(
            f"Map exceedance denominator = valid daily H profiles; threshold = "
            f"{threshold:g}%."
        )

        for ilayer, (top, bottom) in enumerate(zip(tops, bottoms)):
            print(f"Layer {bottom:g}-{top:g} hPa")

            for y0 in range(0, nlat, TILE_LAT):
                y1 = min(y0 + TILE_LAT, nlat)
                ys = slice(y0, y1)

                for x0 in range(0, nlon, TILE_LON):
                    x1 = min(x0 + TILE_LON, nlon)
                    xs = slice(x0, x1)

                    hblock = _load_time_tile(
                        datasets, "H", ilayer, ys, xs, ntime
                    )
                    valid = np.isfinite(hblock)
                    nvalid = valid.sum(axis=0)
                    nexceed = ((hblock > threshold) & valid).sum(axis=0)

                    pct = np.full(nvalid.shape, np.nan, dtype=np.float32)
                    good = nvalid >= MIN_VALID_DAYS
                    pct[good] = (
                        100.0 * nexceed[good] / nvalid[good]
                    ).astype(np.float32)
                    pct_H_gt[ilayer, ys, xs] = pct

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        hmed = np.nanmedian(
                            hblock, axis=0, overwrite_input=True
                        ).astype(np.float32)
                    hmed[~good] = np.nan
                    med_H[ilayer, ys, xs] = hmed
                    del hblock, valid, nvalid, nexceed, pct, hmed

                    rblock = _load_time_tile(
                        datasets, "R", ilayer, ys, xs, ntime
                    )
                    rvalid = np.isfinite(rblock)
                    nrvalid = rvalid.sum(axis=0)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        rmed = np.nanmedian(
                            rblock, axis=0, overwrite_input=True
                        ).astype(np.float32)
                    rmed[nrvalid < MIN_VALID_DAYS] = np.nan
                    med_R[ilayer, ys, xs] = rmed
                    del rblock, rvalid, nrvalid, rmed

                print(f"  latitude rows {y0:3d}:{y1:3d}", flush=True)

        return lat, lon, tops, bottoms, med_H, pct_H_gt, med_R

    finally:
        for nc in datasets:
            try:
                nc.close()
            except Exception:
                pass


# ============================================================
# COMPACT ZONAL CLIMATOLOGY
# ============================================================

def _load_zonal_time_block(datasets, var_name, ntime, nzonal, nlat):
    block = np.empty((ntime, nzonal, nlat), dtype=np.float32)
    t0 = 0
    for nc in datasets:
        nt = len(nc.dimensions[TIME_DIM])
        block[t0:t0 + nt, :, :] = _filled_float32(nc.variables[var_name][:])
        t0 += nt
    return block


def _check_zonal_annual_files(datasets, threshold):
    required = {ZONAL_LEVEL_VAR, ZONAL_H_VAR, ZONAL_F_VAR, ZONAL_R_VAR}
    first = datasets[0]
    if not required.issubset(first.variables):
        missing = sorted(required.difference(first.variables))
        raise ValueError(
            "Annual files predate the zonal-row update and are missing "
            f"{missing}. Run `python Homogen.py compute --threshold "
            f"{threshold:g}` (or mode `all`) once to rebuild them."
        )
    if not hasattr(first, ZONAL_THRESHOLD_ATTR):
        raise ValueError(
            "Annual files do not record the zonal f threshold. Recompute them "
            f"with --threshold {threshold:g}."
        )

    p0 = np.asarray(first.variables[ZONAL_LEVEL_VAR][:], dtype=np.float64)
    for nc in datasets:
        if not required.issubset(nc.variables):
            raise ValueError(
                f"{nc.filepath()} is missing zonal diagnostics. Recompute annual files."
            )
        stored_threshold = float(getattr(nc, ZONAL_THRESHOLD_ATTR, np.nan))
        if not np.isclose(stored_threshold, threshold, rtol=0.0, atol=1.0e-6):
            raise ValueError(
                f"{nc.filepath()} stores zonal f for threshold "
                f"{stored_threshold:g}%, but the requested plot threshold is "
                f"{threshold:g}%. Re-run compute/all with the new threshold."
            )
        p = np.asarray(nc.variables[ZONAL_LEVEL_VAR][:], dtype=np.float64)
        if p.shape != p0.shape or not np.allclose(p, p0, rtol=0.0, atol=1.0e-5):
            raise ValueError(
                "Annual zonal pressure coordinates are inconsistent. Recompute all annual files."
            )
    return p0


def calculate_zonal_climatology(threshold):
    """Temporal climatology of the compact daily zonal products."""
    files = _annual_files()
    datasets = [Dataset(_quiet_path(f), "r") for f in files]

    try:
        first = datasets[0]
        lat = np.asarray(first.variables[LAT_DIM][:], dtype=np.float64)
        p = _check_zonal_annual_files(datasets, threshold)

        ntime = sum(len(nc.dimensions[TIME_DIM]) for nc in datasets)
        nzonal = len(p)
        nlat = len(lat)

        hblock = _load_zonal_time_block(
            datasets, ZONAL_H_VAR, ntime, nzonal, nlat
        )
        hvalid = np.isfinite(hblock)
        nh = hvalid.sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            H = np.nanmedian(hblock, axis=0, overwrite_input=True).astype(np.float32)
        H[nh < MIN_VALID_DAYS] = np.nan
        del hblock, hvalid, nh

        fblock = _load_zonal_time_block(
            datasets, ZONAL_F_VAR, ntime, nzonal, nlat
        )
        fvalid = np.isfinite(fblock)
        nf = fvalid.sum(axis=0)
        fsum = np.sum(np.where(fvalid, fblock, 0.0), axis=0, dtype=np.float64)
        f = np.full((nzonal, nlat), np.nan, dtype=np.float32)
        good = nf >= MIN_VALID_DAYS
        f[good] = (fsum[good] / nf[good]).astype(np.float32)
        del fblock, fvalid, nf, fsum, good

        rblock = _load_zonal_time_block(
            datasets, ZONAL_R_VAR, ntime, nzonal, nlat
        )
        rvalid = np.isfinite(rblock)
        nr = rvalid.sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            R = np.nanmedian(rblock, axis=0, overwrite_input=True).astype(np.float32)
        R[nr < MIN_VALID_DAYS] = np.nan
        del rblock, rvalid, nr

        print(
            f"Zonal climatology from {ntime} daily records; "
            f"integration bottom={ZONAL_INTEGRATION_BOTTOM_HPA:g} hPa."
        )
        return lat, p, H, f, R

    finally:
        for nc in datasets:
            try:
                nc.close()
            except Exception:
                pass


# ============================================================
# JOURNAL-STYLE PLOTTING
# ============================================================

def _to_minus180_180(lon, *fields):
    lon = np.asarray(lon, dtype=np.float64)
    if lon.size and np.nanmax(lon) > 180.0:
        new_lon = ((lon + 180.0) % 360.0) - 180.0
        order = np.argsort(new_lon)
        new_lon = new_lon[order]
        out_fields = [f[..., order] for f in fields]
        return (new_lon, *out_fields)
    return (lon, *fields)


def _lat_weighted_global_mean(field, lat):
    field = np.asarray(field, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    weights = np.cos(np.deg2rad(lat))[:, None]
    valid = np.isfinite(field)
    denom = np.sum(np.where(valid, weights, 0.0))
    if denom <= 0.0:
        return np.nan
    return float(np.sum(np.where(valid, field * weights, 0.0)) / denom)


def plot_climatology(threshold, out_png=OUT_PNG):
    lat, lon, tops, bottoms, med_H, pct_H_gt, med_R = calculate_climatology(
        threshold
    )
    zlat, zpressure, zonal_H, zonal_f, zonal_R = calculate_zonal_climatology(
        threshold
    )

    if lat.shape != zlat.shape or not np.allclose(lat, zlat):
        raise ValueError("Map and zonal latitude coordinates do not match.")

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

    column_titles = [
        r"Median $H$",
        rf"Daily $H > {threshold:g}\%$",
        r"Median $R$",
    ]
    right_title_formats = [
        lambda x: rf"$\langle H\rangle$ = {x:.1f}$\%$",
        lambda x: rf"$\langle f\rangle$ = {x:.1f}$\%$",
        lambda x: rf"$\langle R\rangle$ = {x:.2f}",
    ]

    map_columns = [med_H, pct_H_gt, med_R]
    zonal_columns = [zonal_H, zonal_f, zonal_R]

    if len(tops) != 3 or len(bottoms) != 3:
        raise ValueError(
            "This journal layout currently requires exactly three map-layer pairs."
        )

    fig = plt.figure(figsize=(11.6, 9.0))
    gs = GridSpec(
        nrows=5,
        ncols=3,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.0, 0.92, 0.085],
        hspace=0.17,
        wspace=0.12,
        left=0.065,
        right=0.985,
        bottom=0.06,
        top=0.94,
    )

    map_axes = np.empty((3, 3), dtype=object)
    for row in range(3):
        for col in range(3):
            map_axes[row, col] = fig.add_subplot(gs[row, col], projection=proj)
    zonal_axes = [fig.add_subplot(gs[3, col]) for col in range(3)]

    panel_letters = list(string.ascii_lowercase[:12])

    for col, title in enumerate(column_titles):
        map_axes[0, col].text(
            0.5,
            1.19,
            title,
            transform=map_axes[0, col].transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )

    for row, (top, bottom) in enumerate(zip(tops, bottoms)):
        for col in range(3):
            ax = map_axes[row, col]
            field = map_columns[col][row]
            letter = panel_letters[row * 3 + col]

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
                f"({letter})  {bottom:g}-{top:g} hPa",
                loc="left",
                fontsize=9.5,
                pad=3,
            )

            gmean = _lat_weighted_global_mean(field, lat)
            ax.set_title(
                right_title_formats[col](gmean),
                loc="right",
                fontsize=9.5,
                pad=3,
            )

    for col, ax in enumerate(zonal_axes):
        field = zonal_columns[col]
        letter = panel_letters[9 + col]

        ax.pcolormesh(
            lat,
            zpressure,
            field,
            shading="auto",
            cmap=cmaps[col],
            norm=norms[col],
        )

        ax.set_yscale("linear")
        ax.set_ylim(ZONAL_PLOT_BOTTOM_HPA, ZONAL_PLOT_TOP_HPA)
        ax.set_yticks(ZONAL_PRESSURE_TICKS)
        ax.set_yticklabels([f"{p:g}" for p in ZONAL_PRESSURE_TICKS])

        ax.set_xlim(-90.0, 90.0)
        ax.set_xticks(ZONAL_LAT_TICKS)
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
            ax.set_ylabel(r"Layer-top pressure $p$ (hPa)", fontsize=8.5, labelpad=2)
        else:
            ax.set_ylabel("")

        ax.set_title(
            rf"({letter})  Zonal mean, 1000-$p$ hPa",
            loc="left",
            fontsize=9.5,
            pad=3,
        )

    cbar_ticks = [H_CBAR_TICKS, FREQ_CBAR_TICKS, R_CBAR_TICKS]
    cbar_labels = [
        r"$H$ ($\%$)",
        rf"Days / zonal fraction with $H>{threshold:g}\%$ ($\%$)",
        r"$R$",
    ]

    for col in range(3):
        cax = fig.add_subplot(gs[4, col])
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
        description="Compute and plot ln(p)-weighted vertical Nd heterogeneity."
    )
    parser.add_argument(
        "mode",
        choices=("compute", "plot", "all"),
        help="compute annual daily H/R + zonal files, plot climatology, or both",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        help="H exceedance threshold in percent for f (default: 30)",
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

    if args.threshold < 0.0:
        parser.error("--threshold must be nonnegative")

    if args.mode in ("compute", "all"):
        compute_all_years(threshold=args.threshold, workers=args.workers)

    if args.mode in ("plot", "all"):
        plot_climatology(threshold=args.threshold, out_png=args.out)


if __name__ == "__main__":
    main()
