# cache_utils.py

import os
import json
import numpy as np
import xarray as xr
import hashlib

# ============================
# Fast Array Cache (.npy+.json)
# ============================

def save_array_cache(path_base, dataarray):
    """
    Save a DataArray using:
      - <path>.npy  \u2192 binary float32 array
      - <path>.json \u2192 full dimension + coordinate metadata
    """
    os.makedirs(os.path.dirname(path_base), exist_ok=True)

    arr = np.ascontiguousarray(dataarray.data.astype(np.float32))
    np.save(path_base + ".npy", arr)

    meta = {
        "dims": list(dataarray.dims),
        "shape": arr.shape,
        "dtype": "float32",
        "coords": {
            dim: dataarray.coords[dim].values.tolist()
            for dim in dataarray.dims
        },
        "name": dataarray.name,
    }
    with open(path_base + ".json", "w") as f:
        json.dump(meta, f)

    print(f"[CACHE] Saved array \u2192 {path_base}.npy (+metadata)")


def load_array_cache(path_base):
    """
    Load .npy + .json into an xarray.DataArray.
    Returns None if the cache is missing.
    """
    npy = path_base + ".npy"
    js  = path_base + ".json"

    if not (os.path.exists(npy) and os.path.exists(js)):
        return None

    with open(js, "r") as f:
        meta = json.load(f)

    arr = np.load(npy)

    coords = {
        dim: (dim, np.array(meta["coords"][dim]))
        for dim in meta["dims"]
    }

    print(f"[CACHE] Loaded fast array cache from {path_base}")

    return xr.DataArray(arr, dims=meta["dims"], coords=coords, name=meta["name"])


# ============================
# Grid Hashing
# ============================

def hash_grid(ds, keys=("lat", "lon")):
    """
    Compute reproducible hash for caching fields on a given grid.
    Prevents mixing data from different spatial grids.
    """
    h = hashlib.md5()
    for k in keys:
        h.update(ds[k].values.tobytes())
    return h.hexdigest()[:10]


# ============================
# Cache path generator
# ============================

def cache_path(base_dir, varname, prefix, grid_hash, tag=""):
    """
    Return *base* cache path (no extension).
    All caches use:  <path>.npy and <path>.json
    """
    safe_prefix = prefix.replace("*", "")
    fname = f"{varname}__{safe_prefix}__{grid_hash}__{tag}"
    return os.path.join(base_dir, fname)


# ============================
# (Optional) Visualization
# ============================

def save_vis_nc(path, da, time_slice=None, modes=None, stride_lat=1, stride_lon=1):
    da_vis = da

    if time_slice is not None:
        da_vis = da_vis.isel(time=time_slice)
    if modes is not None:
        da_vis = da_vis.isel(mode=modes)

    if stride_lat > 1:
        da_vis = da_vis.isel(lat=slice(0, None, stride_lat))
    if stride_lon > 1:
        da_vis = da_vis.isel(lon=slice(0, None, stride_lon))

    enc = {da_vis.name: {"zlib": True, "complevel": 4}}
    da_vis.to_netcdf(path, engine="netcdf4", encoding=enc)

    print(f"[VIS] Saved visualization file \u2192 {path}")
    
# ============================================================
# TIME NORMALIZATION (critical for exact alignment)
# ============================================================


# ============================================================
# TIME UTILITIES
# ============================================================

def force_datetime64_ns(t):
    """
    Convert any time array to datetime64[ns].
    Supports:
        - datetime64
        - CFTime
        - int64 nanoseconds / seconds since epoch
        - cached values storing ints
    """
    t = np.asarray(t)

    # Already datetime64
    if np.issubdtype(t.dtype, np.datetime64):
        return t.astype("datetime64[ns]")

    # Try CF decode through xarray
    try:
        tt = xr.conventions.decode_cf_datetime(
            t, units="seconds since 1970-01-01", calendar="standard"
        )
        return tt.astype("datetime64[ns]")
    except Exception:
        pass

    # If integer, interpret as nanoseconds since epoch
    if np.issubdtype(t.dtype, np.integer):
        # try interpreting as ns, then as s
        try:
            tt = t.astype("datetime64[ns]")
            return tt
        except Exception:
            tt = t.astype("datetime64[s]").astype("datetime64[ns]")
            return tt

    raise RuntimeError(f"Could not convert time dtype {t.dtype} into datetime64[ns]")


def normalize_time_coord(ds):
    """
    Force dataset 'time' coord to datetime64[ns] floor(D).
    Works even if incoming time was integer, CFTime, stale cache, etc.
    """
    if "time" not in ds.coords:
        return ds

    t = ds["time"].values
    t = force_datetime64_ns(t)
    t = t.astype("datetime64[D]").astype("datetime64[ns]")  # floor to day

    return ds.assign_coords(time=t)


def safe_time_intersection(t1, t2):
    """
    Robust intersection of two time arrays that may be datetime64, int64, or mixed.
    Always returns datetime64[ns].
    """
    t1_dt = force_datetime64_ns(t1)
    t2_dt = force_datetime64_ns(t2)

    # Convert to int nanoseconds for safe NumPy intersection
    t1_ns = t1_dt.astype("int64")
    t2_ns = t2_dt.astype("int64")

    inter = np.intersect1d(t1_ns, t2_ns)
    return inter.astype("datetime64[ns]")

