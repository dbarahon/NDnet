# cache_utils.py

import os
import json
import numpy as np
import xarray as xr
import hashlib
import torch
import dask.array as da


# ============================
# Visualization
# ============================

def safe_to_netcdf_chunked(
    ds,
    path,
    engine="netcdf4",
    complevel=1,
    chunk_t=1,
    chunk_lev=22,
    chunk_lat=181,
    chunk_lon=360,
):
    import os, shutil
    from dask.diagnostics import ProgressBar

    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # write to local scratch first
    #tmpdir = os.environ.get("SLURM_TMPDIR", "/temp")
    tmpdir =  "./temp"
    tmp_path = os.path.join(tmpdir, os.path.basename(path) + ".tmp")

    # Ensure float32 and reasonable chunking on-disk
    ds2 = ds.copy()
    for v in ds2.data_vars:
        if ds2[v].dtype.kind == "f":
            ds2[v] = ds2[v].astype("float32")

    # Rechunk for IO (important)
    # You can tune these; below keeps whole lat/lon planes and chunks in time.
    ds2 = ds2.chunk({"time": chunk_t, "lev": chunk_lev, "lat": chunk_lat, "lon": chunk_lon})

    encoding = {}
    for v in ds2.data_vars:
        if set(ds2[v].dims) >= {"time", "lev", "lat", "lon"}:
            encoding[v] = {
                "zlib": True,
                "complevel": int(complevel),
                "shuffle": True,
                "chunksizes": (chunk_t, chunk_lev, chunk_lat, chunk_lon),
            }
        elif ds2[v].dtype.kind == "f":
            encoding[v] = {"zlib": True, "complevel": int(complevel), "shuffle": True}

    # Make sure tmp doesn't exist from a crash
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception:
        pass

    print(f"[NETCDF] Streaming write to temp: {tmp_path}", flush=True)
    with ProgressBar():
        ds2.to_netcdf(tmp_path, engine=engine, format="NETCDF4", encoding=encoding, compute=True)

    print(f"[NETCDF] Moving to final: {path}", flush=True)
    shutil.move(tmp_path, path)

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

    print(f"[VIS] Saved visualization file {path}")

#============================================================
## General Utilities 
#=============================================================


def _to_numpy(data, dtype=None):
    """
    Robustly convert various backends (dask, netCDF buffers, memoryview, lists)
    to a real numpy ndarray.
    """
    # dask support (optional)
    try:
        import dask.array as da
        if isinstance(data, da.Array):
            data = data.compute()
    except Exception:
        pass

    # netCDF / memoryview / numpy / torch storage views -> ndarray
    arr = np.asarray(data)

    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr



def safe_to_netcdf(ds, path, engine="netcdf4"):
    import os, shutil
    from dask.diagnostics import ProgressBar

    # ensure directory exists
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # cast floats to float32 to reduce IO
    for v in ds.data_vars:
        if ds[v].dtype.kind == "f":
            ds[v] = ds[v].astype("float32")

    # 1) force compute in a controlled way
    print(f"[NETCDF] Computing dataset in memory before writing: {path}", flush=True)
    with ProgressBar():
        ds2 = ds.load()

    # 2) write to local scratch first, then move
    tmpdir = os.environ.get("SLURM_TMPDIR", "/tmp")
    tmp_path = os.path.join(tmpdir, os.path.basename(path) + ".tmp")

    print(f"[NETCDF] Writing to temp: {tmp_path}", flush=True)
    ds2.to_netcdf(tmp_path, engine=engine, format="NETCDF4")

    print(f"[NETCDF] Moving to final: {path}", flush=True)
    shutil.move(tmp_path, path)


    
def nb(x, prec=2):
    return np.format_float_positional(x, precision=prec)


def amean(ds):
    weights = np.cos(np.deg2rad(ds.lat))
    ds_weighted = ds.weighted(weights)
    return ds_weighted.mean(("lon", "lat"), skipna=True).load()


def corrcoefxy(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x)
    std_y = np.std(y)
    n = len(x)
    return ((x - mean_x) * (y - mean_y)).sum() / n / (std_x * std_y)


    
def log1p_scaler(ds):
    return xr.apply_ufunc(np.log1p, ds, dask='parallelized')


def inv_log1p_scaler(ds):
    return xr.apply_ufunc(np.expm1, ds, dask='parallelized')


def power_scaler(ds, nexp):
    func = lambda x, n: np.power(np.where(x < 0, 0, x), n)
    return xr.apply_ufunc(func, ds, nexp, dask='parallelized')


def linear_scaler(ds, cst):
    func = lambda x, n: x * n
    return xr.apply_ufunc(func, ds, cst, dask='parallelized')


def calcRF(ds):
    ds['net'] = ds.SWTNET - ds.OLR
    ds['swcf'] = ds.SWTNETC - ds.SWTNET
    ds['lwcf'] = ds.OLRC - ds.OLR
    return ds


def shift_lon(lon):
    return xr.where(lon > 180., lon - 360., lon)
    
    
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

# utils_tensorize.py  (or utils.py)




def tensorize_inputs(
    Nin_2d: xr.Dataset,
    Ares_base: xr.DataArray,   # (time, mode, lat, lon) residual coeffs (anchor)
    pin_memory=True,
    verbose=False,
):
    """
    Returns:
      X2d_t : torch.float32 [T, C, H, W]
      MU0_t : torch.float32 [T, M, H, W]
    """

    # ---- X2D (time, lat, lon, var) -> [T,C,H,W] ----
    varnames = list(Nin_2d.data_vars)
    X = xr.concat([Nin_2d[v] for v in varnames], dim="channel")  # (channel, time, lat, lon)
    X = X.transpose("time", "channel", "lat", "lon")            # (time, channel, lat, lon)
    X_np = np.asarray(X.data, dtype=np.float32)                 # triggers compute if not already
    X2d_t = torch.from_numpy(X_np)

    # ---- MU0 (base coeff residuals) -> [T,M,H,W] ----
    MU0 = Ares_base.transpose("time", "mode", "lat", "lon")
    MU0_np = np.asarray(MU0.data, dtype=np.float32)
    MU0_t = torch.from_numpy(MU0_np)

    if pin_memory and torch.cuda.is_available():
        X2d_t = X2d_t.pin_memory()
        MU0_t = MU0_t.pin_memory()
   
    if verbose:
        print("[TENSORIZE v2]")
        print("  X2d_t:", tuple(X2d_t.shape), X2d_t.dtype)
        print("  MU0_t:", tuple(MU0_t.shape), MU0_t.dtype)

    return X2d_t, MU0_t

def A_sites_from_infer_grid(coeffs_resid_nn, sig_grid, site_meta, time_coord, mode_coord=None):
    """
    coeffs_resid_nn: np.ndarray (T,M,H,W)  residual coeff field (A_abs - A_mean)
    sig_grid:        np.ndarray (T,M,H,W)  sigma field (optional; can be None)

    site_meta: xr.Dataset with:
      - ilat (site,)
      - ilon (site,)
      - site (site,)  (names)
      - site_lat, site_lon (site,) optional for plotting

    Returns:
      A_sites_pred: xr.DataArray (time, mode, site)
      sig_sites:    xr.DataArray (time, mode, site) or None
    """
    ilat = site_meta["ilat"].values.astype(np.int64)
    ilon = site_meta["ilon"].values.astype(np.int64)
    site = site_meta["site"].values

    T, M, H, W = coeffs_resid_nn.shape
    S = ilat.size

    A = np.full((T, M, S), np.nan, dtype=np.float32)
    Sig = None if sig_grid is None else np.full((T, M, S), np.nan, dtype=np.float32)

    # simple + correct: loop over sites (S is small)
    for s in range(S):
        A[:, :, s] = coeffs_resid_nn[:, :, ilat[s], ilon[s]]
        if Sig is not None:
            Sig[:, :, s] = sig_grid[:, :, ilat[s], ilon[s]]

    if mode_coord is None:
        mode_coord = np.arange(M, dtype=np.int32)

    coords = {
        "time": time_coord,
        "mode": mode_coord,
        "site": xr.DataArray(site, dims="site", name="site"),
    }

    A_da = xr.DataArray(A, dims=("time", "mode", "site"), coords=coords, name="A_sites_resid")

    if Sig is None:
        return A_da, None
    Sig_da = xr.DataArray(Sig, dims=("time", "mode", "site"), coords=coords, name="sigma_sites")
    return A_da, Sig_da


def A_sites_from_Ares_base(Ares_base, site_meta):
    """
    Ares_base: xr.DataArray (time, mode, lat, lon) residual field
    site_meta: xr.Dataset with ilat, ilon, site

    Returns:
      A_base_sites_resid: xr.DataArray (time, mode, site)
    """
    ilat = site_meta["ilat"].values.astype(np.int64)
    ilon = site_meta["ilon"].values.astype(np.int64)

    A_base_sites = Ares_base.isel(
        lat=xr.DataArray(ilat, dims="site"),
        lon=xr.DataArray(ilon, dims="site"),
    ).transpose("time", "mode", "site").astype(np.float32)

    # ensure site coord is the same as site_meta
    if "site" in site_meta:
        A_base_sites = A_base_sites.assign_coords(site=("site", site_meta["site"].values))

    return A_base_sites.rename("A_base_sites_resid")

