# ============================================================
# NEW data_handling.py for site-only coeff targets (20y streaming)
# ============================================================
import os
import time
import json
import numpy as np
import xarray as xr
import xesmf as xe
import dask.array as da
import config as cfg
import glob
import re


# ============================================================
# Small utilities (safe, deterministic)
# ============================================================
def align_on_time(*objs):
    times = [o["time"] for o in objs]
    t_common = xr.IndexVariable("time", np.intersect1d.reduce([t.values for t in times]))
    return [o.sel(time=t_common) for o in objs]
    
    
def force_datetime64_ns(tvals):
    t = np.asarray(tvals)
    return t.astype("datetime64[ns]")

def normalize_time_coord(ds):
    if "time" not in ds.coords:
        return ds
    # avoid full ds["time"].values
    ds = ds.sortby("time")
    return ds

def trim_to_overlap_time(a, b):
    t0 = max(a.time.isel(time=0).item(), b.time.isel(time=0).item())
    t1 = min(a.time.isel(time=-1).item(), b.time.isel(time=-1).item())
    return a.sel(time=slice(t0, t1)), b.sel(time=slice(t0, t1))


def _print_da_info(name, obj):
    if isinstance(obj, xr.Dataset):
        print(f"[DBG] {name}: Dataset vars={list(obj.data_vars)} dims={dict(obj.sizes)}")
        if "time" in obj.coords:
            nt = obj.sizes.get("time", None)
            t0 = obj["time"].isel(time=0).item() if nt and nt > 0 else None
            t1 = obj["time"].isel(time=-1).item() if nt and nt > 0 else None
            print(f"      time: n={nt}, first={t0}, last={t1}")
        return

    if isinstance(obj, xr.DataArray):
        print(f"[DBG] {name}: DataArray dims={obj.dims} shape={obj.shape} dtype={obj.dtype}")
        if "time" in obj.coords:
            nt = obj.sizes.get("time", None)
            t0 = obj["time"].isel(time=0).item() if nt and nt > 0 else None
            t1 = obj["time"].isel(time=-1).item() if nt and nt > 0 else None
            print(f"      time: n={nt}, first={t0}, last={t1}")
        return

    print(f"[DBG] {name}: type={type(obj)}")


# ============================================================
# Regridding with cached weights (unchanged)
# ============================================================

def regrid_2d_to_nout(ds2d, Nout, chk2d, weights_file="cache/weights_modis_to_nd.nc"):
    wdir = os.path.dirname(weights_file) or "."
    os.makedirs(wdir, exist_ok=True)
    lock = weights_file + ".lock"

    if not os.path.exists(weights_file):
        got_lock = False
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            got_lock = True
        except FileExistsError:
            got_lock = False

        if got_lock:
            print(f"[REGRID] Building weights -> {weights_file}")
            xe.Regridder(
                ds2d, Nout, "bilinear",
                periodic=True,
                filename=weights_file,
                reuse_weights=False
            )
            try:
                os.remove(lock)
            except FileNotFoundError:
                pass
        else:
            print(f"[REGRID] Waiting for weights: {weights_file}")
            while (not os.path.exists(weights_file)) and os.path.exists(lock):
                time.sleep(1.0)

    #print(f"[REGRID] Using weights: {weights_file}")
    R2 = xe.Regridder(
        ds2d, Nout, "bilinear",
        periodic=True,
        filename=weights_file,
        reuse_weights=True
    )
    return R2(ds2d, output_chunks=chk2d)

# ============================================================
# Main loader for the new training setup
# ============================================================

def get_data(
    vars_out=None,
    pth_all="",
    vars_in_2d=None,
    modis_source=None,
    day="",
    cache_folder="cache",
    minP=250,
):
    """
    Returns objects suitable for streaming IterableDataset:

      Nin_2d    : xr.Dataset   (time, lat, lon)        dask-backed, standardized, regridded
      Ares_base : xr.DataArray (time, mode, lat, lon)  dask-backed          [dense base residual coeffs]
    """

    ymd = day.strftime("%Y%m%d")
    ym = day.strftime("%Y%m")  
    if vars_out is None:
        vars_out = {}
    if vars_in_2d is None:
        vars_in_2d = {}

    EOF     = cfg.EOF
    Nd_mean = cfg.Nd_mean
    A_mean  = cfg.A_mean  # used downstream; we attach site_meta so you can sample it

    # IMPORTANT: chunk time larger than 3 to reduce scheduler overhead for 20y
    chk2d = {"time": cfg.t_chk, "lat": -1, "lon": -1}
    chk3d = {"time": cfg.t_chk, "lev": -1, "lat": -1, "lon": -1}

    # ------------------------------------------------------------
    # 2) Load base residual coefficients (dense), used for anchor + grad
    # ------------------------------------------------------------
    varname_base = list(vars_out.keys())[1]
    pth_base = f"{vars_out[varname_base]}/*{ymd}*"
    #print("[LOAD 3D OUT base]", pth_base, len(sorted(glob.glob(pth_base))), "files")

    Ab = xr.open_mfdataset(pth_base, parallel=False, chunks=chk3d)[varname_base]
    Ares_base = normalize_time_coord(Ab)
    Ares_base = Ares_base.where(Ares_base > -1e9).where(Ares_base < 1e9)

    # ------------------------------------------------------------
    # 3) Load MODIS 2D inputs, normalize
    # ------------------------------------------------------------
    pth2d = f"{pth_all}dailies/{modis_source}/on_Noutgrid/{modis_source}.*{ym}*.nc"
    #print("[LOAD 2D IN]", pth2d)
    t0 = time.time()

    ds2d = xr.open_mfdataset(pth2d, parallel=False, chunks=chk2d).sel(time = Ares_base.time) 

    # standardize (lazy)
    for v in ds2d.data_vars:
        mu = float(vars_in_2d[v]["mean"])
        sd = float(vars_in_2d[v]["std"])
        ds2d[v] = (ds2d[v] - mu) / sd

    Nin_2d = ds2d
    #print(f"[DBG] open_mfdataset(ds2d) done in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------
    # 4) Final alignment: Nin_2d, Ares_base, Nout share exact time
    # ------------------------------------------------------------
    #Nin_2d, Ares_base = xr.align(Nin_2d, Ares_base, join="inner")

    Nin_2d    = Nin_2d.transpose("time", "lat", "lon")
    Ares_base = Ares_base.transpose("time", "mode", "lat", "lon")

    #_print_da_info("Nin_2d", Nin_2d)
    #_print_da_info("Ares_base", Ares_base)

    # ------------------------------------------------------------
  
    # Fill NaNs in 2D inputs only (typical for NN); leave Nd_sites as-is (you'll infer masks from it)
    Nin_2d = Nin_2d.fillna(0)
    Ares_base =Ares_base.fillna(0)

    return Nin_2d, Ares_base

    
    
    #############################################


# ---------------------------------------------
def reconstruct_Nd_from_coeffs_chunked(
    coeffs_abs,          # (T, M, lat, lon)
    EOF,                 # (mode, lev)
    Nd_mean,             # (lev)
    time_coord,
    lat_coord,
    lon_coord,
    sigma_coeffs_abs=None,  # OPTIONAL: (T, M, lat, lon) std-dev of coeffs_abs
    chunk_t=64,
    sigma_floor=0.0,
):
    """
    Mean:
      Nd(t,lev,lat,lon) = sum_m coeffs_abs(t,m,lat,lon) * EOF(m,lev) + Nd_mean(lev)

    Uncertainty propagation (if sigma_coeffs_abs provided), assuming independent modes:
      sigmaNd^2(t,lev,lat,lon) = sum_m sigma_coeffs_abs^2(t,m,lat,lon) * EOF^2(m,lev)

    Returns:
      xr.Dataset with variables:
        - Nd_rec       (time, lev, lat, lon) float32 clipped >= 0
        - sigmaNd_rec  (time, lev, lat, lon) float32 clipped >= sigma_floor   [only if sigma provided]
    """

    def _to_dask_4d(x, chunk_t):
        if x is None:
            return None
        if isinstance(x, xr.DataArray):
            x = x.data
        if not isinstance(x, da.Array):
            x = da.from_array(np.asarray(x), chunks=(chunk_t, -1, -1, -1))
        else:
            x = x.rechunk((chunk_t, -1, -1, -1))
        return x

    # --- coeffs data ---
    coeffs_data = _to_dask_4d(coeffs_abs, chunk_t)

    # --- sigma coeffs (optional) ---
    sigma_data = _to_dask_4d(sigma_coeffs_abs, chunk_t)

    # --- EOF + mean (small) ---
    EOF_mz = np.asarray(EOF.values, dtype=np.float32)       # (M_eof, Z)
    mean_z = np.asarray(Nd_mean.values, dtype=np.float32)   # (Z,)

    T, M_in, Ny, Nx = coeffs_data.shape
    M_eof = EOF_mz.shape[0]

    # crop modes if needed (keep consistent across mean + sigma)
    M_use = min(M_in, M_eof)
    if M_use != M_in:
        coeffs_data = coeffs_data[:, :M_use, :, :]
        if sigma_data is not None:
            sigma_data = sigma_data[:, :M_use, :, :]
    if M_use != M_eof:
        EOF_mz = EOF_mz[:M_use, :]

    # --- mean reconstruction ---
    Nd_tylz = da.tensordot(coeffs_data, EOF_mz, axes=([1], [0]))   # (T, lat, lon, lev)
    Nd_tylz = Nd_tylz + mean_z[None, None, None, :]
    Nd_tzyl = da.transpose(Nd_tylz, (0, 3, 1, 2))                  # (T, lev, lat, lon)
    Nd_tzyl = da.maximum(Nd_tzyl, 0.0).astype(np.float32)

    lev_coord = Nd_mean["lev"] if "lev" in Nd_mean.coords else Nd_mean.coords.get("lev", None)

    Nd_da = xr.DataArray(
        Nd_tzyl,
        dims=("time", "lev", "lat", "lon"),
        coords={"time": time_coord, "lev": lev_coord, "lat": lat_coord, "lon": lon_coord},
        name="Nd_rec",
    )

    out = xr.Dataset({"Nd_rec": Nd_da})

    # --- sigmaNd reconstruction (optional) ---
    if sigma_data is not None:
        # variance propagation: Var(Nd) = sum_m Var(A_m)*EOF^2
        EOF2_mz = (EOF_mz ** 2).astype(np.float32)                  # (M_use, Z)
        varA = (sigma_data.astype(np.float32) ** 2)                 # (T, M, lat, lon)

        varNd_tylz = da.tensordot(varA, EOF2_mz, axes=([1], [0]))   # (T, lat, lon, lev)
        varNd_tzyl = da.transpose(varNd_tylz, (0, 3, 1, 2))         # (T, lev, lat, lon)

        sigmaNd_tzyl = da.sqrt(da.maximum(varNd_tzyl, 0.0))
        if sigma_floor and sigma_floor > 0.0:
            sigmaNd_tzyl = da.maximum(sigmaNd_tzyl, float(sigma_floor))

        sigma_da = xr.DataArray(
            sigmaNd_tzyl.astype(np.float32),
            dims=("time", "lev", "lat", "lon"),
            coords={"time": time_coord, "lev": lev_coord, "lat": lat_coord, "lon": lon_coord},
            name="sigmaNd_rec",
        )

        out["sigmaNd_rec"] = sigma_da

    return out

# ---------------------------------------------------
# Reconstruct Nd ONLY at sites (expects 3-return loader)
# ---------------------------------------------------

def reconstruct_Nd_from_A_sites(
    A_sites_resid,   # xr.DataArray (time, mode, site) residuals
    site_meta,       # xr.Dataset with site coord; used only for coords
    EOF,             # xr.DataArray (mode, lev)
    Nd_mean,         # xr.DataArray (lev,)
    A_mean,          # (mode, lat, lon) climatological mean coeff field
    valid_ref=None,  # xr.DataArray like Nd_sites (time, lev, site)
    mask_mode="obs_dates",  # "obs_levels" | "obs_dates" | "none"
):
    """
    Returns:
      Nd_rec: xr.DataArray (lev, time, site)
      A_abs_sites: xr.DataArray (time, mode, site)
      A_mean_sites: xr.DataArray (mode, site)
    """

    # -------------------------
    # 0) validate mask_mode
    # -------------------------
    if mask_mode not in ("obs_levels", "obs_dates", "none"):
        raise ValueError(f"mask_mode must be one of: 'obs_levels', 'obs_dates', 'none'. Got: {mask_mode}")

    # -------------------------
    # 1) sample A_mean at sites (using ilat/ilon)
    # -------------------------
    Ares = A_sites_resid.transpose("time", "mode", "site")

    ilat = site_meta["ilat"].values.astype(np.int64)
    ilon = site_meta["ilon"].values.astype(np.int64)

    if isinstance(A_mean, xr.DataArray):
        Amean_mhw = A_mean
    else:
        Amean_mhw = xr.DataArray(
            np.asarray(A_mean, dtype=np.float32),
            dims=("mode", "lat", "lon"),
            coords={"mode": EOF["mode"]},
            name="A_mean",
        )

    A_mean_sites = Amean_mhw.isel(
        lat=xr.DataArray(ilat, dims="site"),
        lon=xr.DataArray(ilon, dims="site"),
    ).astype(np.float32)

    # -------------------------
    # 2) absolute coeffs at sites
    # -------------------------
    A_abs_sites = (A_mean_sites + Ares).astype(np.float32)

    # -------------------------
    # 3) reconstruct Nd at sites
    # -------------------------
    Nd_tls = xr.dot(A_abs_sites, EOF, dims="mode")      # (time, site, lev)
    Nd_tsl = Nd_tls.transpose("time", "lev", "site")    # (time, lev, site)
    Nd_tsl = (Nd_tsl + Nd_mean).astype(np.float32)
    Nd_tsl = Nd_tsl.clip(min=0.0)

    # -------------------------
    # 4) mask using valid_ref = Nd_sites (time, lev, site) OR (time, site)
    # -------------------------
    if (mask_mode != "none") and (valid_ref is not None):
        # If valid_ref is Nd_sites, it should already be (time,lev,site)
        v_obs = xr.ufuncs.isfinite(valid_ref)

        # Ensure dims match expected
        if ("lev" in valid_ref.dims) and ("lev" in Nd_tsl.dims):
            # align lev if needed
            try:
                v_obs = v_obs.sel(lev=Nd_tsl["lev"])
            except Exception:
                # fallback by position
                v_obs = v_obs.isel(lev=slice(0, Nd_tsl.sizes["lev"]))

        if mask_mode == "obs_levels":
            # want mask3d (time,lev,site)
            if "lev" in v_obs.dims:
                mask3d = v_obs
            else:
                # v_obs is (time,site) -> broadcast
                mask3d = v_obs.broadcast_like(Nd_tsl)
            Nd_tsl = Nd_tsl.where(mask3d)

        elif mask_mode == "obs_dates":
            # want (time,site) validity
            if "lev" in v_obs.dims:
                v_ts = v_obs.any("lev")
            else:
                v_ts = v_obs
            Nd_tsl = Nd_tsl.where(v_ts.broadcast_like(Nd_tsl))

        # -------------------------
    # 5) keep permanently-missing (lev,site) as NaN
    # -------------------------
    if valid_ref is not None and ("lev" in valid_ref.dims):
        # valid_ref expected like Nd_sites: (time, lev, site)
        v_obs = xr.ufuncs.isfinite(valid_ref)

        # align lev if needed
        try:
            v_obs = v_obs.sel(lev=Nd_tsl["lev"])
        except Exception:
            v_obs = v_obs.isel(lev=slice(0, Nd_tsl.sizes["lev"]))

        # levels that EVER have data at that site (lev,site)
        v_level_site = v_obs.any("time")  # (lev, site)

        # broadcast to (time,lev,site) and apply
        Nd_tsl = Nd_tsl.where(v_level_site.broadcast_like(Nd_tsl))

    # final output requested: (lev,time,site)
    Nd_lts = Nd_tsl.transpose("lev", "time", "site").rename("Nd_rec")

    # attach convenient coords if present
    for c in ["site", "site_lat", "site_lon"]:
        if c in site_meta and c not in Nd_lts.coords:
            Nd_lts = Nd_lts.assign_coords({c: ("site", site_meta[c].values)})

    return Nd_lts, A_abs_sites, A_mean_sites

