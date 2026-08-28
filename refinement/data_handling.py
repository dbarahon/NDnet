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
from torch.utils.data import Dataset

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

    print(f"[REGRID] Using weights: {weights_file}")
    R2 = xe.Regridder(
        ds2d, Nout, "bilinear",
        periodic=True,
        filename=weights_file,
        reuse_weights=True
    )
    return R2(ds2d, output_chunks=chk2d)


def save_site_fill_debug_netcdfs(
    Nd_sites: xr.DataArray,    # raw (time, lev, site)
    Nd_filled: xr.DataArray,   # (time, lev, site)
    Nd_anom: xr.DataArray,     # (time, lev, site)
    valid_ts: xr.DataArray,    # (time, site)
    site_names,                # list/array length nsite
    out_dir: str = ".",
    prefix: str = "sites_fill_debug",
):
    """
    Writes one NetCDF per site with CF/CDO-friendly dimension order:
      - Nd_raw(time, lev)
      - Nd_filled(time, lev)
      - Nd_anom(time, lev)
      - valid_ts(time)
      - *_tmean(lev)
      - valid_frac(scalar)
    """

    def _safe(s: str) -> str:
        s = str(s)
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
        return s[:80] if len(s) > 80 else s

    os.makedirs(out_dir, exist_ok=True)

    # --- checks ---
    for da, nm in [(Nd_sites, "Nd_sites"), (Nd_filled, "Nd_filled"), (Nd_anom, "Nd_anom")]:
        if not all(d in da.dims for d in ("time", "lev", "site")):
            raise ValueError(f"{nm} must have dims including ('time','lev','site'), got {da.dims}")
    if not all(d in valid_ts.dims for d in ("time", "site")):
        raise ValueError(f"valid_ts must have dims including ('time','site'), got {valid_ts.dims}")

    nsite = Nd_sites.sizes["site"]
    site_names = np.asarray(site_names)
    if site_names.size != nsite:
        raise ValueError(f"site_names length ({site_names.size}) != nsite ({nsite})")

    # --- normalize order for indexing (keep time,lev for CDO) ---
    Nd_raw_o    = Nd_sites.transpose("site", "time", "lev")
    Nd_filled_o = Nd_filled.transpose("site", "time", "lev")
    Nd_anom_o   = Nd_anom.transpose("site", "time", "lev")
    valid_ts_o  = valid_ts.transpose("site", "time")

    # --- summaries ---
    Nd_raw_tmean    = Nd_raw_o.mean("time", skipna=True)        # (site, lev)
    Nd_filled_tmean = Nd_filled_o.mean("time", skipna=True)     # (site, lev)
    Nd_anom_tmean   = Nd_anom_o.mean("time", skipna=True)       # (site, lev)
    valid_frac      = valid_ts_o.astype(np.float32).mean("time")# (site,)

    # helper to drop *all* non-dimension coords (lat/lon/site/etc.)
    def _strip_coords(da: xr.DataArray) -> xr.DataArray:
        # keep only dimension coordinates (time/lev); drop any others
        return da.reset_coords(drop=True)

    for si, sname in enumerate(site_names):
        nd_raw    = _strip_coords(Nd_raw_o.isel(site=si)).astype(np.float32)       # (time, lev)
        nd_fill   = _strip_coords(Nd_filled_o.isel(site=si)).astype(np.float32)    # (time, lev)
        nd_anom_s = _strip_coords(Nd_anom_o.isel(site=si)).astype(np.float32)      # (time, lev)
        vts       = _strip_coords(valid_ts_o.isel(site=si)).astype("uint8")        # (time,)
        nd_raw_m  = _strip_coords(Nd_raw_tmean.isel(site=si)).astype(np.float32)   # (lev,)
        nd_fill_m = _strip_coords(Nd_filled_tmean.isel(site=si)).astype(np.float32)# (lev,)
        nd_anom_m = _strip_coords(Nd_anom_tmean.isel(site=si)).astype(np.float32)  # (lev,)
        vfrac     = float(valid_frac.isel(site=si).values)

        # enforce dim order explicitly (time first)
        nd_raw    = nd_raw.transpose("time", "lev")
        nd_fill   = nd_fill.transpose("time", "lev")
        nd_anom_s = nd_anom_s.transpose("time", "lev")

        ds = xr.Dataset(
            data_vars=dict(
                Nd_raw=nd_raw,
                Nd_filled=nd_fill,
                Nd_anom=nd_anom_s,
                valid_ts=vts,
                Nd_raw_tmean=nd_raw_m,
                Nd_filled_tmean=nd_fill_m,
                Nd_anom_tmean=nd_anom_m,
                valid_frac=xr.DataArray(vfrac),
            ),
            coords=dict(
                time=Nd_sites["time"],   # clean time coord
                lev=Nd_sites["lev"],     # clean lev coord
            ),
            attrs=dict(
                site_name=str(sname),
                site_index=int(si),
            ),
        )

        fn = os.path.join(out_dir, f"{prefix}_{si:03d}_{_safe(sname)}.nc")
        ds.to_netcdf(
            fn,
            unlimited_dims=["time"],  # helps CDO and CF readers
        )


def fillna_with_site_mean_if_min_lev(
    Nd_sites: xr.DataArray,
    *,
    min_valid_lev: int = 2,
    min_valid_times_per_level_for_mean: int = 3,
    min_valid_days_for_site_mean: int = 3,
    dim_time: str = "time",
    dim_lev: str = "lev",
    dim_site: str = "site",
):
    """
    Fill NaNs in Nd_sites with the site-mean vertical profile (mean over time),
    but ONLY when both conditions hold:

      1) Site-level gate:
         the site must have at least `min_valid_days_for_site_mean` days
         whose (time,site) profile has at least `min_valid_lev` finite levels.

      2) Level-level gate:
         a given (lev,site) mean is only defined if that level has at least
         `min_valid_times_per_level_for_mean` finite time points.

    If either gate fails, the corresponding mean is NaN and nothing is filled there.
    """
    for d in (dim_time, dim_lev, dim_site):
        if d not in Nd_sites.dims:
            raise ValueError(f"Nd_sites must contain dim '{d}'. Got dims={Nd_sites.dims}")

    finite = xr.ufuncs.isfinite(Nd_sites)

    # ------------------------------------------------------------
    # Site-level gate:
    # valid day = at least `min_valid_lev` finite levels in that (time,site) profile
    # ------------------------------------------------------------
    n_valid_lev = finite.sum(dim_lev)  # (time,site)
    valid_day_for_site_mean = (n_valid_lev >= int(min_valid_lev))  # (time,site)
    n_valid_days_site = valid_day_for_site_mean.sum(dim_time)  # (site,)
    site_ok_for_mean = (n_valid_days_site >= int(min_valid_days_for_site_mean))  # (site,)

    # ------------------------------------------------------------
    # Raw mean over time
    # ------------------------------------------------------------
    Nd_mean_site = Nd_sites.mean(dim_time, skipna=True)  # (lev,site)

    # ------------------------------------------------------------
    # Level-level gate:
    # need at least `min_valid_times_per_level_for_mean` finite time points at this level
    # ------------------------------------------------------------
    n_valid_times_per_level = finite.sum(dim_time)  # (lev,site)
    level_ok_for_mean = (n_valid_times_per_level >= int(min_valid_times_per_level_for_mean))

    # Apply both gates to the mean profile
    Nd_mean_site = Nd_mean_site.where(level_ok_for_mean)
    Nd_mean_site = Nd_mean_site.where(site_ok_for_mean)

    # ------------------------------------------------------------
    # Filling eligibility:
    # only fill columns that already have at least `min_valid_lev` finite levels
    # and belong to sites that passed the site-level gate
    # ------------------------------------------------------------
    eligible = valid_day_for_site_mean & site_ok_for_mean  # (time,site)
    eligible_3d = eligible.broadcast_like(Nd_sites)

    Nd_filled_candidate = Nd_sites.fillna(Nd_mean_site)
    Nd_filled = xr.where(eligible_3d, Nd_filled_candidate, Nd_sites)

    return Nd_filled, eligible, Nd_mean_site

###############################3

def fillna_with_site_level_normal_site_mean_mask(
    Nd_sites: xr.DataArray,
    *,
    seed: int = 0,
    min_count: int = 3,
    ddof: int = 1,
    clip_min: float = 0.0,
) -> xr.DataArray:
    """
    Fill NaNs in Nd_sites(time, lev, site) by sampling N(mu, sigma) estimated per (lev, site).
    Mask rule: do NOT fill if the site+level mean mu(lev,site) is NaN.

    Parameters
    ----------
    Nd_sites : xr.DataArray
        dims: ('time','lev','site')
    seed : int
        RNG seed for reproducibility.
    min_count : int
        Minimum finite samples along time to trust sigma; otherwise sigma->0 (fill with mu).
    ddof : int
        ddof for std.
    clip_min : float
        Clip finite filled values to >= clip_min (Nd should be >=0).

    Returns
    -------
    xr.DataArray with same shape/coords as Nd_sites
    """

    # --- per-(lev,site) climatology
    mu  = Nd_sites.mean("time", skipna=True)                     # (lev, site)
    cnt = xr.ufuncs.isfinite(Nd_sites).sum("time")               # (lev, site)
    sig = Nd_sites.std("time", skipna=True, ddof=ddof)           # (lev, site)

    # if not enough samples or sigma invalid, force sigma=0 => fill becomes exactly mu
    sig = sig.where((cnt >= min_count) & xr.ufuncs.isfinite(sig) & (sig > 0), other=0.0)

    # --- RNG samples with broadcasting
    rng = np.random.default_rng(seed)
    mu_np  = mu.transpose("lev", "site").values[None, :, :]      # (1, lev, site)
    sig_np = sig.transpose("lev", "site").values[None, :, :]     # (1, lev, site)

    samp = rng.normal(loc=mu_np, scale=sig_np, size=Nd_sites.shape)
    samp = xr.DataArray(samp, dims=Nd_sites.dims, coords=Nd_sites.coords)

    # If mu is NaN, keep samp NaN (enforces "mask where Nd_mean_site is NaN")
    samp = samp.where(xr.ufuncs.isfinite(mu.broadcast_like(Nd_sites)))

    # --- fill only where Nd_sites is missing
    miss = ~xr.ufuncs.isfinite(Nd_sites)
    Nd_filled = Nd_sites.where(~miss, other=samp)

    # --- clip to enforce Nd >= 0
    if clip_min is not None:
        Nd_filled = xr.where(
            xr.ufuncs.isfinite(Nd_filled),
            xr.ufuncs.maximum(Nd_filled, clip_min),
            Nd_filled
        )

    return Nd_filled


def project_coeffs_at_sites_only(
    Nout, varname, EOF, Nd_mean, A_mean,
    site_json_path,
    min_valid_lev=1,
    debug_out_dir="sites",
    debug_prefix="filldbg",
    write_debug=True,
    only_site=None,
    site_var_dir=None,
    site_prefix="",
    minP=250,
):
    """
    Site-only EOF projection using per-site NetCDFs.

    Returns
    -------
    A_resid_sites : (time, mode, site) float32
    Ns            : (time, lev, site) float32   (conditioned Nd used for projection)
    Es            : (time, lev, site) float32   (conditioned Nd 1-sigma error, filled with level mean)
    site_meta     : xr.Dataset(site) with ilat/ilon and lat/lon coords
    """
    # -------------------------
    # local helpers
    # -------------------------
    def _maybe_wrap_lon_to_grid(lon_site, lon_grid_vals):
        lg = np.asarray(lon_grid_vals, dtype=np.float64)
        lo = float(np.nanmin(lg))
        hi = float(np.nanmax(lg))
        x = float(lon_site)

        if lo >= 0.0 and hi > 180.0:      # 0..360 grid
            if x < 0.0:
                x += 360.0
            if x >= 360.0:
                x -= 360.0
            return x

        if lo < 0.0 and hi <= 180.0:      # -180..180 grid
            if x > 180.0:
                x -= 360.0
            return x

        return x

    def _nearest_1d_index(grid_vals, x):
        gv = np.asarray(grid_vals, dtype=np.float64)
        return int(np.argmin(np.abs(gv - float(x))))

    def _latlon_to_ilat_ilon(lat_grid, lon_grid, site_lats, site_lons):
        latg = np.asarray(lat_grid)
        long = np.asarray(lon_grid)
        ilat = np.empty(len(site_lats), dtype=np.int64)
        ilon = np.empty(len(site_lons), dtype=np.int64)
        for i, (la, lo) in enumerate(zip(site_lats, site_lons)):
            ilat[i] = _nearest_1d_index(latg, la)
            lo2 = _maybe_wrap_lon_to_grid(lo, long)
            ilon[i] = _nearest_1d_index(long, lo2)
        return ilat, ilon

    def _load_sites_catalog(path):
        site_names, site_lats, site_lons = [], [], []
        ilat, ilon = [], []

        if str(path).endswith(".jsonl"):
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    if "site" not in d:
                        continue
                    site_names.append(str(d["site"]))
                    site_lats.append(float(d.get("lat", np.nan)))
                    site_lons.append(float(d.get("lon", np.nan)))
            return site_names, np.asarray(site_lats, np.float64), np.asarray(site_lons, np.float64), None, None

        with open(path, "r") as f:
            J = json.load(f)

        for name, d in J.items():
            if ("geos_ilat" in d) and ("geos_ilon" in d):
                site_names.append(str(name))
                ilat.append(int(d["geos_ilat"]))
                ilon.append(int(d["geos_ilon"]))
                site_lats.append(float(d.get("lat", np.nan)))
                site_lons.append(float(d.get("lon", np.nan)))
            elif ("lat" in d) and ("lon" in d):
                site_names.append(str(name))
                site_lats.append(float(d["lat"]))
                site_lons.append(float(d["lon"]))

        ilat = np.asarray(ilat, np.int64) if len(ilat) else None
        ilon = np.asarray(ilon, np.int64) if len(ilon) else None
        return site_names, np.asarray(site_lats, np.float64), np.asarray(site_lons, np.float64), ilat, ilon

    def _resolve_site_nc(var_dir, site, varname, prefix):
        if prefix and (("*" in prefix) or ("?" in prefix) or str(prefix).endswith(".nc")):
            cand = os.path.join(var_dir, prefix)
            hits = sorted(glob.glob(cand))
            if hits:
                for h in hits:
                    bn = os.path.basename(h)
                    if f".{site}." in bn and f".{varname}." in bn:
                        return h
                for h in hits:
                    if f".{site}." in os.path.basename(h):
                        return h
                return hits[0]

        if prefix:
            fn = os.path.join(var_dir, f"Cloudnet.{site}.{varname}.tavg1D.{prefix}.nc")
            if os.path.exists(fn):
                return fn

        pat = os.path.join(var_dir, f"Cloudnet.{site}.{varname}.tavg1D.*.nc")
        hits = sorted(glob.glob(pat))
        if not hits:
            raise FileNotFoundError(f"No per-site file for site='{site}' var='{varname}' under {var_dir}")
        return hits[0]

    def _open_dataset_compat(fn, chunks):
        try:
            return xr.open_dataset(fn, chunks=chunks, use_cftime=False)
        except TypeError:
            return xr.open_dataset(fn, chunks=chunks)

    def _fill_error_with_level_mean(E):
        """
        E: (time, lev, site)
        Fill NaNs by mean over time per (lev,site).
        Do NOT fill where that mean is NaN.
        """
        Em = E.mean("time", skipna=True)  # (lev,site)
        Ef = E.fillna(Em)                 # fill where mean exists
        Ef = Ef.where(xr.ufuncs.isfinite(Em.broadcast_like(E)))  # keep NaN if no mean
        return Ef

    # -------------------------
    # 0) sanity
    # -------------------------
    if site_var_dir is None:
        raise ValueError("project_coeffs_at_sites_only: site_var_dir must be provided.")
    if ("lat" not in Nout.coords) or ("lon" not in Nout.coords) or ("time" not in Nout.coords):
        raise ValueError("project_coeffs_at_sites_only: Nout must have coords: time, lat, lon.")

    # -------------------------
    # 1) Load sites + map to grid
    # -------------------------
    site_names, site_lats, site_lons, ilat0, ilon0 = _load_sites_catalog(site_json_path)

    if only_site is not None:
        key = only_site.strip().lower()
        keep = [i for i, s in enumerate(site_names) if str(s).strip().lower() == key]
        if not keep:
            raise ValueError(f"only_site='{only_site}' not found. Example sites: {site_names[:20]}")
        i0 = keep[0]
        site_names = [site_names[i0]]
        site_lats  = np.asarray([site_lats[i0]], dtype=np.float64)
        site_lons  = np.asarray([site_lons[i0]], dtype=np.float64)
        if ilat0 is not None and ilon0 is not None:
            ilat0 = np.asarray([ilat0[i0]], dtype=np.int64)
            ilon0 = np.asarray([ilon0[i0]], dtype=np.int64)

    Ny, Nx = Nout.sizes["lat"], Nout.sizes["lon"]
    if (ilat0 is not None) and (ilon0 is not None) and (len(ilat0) == len(site_names)):
        ilat = np.asarray(ilat0, dtype=np.int64)
        ilon = np.asarray(ilon0, dtype=np.int64)
    else:
        if np.any(~np.isfinite(site_lats)) or np.any(~np.isfinite(site_lons)):
            raise ValueError("Site catalog missing lat/lon; cannot compute ilat/ilon.")
        ilat, ilon = _latlon_to_ilat_ilon(Nout["lat"].values, Nout["lon"].values, site_lats, site_lons)

    inb = (ilat >= 0) & (ilat < Ny) & (ilon >= 0) & (ilon < Nx)
    if not np.any(inb):
        raise ValueError("No in-bounds sites after mapping to Abase grid.")

    site_names = [s for s, ok in zip(site_names, inb) if ok]
    site_lats  = site_lats[inb]
    site_lons  = site_lons[inb]
    ilat       = ilat[inb]
    ilon       = ilon[inb]

    # -------------------------
    # 2) Load Nd + Nd_error from per-site files
    # -------------------------
    da_list = []
    ea_list = []
    file_lat = []
    file_lon = []

    t_template = Nout["time"]
    errname = f"{varname}_error"

    for s in site_names:
        fn = _resolve_site_nc(site_var_dir, s, varname, site_prefix)
        ds = _open_dataset_compat(fn, chunks={"time": cfg.t_chk, "lev": -1})
        ds = normalize_time_coord(ds)

        if "lev" in ds.coords:
            ds = ds.sel(lev=ds["lev"].where(ds["lev"] >= minP, drop=True))

        if varname not in ds:
            raise KeyError(f"{fn} missing variable '{varname}'")

        Nd = ds[varname].reindex(time=t_template)                  # (time,lev)
        Nd = Nd.expand_dims(site=[s]).astype(np.float32)
        da_list.append(Nd)

        if errname in ds:
            E = ds[errname].reindex(time=t_template)
        else:
            # allow missing error var; will remain NaN and later filling will keep NaN if no mean
            E = xr.full_like(ds[varname], np.nan).reindex(time=t_template)

        E = E.expand_dims(site=[s]).astype(np.float32)
        ea_list.append(E)

        file_lat.append(float(ds["lat"].values) if "lat" in ds else np.nan)
        file_lon.append(float(ds["lon"].values) if "lon" in ds else np.nan)

    Nd_sites = xr.concat(da_list, dim="site").transpose("time", "lev", "site")
    E_sites  = xr.concat(ea_list, dim="site").transpose("time", "lev", "site")

    Nd_sites = Nd_sites.assign_coords(site=("site", np.asarray(site_names, dtype="U")))
    E_sites  = E_sites.assign_coords(site=("site", np.asarray(site_names, dtype="U")))

    # interp to EOF lev if needed (do same for error so shapes match)
    if ("lev" in Nd_sites.coords) and (not np.array_equal(Nd_sites["lev"].values, EOF["lev"].values)):
        Nd_sites = Nd_sites.interp(lev=EOF["lev"])
        E_sites  = E_sites.interp(lev=EOF["lev"])

    # -------------------------
    # 3) Nd_mean on lev (as before)
    # -------------------------
    if isinstance(Nd_mean, xr.DataArray):
        Nd_mean_da = Nd_mean
        if "lev" not in Nd_mean_da.dims and Nd_mean_da.ndim == 1 and Nd_mean_da.size == Nd_sites.sizes["lev"]:
            Nd_mean_da = xr.DataArray(Nd_mean_da.values, dims=("lev",), coords={"lev": Nd_sites["lev"]})
    else:
        Nd_mean_arr = np.asarray(Nd_mean)
        if Nd_mean_arr.ndim == 0:
            Nd_mean_da = xr.DataArray(float(Nd_mean_arr))
        elif Nd_mean_arr.ndim == 1 and Nd_mean_arr.size == Nd_sites.sizes["lev"]:
            Nd_mean_da = xr.DataArray(Nd_mean_arr, dims=("lev",), coords={"lev": Nd_sites["lev"]})
        else:
            Nd_mean_da = xr.DataArray(float(np.nanmean(Nd_mean_arr)))

    # -------------------------
    # 4) FILLING LOGIC for Nd + error
    # -------------------------
    Ns = fillna_with_site_level_normal_site_mean_mask(
        Nd_sites, seed=42, min_count=10, ddof=1, clip_min=0.0
    )

    # error: fill with mean error at that (lev,site)
    Es = _fill_error_with_level_mean(E_sites)

    # keep error NaN wherever Ns is NaN (site/level completely missing)
    Es = Es.where(xr.ufuncs.isfinite(Ns))

    # -------------------------
    # 5) Projection to coefficients (unchanged)
    # -------------------------
    NsL = Ns.load()
    Nd_anom = (NsL - Nd_mean_da).transpose("time", "lev", "site")

    E = EOF.transpose("mode", "lev").values.astype(np.float32)   # (M,Z)
    T = Nd_anom.sizes["time"]
    M = EOF.sizes["mode"]
    S = Nd_anom.sizes["site"]

    A_abs = np.full((T, M, S), np.nan, dtype=np.float32)

    v_level_site = xr.ufuncs.isfinite(NsL).any("time")  # (lev,site)
    for si in range(S):
        lev_mask = v_level_site[:, si].values
        if lev_mask.sum() < int(min_valid_lev):
            continue
        Ek = E[:, lev_mask]          # (M,Zk)
        G = Ek @ Ek.T
        Ginv = np.linalg.pinv(G, rcond=1e-6)
        y = Nd_anom[:, lev_mask, si].values
        A_abs[:, :, si] = (y @ Ek.T) @ Ginv

    A_sites = xr.DataArray(
        A_abs,
        dims=("time", "mode", "site"),
        coords={"time": NsL["time"], "mode": EOF["mode"], "site": NsL["site"]},
    ).astype(np.float32)

    valid_lev = xr.ufuncs.isfinite(NsL)
    n_valid_lev = valid_lev.sum("lev")
    valid_ts = (n_valid_lev >= int(min_valid_lev))
    A_sites = A_sites.where(valid_ts)

    # -------------------------
    # 6) Subtract A_mean sampled at sites (unchanged)
    # -------------------------
    A_mean_xr = xr.DataArray(
        np.asarray(A_mean, dtype=np.float32),
        dims=("mode", "lat", "lon"),
        coords={"mode": EOF["mode"], "lat": Nout["lat"], "lon": Nout["lon"]},
        name="A_mean",
    )
    A_mean_sites = A_mean_xr.isel(
        lat=xr.DataArray(ilat, dims="site"),
        lon=xr.DataArray(ilon, dims="site"),
    ).assign_coords(site=("site", np.asarray(site_names, dtype="U")))

    A_resid_sites = (A_sites - A_mean_sites).where(valid_ts).astype(np.float32)

    # -------------------------
    # 7) site_meta (unchanged)
    # -------------------------
    geos_lat = Nout["lat"].isel(lat=xr.DataArray(ilat, dims="site")).values
    geos_lon = Nout["lon"].isel(lon=xr.DataArray(ilon, dims="site")).values

    site_meta = xr.Dataset(
        data_vars=dict(
            ilat=("site", ilat.astype(np.int64)),
            ilon=("site", ilon.astype(np.int64)),
            site_lat=("site", geos_lat.astype(np.float64)),
            site_lon=("site", geos_lon.astype(np.float64)),
            obs_lat=("site", np.asarray(site_lats, np.float64)),
            obs_lon=("site", np.asarray(site_lons, np.float64)),
            file_lat=("site", np.asarray(file_lat, np.float64)),
            file_lon=("site", np.asarray(file_lon, np.float64)),
        ),
        coords=dict(site=("site", np.asarray(site_names, dtype="U"))),
    )

    # -------------------------
    # 8) debug (optional)
    # -------------------------
    if write_debug:
        try:
            os.makedirs(debug_out_dir, exist_ok=True)
            save_site_fill_debug_netcdfs(
                Nd_sites=Nd_sites,
                Nd_filled=Ns,
                Nd_anom=Nd_anom,
                valid_ts=valid_ts,
                site_names=site_names,
                out_dir=debug_out_dir,
                prefix=debug_prefix
            )
        except Exception as e:
            print("[WARN] debug netcdf save failed:", repr(e))

    return A_resid_sites, Ns, Es, site_meta

# ============================================================
# Main loader for the new training setup
# ============================================================

def get_data(
    vars_out=None,
    pth_all="",
    vars_in_2d=None,
    modis_source=None,
    prefix="",
    dense_prefix=None,
    site_prefix=None,
    cache_folder="cache",
    minP=250,
    site_json_path=None,
):
    """
    Returns objects suitable for streaming IterableDataset:

      Nin_2d    : xr.Dataset   (time, lat, lon)
      A_sites   : xr.DataArray (time, mode, site)      site-projected coeff residuals
      Nd_sites  : xr.DataArray (time, lev, site)       conditioned Nd used for projection (Ns)
      Ares_base : xr.DataArray (time, mode, lat, lon)  dense base residual coeffs
      site_meta : xr.Dataset   (site) with ilat/ilon and lat/lon info

    Notes
    -----
    - dense_prefix controls the dense base + MODIS inputs
    - site_prefix controls which per-site files are selected
    - if not provided, both default to the legacy `prefix` behavior
    """
    if vars_out is None:
        vars_out = {}
    if vars_in_2d is None:
        vars_in_2d = {}
    if site_json_path is None:
        site_json_path = cfg.SITE_JSON

    # backward-compatible defaults
    if dense_prefix is None:
        dense_prefix = prefix
    if site_prefix is None:
        site_prefix = prefix

    EOF     = cfg.EOF
    Nd_mean = cfg.Nd_mean
    A_mean  = cfg.A_mean

    chk2d = {"time": cfg.t_chk, "lat": -1, "lon": -1}
    chk3d = {"time": cfg.t_chk, "lev": -1, "lat": -1, "lon": -1}

    print(f"[INFO] dense_prefix={dense_prefix!r}")
    print(f"[INFO] site_prefix={site_prefix!r}")

    # ------------------------------------------------------------
    # 1) Load base residual coefficients (dense), defines template grid/time
    # ------------------------------------------------------------
    varname_base = list(vars_out.keys())[1]
    pth_base = f"{vars_out[varname_base]}/{dense_prefix}"
    print("[LOAD 3D OUT base]", pth_base, len(sorted(glob.glob(pth_base))), "files")

    Ab = xr.open_mfdataset(pth_base, parallel=False, chunks=chk3d)[varname_base]
    Ares_base = normalize_time_coord(Ab)
    Ares_base = Ares_base.where(Ares_base > -1e9).where(Ares_base < 1e9)

    # Template Dataset (Abase reference) for mapping sites
    Nout_template = xr.Dataset(coords=dict(
        time=Ares_base["time"],
        lat=Ares_base["lat"],
        lon=Ares_base["lon"],
    ))

    # ------------------------------------------------------------
    # 2) Load MODIS 2D inputs, normalize
    # ------------------------------------------------------------
    pth2d = f"{pth_all}dailies/{modis_source}/on_Noutgrid/{modis_source}.{dense_prefix}.nc"
    print("[LOAD 2D IN]", pth2d)
    t0 = time.time()

    ds2d = xr.open_mfdataset(pth2d, parallel=False, chunks=chk2d)

    for v in ds2d.data_vars:
        mu = float(vars_in_2d[v]["mean"])
        sd = float(vars_in_2d[v]["std"])
        ds2d[v] = (ds2d[v] - mu) / sd

    Nin_2d = ds2d
    print(f"[DBG] open_mfdataset(ds2d) done in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------
    # 3) Final alignment: Nin_2d and Ares_base share exact time
    # ------------------------------------------------------------
    Nin_2d, Ares_base = xr.align(Nin_2d, Ares_base, join="inner")

    # keep template time consistent after alignment
    Nout_template = Nout_template.sel(time=Ares_base["time"])

    Nin_2d    = Nin_2d.transpose("time", "lat", "lon")
    Ares_base = Ares_base.transpose("time", "mode", "lat", "lon")

    _print_da_info("Nin_2d", Nin_2d)
    _print_da_info("Ares_base", Ares_base)

    # ------------------------------------------------------------
    # 4) Site-only projection: observed Nd comes from per-site files
    # ------------------------------------------------------------
    varname_obs = list(vars_out.keys())[0]

    A_sites, Nd_sites, Err_sites, site_meta = project_coeffs_at_sites_only(
        Nout=Nout_template,                  # Abase reference
        varname=varname_obs,
        EOF=EOF,
        Nd_mean=Nd_mean,
        A_mean=A_mean,
        site_json_path=site_json_path,
        min_valid_lev=cfg.min_valid_lev,
        write_debug=cfg.write_site_debug,
        site_var_dir=vars_out[varname_obs],  # directory holding per-site NetCDFs
        site_prefix=site_prefix,             # <-- decoupled from dense_prefix
        minP=minP,
    )

    # Ensure consistent ordering & dtypes for training
    A_sites  = A_sites.transpose("time", "mode", "site").astype(np.float32)
    Nd_sites = Nd_sites.transpose("time", "lev", "site").astype(np.float32)

    _print_da_info("A_sites", A_sites)
    _print_da_info("Nd_sites", Nd_sites)
    print(f"[INFO] #sites: {site_meta.dims.get('site', '??')}")

    # Fill NaNs in 2D inputs only
    Nin_2d = Nin_2d.fillna(0)

    return Nin_2d, A_sites, Nd_sites, Err_sites, Ares_base, site_meta
    
    
    #############################################

# ---------------------------------------------
# Reconstruct Nd on full grid (chunked in time)
# ---------------------------------------------
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
    site_meta,       # xr.Dataset with site coord + ilat/ilon
    EOF,             # xr.DataArray (mode, lev)
    Nd_mean,         # xr.DataArray (lev,) or array-like
    A_mean,          # (mode, lat, lon) climatological mean coeff field OR xr.DataArray
    valid_ref=None,  # xr.DataArray like Nd_sites (time, lev, site) OR (time, site)
    mask_mode="obs_dates",  # "obs_levels" | "obs_dates" | "none"
):
    """
    Returns:
      Nd_rec       : xr.DataArray (lev, time, site)
      A_abs_sites  : xr.DataArray (time, mode, site)
      A_mean_sites : xr.DataArray (mode, site)
    """

    if mask_mode not in ("obs_levels", "obs_dates", "none"):
        raise ValueError(f"mask_mode must be one of: 'obs_levels', 'obs_dates', 'none'. Got: {mask_mode}")

    # --- align sites defensively (avoids silent reordering issues) ---
    if "site" in A_sites_resid.coords and "site" in site_meta.coords:
        common = np.intersect1d(A_sites_resid["site"].values, site_meta["site"].values)
        if common.size == 0:
            raise ValueError("No common site labels between A_sites_resid and site_meta.")
        A_sites_resid = A_sites_resid.sel(site=common)
        site_meta = site_meta.sel(site=common)
        if valid_ref is not None and "site" in valid_ref.dims:
            valid_ref = valid_ref.sel(site=common)

    # standard order
    Ares = A_sites_resid.transpose("time", "mode", "site")

    ilat = site_meta["ilat"].values.astype(np.int64)
    ilon = site_meta["ilon"].values.astype(np.int64)

    # --- wrap A_mean as DataArray ---
    if isinstance(A_mean, xr.DataArray):
        Amean_mhw = A_mean
    else:
        A_mean_np = np.asarray(A_mean, dtype=np.float32)
        if A_mean_np.ndim != 3:
            raise ValueError(f"A_mean must be 3D (mode,lat,lon). Got shape {A_mean_np.shape}")
        mode_coord = EOF["mode"] if (EOF.sizes["mode"] == A_mean_np.shape[0]) else np.arange(A_mean_np.shape[0])
        Amean_mhw = xr.DataArray(
            A_mean_np,
            dims=("mode", "lat", "lon"),
            coords={"mode": mode_coord},
            name="A_mean",
        )

    # --- identify spatial dims robustly ---
    if "lat" in Amean_mhw.dims and "lon" in Amean_mhw.dims:
        lat_dim, lon_dim = "lat", "lon"
    else:
        spatial_dims = [d for d in Amean_mhw.dims if d != "mode"]
        if len(spatial_dims) < 2:
            raise ValueError(f"A_mean must have 2 spatial dims besides mode. Got dims {Amean_mhw.dims}")
        lat_dim, lon_dim = spatial_dims[0], spatial_dims[1]

    # --- align/crop mode if needed ---
    try:
        Amean_mhw = Amean_mhw.sel(mode=EOF["mode"])
    except Exception:
        M = min(Amean_mhw.sizes["mode"], EOF.sizes["mode"])
        Amean_mhw = Amean_mhw.isel(mode=slice(0, M))
        Ares = Ares.isel(mode=slice(0, M))
        EOF = EOF.isel(mode=slice(0, M))

    # --- sample A_mean at sites ---
    A_mean_sites = Amean_mhw.isel(
        {
            lat_dim: xr.DataArray(ilat, dims="site"),
            lon_dim: xr.DataArray(ilon, dims="site"),
        }
    ).astype(np.float32)

    # attach site coord if needed
    if "site" not in A_mean_sites.coords and "site" in Ares.coords:
        A_mean_sites = A_mean_sites.assign_coords(site=Ares["site"])

    # ============================================================
    # FIX: rename sampled-grid aux coords lat/lon -> site_lat/site_lon
    # This prevents collisions with per-site file aux coords lat/lon.
    # ============================================================
    rename_map = {}
    if "lat" in A_mean_sites.coords and "lat" not in A_mean_sites.dims:
        rename_map["lat"] = "site_lat"
    if "lon" in A_mean_sites.coords and "lon" not in A_mean_sites.dims:
        rename_map["lon"] = "site_lon"
    if rename_map:
        A_mean_sites = A_mean_sites.rename(rename_map)

    # --- absolute coeffs ---
    A_abs_sites = (A_mean_sites + Ares).astype(np.float32)

    # --- make Nd_mean broadcastable on lev ---
    if not isinstance(Nd_mean, xr.DataArray):
        Nd_mean = xr.DataArray(np.asarray(Nd_mean, dtype=np.float32), dims=("lev",), coords={"lev": EOF["lev"]})
    else:
        try:
            Nd_mean = Nd_mean.sel(lev=EOF["lev"])
        except Exception:
            Nz = min(Nd_mean.sizes.get("lev", Nd_mean.size), EOF.sizes["lev"])
            Nd_mean = Nd_mean.isel(lev=slice(0, Nz))
            EOF = EOF.isel(lev=slice(0, Nz))

    # --- reconstruct Nd (time,lev,site) ---
    Nd_tsl = xr.dot(A_abs_sites, EOF, dims="mode")          # (time, site, lev)
    Nd_tsl = Nd_tsl.transpose("time", "lev", "site")
    Nd_tsl = (Nd_tsl + Nd_mean).astype(np.float32).clip(min=0.0)

    # --- masking using valid_ref ---
    if (mask_mode != "none") and (valid_ref is not None):
        # reindex onto Nd_tsl grid (prevents silent misalignment)
        try:
            if "time" in valid_ref.dims:
                valid_ref = valid_ref.reindex(time=Nd_tsl["time"])
            if "site" in valid_ref.dims:
                valid_ref = valid_ref.reindex(site=Nd_tsl["site"])
            if "lev" in valid_ref.dims:
                valid_ref = valid_ref.reindex(lev=Nd_tsl["lev"])
        except Exception:
            pass

        v_obs = xr.ufuncs.isfinite(valid_ref)

        if mask_mode == "obs_levels":
            mask3d = v_obs if "lev" in v_obs.dims else v_obs.broadcast_like(Nd_tsl)
            Nd_tsl = Nd_tsl.where(mask3d)

        elif mask_mode == "obs_dates":
            v_ts = v_obs.any("lev") if "lev" in v_obs.dims else v_obs
            Nd_tsl = Nd_tsl.where(v_ts.broadcast_like(Nd_tsl))

    # --- keep permanently-missing (lev,site) as NaN ---
    if valid_ref is not None and ("lev" in valid_ref.dims):
        v_obs2 = xr.ufuncs.isfinite(valid_ref)
        try:
            v_obs2 = v_obs2.reindex(lev=Nd_tsl["lev"], site=Nd_tsl["site"])
        except Exception:
            pass
        v_level_site = v_obs2.any("time")  # (lev,site)
        Nd_tsl = Nd_tsl.where(v_level_site.broadcast_like(Nd_tsl))

    Nd_lts = Nd_tsl.transpose("lev", "time", "site").rename("Nd_rec")

    # attach site_meta coords (won't collide with 'lat' anymore)
    for c in ["site_lat", "site_lon", "obs_lat", "obs_lon", "file_lat", "file_lon"]:
        if c in site_meta and c not in Nd_lts.coords:
            Nd_lts = Nd_lts.assign_coords({c: ("site", site_meta[c].values)})

    return Nd_lts, A_abs_sites, A_mean_sites

