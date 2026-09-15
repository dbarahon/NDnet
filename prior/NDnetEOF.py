# quick subroutine to analyze the W data
import sys
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import hashlib
#import netCDF4 as nc
import os
import glob
import xarray as xr
import time
from dask.distributed import Client, LocalCluster
from datetime import datetime, timedelta
#from scipy import interpolate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchsummary import summary as torchsummary_summary
import json
import gc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import xesmf as xe
from torchinfo import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
from numba import njit
from cache_utils import (
    hash_grid,
    cache_path,
    load_array_cache,
    save_array_cache,
    save_vis_nc,
    force_datetime64_ns, 
    normalize_time_coord,
    safe_time_intersection
)

from matplotlib.gridspec import GridSpec
np.set_printoptions(precision=3)

# -----------------------------
# EOF BASIS (vertical only)
# -----------------------------
EOF_ds = xr.open_dataset("Nd_EOF.nc")

EOF = EOF_ds["EOFs"]          # (mode, lev)
Nd_mean = EOF_ds["Nd_mean"]   # (lev)

num_modes = EOF.sizes["mode"]


# -----------------------------------
# Mode weights (variance-aware)
# -----------------------------------
# If eigenvalues were saved, use them.
# Otherwise approximate using EOF energy.

mode_var = np.sum(EOF.values**2, axis=1)

mode_weights = torch.tensor(
    mode_var / mode_var.mean(),
    dtype=torch.float32
)


# Utilities
# -----------------------------


# ============================================================
# SIMPLE, ROBUST MODEL SAVE / LOAD (WEIGHTS ONLY)
# ============================================================

def save_model_weights(model, path):
    """
    Save ONLY model weights (DP-safe, pickle-free).
    """
    sd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(sd, path)
    print(f"[SAVE] Model weights saved to {path}")


def load_model_weights(model, path, device):
    """
    Load ONLY model weights into a NON-DataParallel model.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weights not found: {path}")

    sd = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()

    #print(f"[LOAD] Model weights loaded from {path}")

def save_training_checkpoint(model, optimizer, criterion, epoch, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "alpha": criterion.alpha.detach().cpu().numpy(),
        },
        path
    )
    #print(f"[CKPT] Training checkpoint saved at epoch {epoch}")


def load_training_checkpoint(
    model,
    optimizer,
    criterion,
    ckpt_path,
    device,
):
    """
    Load training checkpoint BEFORE DataParallel wrapping.

    Returns:
        start_epoch (int)
    """
    if not os.path.exists(ckpt_path):
        print("[CKPT] No training checkpoint found")
        return 0

    print(f"[CKPT] Loading training checkpoint: {ckpt_path}")

    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False   # trusted, training only
    )

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if "alpha" in ckpt:
        criterion.alpha.data = torch.tensor(
            ckpt["alpha"],
            dtype=torch.float32,
            device=device
        )

    start_epoch = ckpt["epoch"] + 1
    print(f"[CKPT] ----------------Resuming from epoch {start_epoch}")

    return start_epoch


@njit
def project_masked_profiles(Nd2d, EOFm):
    """
    Nd2d: [Ncells, Nz]
    EOFm: [Nmodes, Nz]
    Returns: [Ncells, Nmodes]
    """
    Ncells, Nz = Nd2d.shape
    Nmodes = EOFm.shape[0]

    # output
    A = np.full((Ncells, Nmodes), np.nan, dtype=np.float64)

    for i in range(Ncells):
        # find valid levels
        n_valid = 0
        for k in range(Nz):
            if Nd2d[i, k] == Nd2d[i, k]:  # fast isnan check
                n_valid += 1

        # fully missing column \u2192 keep NaNs
        if n_valid == 0:
            continue

        # allocate temp arrays for valid levels
        a_tmp = np.empty(n_valid, dtype=np.float64)
        E_tmp = np.empty((Nmodes, n_valid), dtype=np.float64)

        # fill them
        pos = 0
        for k in range(Nz):
            if Nd2d[i, k] == Nd2d[i, k]:
                a_tmp[pos] = Nd2d[i, k]
                for m in range(Nmodes):
                    E_tmp[m, pos] = EOFm[m, k]
                pos += 1

        # compute coefficients
        for m in range(Nmodes):
            num = 0.0
            den = 0.0
            for j in range(n_valid):
                num += E_tmp[m, j] * a_tmp[j]
                den += E_tmp[m, j] * E_tmp[m, j]
            if den > 0.0:
                A[i, m] = num / den
            else:
                A[i, m] = np.nan

    return A


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


def stand_by_cell(ds, invert=False, var='N', lv=[], pth_all='', min_std=0.0):
    # assume we have std and mean saved in the same folder
    pth = f'{pth_all}stats/{var}_stats.nc4'
    dst = xr.open_mfdataset(pth, parallel=True)
   
    if lv:
        ds_mean = dst[var + '_mean'].sel(lev=lv).mean()
        ds_std = dst[var + '_std'].sel(lev=lv).mean()
    else:
        ds_mean = dst[var + '_mean'].mean()
        ds_std = np.clip(dst[var + '_std'], min_std, None).mean()
    
    print(var, ' mean = ', ds_mean.load(), ' std = ', ds_std.load())
     
    if not invert:
        ds = (ds - ds_mean) / ds_std
    else:
        ds = ds * ds_std + ds_mean
    return ds
    

def stand_by_cell_out(ds, invert=False, var='NCPL_VOL'):
    """
    Standardization / de-standardization for Nd for output.
    """
    pth = f'../stats/{var}_stats.nc4'
    dst = xr.open_mfdataset(pth, parallel=True)
    ds_mean = dst[f'{var}_mean'] * 1e-6
    ds_std = dst[f'{var}_std'].where(dst[f'{var}_std'] > 0.) * 1e-6

    regridder = xe.Regridder(ds_mean, dst, 'bilinear', periodic=True)  # ensure same grid

    ds_mean = regridder(ds_mean)
    ds_std = regridder(ds_std)

    if not invert:
        ds_out = (ds - ds_mean) / ds_std
    else:
        ds_out = ds * ds_std + ds_mean

    return ds_out


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
#                         GET_DATA 
# ============================================================

def get_data(
    vars_out=None,
    pth_all="",
    vars_in_2d=None,
    vars_in_3d=None,
    prefix="",
    nexp=1,
    visualize_cache=True,
):
    """
    Build Nd outputs, 2D/3D inputs, and EOF coefficients with fast caching.
    """

    if vars_out is None: vars_out = {}
    if vars_in_2d is None: vars_in_2d = {}
    if vars_in_3d is None: vars_in_3d = {}

    chk2d = {"time": 3, "lat": -1, "lon": -1}
    chk3d = {"time": 3, "lev": -1, "lat": -1, "lon": -1}
    cache_dir = "./cache/"


    # ================================================================
    # 1. LOAD + NORMALIZE NOUT (Nd)
    # ================================================================
    N_list = []
    minP = 250


    for v in vars_out.keys():
        pth = f"{pth_all}dailies/{vars_out[v]}/{prefix}"
        print("[LOAD 3D OUT]", pth)

        D = xr.open_mfdataset(pth, parallel=True, chunks=chk3d)[[v]]
        D = normalize_time_coord(D)
        D = D.sel(lev=D.lev.where(D.lev >= minP, drop=True))
        N_list.append(D)

    Nout = xr.merge(N_list)
    varname = list(vars_out.keys())[0]
    Nout = Nout.where(Nout >= 0)
   
   
    # ----------------------------------------------------


    # Cache Nd arrays
    grid_hash = hash_grid(Nout)
    p_cache = cache_path(cache_dir, "NOUT", prefix, grid_hash, "raw")

    nout_vars = []
    for v in Nout.data_vars:
        base = p_cache + f"_{v}"
        cached = load_array_cache(base)
        if cached is not None:
            print(f"[CACHE HIT] NOUT {v}")
            nout_vars.append(cached.to_dataset())
        else:
            save_array_cache(base, Nout[v])
            nout_vars.append(Nout[[v]])
            print(f"[CACHE] Saved NOUT {v}")

    Nout = xr.merge(nout_vars)
    Nout = normalize_time_coord(Nout)   # <--- CRITICAL (cached may be stale)


    # ================================================================
    # 2. LOAD + NORMALIZE + CACHE 2D INPUTS
    # ================================================================
    p2_cache = cache_path(cache_dir, "NIN2D", prefix, grid_hash, "std_regridded")

    nin_cached = []
    missing_2d = []

    for v in vars_in_2d.keys():
        base = p2_cache + f"_{v}"
        cached = load_array_cache(base)
        if cached is not None:
            print(f"[CACHE HIT] NIN2D {v}")
            nin_cached.append(cached.to_dataset())
        else:
            missing_2d.append(v)

    if missing_2d:
        Nin_list = []
        for v in missing_2d:
            pth = f"{pth_all}dailies/{vars_in_2d[v]}/{prefix}"
            print("[LOAD 2D]", pth)

            D = xr.open_mfdataset(pth, parallel=True, chunks=chk2d)[[v]]
            D = normalize_time_coord(D)
            D = stand_by_cell(D, var=v, pth_all=pth_all)
            Nin_list.append(D)

        Nin_new = xr.merge(Nin_list)

        # Regrid to Nd grid
        print("[REGRID] 2D \u2192 Nd grid")
        R2 = xe.Regridder(Nin_new, Nout, "bilinear", periodic=True)
        Nin_new = R2(Nin_new, output_chunks=chk2d)

        for v in missing_2d:
            save_array_cache(p2_cache + f"_{v}", Nin_new[v])
            nin_cached.append(Nin_new[[v]])
            print(f"[CACHE] Saved NIN2D {v}")

    Nin_2d = xr.merge(nin_cached) if nin_cached else xr.Dataset()
    if "time" in Nin_2d.coords:
        Nin_2d = normalize_time_coord(Nin_2d)


    # ================================================================
    # 3. TIME ALIGNMENT (SAFE)
    # ================================================================
    if "time" in Nin_2d.dims:
        shared_time = safe_time_intersection(Nout.time.values, Nin_2d.time.values)
        print("[INFO] Shared time count:", len(shared_time))

        Nout   = Nout.sel(time=shared_time)
        Nin_2d = Nin_2d.sel(time=shared_time)
    else:
        shared_time = force_datetime64_ns(Nout.time.values)

    Nin_2d = Nin_2d.transpose("time", "lat", "lon") if "time" in Nin_2d.dims else Nin_2d
    Nout   = Nout.transpose("time", "lev", "lat", "lon")


    # ================================================================
    # 4. LOAD 3D INPUT VARIABLES
    # ================================================================
    extracted_vars = []

    for var3d, cfg in vars_in_3d.items():
        levels = cfg["levels"]
        path3d = cfg["path"]
        std_var = cfg.get("std_varname", var3d)

        pth3d = f"{pth_all}dailies/{path3d}/{prefix}"
        print(f"[LOAD 3D INPUT] {var3d} \u2192 levels {levels} from {pth3d}")

        # Load 3D data
        D = xr.open_mfdataset(pth3d, parallel=True, chunks=chk3d)[[var3d]]
        D = normalize_time_coord(D)

        # Interpolate in vertical
        Dsel = D.interp(lev=levels, kwargs={"fill_value": "extrapolate"})
        Dsel = normalize_time_coord(Dsel)
        Dsel = Dsel.transpose("time", "lev", "lat", "lon")

        # Align on time
        t_3d = Dsel.time.values
        shared_time_3d = safe_time_intersection(shared_time, t_3d)
        print(f"[INFO] {var3d} shared timestamps:", len(shared_time_3d))

        level_vars = []

        for lev in levels:
            outname = f"{var3d}_{lev}hPa"
            cache_file = cache_path(cache_dir, outname, prefix, grid_hash, "std_regridded")

            cached = load_array_cache(cache_file)
            if cached is not None:
                print(f"[CACHE HIT] 3D {outname}")
                level_vars.append(cached.to_dataset())
                continue

            # Extract + align + standardize
            Da = Dsel.sel(lev=lev).drop_vars("lev")
            Da = Da.sel(time=shared_time_3d)

            if isinstance(Da, xr.DataArray):
                Da = Da.to_dataset(name=outname)
            else:
                Da = Da.rename({list(Da.data_vars)[0]: outname})

            Da = stand_by_cell(Da, var=std_var, pth_all=pth_all)
            print('Da=============', Da)
            

            # Regrid
            print(f"[REGRID] \u2192 Nd grid: {outname}")
            R3 = xe.Regridder(Da, Nout, "bilinear", periodic=True)
            Da = R3(Da, output_chunks=chk2d)
            # xESMF often drops time coord \u2192 restore it
            if "time" in Da.dims and "time" not in Da.coords:
                print("[WARN] Regridder dropped time coord \u2192 restoring datetime coords")
                Da = Da.assign_coords(time=shared_time_3d)


            save_array_cache(cache_file, Da[outname])
            level_vars.append(Da)

        extracted_vars.append(xr.merge(level_vars))

    if extracted_vars:
   
        Nin_2d = xr.merge([Nin_2d] + extracted_vars)
        
        Nin_2d = normalize_time_coord(Nin_2d)


    # =====================================================
    # NEW PROJECTION USING PHYSICAL EOFs
    # =====================================================

    Nd = Nout[varname]

    # Interpolate Nd to EOF vertical grid if needed
    if not np.array_equal(Nd.lev.values, EOF.lev.values):
        Nd = Nd.interp(lev=EOF.lev)

    # Non-nan Anomaly relative to global mean
    Nd_filled = Nd.fillna(Nd_mean)
    Nd_anom   = Nd_filled - Nd_mean


    # Flatten: (time, lat, lon, lev) \u2192 (Ncells, lev)
    Nt, Nz, Ny, Nx = Nd_anom.transpose("time","lev","lat","lon").shape
    NdA = Nd_anom.transpose("time","lat","lon","lev").values.reshape(-1, Nz)

    EOFm = EOF.values.astype(np.float64)   # (mode, lev)
    den  = np.sum(EOFm**2, axis=1)

    A = NdA @ EOFm.T / den                 # (Ncells, mode)

    # Reshape back
    A = A.reshape(Nt, Ny, Nx, num_modes)
    A = np.moveaxis(A, -1, 1)              # (time, mode, lat, lon)

    
    
    A_mean = np.nanmean(A, axis=0)

    # Fallback: zero mean where no samples exist
    A_mean = np.where(np.isnan(A_mean), 0.0, A_mean)


    # Residual coefficients (training target)
    A_resid = A - A_mean
    np.save(hp["model_name"] + "_coeff_mean.npy", A_mean.astype(np.float32))

    
    Ncoeff = xr.Dataset(
        {
            "a": (("time","mode","lat","lon"), A_resid)
        },
        coords={
            "time": Nd.time,
            "mode": np.arange(num_modes),
            "lat": Nd.lat,
            "lon": Nd.lon,
        }
    )


    
    # =============================================================
    # OPTIONAL: SAVE SMALL VISUALIZATION NETCDF FOR DEBUGGING
    # =============================================================
    pcoeff = cache_path(cache_dir, "NCOEFF_res", prefix, grid_hash, "raw")

    if visualize_cache:
        print("[VIS] Saving coefficient preview file")
        save_vis_nc(
            pcoeff + "_viz.nc",
            Ncoeff["a"],
            time_slice=slice(0, 5),
            modes=slice(0, 5),
            stride_lat=2,
            stride_lon=2,
        )



    # ================================================================
    # FINAL RETURN
    # ================================================================
    return Nin_2d.fillna(0).load(), Ncoeff.load(), Nout.load()


class EOFNdDataset(Dataset):
    def __init__(self, Nin_2d, Ncoeff, Nout):
        self.X2D = Nin_2d
        self.Y   = Ncoeff
        self.Nd  = Nout
        self.vars_2d = list(Nin_2d.data_vars)
        self.num_modes = Ncoeff.sizes["mode"]

    def __len__(self):
        return self.X2D.sizes["time"]

    def __getitem__(self, idx):
        idx = int(idx)

        # -----------------------------
        # Inputs
        # -----------------------------
        x2d_vars = [
            self.X2D[v].isel(time=idx).values
            for v in self.vars_2d
        ]
        x2d = torch.tensor(
            np.stack(x2d_vars, axis=0),
            dtype=torch.float32
        )  # [C, H, W]

        # -----------------------------
        # EOF coefficients (target)
        # -----------------------------
        y = torch.tensor(
            self.Y['a'].isel(time=idx).values,
            dtype=torch.float32
        )  # [M, H, W]

        # -----------------------------
        # TRUE Nd (physical space)
        # -----------------------------
        nd = torch.tensor(
            self.Nd.isel(time=idx).values,
            dtype=torch.float32
        )  # [Z, H, W]

        return x2d, y, nd


# ------------------------------------------------
# Neural Network: 2D conv net predicting EOF coeffs
# ------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, n_groups=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=n_groups, num_channels=out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=n_groups, num_channels=out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        # Decoder input has 2*out_ch channels
        self.up = nn.ConvTranspose2d(
            out_ch * 2, out_ch * 2, kernel_size=2, stride=2
        )

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        """
        x1: decoder tensor (B, 2*out_ch, H1, W1)
        x2: encoder skip   (B, out_ch,  H2, W2)
        """

        x1 = self.up(x1)

        # --- SAFE ALIGNMENT (critical for odd sizes) ---
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        if diffY != 0 or diffX != 0:
            x1 = F.pad(
                x1,
                [
                    diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2,
                ]
            )

        # Concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)






class UNetEOF(nn.Module):
    """
    U-Net predicting EOF coefficients directly.

    Input:
        x : [B, Cin, H, W]

    Output:
        coeffs : [B, Nmodes, H, W]
    """
    def __init__(self, in_channels, num_modes, base_ch=48):
        super().__init__()

        # ---------------- Encoder ----------------
        self.inc   = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)

        # ---------------- Bottleneck ----------------
        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 8)

        # ---------------- Decoder ----------------
        self.up3 = Up(base_ch * (8 + 4), base_ch * 4)
        self.up2 = Up(base_ch * (4 + 2), base_ch * 2)
        self.up1 = Up(base_ch * (2 + 1), base_ch)

        # ---------------- Output head ----------------
        self.out_conv = nn.Conv2d(
            base_ch,
            num_modes,
            kernel_size=1
        )

    def forward(self, x):
        # ---------- Encoder ----------
        x1 = self.inc(x)       # [B, C, H, W]
        x2 = self.down1(x1)    # [B, 2C, H/2, W/2]
        x3 = self.down2(x2)    # [B, 4C, H/4, W/4]
        x4 = self.down3(x3)    # [B, 8C, H/8, W/8]

        # ---------- Bottleneck ----------
        xb = self.bottleneck(x4)

        # ---------- Decoder ----------
        x = self.up3(xb, x3)
        x = self.up2(x,  x2)
        x = self.up1(x,  x1)

        # ---------- Output ----------
        out = self.out_conv(x)   # [B, Nmodes, H, W]
        return out


# ======================================
# 1. Losses
# ======================================

def reconstruct_Nd_torch(coeffs, EOF, Nd_mean, beta=1.0):
    B, M, H, W = coeffs.shape
    Z = EOF.shape[1]

    coeffs = torch.nan_to_num(coeffs, nan=0.0)
    EOF    = torch.nan_to_num(EOF, nan=0.0)
    Nd_mean = torch.nan_to_num(Nd_mean, nan=0.0)

    coeffs_f = coeffs.permute(0,2,3,1).reshape(-1, M)

    Nd_f = coeffs_f @ EOF + Nd_mean
    Nd   = Nd_f.reshape(B, H, W, Z).permute(0,3,1,2)

    return F.softplus(Nd, beta=beta)





class WeightedCoeffMSE(nn.Module):
    def __init__(self, lat_tensor, mode_weights):
        super().__init__()
        self.register_buffer(
            "lat_weights",
            torch.cos(torch.deg2rad(lat_tensor))  # (H,)
        )
        self.register_buffer(
            "mode_weights",
            mode_weights  # (K,)
        )

    def forward(self, pred, target):
        # pred, target: [B, K, H, W]
        mask = ~torch.isnan(target)

        # build full weights with broadcasting
        lat_w  = self.lat_weights.view(1, 1, -1, 1)   # [1,1,H,1]
        mode_w = self.mode_weights.view(1, -1, 1, 1)  # [1,K,1,1]

        w = lat_w * mode_w                            # [1,K,H,1]
        w = w.expand_as(pred)                         # [B,K,H,W]

        diff2 = (pred - target)**2

        # mask via multiplication (safe)
        weighted_diff2 = w * diff2 * mask

        # normalize by number of valid points
        return weighted_diff2.sum() / mask.sum().clamp(min=1)



class MaskedNaNMSELossLatWeighted(nn.Module):
    def __init__(self, lat_tensor):
        """
        lat_tensor: (H,) latitude values
        """
        super().__init__()
        self.register_buffer("lat_weights",
                             torch.cos(torch.deg2rad(lat_tensor)))  # (H,)

    def forward(self, pred, target):
        # pred, target: [B, K, H, W]
        mask = ~torch.isnan(target)  # valid pixels: True

        # broadcast weights to [B, K, H, W]
        weights = self.lat_weights.view(1, 1, -1, 1).expand_as(pred)

        pred_valid = pred[mask]
        target_valid = target[mask]
        weights_valid = weights[mask]

        loss = torch.mean(weights_valid * (pred_valid - target_valid) ** 2)
        return loss



class LaplacianLoss(nn.Module):
    """
    Penalizes mismatch in curvature (second derivatives).
    Encourages sharper spatial structure.
    """
    def forward(self, pred, target):
        # pred, target: [B, K, H, W]

        # Replace NaNs in target (safe for derivatives)
        target = torch.nan_to_num(target, nan=0.0)
        pred   = torch.nan_to_num(pred,   nan=0.0)

        lap_p = (
            -4 * pred
            + F.pad(pred[..., 1:, :], (0, 0, 0, 1))
            + F.pad(pred[..., :-1, :], (0, 0, 1, 0))
            + F.pad(pred[..., :, 1:], (0, 1, 0, 0))
            + F.pad(pred[..., :, :-1], (1, 0, 0, 0))
        )

        lap_t = (
            -4 * target
            + F.pad(target[..., 1:, :], (0, 0, 0, 1))
            + F.pad(target[..., :-1, :], (0, 0, 1, 0))
            + F.pad(target[..., :, 1:], (0, 1, 0, 0))
            + F.pad(target[..., :, :-1], (1, 0, 0, 0))
        )

        mask = ~torch.isnan(target)
        return ((lap_p - lap_t) ** 2)[mask].mean()



class GradientLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # pred,target: [B, K, H, W]
        
        # Sanitize predictions: no NaNs allowed in pred
        pred = torch.nan_to_num(pred, nan=0.0)
        
        # Mask: valid target pixels
        mask = ~torch.isnan(target)

        # --------- Horizontal gradients ---------
        # Compute gradient, but replace NaNs from target BEFORE using
        target_safe = torch.nan_to_num(target, nan=0.0)

        p_dx = pred[..., :, 1:] - pred[..., :, :-1]
        t_dx = target_safe[..., :, 1:] - target_safe[..., :, :-1]

        mask_dx = mask[..., :, 1:] & mask[..., :, :-1]

        # --------- Vertical gradients ---------
        p_dy = pred[..., 1:, :] - pred[..., :-1, :]
        t_dy = target_safe[..., 1:, :] - target_safe[..., :-1, :]

        mask_dy = mask[..., 1:, :] & mask[..., :-1, :]

        # --------- Charbonnier losses ---------
        dx_loss_map = torch.sqrt((p_dx - t_dx)**2 + self.eps**2)
        dy_loss_map = torch.sqrt((p_dy - t_dy)**2 + self.eps**2)

        # Mask invalid gradient pixels
        dx_valid = dx_loss_map[mask_dx]
        dy_valid = dy_loss_map[mask_dy]

        # Avoid computing mean of empty tensor ? return 0
        loss_dx = dx_valid.mean() if dx_valid.numel() > 0 else pred.new_tensor(0.0)
        loss_dy = dy_valid.mean() if dy_valid.numel() > 0 else pred.new_tensor(0.0)

        return loss_dx + loss_dy


class NdGradientLoss(nn.Module):
    def forward(self, pred_nd, true_nd):
       

        eps = 1e-6
        mask = true_nd > 0

        pred = torch.log(torch.clamp(pred_nd, min=eps))
        true = torch.log(torch.clamp(true_nd, min=eps))

        
        dx_p = pred[...,1:] - pred[...,:-1]
        dy_p = pred[...,1:,:] - pred[...,:-1,:]

        dx_t = true[...,1:] - true[...,:-1]
        dy_t = true[...,1:,:] - true[...,:-1,:]

        m_dx = mask[...,1:] & mask[...,:-1]
        m_dy = mask[...,1:,:] & mask[...,:-1,:]

        loss_dx = (dx_p[m_dx] - dx_t[m_dx]).abs().mean()
        loss_dy = (dy_p[m_dy] - dy_t[m_dy]).abs().mean()

        return loss_dx + loss_dy


def build_pressure_weights(lev, p0=500.0, alpha=2.0):
    """
    lev: tensor [Z] in hPa
    """
    p = lev.float()
    w = torch.ones_like(p)
    w[p < p0] = (p0 / p[p < p0]) ** alpha
    return w / w.mean()


# ======================================
# 3. COMBINED LOSS WITH TRAINABLE LAMBDA
# ======================================



class CombinedCoeffLoss(nn.Module):
    def __init__(self, lat_tensor, init_lambda=0.05, lambda_max=0.3):
        super().__init__()
        #self.mse_lat = MaskedNaNMSELossLatWeighted(lat_tensor)
        self.mse_lat   = WeightedCoeffMSE(lat_tensor, mode_weights)
        self.grad_loss = GradientLoss()
        self.lap_loss  = LaplacianLoss()

        # Trainable alpha ? lambda = sigmoid(alpha)
        alpha_init = np.log(init_lambda / (1 - init_lambda))  # inverse sigmoid
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        self.current_lambda = None  # for logging
        self.lambda_max = lambda_max
        self.nd_grad = NdGradientLoss()
        

        self.register_buffer(
            "EOF",
            torch.as_tensor(EOF.values, dtype=torch.float32)
        )
        self.register_buffer(
            "Nd_mean",
            torch.as_tensor(Nd_mean.values, dtype=torch.float32)
        )

        A_mean = np.load(hp["model_name"] + "_coeff_mean.npy")
        A_mean = np.nan_to_num(A_mean, nan=0.0)

        self.register_buffer(
            "A_mean",
            torch.as_tensor(A_mean, dtype=torch.float32)
        )
        
        self.register_buffer(
            "p_weights",
            build_pressure_weights(
                torch.as_tensor(EOF_ds.lev.values, dtype=torch.float32),
                p0=500.0,
                alpha=2.0
            )
        )


    def nd_mse_pressure_weighted(self, pred_nd, true_nd):
        pred_nd = torch.nan_to_num(pred_nd, nan=0.0)
        true_nd = torch.nan_to_num(true_nd, nan=0.0)

        mask = true_nd > 0
        if mask.sum() == 0:
            return pred_nd.new_tensor(0.0)

        w = self.p_weights.view(1, -1, 1, 1)

        diff2 = (pred_nd - true_nd) ** 2
        weighted = diff2 * w * mask

        return weighted.sum() / mask.sum().clamp(min=1)


    def forward(self, pred, target, true_nd):
        # ---- HARDEN EVERYTHING ----
        pred   = torch.nan_to_num(pred,   nan=0.0)
        target = torch.nan_to_num(target, nan=0.0)
        true_nd = torch.nan_to_num(true_nd, nan=0.0)

        lambda_grad = torch.sigmoid(self.alpha) * self.lambda_max
        self.current_lambda = lambda_grad.detach()

        # ---- Coefficient-space losses ----
        L_coeff = 1e-2 * self.mse_lat(pred, target)
        L_grad  = self.grad_loss(pred, target)
        L_lap   = 1e-4 * self.lap_loss(pred, target)
        
        
        pred_abs = pred + self.A_mean.unsqueeze(0)

        pred_nd = reconstruct_Nd_torch(
            pred_abs,
            self.EOF,
            self.Nd_mean
        )


        #L_nd = self.nd_grad(pred_nd, true_nd)

        L_nd  = 1e-2*self.nd_mse_pressure_weighted(pred_nd, true_nd)
         
        if torch.rand(1).item() < 0.01:
            print(
                f"L_coeff={L_coeff.item():.3e}, "
                f"L_grad={L_grad.item():.3e}, "
                f"L_nd={L_nd.item():.3e}, "
                f"lambda={lambda_grad.item():.3f} "
                f"L_lap={L_lap.item():.3e}, "
            )
        
        return (L_coeff + lambda_grad * L_grad + L_nd + L_lap)

        #return L_coeff + lambda_grad * L_lap
        





def visualize_model_graph(model, in_channels_2d, device, out_file='CoeffNet_Graph'):
    model = model.to(device)
    model.eval()

    dummy_x2d = torch.randn(1, in_channels_2d, 32, 32, requires_grad=True).to(device)
    output = model(dummy_x2d)

    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render(out_file)

    print(f"[INFO] Model architecture graph saved as {out_file}.png")



# ------------------------------------------------
# Training Routine (keeps scheduler, early stopping, logging)
# ------------------------------------------------
def training_val(model, train_loader, val_loader, hp, device, latitudes=None, start_epoch=0):
    #criterion = MaskedNaNMSELossLatWeighted(latitudes).to(device)
    criterion = CombinedCoeffLoss(latitudes).to(device)
    #criterion = WeightedCoeffMSE(latitudes, mode_weights).to(device)
    
    
    optimizer = optim.Adam(
        [
            {'params': model.parameters()},
            {'params': criterion.parameters()}   # includes alpha
        ],
        lr=hp['l_r'],
        weight_decay=1e-5
	)

    train_losses, val_losses = [], []
    patience_counter, min_rolling_val_loss = 0, float('inf')
    Nr = hp.get('rolling_Nr', 5)
    num_epochs = hp['nepochs']
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    # Print shapes for first batch
    x2d_sample, y_sample, nd_sample = next(iter(train_loader))
    print(
    f"Sample shapes -> x2d: {x2d_sample.shape}, "
    f"y (coeffs): {y_sample.shape}, "
    f"nd: {nd_sample.shape}")
    
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for x2d, y, nd in train_loader:
            x2d, y, nd = x2d.to(device), y.to(device), nd.to(device)
            optimizer.zero_grad()
            y_pred = model(x2d)
            #print("Per-mode std:", y_pred.std(dim=(0,2,3)))
            loss = criterion(y_pred, y, nd)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * x2d.size(0)
        train_epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_epoch_loss)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x2d, y, nd in val_loader:
                x2d, y, nd = x2d.to(device), y.to(device), nd.to(device)
                y_pred = model(x2d)
                val_loss_sum += criterion(y_pred, y, nd).item() * x2d.size(0)
        val_epoch_loss = val_loss_sum / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)

        # Rolling avg
        if len(val_losses) >= Nr:
            rolling_avg_val_loss = sum(val_losses[-Nr:]) / Nr
        else:
            rolling_avg_val_loss = sum(val_losses) / len(val_losses)

        scheduler.step(rolling_avg_val_loss)
        

        epoch_time = time.time() - start_time
        print(
            f'Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f} s, '
            f'Train Loss: {train_epoch_loss:.6f}, Val Loss: {val_epoch_loss:.6f}, '
            f'Current LR: {optimizer.param_groups[0]["lr"]}, '
            f'lambda_grad = {criterion.current_lambda.item():.3f}')



        # Early stopping based on rolling average
        if rolling_avg_val_loss < min_rolling_val_loss:
            min_rolling_val_loss = rolling_avg_val_loss
            patience_counter = 0
            #print('model saved')
            save_model_weights(model, hp['model_name'] + "_weights.pt")
            save_training_checkpoint(
                                        model,
                                        optimizer,
                                        criterion,
                                        epoch,
                                        hp['model_name'] + "_train.ckpt"
                                    )

        else:
            patience_counter += 1
            if patience_counter >= hp['patience']:
                print("Early stopping")
                break

    return train_losses, val_losses



# ------------------------------------------------
# Main
# ------------------------------------------------
if __name__ == "__main__":
    
    
    prefix = '*20[0-1]*'
    #prefix = '*20*'
    #prefix = '*2010[1-6]*'
    test_pr = '*2020*' 
    
    pth_all = '/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/'

    vars_in_3d = {
            "T": {
                "path": "atm_inst_6hr_glo_L720x361_p49",
                "levels": [900, 500, 300],
                "std_varname": "T"      # name used in stats/T_stats.nc4
            },
        }

    vars_in_3d = {}
    vars_in_2d = {
        'MDSOPTHCKWTR': 'mod_inst_6hr_glo_L720x361_sfc',
        'MDSCLDSZWTR': 'mod_inst_6hr_glo_L720x361_sfc',
        'MDSCLDFRCWTR': 'mod_inst_6hr_glo_L720x361_sfc', 
        'SWCLDPRS': 'mod_inst_6hr_glo_L720x361_sfc'
    }
    vars_out = {
        'NDinc_cf01p': 'mkincloudNd'
    }

    hp = {
        'nepochs': 1500,
        'batch_sz': 16,
        'l_r': 0.0002,
        'patience': 120,
        'model_name': "NDnet_EOFcoeff",
        'rolling_Nr': 5, 
        'basech': 64
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device===', device)
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs" if num_gpus > 1 else "Using a single GPU or CPU")

    nexp = 1.
    do_train =  True
    do_test = True

    # --------------------
    # TRAIN
    # --------------------
    if do_train:
        Nin_2d, Ncoeff, Nout  = get_data(
            vars_out=vars_out,
            pth_all=pth_all,
            vars_in_2d=vars_in_2d,
            vars_in_3d=vars_in_3d,
            prefix=prefix,
            nexp=nexp
        )

        Ndd =  Nout[list(vars_out.keys())[0]]
        dataset = EOFNdDataset(Nin_2d, Ncoeff, Ndd)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=hp['batch_sz'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hp['batch_sz'], shuffle=False)

        #model = CoeffNet(
         #   in_channels=len(dataset.vars_2d),
         #   num_modes=num_modes
        #)

        model = UNetEOF(
            in_channels=len(dataset.vars_2d),
            num_modes=num_modes,
            base_ch=hp['basech']
        ).to(device)
        


        latitudes = torch.tensor(Nin_2d.lat.values, dtype=torch.float32)
        criterion = CombinedCoeffLoss(latitudes).to(device)
        optimizer = optim.Adam(
            [
                {'params': model.parameters()},
                {'params': criterion.parameters()}
            ],
            lr=hp['l_r'],
            weight_decay=1e-5
        )

        # ------------------------------------------------
        # Resume training if checkpoint exists (BEFORE DP)
        # ------------------------------------------------
        start_epoch = load_training_checkpoint(
            model,
            optimizer,
            criterion,
            hp['model_name'] + "_train.ckpt",
            device)
     
        if num_gpus > 1:
            model = nn.DataParallel(model)


        train_losses, val_losses = training_val(
            model,
            train_loader,
            val_loader,
            hp,
            device,
            latitudes=latitudes,
            start_epoch=start_epoch
        )


        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Val')
        plt.xlim(left=5)
        plt.legend()
        plt.savefig(hp['model_name'] + '_loss.png')

    # --------------------
    # TEST / DIAGNOSTICS
    # --------------------
    if do_test:

        Nin_2d, Ncoeff_test, Y_true = get_data(
            vars_out=vars_out,
            pth_all=pth_all,
            vars_in_2d=vars_in_2d,
            vars_in_3d=vars_in_3d,
            prefix=test_pr,
            nexp=nexp
        )

        Ndd =  Y_true[list(vars_out.keys())[0]]
        dataset = EOFNdDataset(Nin_2d, Ncoeff_test, Ndd)
        test_loader = DataLoader(dataset, batch_size=hp['batch_sz'], shuffle=False)

        model = UNetEOF(
            in_channels=len(dataset.vars_2d),
            num_modes=num_modes,
            base_ch=hp['basech']
        ).to(device)

        # load weights BEFORE DataParallel
        load_model_weights(
            model,
            hp['model_name'] + "_weights.pt",
            device
        )

        if num_gpus > 1:
            model = nn.DataParallel(model)

        summary(
            model,
            input_data=[
                torch.randn(1, len(dataset.vars_2d), 32, 32).to(device)
            ],
            col_names=["input_size", "output_size", "num_params", "kernel_size"],
            depth=3,
            device=device
        )



        # === Inference (residual EOF coefficients) ===
        all_coeffs_pred = []

        with torch.no_grad():
            for x2d, y, nd in test_loader:
                x2d = x2d.to(device)
                outputs = model(x2d)              # [B, M, lat, lon]  (RESIDUALS)
                all_coeffs_pred.append(outputs.cpu())

        all_coeffs_pred = torch.cat(all_coeffs_pred, dim=0)  # [T, M, lat, lon]
        coeffs_resid = all_coeffs_pred.numpy().astype(np.float32)

        # ============================================================
        # ADD BACK MEAN EOF COEFFICIENTS
        # ============================================================
        A_mean = np.load(hp["model_name"] + "_coeff_mean.npy")   # (M, lat, lon)

        # Absolute coefficients
        coeffs_abs = coeffs_resid + A_mean[None, :, :, :]        # (T, M, lat, lon)

        # ============================================================
        # RECONSTRUCT Nd FROM ABSOLUTE COEFFICIENTS
        # ============================================================
        EOF_m   = EOF.values.astype(np.float64)        # (M, Z)
        Nd_mean_v = Nd_mean.values.astype(np.float64)  # (Z,)

        Nt, Nm, Ny, Nx = coeffs_abs.shape
        Nz = EOF_m.shape[1]

        Nd_rec = np.zeros((Nt, Nz, Ny, Nx), dtype=np.float32)

        for t in range(Nt):
            # (M, lat*lon)
            coeff_t = coeffs_abs[t].reshape(Nm, Ny * Nx)

            # Nd anomaly + mean
            Nd_t = (EOF_m.T @ coeff_t).T + Nd_mean_v    # (lat*lon, Z)

            Nd_rec[t] = Nd_t.reshape(Ny, Nx, Nz).transpose(2, 0, 1)

        # ============================================================
        # PHYSICAL CONSTRAINT
        # ============================================================
        Nd_rec = np.maximum(Nd_rec, 0.0)


        
      
        # === Create prediction xarray with same coords as Y_true ===
        pred_xr = Y_true.copy()
        pred_xr['NDinc_cf01p'].data = Nd_rec
        
        
        minN =  0.1 
        Y_true =  Y_true.where(Y_true >= minN)
        pred_xr =  pred_xr.where(Y_true >= minN)
        pred_xr =  pred_xr.where(pred_xr >= minN)
        pred_xr = pred_xr.rename({'NDinc_cf01p': 'NDinc_cf01p_pred'})
        
       

        ## === de-standardize Nd (true + pred) ===
        #Y_true = stand_by_cell_out(Y_true, invert=True, var='NCPL_VOL')
        #pred_xr = stand_by_cell_out(pred_xr, invert=True, var='NCPL_VOL')

        # === Time-mean and diagnostics as before ===
        avg_pred = pred_xr.mean(dim='time').to_array().squeeze()  # [lev, lat, lon]
        avg_true = Y_true.mean(dim='time').to_array().squeeze()
        avg_diff = 100 * (avg_pred - avg_true) / np.clip(avg_true, a_min=0.01, a_max=None)  # Percent difference
        
        
                # ============================================================
        # RMS (Pred - True) \u2014 SAFE VERSION
        # ============================================================

        var = 'NDinc_cf01p'   # true variable name
        varp = 'NDinc_cf01p_pred'

        rms_3d = 100*(np.sqrt(
            (pred_xr[varp] - Y_true[var])**2
        ).mean(dim='time', skipna=True)/ Y_true[var].mean(dim='time', skipna=True))  # DataArray [lev, lat, lon]



        output_prefix = 'NDnet_pred'
        pred_xr.to_netcdf(f"{output_prefix}.nc", format='NETCDF4', engine='netcdf4', mode='w')
        Y_true.to_netcdf(f"{output_prefix}.nc", format='NETCDF4', engine='netcdf4', mode='a')

        # ============================================================
        # PLOTTING
        # ============================================================

        levels_to_plot = [900, 700, 500]

        proj = ccrs.PlateCarree()
        nrows = len(levels_to_plot) + 1  # +1 zonal row

        fig = plt.figure(figsize=(14, 3.2 * len(levels_to_plot) + 2))

        gs = GridSpec(
            nrows=nrows,
            ncols=4,                          # <-- was 3
            figure=fig,
            height_ratios=[1.0] * len(levels_to_plot) + [0.6],
            hspace=0.15,
            wspace=0.05
        )

        axes = np.empty((nrows, 4), dtype=object)


        # ------------------------------------------------------------
        # MAP ROWS
        # ------------------------------------------------------------
        for i in range(len(levels_to_plot)):
            for j in range(4):
                axes[i, j] = fig.add_subplot(gs[i, j], projection=proj)

        # ------------------------------------------------------------
        # ZONAL ROW (no projection)
        # ------------------------------------------------------------
        for j in range(4):
            axes[nrows - 1, j] = fig.add_subplot(gs[nrows - 1, j])

        Nmax = 1000
        Nmin = 10
        log_norm = mcolors.LogNorm(vmin=Nmin, vmax=Nmax)
        shk = 0.4

        # ============================================================
        # MAP PANELS
        # ============================================================
        for i, lev in enumerate(levels_to_plot):

            true_slice = avg_true.sel(lev=lev)
            pred_slice = avg_pred.sel(lev=lev)
            diff_slice = avg_diff.sel(lev=lev)
            rms_slice = rms_3d.sel(lev=lev)

            # --- True ---
            im_true = true_slice.plot.pcolormesh(
                ax=axes[i, 0],
                transform=proj,
                cmap='turbo',
                norm=log_norm,
                add_colorbar=False
            )
            mn = nb(true_slice.mean().load())
            axes[i, 0].set_title(f'True @ {lev} hPa Mean: {mn}')
            axes[i, 0].coastlines()
            axes[i, 0].add_feature(cfeature.BORDERS, linewidth=0.5)
            fig.colorbar(im_true, ax=axes[i, 0], orientation='vertical', shrink=shk)

            # --- Prediction ---
            im_pred = pred_slice.plot.pcolormesh(
                ax=axes[i, 1],
                transform=proj,
                cmap='turbo',
                norm=log_norm,
                add_colorbar=False
            )
            mn = nb(pred_slice.mean().load())
            axes[i, 1].set_title(f'Prediction @ {lev} hPa Mean: {mn}')
            axes[i, 1].coastlines()
            axes[i, 1].add_feature(cfeature.BORDERS, linewidth=0.5)
            fig.colorbar(im_pred, ax=axes[i, 1], orientation='vertical', shrink=shk)

            # --- % Difference ---
            im_diff = diff_slice.plot.pcolormesh(
                ax=axes[i, 2],
                transform=proj,
                cmap='RdBu_r',
                vmin=-50, vmax=50,
                add_colorbar=False
            )
            
            mn = nb(diff_slice.mean().load())
            axes[i, 2].set_title(f'% (Pred-True) Mean: {mn}')
            axes[i, 2].coastlines()
            axes[i, 2].add_feature(cfeature.BORDERS, linewidth=0.5)
            fig.colorbar(im_diff, ax=axes[i, 2], orientation='vertical', shrink=shk)
            
            
            
            # --- RMS ---
            im_rms = rms_slice.plot.pcolormesh(
                ax=axes[i, 3],
                transform=proj,
                cmap='RdBu_r',
                vmin=-50, vmax=50,
                add_colorbar=False
            )

            mn = nb(rms_slice.mean().load())
            axes[i, 3].set_title(f'RMS Error @ {lev} hPa Mean: {mn}')
            axes[i, 3].coastlines()
            axes[i, 3].add_feature(cfeature.BORDERS, linewidth=0.5)
            fig.colorbar(im_rms, ax=axes[i, 3], orientation='vertical', shrink=shk)
            
            
            

        # ============================================================
        # ZONAL MEAN ROW
        # ============================================================

        row = len(levels_to_plot)

        zonal_true_2d = avg_true.mean(dim='lon', skipna=True)
        zonal_pred_2d = avg_pred.mean(dim='lon', skipna=True)
        zonal_diff_2d = 100 * (zonal_pred_2d - zonal_true_2d) / np.clip(
            zonal_true_2d, a_min=0.1, a_max=None
        )

        
        
        # Zonal RMS
        rms_zonal_2d = 100*np.sqrt(
            ((zonal_pred_2d - zonal_true_2d) ** 2)
        )/ zonal_true_2d.where(zonal_true_2d > minN)


        
        ZT = np.clip(zonal_true_2d.values, Nmin, None)
        ZP = np.clip(zonal_pred_2d.values, Nmin, None)
        ZD = zonal_diff_2d.values

        # --- Pressure from Nd_mean ---
        p_lev = Nd_mean.lev.values

        # Standard pressure ticks
        p_ticks = np.array([1000, 850, 700, 500, 300, 200])
        yticks = [np.argmin(np.abs(p_lev - p)) for p in p_ticks]
        yticklabels = [f"{int(p)}" for p in p_ticks]

        # --- True zonal ---
        im0 = axes[row, 0].imshow(
            ZT,
            aspect='auto',
            cmap='turbo',
            norm=log_norm
        )
        axes[row, 0].set_title('Zonal Mean (True)')
        axes[row, 0].set_xlabel('Latitude')
        axes[row, 0].set_ylabel('Pressure (hPa)')
        axes[row, 0].set_yticks(yticks)
        axes[row, 0].set_yticklabels(yticklabels)
        axes[row, 0].invert_yaxis()
        fig.colorbar(im0, ax=axes[row, 0], orientation='vertical', shrink=shk)

        # --- Pred zonal ---
        im1 = axes[row, 1].imshow(
            ZP,
            aspect='auto',
            cmap='turbo',
            norm=log_norm
        )
        axes[row, 1].set_title('Zonal Mean (Prediction)')
        axes[row, 1].set_xlabel('Latitude')
        axes[row, 1].set_ylabel('')
        axes[row, 1].set_yticks(yticks)
        axes[row, 1].set_yticklabels(yticklabels)
        axes[row, 1].invert_yaxis()
        fig.colorbar(im1, ax=axes[row, 1], orientation='vertical', shrink=shk)

        # --- % diff zonal ---
        im2 = axes[row, 2].imshow(
            ZD,
            aspect='auto',
            cmap='RdBu_r',
            vmin=-50, vmax=50
        )
        axes[row, 2].set_title('% Difference (Zonal)')
        axes[row, 2].set_xlabel('Latitude')
        axes[row, 2].set_ylabel('')
        axes[row, 2].set_yticks(yticks)
        axes[row, 2].set_yticklabels(yticklabels)
        axes[row, 2].invert_yaxis()
        fig.colorbar(im2, ax=axes[row, 2], orientation='vertical', shrink=shk)
        
        
        # --- RMS zonal ---
        im3 = axes[row, 3].imshow(
            rms_zonal_2d.values,
            aspect='auto',
            cmap='RdBu_r',
            norm=mcolors.LogNorm(vmin=1, vmax=300)
        )

        axes[row, 3].set_title('RMS Error (Zonal)')
        axes[row, 3].set_xlabel('Latitude')
        axes[row, 3].set_yticks(yticks)
        axes[row, 3].set_yticklabels(yticklabels)
        axes[row, 3].invert_yaxis()
        fig.colorbar(im3, ax=axes[row, 3], orientation='vertical', shrink=shk)

        

        # ============================================================
        # SAVE
        # ============================================================

        plt.savefig(hp['model_name'] + '_test_levels.png', dpi=100, bbox_inches='tight')
        plt.close(fig)

        print('------------done')
        
