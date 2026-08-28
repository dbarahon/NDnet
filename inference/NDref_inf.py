from config import *
from data_handling import *
from utils import *
from NN_construct import *
from train_infer import *
from plotting import plot_global_level_comparison, plot_vertical_profiles_sites, plot_global_level_sigma_inference
from profile_taper import apply_nd_pressure_taper, apply_sigma_pressure_taper, normalize_taper_mode

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
import torch.multiprocessing as mp
import dask.array as da
import matplotlib.pyplot as plt

from torch.utils.data import WeightedRandomSampler
import torch.multiprocessing as m

from datetime import datetime, timedelta
import os, glob, sys, gc, time


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  # PyTorch 2.x




def build_model(in_channels, numgpu, device):
    model = ProbRefineFromBaseAndX(
        in_channels=in_channels,
        num_modes=num_modes,
        feat_ch=32,
        head_ch=64,
        sigma_min=1e-3,
    ).to(device)

    if numgpu > 1:
        model = nn.DataParallel(model)
    return model

def load_PS_on_Nout(day, Nout_template, PS_regridder_cache):
    """
    Load daily-mean surface pressure, regrid to Nout grid.

    Returns
    -------
    PS_on : xr.DataArray
        dims: (time=1, lat, lon)
        units: Pa
    """

    yyyy = day.strftime("%Y")
    yyyymm = day.strftime("%Y%m")
    t0 = np.datetime64(day.strftime("%Y-%m-%d"))

    pth = (
        "/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/dailies/"
        "atm_inst_6hr_glo_L720x361_p49/"
        f"GiOCEAN_e1.atm_inst_6hr_glo_L720x361_p49.tavg1D.{yyyymm}.nc"
    )

    if not os.path.exists(pth):
        raise FileNotFoundError(f"Missing PS file: {pth}")

    ds = xr.open_dataset(pth, chunks={"time": 1})[["PS"]]
    ds = normalize_time_coord(ds)

    # select the day
    ds = ds.sel(time=t0)

    # build regridder ONCE
    if PS_regridder_cache.get("Rps") is None:
        PS_regridder_cache["Rps"] = xe.Regridder(
            ds,
            Nout_template.isel(time=0),
            "bilinear",
            periodic=True,
            reuse_weights=False
        )

    PS_on = PS_regridder_cache["Rps"](ds)
    PS_on = PS_on.expand_dims(time=[t0])

    # mask fill values
    PS_on = PS_on.where(PS_on < 1e14)
    
    #print('-------------', PS_on)

    return PS_on["PS"]


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    numgpu = torch.cuda.device_count()
    mp.set_start_method("spawn", force=True)

    # IMPORTANT: for dask/xarray streaming, start with 0 workers (avoid duplicate IO)
    numwork = 0

    print("Using DataParallel on", numgpu, "GPUs")

    taper_mode = normalize_taper_mode(UPPER_TAPER_MODE)
    taper_start_hpa = float(UPPER_TAPER_START_HPA)
    taper_top_hpa = float(UPPER_TAPER_TOP_HPA)
    print(
        f"[TAPER] mode={taper_mode} start={taper_start_hpa:g} hPa "
        f"top={taper_top_hpa:g} hPa"
    )
    
    in_channels = 4
    model = build_model(in_channels, numgpu, device)

    # Load weights
    load_model_weights_for_inference(
        model,
        model_path,
        device=device,
        strict=True,     # set False if you changed heads
        eval_mode=True
    )
    

    # ----------------------------
    # Daily loop
    # ----------------------------
    R2_cache = {}  # holds persistent MODIS->Nout regridder
    d0 = datetime.fromisoformat(START_DATE)
    d1 = datetime.fromisoformat(END_DATE)
    PS_regridder_cache={}

    day = d0
    
    while day <= d1:
        ymd = day.strftime("%Y%m%d")        
        out_nc = os.path.join(OUTDIR, f"NDnet_infer_{ymd}.nc4")

         
        
        # -------------------------------
        # RESUME LOGIC
        # -------------------------------
        if os.path.exists(out_nc):
            try:
                # light sanity check (won't load data)
                xr.open_dataset(out_nc, engine="netcdf4").close()
                #print(f"[SKIP] {ymd} already processed")
                day += timedelta(days=1)
                continue
            except Exception:
                print(f"[WARN] {ymd} exists but is corrupted → recomputing")

        try:
            print(f"\n=== {ymd} ===")
            Nin_2d, Ares_base = get_data(
            vars_out=vars_out,
            pth_all=pth_all,
            vars_in_2d=vars_in_2d,
            modis_source=modis_source,
            day=day,
            cache_folder="cache",
            minP=minP,
            )
        except Exception:
                print(f"XXXXXX[ERROR]XXXXXXXXXXXXXXX    {ymd}")
                day += timedelta(days=1)
                continue    

        X2d_t, MU0_t = tensorize_inputs(
                    Nin_2d=Nin_2d,
                    Ares_base=Ares_base,
                    pin_memory=True,
                    verbose=False,
                )

        
        nt = Nin_2d.sizes["time"]
        all_idx = np.arange(nt, dtype=np.int64)
        
        test_ds = DTS_in(X2d_t, MU0_t, all_idx)
        test_loader   = DataLoader(test_ds, 
              shuffle=False, num_workers=0, pin_memory=True)
           
        # ============================================================
        #  inference + Nd and sigmaNd reconstruction
        # ============================================================

        coeffs_resid_nn, sig = run_infer_x_mu0(device, test_loader, model)  # numpy (T,M,lat,lon)

        # absolute coeffs = residual + Amean
        coeffs_abs_nn   = coeffs_resid_nn + Amean4               # numpy (T,M,lat,lon)

        # daskify NN coeffs (+ sigma)
        coeffs_abs_nn_da = da.from_array(coeffs_abs_nn.astype(np.float32), chunks=(64, -1, -1, -1))
        sig_nn_da        = da.from_array(sig.astype(np.float32),           chunks=(64, -1, -1, -1))

        # coords
        time_coord = Ares_base["time"]
        lat_coord  = Ares_base["lat"]
        lon_coord  = Ares_base["lon"]

        # reconstruct Nd globally + sigmaNd for NN
        ds_nn = reconstruct_Nd_from_coeffs_chunked(
            coeffs_abs_nn_da,
            EOF,
            Nd_mean,
            time_coord,
            lat_coord,
            lon_coord,
            sigma_coeffs_abs=sig_nn_da,
            chunk_t=64,
        )

        # Keep literal EOF reconstruction available for optional diagnostics,
        # but use the tapered fields as the final product.
        Nd_raw = ds_nn["Nd_rec"].rename("NDinc_cf01p_raw")
        sigma_raw = ds_nn["sigmaNd_rec"].rename("sigmaNd_raw")

        Nd_final = apply_nd_pressure_taper(
            Nd_raw,
            Nd_mean=Nd_mean,
            mode=taper_mode,
            p_start=taper_start_hpa,
            p_top=taper_top_hpa,
            name="NDinc_cf01p",
        )
        sigma_final = apply_sigma_pressure_taper(
            sigma_raw,
            mode=taper_mode,
            p_start=taper_start_hpa,
            p_top=taper_top_hpa,
            name="sigmaNd",
        )

        ds_nn = xr.Dataset({"Nd_rec": Nd_final, "sigmaNd_rec": sigma_final})
        if SAVE_RAW_TAPER_FIELDS:
            ds_nn["Nd_rec_raw"] = Nd_raw
            ds_nn["sigmaNd_rec_raw"] = sigma_raw

        
        # --------------------------------------------------
        # Surface-pressure mask (p > Ps) MERRA
        # --------------------------------------------------
        PS_on = load_PS_on_Nout(day, ds_nn, PS_regridder_cache)
        
        # If load_PS_on_Nout returns time as datetime64[s] and ds_nn is datetime64[ns], force match:
        PS_on = PS_on.assign_coords(time=ds_nn.time)

        # Build 4D fields on ds_nn's exact dims (time, lev, lat, lon)
        tmpl  = ds_nn["Nd_rec"]                 # (time, lev, lat, lon)
        lev4d = (ds_nn["lev"] * 100.0).broadcast_like(tmpl)   # Pa
        ps4d  = PS_on.broadcast_like(tmpl)                   # Pa

        mask = lev4d < ps4d                                  # True where level is below surface
        ds_nn = ds_nn.where(mask)

        
        # Ensure these are DataArrays with correct names
        Nd_nn_da      = ds_nn["Nd_rec"].rename("NDinc_cf01p")
        sigmaNd_nn_da = ds_nn["sigmaNd_rec"].rename("sigmaNd")

        # If you want to force time to exactly "day" (optional):
        tcoord = np.array([np.datetime64(day.strftime("%Y-%m-%d"))], dtype="datetime64[ns]")
        Nd_nn_da      = Nd_nn_da.assign_coords(time=tcoord)
        sigmaNd_nn_da = sigmaNd_nn_da.assign_coords(time=tcoord)

        # Build coeff DataArrays with coords
        mode = np.arange(num_modes)
        a_resid_da = xr.DataArray(
            coeffs_resid_nn.astype(np.float32),   # shape (T,M,lat,lon) already
            dims=("time", "mode", "lat", "lon"),
            coords={"time": tcoord, "mode": mode, "lat": lat_coord, "lon": lon_coord},
            name="a_resid_pred",
        )
        a_abs_da = xr.DataArray(
            coeffs_abs_nn.astype(np.float32),
            dims=("time", "mode", "lat", "lon"),
            coords={"time": tcoord, "mode": mode, "lat": lat_coord, "lon": lon_coord},
            name="a_abs_pred",
        )

        out = xr.Dataset(
            {
                "NDinc_cf01p": Nd_nn_da,
                "sigmaNd": sigmaNd_nn_da,
                "a_resid_pred": a_resid_da,
                "a_abs_pred": a_abs_da,
            }
        )
        if SAVE_RAW_TAPER_FIELDS:
            out["NDinc_cf01p_raw"] = ds_nn["Nd_rec_raw"].assign_coords(time=tcoord)
            out["sigmaNd_raw"] = ds_nn["sigmaNd_rec_raw"].assign_coords(time=tcoord)

        out.attrs.update(
            upper_taper_mode=taper_mode,
            upper_taper_start_hpa=taper_start_hpa,
            upper_taper_top_hpa=taper_top_hpa,
        )

        out = out.chunk({"time": 1})
        #print(out)
        safe_to_netcdf_chunked(out, out_nc)
        day += timedelta(days=1)


 
    print("DONE.")
