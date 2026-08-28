import sys
import numpy as np
import torch
import xarray as xr
import os, glob
from utils import normalize_time_coord
MODEL_NAME =  "NDnet_EOFcoeff_sigmaFT"
MODEL_BASE =  "NDnet_EOFcoeff"
min_valid_lev =  2

bypass_infer= False

START_DATE = "2003-01-01"
END_DATE   = "2024-12-31"   # inclusive
OUTDIR     = "./NDnet_infer_daily/"
os.makedirs(OUTDIR, exist_ok=True)

# -------------------------------------------------
# Post-inference upper-pressure taper
# -------------------------------------------------
# Options:
#   "none"        -> original EOF reconstruction
#   "climatology" -> relax refined Nd toward Nd_mean aloft
#   "zero"        -> relax total refined Nd toward zero aloft
#   "base"        -> relax refined Nd toward the physically based
#                    reanalysis-emulation/base reconstruction aloft
UPPER_TAPER_MODE = "climatology"
UPPER_TAPER_START_HPA = 500.0
UPPER_TAPER_TOP_HPA = 350.0
SAVE_RAW_TAPER_FIELDS = False




# Paths / constants
# -----------------------------
MODELROOT = "/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/in_cloudM2/" #base model
SITE_JSON = "/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/dailies/Cloudnet_sites_multiyear/site_timeprofile_diagnostics.jsonl"

model_path = "/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/refinement/push8_sigma2/" + MODEL_NAME + "_weights.pt"
train_cache_path =  "cache_train"
test_cache_path =  "cache_test"
    

pth_all = '/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/'

t_chk = 16  
vars_in_3d = {}
# MODIS folder name under pth_all/dailies/
modis_source = "MCD06COSP_D3_MODIS"   # <-- no trailing slash

# Stats 
vars_in_2d = {
"Cloud_Optical_Thickness_Liquid_Mean": {"mean": 13.945,   "std": 9.665},
"Cloud_Particle_Size_Liquid_Mean":     {"mean": 12.08,"std": 3.126},  # <-- use your real values here
"Cloud_Retrieval_Fraction_Liquid_Mean":{"mean": 0.268,    "std": 0.205},
"Cloud_Top_Pressure_Mean":             {"mean": 551.241,  "std": 177.277},
}

#here we have two vars out var:path
vars_out = {
'NDinc_cf01p_pred': '/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/inference/NDnet_infer_daily/', #refer data 
'a_resid_pred': '/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/inference/NDnet_infer_daily/', #precomputed residual alphas from base model
}




EOF_ds  = xr.open_dataset(f"{MODELROOT}Nd_EOF.nc")
EOF     = EOF_ds["EOFs"]          # (mode, lev)
Nd_mean = EOF_ds["Nd_mean"]       # (lev)
num_modes = EOF.sizes["mode"]
 # Load A_mean: IMPORTANT — this was saved on Nout (training) grid
A_mean = np.load(MODELROOT + MODEL_BASE + "_coeff_mean.npy")  # (M, lat, lon) on Nout grid
# Fallback: zero mean where no samples exist
A_mean = np.where(np.isnan(A_mean), 0.0, A_mean)
Amean4 = A_mean[None, :, :, :]
minP =250     

# ----------------------------
# Templates / EOF reconstruction constants
# ----------------------------

chk3d = {"time": 1, "lev": -1, "lat": -1, "lon": -1}
sample_date="201001"
# Load one sample of Ndout to define Nout grid
v0 = list(vars_out.keys())[0]
pth = f"{vars_out[v0]}/*{sample_date}*"
files = sorted(glob.glob(pth))
if len(files) == 0:
    raise FileNotFoundError(f"No Nout sample found with pattern: {pth}")

D = xr.open_mfdataset(files, parallel=True, chunks=chk3d)[[v0]]
D = normalize_time_coord(D)
D = D.sel(lev=D.lev.where(D.lev >= minP, drop=True))
D = D.where(D >= 0)
Nout_template = D  # has (time, lev, lat, lon) but time typically 1

        

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

    
    
