import sys
import numpy as np
import torch
import xarray as xr
import os, glob
from utils import normalize_time_coord

# -------------------------------------------------
# Model / representation identities
# -------------------------------------------------
# MODEL_BASE names the fixed EOF/base representation stored under BASEROOT.
# MODEL_NAME names the trained refinement model in the current directory.
MODEL_BASE = "NDnet_EOFcoeff"
MODEL_NAME = "NDnet_EOFcoeff_sigmaFT"
BASEROOT = "/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/in_cloudM2/"

min_valid_lev = 2
bypass_infer = False

START_DATE = "2003-01-01"
END_DATE = "2024-12-31"   # inclusive
OUTDIR = "./NDnet_infer_daily/"
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
SITE_JSON = "/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/dailies/Cloudnet_sites_multiyear/site_timeprofile_diagnostics.jsonl"

# The trained/refined model is always loaded from the current run directory.
model_path = "./" + MODEL_NAME + "_weights.pt"
train_cache_path = "cache_train"
test_cache_path = "cache_test"

pth_all = '/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/'

t_chk = 16
vars_in_3d = {}
# MODIS folder name under pth_all/dailies/
modis_source = "MCD06COSP_D3_MODIS"   # <-- no trailing slash

# Stats
vars_in_2d = {
    "Cloud_Optical_Thickness_Liquid_Mean": {"mean": 13.945, "std": 9.665},
    "Cloud_Particle_Size_Liquid_Mean": {"mean": 12.08, "std": 3.126},
    "Cloud_Retrieval_Fraction_Liquid_Mean": {"mean": 0.268, "std": 0.205},
    "Cloud_Top_Pressure_Mean": {"mean": 551.241, "std": 177.277},
}

# here we have two vars out var:path
vars_out = {
    'NDinc_cf01p_pred': '/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/inference/NDnet_infer_daily/',
    'a_resid_pred': '/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/inference/NDnet_infer_daily/',
}


# -------------------------------------------------
# Fixed base representation: ALWAYS from BASEROOT / MODEL_BASE
# -------------------------------------------------
EOF_ds = xr.open_dataset(f"{BASEROOT}Nd_EOF.nc")
EOF = EOF_ds["EOFs"]          # (mode, lev)
Nd_mean = EOF_ds["Nd_mean"]   # (lev)
num_modes = EOF.sizes["mode"]

# A_mean belongs to the base EOF representation, not to MODEL_NAME.
A_mean = np.load(BASEROOT + MODEL_BASE + "_coeff_mean.npy")
A_mean = np.where(np.isnan(A_mean), 0.0, A_mean)
Amean4 = A_mean[None, :, :, :]
minP = 250

# ----------------------------
# Templates / EOF reconstruction constants
# ----------------------------
chk3d = {"time": 1, "lev": -1, "lat": -1, "lon": -1}
sample_date = "201001"

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
mode_var = np.sum(EOF.values**2, axis=1)

mode_weights = torch.tensor(
    mode_var / mode_var.mean(),
    dtype=torch.float32
)
