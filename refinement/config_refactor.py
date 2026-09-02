import sys
import numpy as np
import torch
import xarray as xr

# -------------------------------------------------
# Model / representation identities
# -------------------------------------------------
# MODEL_BASE names the fixed base-representation artifacts (EOF coefficient
# climatology, etc.) stored under BASEROOT.
# MODEL_NAME names the refinement model/checkpoint in the current directory.
MODEL_BASE = "NDnet_EOFcoeff"
MODEL_NAME = "NDnet_EOFcoeff_sigmaFT"
BASEROOT = "/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/in_cloudM2/best/"


do_train = False
do_sites = True
do_campaigns = True
do_test = True
min_valid_lev = 2
write_site_debug = False
print_every = 50
bypass_infer = False

prefix = '*2020*'
prefix_dense = '*20[0-2]*'
prefix_sites = '*20[1-2]*'
test_pr = '*2009*'

deb = 0
if deb > 0:
    prefix_dense = '*2020*'
    prefix_sites = '*2020*'
    test_pr = '*200901*'
    print_every = 10

sitemaskmode = "obs_dates" # "obs_levels" | "obs_dates" | "none" makes little difference, use dates though

# -------------------------------------------------
# NetCDF temporary writes
# -------------------------------------------------
# safe_to_netcdf() and safe_to_netcdf_chunked() stage files here before
# moving them to their final path. Use a writable project/scratch directory.
# An absolute path is also fine, e.g. "/panfs/.../my_scratch/netcdf_tmp".
NETCDF_TMPDIR = "./temp"

# -------------------------------------------------
# Fast repeated site/campaign diagnostics
# -------------------------------------------------
# The first evaluation run computes the raw (untapered) site/campaign products
# and stores them here. Subsequent taper experiments reuse those small files,
# so the model, site projection, and dense diagnostic inference are skipped.
# Cache validity includes MODEL_NAME weights, EOF/base files, prefixes, and
# relevant diagnostic settings; stale caches are rejected automatically.
REUSE_DIAGNOSTIC_CACHE = True
WRITE_DIAGNOSTIC_CACHE = True
DIAGNOSTIC_CACHE_DIR = "./cache/diagnostics"

# On a cache miss, run the dense NN only on the union of days required by the
# requested site/campaign diagnostics. Set False for the legacy all-day pass.
SPARSE_DIAGNOSTIC_INFERENCE = True

# Use the actual unfilled CloudNet observation dates for the site diagnostic
# selector. The training/projection target is intentionally filled in time and
# therefore cannot be used to identify observed days. Set False only if you
# deliberately want the old dense training-validity behavior.
SITE_DIAGNOSTIC_OBS_DAYS_ONLY = True

# -------------------------------------------------
# Plotting thresholds
# -------------------------------------------------
# Minimum Nd retained by the global Nd comparison diagnostic. Keeping this in
# config makes taper experiments directly comparable without editing plotting.py.
MIN_ND = 1.0

# -------------------------------------------------
# Post-inference upper-pressure taper
# -------------------------------------------------
# Options:
#   "none"        -> original EOF reconstruction
#   "climatology" -> relax refined Nd toward Nd_mean aloft
#   "zero"        -> relax total refined Nd toward zero aloft
#   "base"        -> relax refined Nd toward the untouched
#                    reanalysis-emulation/base reconstruction aloft
#
# The trained model and inferred EOF coefficients are unchanged.
# Nd_base is never tapered or otherwise modified.
UPPER_TAPER_MODE = "climatology"
UPPER_TAPER_START_HPA = 500.0
UPPER_TAPER_TOP_HPA = 350.0
SAVE_RAW_TAPER_FIELDS = True


hp = {
    "nepochs": 3000,
    "batch_sz": 16,
    "l_r": 5e-5,
    "patience": 20,

    # MODEL_NAME is always the model being produced/evaluated in ./
    "model_name": MODEL_NAME,
    "rolling_Nr": 10,
    "basech": 64,

    "site_sampling_beta": 1.0,
    "use_site_sampler": False,
    "min_sites_for_sampling": 1,

    # Mean constraints
    "lambda_sites": 0.05,
    "lambda_anchor": 1.0,
    "lambda_anchor_decay": 1.0,
    "lambda_anchor_min": 0.1,
    "lambda_grad": 0.0,

    "lambda_nd": 0.5,
    "lambda_campaign": 1.0,

    # Probabilistic terms
    "lambda_Ndgauss": 0.25,
    "lambda_campaign_gauss": 0.20,

    # Variance-relative sigma penalty
    "lambda_sigma": 2.0,
    "lambda_campaign_sigma": 1.0,
    "sigma_mode_var_floor": 0.05,

    "min_sigma": 0.1,

    # Protect shared mean features from sigma branch
    "detach_sigma_features": True,

    # -------------------------------------------------
    # Optional sigma-only fine tuning
    # -------------------------------------------------
    "fine_tune_sigma": True,

    # Keep MODEL_NAME as the output/current model name rather than appending
    # another _sigmaFT suffix in active_model_name().
    "sigma_finetune_model_name": MODEL_NAME,

    # Source checkpoint for sigma-only fine tuning. This is a MODEL checkpoint,
    # not a MODEL_BASE representation artifact. It also lives in ./.
    "sigma_finetune_source_weights": "./NDnet_EOFcoeff_weights.pt",

    "sigma_finetune_lr": 5e-5,
    "sigma_finetune_epochs": 50,
    "sigma_finetune_patience": 15,

    "sigma_finetune_lambda_sigma": 20.0,
    "sigma_finetune_lambda_campaign_sigma": 6.0,

    # Optional: otherwise retain normal Gaussian-loss values
    "sigma_finetune_lambda_sites": 0.04,
    "sigma_finetune_lambda_Ndgauss": 0.20,
    "sigma_finetune_lambda_campaign_gauss": 0.15,

    # Leave this off for the calibration pass.
    # The normal validation loss already contains the sigma penalties.
    "sigma_finetune_use_campaign_monitor": False,
}


# Paths / constants
# -----------------------------
SITE_JSON = "/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/dailies/Cloudnet_sites_multiyear/site_timeprofile_diagnostics.jsonl"

# Model checkpoints are local to the run directory.
legacy_path = "./" + MODEL_NAME + "_weights.pt"
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
    'NDinc_cf01p': '/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/dailies/Cloudnet_sites_multiyear',
    'a_resid_pred': '/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/inference/bymonth',
}


CAMPAIGN_TRAIN_DIR = '/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/field_campaigns/training'

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


# -----------------------------------
# Mode weights (variance-aware)
# -----------------------------------
mode_var = np.sum(EOF.values**2, axis=1)

mode_weights = torch.tensor(
    mode_var / mode_var.mean(),
    dtype=torch.float32
)
