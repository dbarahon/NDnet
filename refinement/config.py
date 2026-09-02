import numpy as np
import torch
import xarray as xr

# ============================================================
# Experiment / workflow controls
# ============================================================

MODEL_BASE = "NDnet_EOFcoeff"
MODEL_NAME = "NDnet_EOFcoeff_upperRelax"
BASEROOT = "/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/in_cloudM2/best/"

# Independent actions.
do_train = True
do_sites = True
do_campaigns = True
do_test = True

# Training data selection.  This does not affect campaign diagnostics:
# TRAIN_WITH_CAMPAIGNS=False trains on fixed sites only, while
# do_campaigns=True can still evaluate the resulting model on campaigns.
TRAIN_WITH_CAMPAIGNS = True

# Training initialization:
#   "weights" -> start epoch 0 from TRAIN_INIT_WEIGHTS
#   "scratch" -> random model initialization, ignore checkpoints/weights
#   "resume"  -> resume MODEL_NAME.ckpt when present; otherwise random init
TRAIN_INIT_MODE = "weights"
TRAIN_INIT_WEIGHTS = "./NDnet_EOFcoeff_sigmaFT_weights.pt"

min_valid_lev = 2
write_site_debug = False
print_every = 50
bypass_infer = False

prefix = "*2020*"
prefix_dense = "*20[0-2]*"
prefix_sites = "*20[1-2]*"
test_pr = "*2009*"

deb = 0
if deb > 0:
    prefix_dense = "*2020*"
    prefix_sites = "*2020*"
    test_pr = "*200901*"
    print_every = 10

sitemaskmode = "obs_dates"

# ============================================================
# Diagnostics / output
# ============================================================

NETCDF_TMPDIR = "./temp"

REUSE_DIAGNOSTIC_CACHE = True
WRITE_DIAGNOSTIC_CACHE = True
DIAGNOSTIC_CACHE_DIR = "./cache/diagnostics"
SPARSE_DIAGNOSTIC_INFERENCE = True
SITE_DIAGNOSTIC_OBS_DAYS_ONLY = True

MIN_ND = 1.0

# The refactored diagnostic driver still understands the historical taper
# interface.  Keep it explicitly disabled: learned-upper-relax predictions,
# including site/campaign diagnostics, are always evaluated raw.
UPPER_TAPER_MODE = "none"
UPPER_TAPER_START_HPA = 600.0
UPPER_TAPER_TOP_HPA = 350.0
SAVE_RAW_TAPER_FIELDS = False

# ============================================================
# Learned upper-level relaxation
# ============================================================

# These control only the training loss.  No taper is applied at inference.
UPPER_RELAX_START_HPA = 600.0
UPPER_RELAX_TOP_HPA = 350.0
UPPER_RELAX_LAMBDA = 0.1
UPPER_RELAX_ND_SCALE = 100.0  # cm-3

# ============================================================
# Optimization
# ============================================================

hp = {
    "nepochs": 3000,
    "batch_sz": 16,
    "l_r": 5e-5,
    "patience": 20,

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
    "lambda_sigma": 2.0,
    "lambda_campaign_sigma": 1.0,
    "sigma_mode_var_floor": 0.05,
    "min_sigma": 0.1,
    "detach_sigma_features": True,

    # Learned-relax runs train the full V2 model.
    "fine_tune_sigma": False,

    # Do not add a campaign-only monitor when campaigns are excluded
    # from training.
    "use_campaign_monitor": bool(TRAIN_WITH_CAMPAIGNS),
}

# ============================================================
# Data / representation
# ============================================================

SITE_JSON = (
    "/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/dailies/"
    "Cloudnet_sites_multiyear/site_timeprofile_diagnostics.jsonl"
)

legacy_path = "./" + MODEL_NAME + "_weights.pt"
train_cache_path = "cache_train"
test_cache_path = "cache_test"

pth_all = "/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/"
t_chk = 16
vars_in_3d = {}
modis_source = "MCD06COSP_D3_MODIS"

vars_in_2d = {
    "Cloud_Optical_Thickness_Liquid_Mean": {"mean": 13.945, "std": 9.665},
    "Cloud_Particle_Size_Liquid_Mean": {"mean": 12.08, "std": 3.126},
    "Cloud_Retrieval_Fraction_Liquid_Mean": {"mean": 0.268, "std": 0.205},
    "Cloud_Top_Pressure_Mean": {"mean": 551.241, "std": 177.277},
}

vars_out = {
    "NDinc_cf01p": (
        "/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/dailies/"
        "Cloudnet_sites_multiyear"
    ),
    "a_resid_pred": (
        "/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/inference/bymonth"
    ),
}

CAMPAIGN_TRAIN_DIR = (
    "/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/field_campaigns/training"
)
CAMPAIGN_PATTERN = "*_training.nc"

EOF_ds = xr.open_dataset(f"{BASEROOT}Nd_EOF.nc")
EOF = EOF_ds["EOFs"]
Nd_mean = EOF_ds["Nd_mean"]
num_modes = EOF.sizes["mode"]

A_mean = np.load(BASEROOT + MODEL_BASE + "_coeff_mean.npy")
A_mean = np.where(np.isnan(A_mean), 0.0, A_mean)
Amean4 = A_mean[None, :, :, :]

mode_var = np.sum(EOF.values**2, axis=1)
mode_weights = torch.tensor(
    mode_var / mode_var.mean(),
    dtype=torch.float32,
)
