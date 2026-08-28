import sys
import numpy as np
import torch
import xarray as xr

MODEL_NAME =  "NDnet_EOFcoeff"


do_train =  True
do_sites = True 
do_campaigns= True   
do_test = True
min_valid_lev =  2
write_site_debug =  False 
print_every =50
bypass_infer= False

prefix = '*2020*'
prefix_dense = '*20[0-2]*'
prefix_sites = '*20[1-2]*'
test_pr = '*2009*' 

deb =  0
if deb > 0:
    prefix_dense = '*2020*'
    prefix_sites = '*2020*'
    test_pr = '*200901*' 
    print_every =10

sitemaskmode = "obs_dates" # "obs_levels" | "obs_dates" | "none" makes little difference, use dates though

# -------------------------------------------------
# Post-inference upper-pressure taper
# -------------------------------------------------
# Options:
#   "none"        -> original EOF reconstruction
#   "climatology" -> taper only Nd anomaly to zero aloft
#   "zero"        -> taper total Nd to zero aloft
#
# The trained model and inferred EOF coefficients are unchanged.
UPPER_TAPER_MODE = "climatology"
UPPER_TAPER_START_HPA = 500.0
UPPER_TAPER_TOP_HPA = 350.0
SAVE_RAW_TAPER_FIELDS = True


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


    "sigma_finetune_lr": 5e-5,
    "sigma_finetune_epochs": 50,
    "sigma_finetune_patience": 15,

    "sigma_finetune_lambda_sigma": 20.0,
    "sigma_finetune_lambda_campaign_sigma": 6.0,

    # Optional: otherwise retain normal Gaussian-loss values
     "sigma_finetune_lambda_sites": 0.04,
     "sigma_finetune_lambda_Ndgauss": 0.20,
     "sigma_finetune_lambda_campaign_gauss": 0.15,

    # I recommend leaving this off for the calibration pass.
    # The normal validation loss already contains the sigma penalties.
    "sigma_finetune_use_campaign_monitor": False,
}




# Paths / constants
# -----------------------------
MODELROOT = "/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/in_cloudM2/best/"
SITE_JSON = "/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/dailies/Cloudnet_sites_multiyear/site_timeprofile_diagnostics.jsonl"

legacy_path = MODELROOT + MODEL_NAME + "_weights.pt"
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
'NDinc_cf01p': '/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/dailies/Cloudnet_sites_multiyear', #masked site data  
'a_resid_pred': '/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/inference/bymonth', #precomputed residual alphas from base model
}


CAMPAIGN_TRAIN_DIR =  '/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/field_campaigns/training'

EOF_ds  = xr.open_dataset(f"{MODELROOT}Nd_EOF.nc")
EOF     = EOF_ds["EOFs"]          # (mode, lev)
Nd_mean = EOF_ds["Nd_mean"]       # (lev)
num_modes = EOF.sizes["mode"]
 # Load A_mean: IMPORTANT ? this was saved on Nout (training) grid
A_mean = np.load(MODELROOT + MODEL_NAME + "_coeff_mean.npy")  # (M, lat, lon) on Nout grid
# Fallback: zero mean where no samples exist
A_mean = np.where(np.isnan(A_mean), 0.0, A_mean)
Amean4 = A_mean[None, :, :, :]
     
    

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

    
    
