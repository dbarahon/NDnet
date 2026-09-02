"""Configuration for the learned-upper-relax experiment.

All refactored diagnostic/cache settings are inherited unchanged from
``config_refactor.py``.  Only experiment-specific model/training settings are
overridden here.
"""

from config_refactor import *  # noqa: F401,F403

# Distinct output name for this experiment.
MODEL_NAME = "NDnet_EOFcoeff_upperRelax"

# Keep the independent train/site/campaign/global switches from the refactor.
do_train = True
do_sites = True
do_campaigns = True
do_test = True

# No post-processing taper anywhere in refinement diagnostics.  Site and
# campaign tests therefore evaluate the raw learned model at every valid
# observed pressure, including high-altitude observations.
UPPER_TAPER_MODE = "none"
SAVE_RAW_TAPER_FIELDS = False

# Learned pressure-aware attraction of the raw NN reconstruction back to the
# V2 base/reanalysis-emulation profile.  This is a LOSS schedule, not an
# inference transformation.
UPPER_RELAX_START_HPA = 600.0
UPPER_RELAX_TOP_HPA = 350.0
UPPER_RELAX_LAMBDA = 0.1
UPPER_RELAX_ND_SCALE = 100.0  # cm-3; makes the added quadratic dimensionless

# Keep the known-good V2 mean/sigma objective.  We are testing only the new
# upper-relaxation term, so sigma-only fine tuning is disabled for this run.
hp = dict(hp)
hp["model_name"] = MODEL_NAME
hp["fine_tune_sigma"] = False

# Initialize a fresh relaxation run from the current kinked/high-performing V2
# weights.  NDref_prob.py seeds a clean epoch-0 checkpoint from this file only
# when MODEL_NAME.ckpt does not already exist.
hp["init_from_weights"] = "./NDnet_EOFcoeff_sigmaFT_weights.pt"

legacy_path = "./" + MODEL_NAME + "_weights.pt"
