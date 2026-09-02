"""Production inference configuration for the learned-upper-relax model."""

from config_refactor import *  # noqa: F401,F403

MODEL_NAME = "NDnet_EOFcoeff_upperRelax"
model_path = "./" + MODEL_NAME + "_weights.pt"

# Kept explicit for clarity even though the restored V2-style NDref_inf.py does
# not call the post-processing taper at all.
UPPER_TAPER_MODE = "none"
SAVE_RAW_TAPER_FIELDS = False
