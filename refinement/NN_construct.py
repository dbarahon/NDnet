"""Public refinement model/loss interface for the dimensionless branch.

The network architecture remains in ``NN_construct_v2.py``. The exported loss
adds fixed dimensionless normalization and learned upper-level relaxation.
"""

from NN_construct_v2 import *  # noqa: F401,F403
from NN_relax import SparseSiteCampaignNLLRefineLossRelax

SparseSiteCampaignNLLRefineLoss = SparseSiteCampaignNLLRefineLossRelax
