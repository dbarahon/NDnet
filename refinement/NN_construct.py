"""V2 refinement model with learned upper-level relaxation.

The exact V2 implementation is retained in ``NN_construct_v2.py``.  This
module re-exports that implementation and replaces only the loss class with
the pressure-aware learned-relaxation subclass.
"""

from NN_construct_v2 import *  # noqa: F401,F403
from NN_relax import SparseSiteCampaignNLLRefineLossRelax

SparseSiteCampaignNLLRefineLoss = SparseSiteCampaignNLLRefineLossRelax
