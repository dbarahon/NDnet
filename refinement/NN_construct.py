"""Public network architecture interface for refinement workflows.

The model implementation remains in ``NN_construct_v2.py``. Training modules
import their loss explicitly so the active scientific objective is visible at
the call site rather than installed through an alias.
"""

from NN_construct_v2 import *  # noqa: F401,F403
