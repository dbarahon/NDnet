"""Run refinement with optional post-inference pressure tapering.

Behavior
--------
- ``do_train=True``: execute the existing training workflow unchanged.  Its
  existing site/campaign diagnostics run after training.
- ``do_train=False``: skip all optimization, including sigma fine tuning, run
  the existing global test path (if enabled), then run the requested site and
  campaign diagnostics from the already-trained local MODEL_NAME checkpoint.

The pressure-taper hooks are installed before either path so every diagnostic
uses the same final NN product while the base reconstruction remains untouched.
"""

from pathlib import Path
import runpy

import config as cfg
from taper_hooks import install_refinement_taper_hooks


if __name__ == "__main__":
    install_refinement_taper_hooks()

    # The legacy main already behaves correctly for training and for the
    # global do_test path.  With do_train=False it never enters
    # configure_training_stage(), so sigma fine tuning is not executed.
    runpy.run_path(
        str(Path(__file__).with_name("NDref_prob.py")),
        run_name="__main__",
    )

    # In the legacy main, site/campaign diagnostics are nested inside
    # ``if do_train``.  Supply them explicitly in evaluation-only mode.
    if not bool(cfg.do_train) and (
        bool(getattr(cfg, "do_sites", False))
        or bool(getattr(cfg, "do_campaigns", False))
    ):
        from eval_sites_campaigns import run_eval_sites_campaigns

        run_eval_sites_campaigns()
