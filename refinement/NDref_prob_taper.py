"""Run the existing refinement workflow with post-inference pressure tapering.

The underlying NDref_prob.py, model architecture, weights, and training losses
remain unchanged.  This launcher installs taper hooks before executing the
existing main script so site, campaign, and global diagnostics all use the
same final product.
"""

from pathlib import Path
import runpy

from taper_hooks import install_refinement_taper_hooks


if __name__ == "__main__":
    install_refinement_taper_hooks()
    runpy.run_path(
        str(Path(__file__).with_name("NDref_prob.py")),
        run_name="__main__",
    )
