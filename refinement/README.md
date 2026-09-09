# NDnet refinement

Probabilistic refinement of the NDnet prior against fixed-site and field-campaign observations of cloud droplet number concentration (`Nd`).

This branch uses a **dimensionless training objective** so the `lambda_*` coefficients are interpretable tuning parameters rather than unit-conversion factors. The implementation keeps production inference unchanged: the model predicts EOF coefficients directly and no post-inference pressure taper is applied.

The upper-atmosphere prior constraint is **scale separated**: the longitude-mean correction is regularized strongly to suppress broad latitude-band artifacts, while the longitude-anomaly correction is only weakly constrained so sparse high-reaching observations can still modify the local profile.

Key files:

- `NDref_prob.py` — experiment entry point.
- `dimensionless_training.py` — explicit training and initialization workflow.
- `loss_normalization.py` — fixed loss scales and Nd-space prior metric.
- `NN_relax.py` — scale-separated upper-level relaxation.
- `LOSS_NORMALIZATION.md` — mathematical definition of the normalized objective.
- `UPPER_RELAX.md` — zonal/anomaly upper-atmosphere prior constraint.
- `refinement_driver.py` — shared loading, diagnostics, and inference utilities.

The fixed normalization constants and objective settings derived for each run are saved to `<MODEL_NAME>_loss_scales.json` to support reproducibility.
