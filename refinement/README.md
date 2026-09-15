# NDnet refinement

Probabilistic refinement of the NDnet prior against fixed-site and field-campaign observations of cloud droplet number concentration (`Nd`).

This branch uses a **dimensionless training objective** so the `lambda_*` coefficients are interpretable tuning parameters rather than unit-conversion factors. The implementation keeps production inference unchanged: the model predicts EOF coefficients directly and no post-inference pressure taper is applied.

The upper-atmosphere prior constraint is **scale separated**: the longitude-mean correction is regularized strongly to suppress broad latitude-band artifacts, while the longitude-anomaly correction is only weakly constrained so sparse high-reaching observations can still modify the local profile.

## COSP-to-MODIS input standardization

The base NDnet prior is trained with GiOcean/COSP inputs standardized using the variable-specific COSP statistics stored under `data/stats`. MODIS-driven base inference, probabilistic refinement, and production inference instead standardize the corresponding MODIS retrieval fields with MODIS-specific means and standard deviations (`13.945/9.665` for liquid optical thickness, `12.08/3.126` for liquid particle size, `0.268/0.205` for liquid retrieval fraction, and `551.241/177.277` for cloud-top pressure).

This is therefore a **source-specific (mixed) standardization** at the COSP-to-MODIS domain-transfer step, rather than applying the COSP training normalization directly to the MODIS retrievals. Constant physical-unit differences between the archives cancel within a consistently formed z-score, but differences in the COSP and MODIS climatological means and standard deviations constitute a first-order domain normalization and should be retained as part of the method description. `plots/MODIS_input_comparison.py` converts the raw fields to common physical units for plotting and writes `MODIS_input_comparison_standardization_domain_shift.csv`, which reports the COSP and MODIS normalization constants plus paired common-period mean/std diagnostics, including mean shift in COSP-standard-deviation units and the MODIS/COSP standard-deviation ratio.

Key files:

- `NDref_prob.py` — experiment entry point.
- `dimensionless_training.py` — explicit training and initialization workflow.
- `loss_normalization.py` — fixed loss scales and Nd-space prior metric.
- `NN_relax.py` — scale-separated upper-level relaxation.
- `LOSS_NORMALIZATION.md` — mathematical definition of the normalized objective.
- `UPPER_RELAX.md` — zonal/anomaly upper-atmosphere prior constraint.
- `refinement_driver.py` — shared loading, diagnostics, and inference utilities.

The fixed normalization constants and objective settings derived for each run are saved to `<MODEL_NAME>_loss_scales.json` to support reproducibility.