# Dimensionless refinement objective

This branch replaces unit-dependent loss magnitudes with fixed, explicit
normalizations. The goal is to make each `lambda_*` coefficient a scientific
tuning parameter rather than a compensation for the native numerical units of
a particular loss.

The objective is not dynamically balanced. Losses are allowed to differ when
the corresponding physical misfit differs. A campaign RMS error of 300 cm-3
should contribute more than a site RMS error of 100 cm-3. The normalization
only ensures that both are measured against the same fixed physical reference.

## Fixed scales

All normalizations are fixed for an entire training run. No running-loss or
batch-dependent rescaling is used.

### Nd-space scale

`LOSS_ND_SCALE` is a fixed physical reference concentration (default
100 cm-3). Site and campaign mean-squared errors are evaluated as

`mean[((Nd_pred - Nd_obs) / LOSS_ND_SCALE)^2]`.

The whole-column prior anchor and upper-level relaxation use the same scale.
This puts observational fit and departure from the prior in the same physical
currency.

### Coefficient-space scale

For each EOF mode, a reference coefficient scale is estimated once from the
training-period base residual coefficients. The scale is the
latitude-area-weighted RMS over training time and longitude. A small fixed
floor (`LOSS_COEFF_SCALE_FLOOR`) prevents a numerically negligible mode from
creating an arbitrarily strong penalty.

Sigma regularization uses `(sigma_m / S_m)^2`, where `S_m` is the fixed
training-prior RMS for mode `m`.

The derived coefficient scales are written to
`<MODEL_NAME>_loss_scales.json` for reproducibility.

## Projected site coefficient likelihood

The historical loss also included a Gaussian likelihood against EOF
coefficients inferred from sparse site profiles. Those coefficients are not
direct observations: they are obtained by projecting vertically incomplete
site profiles onto the EOF basis.

A Gaussian NLL also has no fixed order-one magnitude when the predicted sigma
is much smaller than the coefficient residual, even if its logarithmic term is
made dimensionless. In scratch training this term can therefore dominate the
objective before sigma has adapted.

For the publication-oriented dimensionless objective this optional term is
disabled by default:

`lambda_site_coeff_gauss = 0.0`.

Site information remains in the direct Nd-space MSE and Nd-space Gaussian
likelihood (`lambda_nd` and `lambda_Ndgauss`). The projected coefficient
likelihood can still be enabled explicitly for controlled sensitivity tests.

## Prior anchor

The dense prior anchor is evaluated in reconstructed Nd space rather than as a
diagonal coefficient MSE. For correction coefficients `c = dA - mu0`,

`dNd(p) = sum_m c_m EOF_m(p)`.

The anchor is the latitude-area-weighted and log-pressure-weighted vertical
mean of

`(dNd / LOSS_ND_SCALE)^2`.

The vertical quadratic is precomputed as a mode-space Gram matrix, so training
does not allocate a full `(batch, level, lat, lon)` correction tensor.

## Upper relaxation

The upper relaxation uses the same Nd-space metric after multiplying the
log-pressure integration weights by the configured pressure ramp. The upper
correction is separated into its longitude mean and longitude anomaly before
that metric is evaluated:

`L_upper = UPPER_RELAX_ZONAL_LAMBDA * L_upper_zonal`

`        + UPPER_RELAX_ANOMALY_LAMBDA * L_upper_anomaly`.

The zonal term constrains broad latitude-band departures from the prior. The
anomaly term constrains only local deviations from that longitude mean and is
weak by default so sparse high-reaching observations can still modify their
local profiles.

Because both components use the same `LOSS_ND_SCALE`, `lambda_anchor`,
`lambda_nd`, `lambda_campaign`, `UPPER_RELAX_ZONAL_LAMBDA`, and
`UPPER_RELAX_ANOMALY_LAMBDA` all multiply dimensionless squared departures
measured against the same physical reference.

The anchor and upper-relaxation terms are expected to be very small at the
start of a prior-centered refinement because `dA - mu0` is initially small.
They should grow only as the learned solution departs from the base. Their
small initial values are therefore not evidence of failed normalization.

## Gaussian likelihoods

Gaussian standardized residuals are unchanged. The logarithmic scale term is
written relative to the corresponding fixed reference scale, e.g.

`log(sigma_eff / LOSS_ND_SCALE) + 0.5 * z^2`.

This removes dependence on the arbitrary choice of physical units without
changing the standardized residual. Gaussian NLL values are not expected to
match MSE values numerically at every stage of optimization.

## Interpretation of lambdas

After normalization, a lambda near one corresponds to an order-one normalized
constraint for an order-one physical departure from its reference scale.
Lambdas still do not compensate automatically for how often a loss is sampled.
In particular, campaign observations occur only on campaign batches, so
campaign sampling frequency remains an independent aspect of the optimization
design.
