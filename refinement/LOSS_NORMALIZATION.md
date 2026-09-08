# Dimensionless refinement objective

This branch replaces unit-dependent loss magnitudes with fixed, explicit
normalizations. The goal is to make each `lambda_*` coefficient a scientific
tuning parameter rather than a compensation for the native numerical units of
a particular loss.

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

Coefficient Gaussian likelihoods use `sigma_m / S_m` in the logarithmic term,
and sigma regularization uses `(sigma_m / S_m)^2`, where `S_m` is the fixed
training-prior RMS for mode `m`.

The derived coefficient scales are written to
`<MODEL_NAME>_loss_scales.json` for reproducibility.

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
log-pressure integration weights by the configured pressure ramp. Therefore
`lambda_anchor`, `lambda_nd`, `lambda_campaign`, and `UPPER_RELAX_LAMBDA` all
multiply dimensionless squared departures measured against the same Nd scale.

## Gaussian likelihoods

Gaussian standardized residuals are unchanged. The logarithmic scale term is
written relative to the corresponding fixed reference scale, e.g.

`log(sigma_eff / LOSS_ND_SCALE) + 0.5 * z^2`.

This removes dependence on the arbitrary choice of physical units without
changing the standardized residual.

## Interpretation of lambdas

After normalization, a lambda near one corresponds to an order-one normalized
constraint. Lambdas still do not compensate automatically for how often a loss
is sampled. In particular, campaign observations occur only on campaign
batches, so campaign sampling frequency remains an independent aspect of the
optimization design.
