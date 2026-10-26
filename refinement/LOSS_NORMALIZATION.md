# Dimensionless refinement objective

The refinement loss uses fixed physical reference scales so its weights (`lambda_*`) multiply dimensionless terms. Scales are constant throughout training; there is no batch-dependent loss balancing.

## Nd-space terms

`LOSS_ND_SCALE` (default `100 cm-3`) normalizes site and campaign errors:

```
L_Nd = mean[((Nd_pred - Nd_obs) / LOSS_ND_SCALE)^2]
```

The prior anchor and upper-level regularizer use the same concentration scale. For EOF coefficient correction `c = dA - mu0`, the associated concentration correction is `dNd(p) = sum_m c_m EOF_m(p)`. The prior anchor integrates its normalized square using latitude-area and log-pressure weights. A precomputed EOF Gram matrix evaluates this quadratic directly in coefficient space.

## Coefficient scales and likelihoods

Each EOF mode uses a fixed coefficient scale `S_m`, calculated from the latitude-weighted RMS of training-period prior residual coefficients, with a floor `LOSS_COEFF_SCALE_FLOOR`. Coefficient uncertainty regularization uses `(sigma_m / S_m)^2`. The scales are recorded in `<MODEL_NAME>_loss_scales.json`.

Gaussian likelihoods use standardized residuals and dimensionless log-scale terms such as `log(sigma_eff / LOSS_ND_SCALE) + 0.5*z^2`. The projected site-coefficient likelihood is optional; `lambda_site_coeff_gauss = 0` disables it. Direct site Nd MSE and Gaussian likelihood terms remain available.

## Upper-level constraint

Upper-level departures are separated into longitude-mean and longitude-anomaly components, each evaluated with a pressure-weighted EOF Gram matrix:

```
L_upper = UPPER_RELAX_ZONAL_LAMBDA * L_upper_zonal
        + UPPER_RELAX_ANOMALY_LAMBDA * L_upper_anomaly
```

See `UPPER_RELAX.md` and `SITE_ND_PRESSURE_WEIGHTING.md` for the pressure-dependent constraints. Loss weights control relative constraint strength; their effects also depend on sampling frequency.
