# Scale-separated upper-level relaxation

The refinement uses a training-time upper-level regularizer rather than a
post-inference taper. Production inference and diagnostics always use the raw
network output.

For correction coefficients

`c = dA - mu0`,

where `dA` is the refined residual coefficient field and `mu0` is the base
residual coefficient field, the Nd correction is

`dNd(p) = sum_m c_m EOF_m(p)`.

The pressure weight is zero at and below `UPPER_RELAX_START_HPA`, increases
linearly in log-pressure through the transition, and reaches one at and above
`UPPER_RELAX_TOP_HPA`. The default transition is 700 to 500 hPa, consistent
with the observed loss of direct site/campaign pressure support aloft.

## Why the correction is separated spatially

A pointwise upper penalty suppresses latitude-band artifacts, but it also
penalizes legitimate local corrections at the small number of sites and
campaign segments that directly sample 400--350 hPa. The scale-separated
regularizer instead decomposes the correction into a longitude mean and a
longitude anomaly at each time, mode, and latitude:

`c_zonal(t,m,lat) = mean_lon c(t,m,lat,lon)`

`c_anom(t,m,lat,lon) = c - c_zonal`.

The zonal-mean component represents the large-scale latitude-band correction
that produced the diagnosed banding. It is regularized strongly toward the
base solution. The anomaly component represents local departures and is
regularized weakly, allowing observations with genuine high-altitude support
to modify the local profile without imposing the same correction around an
entire latitude circle.

Both components use the same upper-level EOF Gram matrix and the same fixed
`LOSS_ND_SCALE` as the site/campaign Nd MSE and whole-column anchor. Thus the
raw terms are dimensionless and directly comparable in physical scale.

The objective contribution is

`L_upper = lambda_zonal * L_upper_zonal + lambda_anomaly * L_upper_anomaly`.

The defaults on this branch are:

- `UPPER_RELAX_START_HPA = 700.0`
- `UPPER_RELAX_TOP_HPA = 500.0`
- `UPPER_RELAX_ZONAL_LAMBDA = 10.0`
- `UPPER_RELAX_ANOMALY_LAMBDA = 0.1`
- `LOSS_ND_SCALE = 100.0` cm-3

The choice `lambda_zonal=10` follows the short-run experiment in which the
pointwise upper constraint at comparable strength removed the global bands
while retaining strong campaign skill. The anomaly weight is deliberately two
orders of magnitude weaker so that local high-reaching observations remain
able to move the solution. It is a tuning parameter and can be set to zero for
a pure large-scale constraint.

The vertical quadratic is precomputed in mode space, including cross-mode
terms, so no full pressure reconstruction is required inside the loss.
Longitude averaging is performed on the coefficient correction before applying
the Gram matrix, which is exactly equivalent because EOF reconstruction is
linear in the coefficients.

For backward compatibility with the training logger, `L_upper_relax` remains
the total weighted upper contribution. The loss also exposes
`L_upper_zonal`, `L_upper_zonal_raw`, `L_upper_anomaly`, and
`L_upper_anomaly_raw` in `return_parts=True` diagnostics.

See `LOSS_NORMALIZATION.md` for the normalization of the remaining loss terms.
