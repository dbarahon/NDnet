# Learned upper-level relaxation

The refinement uses a training-time upper-level regularizer rather than a
post-inference taper. Production inference and diagnostics use the raw network
output.

For correction coefficients

`c = dA - mu0`,

where `dA` is the refined residual coefficient field and `mu0` is the base
residual coefficient field, the Nd correction is

`dNd(p) = sum_m c_m EOF_m(p)`.

The relaxation weight is zero at and below `UPPER_RELAX_START_HPA`, increases
linearly in log-pressure through the transition, and reaches one at and above
`UPPER_RELAX_TOP_HPA`. The default transition on this branch is 700 to 500 hPa,
consistent with the observed loss of direct site/campaign pressure support
aloft.

The regularizer is evaluated as a log-pressure-weighted vertical mean of

`(dNd / LOSS_ND_SCALE)^2`

after applying the pressure ramp. `LOSS_ND_SCALE` is the same fixed Nd scale
used by the site/campaign MSE terms and the whole-column prior anchor. This is
intentional: `UPPER_RELAX_LAMBDA` is now directly comparable to the other
Nd-space tuning weights instead of compensating for a separate numerical
scale.

The vertical quadratic is precomputed as a mode-space Gram matrix, including
cross-mode terms, so the regularizer is equivalent to reconstructing the Nd
correction without allocating a full four-dimensional level field during
training.

## Main controls

In `refinement/config.py`:

- `UPPER_RELAX_START_HPA = 700.0`
- `UPPER_RELAX_TOP_HPA = 500.0`
- `UPPER_RELAX_LAMBDA = 1.0`
- `LOSS_ND_SCALE = 100.0` cm-3

The training log reports the weighted contribution as `upper` at step level
and `UpperRelax` as its epoch mean. `L_upper_relax` and
`L_upper_relax_raw` are also written to the structured step logs.

See `LOSS_NORMALIZATION.md` for the normalization of the remaining loss terms.
