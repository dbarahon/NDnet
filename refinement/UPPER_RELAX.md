# Learned upper-level relaxation

This branch keeps the V2 refinement network/loss behavior and the refactored
site/campaign diagnostics, but replaces the former post-inference pressure
taper with a training regularizer.

The added term operates on the **raw refinement departure from the V2 base**:

`dNd(p) = sum_m [dA_m - mu0_m] EOF_m(p)`.

A quintic smootherstep defines a relaxation weight that is zero at and below
600 hPa and rises smoothly to one by 350 hPa. The model is penalized for raw
Nd departures from the base according to this weight. Predictions themselves
are never tapered.

Consequences:

- production inference is the original V2-style direct EOF reconstruction;
- site and campaign losses remain raw and retain valid high-altitude data;
- the journal-style `plots/` directory is unchanged;
- `refinement/plotting.py` is unchanged;
- the independent site/campaign diagnostic/cache refactor is retained;
- the exact prior V2 implementation is preserved in `NN_construct_v2.py`;
- the exact refactored driver is preserved in `NDref_prob_refactor.py`.

## Main knobs

In `refinement/config.py`:

- `UPPER_RELAX_START_HPA = 600.0`
- `UPPER_RELAX_TOP_HPA = 350.0`
- `UPPER_RELAX_LAMBDA = 0.1`
- `UPPER_RELAX_ND_SCALE = 100.0`

The first run is initialized from `./NDnet_EOFcoeff_sigmaFT_weights.pt` unless
an `NDnet_EOFcoeff_upperRelax.ckpt` checkpoint already exists. Change
`hp["init_from_weights"]` if the preferred kinked V2 weights use another name.
