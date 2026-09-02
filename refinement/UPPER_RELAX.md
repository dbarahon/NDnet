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
- site and campaign diagnostics remain independent of training;
- the exact prior V2 implementation is preserved in `NN_construct_v2.py`.

## Main controls

In `refinement/config.py`:

- `TRAIN_WITH_CAMPAIGNS = True`: include campaign tensors in training. Set
  `False` for a fixed-sites-only fit; `do_campaigns=True` can still run the
  independent campaign diagnostic after training.
- `TRAIN_INIT_MODE = "weights"`: start epoch 0 from
  `TRAIN_INIT_WEIGHTS` and ignore an existing current-model checkpoint.
- `TRAIN_INIT_MODE = "scratch"`: random initialization, ignoring checkpoints
  and initialization weights.
- `TRAIN_INIT_MODE = "resume"`: resume `MODEL_NAME.ckpt` when it exists;
  otherwise start from the model's random initialization.
- `TRAIN_INIT_WEIGHTS = "./NDnet_EOFcoeff_sigmaFT_weights.pt"`: V2 weights
  used by `TRAIN_INIT_MODE = "weights"`.

Relaxation controls:

- `UPPER_RELAX_START_HPA = 600.0`
- `UPPER_RELAX_TOP_HPA = 350.0`
- `UPPER_RELAX_LAMBDA = 0.1`
- `UPPER_RELAX_ND_SCALE = 100.0`

The training log reports the weighted relaxation contribution as `upper` at
step level and `UpperRelax` as its epoch mean. `L_upper_relax` and
`L_upper_relax_raw` are also written to the structured step logs.
