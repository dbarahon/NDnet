# NDnet probabilistic refinement

The refinement network corrects MODIS-driven prior EOF coefficients using fixed-site and aircraft-campaign cloud droplet concentration (`Nd`) observations. It predicts corrected coefficients and coefficient uncertainty for three-dimensional reconstruction.

## Input processing

GiOcean/COSP prior training uses COSP-specific input statistics. MODIS-driven inference and refinement use MODIS-specific standardization:

| MODIS input | Mean | Standard deviation |
| --- | ---: | ---: |
| Liquid optical thickness | 13.945 | 9.665 |
| Liquid particle size | 12.08 | 3.126 |
| Liquid retrieval fraction | 0.268 | 0.205 |
| Cloud-top pressure | 551.241 | 177.277 |

`plots/MODIS_input_comparison.py` compares the source distributions and normalization diagnostics.

## Training objective

The objective combines deterministic Nd errors, Gaussian likelihood terms, a prior anchor, and upper-level regularization. Fixed physical scales make the objective terms dimensionless. The upper-level constraint penalizes longitude-mean departures more strongly than longitude anomalies, preserving local observational influence. Direct site Nd MSE can apply continuous pressure weighting. Production inference uses network output without a post-inference pressure taper.

## Modules

- `NDref_prob.py`: training entry point.
- `config.py`: datasets, optimization, and loss settings.
- `dimensionless_training.py`: model initialization and training stages.
- `NN_construct_v2.py`: refined mean and uncertainty network.
- `loss_normalization.py`: fixed loss scales and prior metric.
- `NN_relax.py`: upper-level regularization.
- `pressure_weighting.py`: site pressure weighting.
- `campaign.py` and `campaign_prepr.py`: campaign processing.
- `refinement_driver.py`: shared model loading, diagnostics, and inference.

For equations and configuration details, see `LOSS_NORMALIZATION.md`, `UPPER_RELAX.md`, and `SITE_ND_PRESSURE_WEIGHTING.md`. Loss scales are saved to `<MODEL_NAME>_loss_scales.json`.
