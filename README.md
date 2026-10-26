# NDnet

NDnet reconstructs global three-dimensional cloud droplet number concentration (`Nd`) from satellite cloud observations using a neural network and a vertical empirical orthogonal function (EOF) representation. GiOcean reanalysis and COSP MODIS-like diagnostics establish the prior relationship; ground-based sites and aircraft campaigns constrain probabilistic refinement.

## Method

1. **Vertical representation:** `prior/Compute_EOF.py` derives an EOF basis from GiOcean `NDinc_cf01p`. The reconstructed profile is `Nd(p) = Nd_mean(p) + sum_m A_m EOF_m(p)`.
2. **Prior model:** `prior/NDnetEOF.py` trains a U-Net to predict EOF coefficients from COSP liquid cloud optical thickness (`MDSOPTHCKWTR`), particle size (`MDSCLDSZWTR`), retrieval fraction (`MDSCLDFRCWTR`), and cloud-top pressure (`SWCLDPRS`).
3. **MODIS inference:** Observed `Cloud_Optical_Thickness_Liquid_Mean`, `Cloud_Particle_Size_Liquid_Mean`, `Cloud_Retrieval_Fraction_Liquid_Mean`, and `Cloud_Top_Pressure_Mean` provide satellite inputs. COSP and MODIS use source-specific input normalization.
4. **Observational refinement:** `refinement/NDref_prob.py` trains corrections to the prior EOF coefficients and predicts coefficient uncertainty. The objective combines fixed-site and daily aircraft-campaign constraints with prior regularization and Gaussian likelihoods. See `refinement/README.md`.
5. **Production inference:** `inference/NDref_inf.py` writes daily NetCDF reconstructions, including `NDinc_cf01p`, `sigmaNd`, `a_resid_pred`, and `a_abs_pred`. Subsurface levels are masked using surface pressure.

The reconstruction describes grid-scale cloudy-air structure, not an instantaneous profile of an individual cloud. Predicted uncertainty represents uncertainty conditional on the learned reconstruction and is distinct from observational measurement error.

## Code layout

- `prior/`: EOF computation, prior-network training, and evaluation.
- `refinement/`: probabilistic correction, observation handling, loss functions, and regularization.
- `inference/`: daily global inference and NetCDF output.
- `plots/`: maps, zonal statistics, validation, and uncertainty diagnostics.

## Execution

Configure paths, model weights, and parameters in the respective scripts and configuration files. A typical execution sequence is:

```bash
cd prior
python Compute_EOF.py
python NDnetEOF.py
cd ../refinement
python NDref_prob.py
cd ../inference
python NDref_inf.py
```

The scripts require the GiOcean/COSP, MODIS, site, and campaign datasets and trained model weights. Paths are environment-specific; the complete datasets and production weights are not included.

Main dependencies include PyTorch, NumPy, pandas, xarray, Dask, scikit-learn, xESMF, Matplotlib, Cartopy, Numba, and NetCDF libraries.
