# NDnet

NDnet is a neural-network framework for reconstructing the global, vertically resolved cloud droplet number concentration, `Nd`, from satellite-observed cloud properties. The method combines information from the GiOcean reanalysis, COSP MODIS-like diagnostics, actual MODIS retrievals, fixed-site observations, and aircraft field campaigns.

The central idea is to predict a low-dimensional set of vertical Empirical Orthogonal Function (EOF) coefficients instead of predicting a full 3-D `Nd` volume directly. The EOF basis provides a physically informed representation learned from GiOcean, while the neural networks determine how the coefficients vary horizontally and in time.

The analysis associated with this repository produces daily global `Nd` fields over the MODIS record (2003–2024). The manuscript configuration uses a 1° horizontal grid and 27 pressure levels extending to 250 hPa.

## Scientific workflow

The workflow has three main stages.

### 1. EOF representation and NDnet-prior

`prior/Compute_EOF.py` derives a vertical EOF basis from the GiOcean `NDinc_cf01p` field. The current configuration retains 12 modes and reconstructs `Nd` as

```
Nd(p) = Nd_mean(p) + sum_m A_m EOF_m(p)
```

where the coefficients `A_m` vary with time, latitude, and longitude.

`prior/NDnetEOF.py` trains a U-Net to predict residual EOF coefficients from four COSP MODIS-like cloud fields:

| Physical quantity | GiOcean/COSP variable |
| --- | --- |
| liquid cloud optical thickness | `MDSOPTHCKWTR` |
| liquid cloud particle size | `MDSCLDSZWTR` |
| liquid cloud retrieval fraction | `MDSCLDFRCWTR` |
| cloud-top pressure | `SWCLDPRS` |

The resulting model is referred to as **NDnet-prior**. Its role is to emulate the three-dimensional GiOcean `Nd` distribution using only information available from MODIS-like two-dimensional cloud fields. Because the network predicts EOF coefficients rather than individual pressure levels, the vertical reconstruction remains tied to the physically based EOF representation.

### 2. MODIS correction and observational refinement

The trained prior can be driven with actual MODIS retrievals instead of the simulated COSP fields. In the manuscript this is referred to as the **MODIS-corrected prior**. The corresponding MODIS inputs are:

| Physical quantity | MODIS variable |
| --- | --- |
| liquid cloud optical thickness | `Cloud_Optical_Thickness_Liquid_Mean` |
| liquid cloud particle size | `Cloud_Particle_Size_Liquid_Mean` |
| liquid retrieval fraction | `Cloud_Retrieval_Fraction_Liquid_Mean` |
| cloud-top pressure | `Cloud_Top_Pressure_Mean` |

The COSP and MODIS inputs use source-specific standardization. This is an explicit part of the COSP-to-MODIS domain transfer; see `refinement/README.md` and `plots/MODIS_input_comparison.py`.

The MODIS-corrected coefficients are then refined against direct observations in `refinement/`. The current refinement model takes both the MODIS fields and the prior EOF coefficients and predicts:

- a correction to the mean EOF coefficients; and
- a positive coefficient uncertainty, `sigma_A`.

Fixed-site observations provide vertically resolved `Nd` constraints. Aircraft observations are reduced to daily campaign constraints before training. Campaign losses use equal-campaign weighting and equal-flight weighting within each campaign so campaigns with more samples do not dominate the refinement.

The refinement is global: observations constrain the learned relationship between MODIS cloud properties and EOF coefficients rather than being applied as local corrections only at observation locations.

### 3. Probabilistic reconstruction and production inference

The refinement objective combines deterministic and Gaussian terms. Experimental uncertainty is supplied separately in the observational likelihood, while the network predicts the uncertainty of the reconstructed `Nd` conditional on the satellite inputs and learned relationships. The predicted `sigmaNd` therefore should not be interpreted as measurement error alone; it can also reflect unresolved variability, representativeness differences, and limitations of the learned reconstruction.

The refinement objective is normalized using fixed physical scales so the `lambda_*` coefficients are scientific tuning parameters rather than unit-conversion factors. Details are documented in:

- `refinement/LOSS_NORMALIZATION.md`
- `refinement/UPPER_RELAX.md`
- `refinement/SITE_ND_PRESSURE_WEIGHTING.md`

The current workflow also uses a training-time upper-level constraint that limits poorly supported broad-scale corrections aloft while retaining local leverage from observations that genuinely reach high altitude. Production inference uses the raw trained network output; no post-inference pressure taper is applied.

`inference/NDref_inf.py` performs daily production inference. The current configuration spans `2003-01-01` through `2024-12-31` and writes one NetCDF file per day. The main output fields are:

| Variable | Description |
| --- | --- |
| `NDinc_cf01p` | reconstructed cloud droplet number concentration |
| `sigmaNd` | predicted reconstruction uncertainty |
| `a_resid_pred` | predicted residual EOF coefficients |
| `a_abs_pred` | absolute EOF coefficients used in reconstruction |

Values below the local surface are masked using daily surface pressure.

## Repository structure

```
NDnet/
├── prior/
│   ├── Compute_EOF.py       # build the vertical EOF basis
│   ├── NDnetEOF.py          # train/evaluate NDnet-prior
│   └── Nd_EOF.nc            # EOF representation used by the prior
│
├── refinement/
│   ├── NDref_prob.py        # main probabilistic-refinement entry point
│   ├── config.py            # training/data/loss configuration
│   ├── NN_construct_v2.py   # refined mean and uncertainty network
│   ├── refinement_driver.py # shared loading, diagnostics and inference
│   ├── campaign*.py         # campaign preprocessing and handling
│   ├── mode_ablation.py     # EOF-mode sensitivity diagnostics
│   └── *.md                 # detailed loss/regularization documentation
│
├── inference/
│   ├── NDref_inf.py         # daily production inference
│   ├── config.py            # inference dates, paths and model selection
│   └── *.py                 # model/data/inference utilities
│
└── plots/
    ├── NDprioreval.py              # prior evaluation
    ├── MODIS_input_comparison.py   # COSP/MODIS input comparison
    ├── NDcomparison.py             # global Nd comparison
    ├── NDdifferences.py            # differences among reconstruction stages
    ├── sigma.py                    # uncertainty diagnostics
    ├── Homogen.py                  # vertical-homogeneity diagnostics
    ├── Homogen_cloudthickness.py   # cloud-boundary sensitivity
    ├── sites_eval.py               # fixed-site evaluation
    ├── campaing_eval.py            # field-campaign evaluation
    └── plot_GR18_comparison.py     # comparison with 2-D MODIS Nd retrievals
```

## Running the workflow

This repository is research code rather than an installed Python package. The scripts currently assume execution from their respective directories and contain paths configured for the original HPC environment.

A typical sequence is:

```bash
# 1. Build the EOF representation
cd prior
python Compute_EOF.py

# 2. Train/evaluate the GiOcean/COSP prior
python NDnetEOF.py

# 3. Configure and run observational refinement
cd ../refinement
# edit config.py for local data paths, model initialization and run controls
python NDref_prob.py

# 4. Configure and run production inference
cd ../inference
# edit config.py for the selected weights, dates, input paths and output directory
python NDref_inf.py
```

Before running on another system, update the absolute paths in the configuration/scripts and provide the required GiOcean, MODIS, fixed-site, and campaign data. Trained production weights and the full input datasets are not stored in this repository. The model name in `inference/config.py` should also be set to the specific refined weights intended for production.

## Main Python dependencies

The workflow uses PyTorch together with the scientific Python/xarray ecosystem. Core dependencies include `numpy`, `pandas`, `xarray`, `dask`, `scikit-learn`, `PyTorch`, `xESMF`, `Cartopy`, `Matplotlib`, `Numba`, and NetCDF support. Some diagnostics use additional plotting/model-summary utilities.

Because the current repository does not yet provide a locked environment file, package versions should be chosen consistently with the system on which the model weights are trained and evaluated.

## Interpretation

NDnet should be viewed as a data-constrained three-dimensional reconstruction rather than a direct vertically resolved satellite retrieval. MODIS provides the observed cloud-property input, GiOcean supplies the initial physically based EOF representation and prior mapping, and ground-based and aircraft observations refine that mapping. The predicted uncertainty quantifies uncertainty in the reconstruction conditional on this framework and is distinct from the experimental uncertainty supplied with the training observations.

At model-grid scale, reconstructed vertical structure represents the ensemble of cloudy subcolumns sampled within a grid cell and should not be interpreted as the instantaneous vertical profile of a single cloud.

## Status

The repository contains the analysis and diagnostic code used in the development of NDnet and is under active research development. Configuration values, model names, and diagnostic scripts may reflect specific experiments used in the associated analysis.
