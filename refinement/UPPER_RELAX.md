# Scale-separated upper-level relaxation

Upper-level relaxation constrains departures of refined EOF coefficients from the MODIS-driven prior during training. Inference reconstructs the raw network prediction without a pressure taper.

Let `c = dA - mu0` denote the corrected-minus-prior residual coefficients. The corresponding concentration correction is `dNd(p) = sum_m c_m EOF_m(p)`.

## Pressure and spatial structure

The pressure ramp is zero below the upper transition (at pressures greater than or equal to `UPPER_RELAX_START_HPA`), increases linearly in log-pressure, and reaches one at pressures less than or equal to `UPPER_RELAX_TOP_HPA`.

Corrections are decomposed at each time, EOF mode, and latitude:

```
c_zonal = mean_lon(c)
c_anomaly = c - c_zonal
L_upper = lambda_zonal * L_upper_zonal
        + lambda_anomaly * L_upper_anomaly
```

The zonal term constrains broad latitude-band changes; the anomaly term constrains smaller-scale departures. Both use a pressure-weighted EOF Gram matrix and the fixed `LOSS_ND_SCALE`, avoiding explicit full-level reconstruction inside the loss.

## Configuration

- `UPPER_RELAX_START_HPA = 700.0`
- `UPPER_RELAX_TOP_HPA = 500.0`
- `UPPER_RELAX_ZONAL_LAMBDA = 10.0`
- `UPPER_RELAX_ANOMALY_LAMBDA = 0.1`
- `LOSS_ND_SCALE = 100.0` cm-3

Diagnostic components include `L_upper_relax`, `L_upper_zonal`, `L_upper_zonal_raw`, `L_upper_anomaly`, and `L_upper_anomaly_raw`. See `LOSS_NORMALIZATION.md` for normalization definitions.
