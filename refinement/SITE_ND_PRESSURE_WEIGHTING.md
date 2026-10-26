# Pressure weighting of site Nd observations

The direct deterministic site Nd mean-squared error supports continuous pressure weighting to increase the relative influence of valid upper-level observations.

For pressure `p`, transition bounds `p_start` and `p_top`, and upper multiplier `w_top`:

```
r = clip((p_start - p) / (p_start - p_top), 0, 1)
w(p) = 1 + (w_top - 1) * r
```

Configuration defaults:

- `SITE_ND_PRESSURE_WEIGHT_START_HPA = 500`
- `SITE_ND_PRESSURE_WEIGHT_TOP_HPA = 350`
- `SITE_ND_PRESSURE_WEIGHT_TOP_FACTOR = 3`

Weights progress from 1 at 500 hPa to 3 at 350 hPa, remaining constant beyond the transition. Each site's weighted MSE is divided by its sum of valid pressure weights before sites are averaged, preserving equal site contributions. This adjustment applies to the deterministic site Nd term (`lambda_nd`), not the site Gaussian likelihood. Set `SITE_ND_PRESSURE_WEIGHT_TOP_FACTOR = 1` to disable weighting.
