# Pressure weighting of fixed-site Nd observations

The scale-separated upper relaxation constrains the longitude-mean upper-level correction while allowing local departures. A small number of fixed sites nevertheless provide direct Nd observations at 400--350 hPa, where the upper constraint is strongest and most sites provide little or no support.

To retain leverage from those real upper-level observations without introducing a discrete pressure boundary, the direct fixed-site Nd MSE uses a continuous pressure multiplier.

For pressure `p`, lower transition pressure `p_start`, upper transition pressure `p_top`, and maximum multiplier `w_top`,

`r = clip((p_start - p) / (p_start - p_top), 0, 1)`

and

`w(p) = 1 + (w_top - 1) r`.

The defaults are:

- `SITE_ND_PRESSURE_WEIGHT_START_HPA = 500`
- `SITE_ND_PRESSURE_WEIGHT_TOP_HPA = 350`
- `SITE_ND_PRESSURE_WEIGHT_TOP_FACTOR = 3`

On the native EOF levels this gives weights of 1.00 at 500 hPa, 1.67 at 450 hPa, 2.33 at 400 hPa, and 3.00 at 350 hPa. The weight remains 1 below the transition and 3 above it.

The weighting is applied only to the direct deterministic site Nd MSE (`lambda_nd`). The site Gaussian Nd likelihood is intentionally unchanged so this remains a targeted adjustment to the deterministic site constraint.

Crucially, the weighted MSE is renormalized separately within each site by the sum of its valid pressure weights before the equal-site average is taken. Thus a high-reaching site does not receive a larger total site weight merely because some of its levels have multipliers greater than one. The pressure weighting only redistributes that site's influence toward its upper-level observations.

Setting `SITE_ND_PRESSURE_WEIGHT_TOP_FACTOR = 1` exactly disables this pressure reweighting.
