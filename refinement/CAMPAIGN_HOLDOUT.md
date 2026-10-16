# Independent observational holdout: campaign flight selection

## Purpose

The reviewer-facing evaluation withholds 20% of **observed profiles** at each
fixed site and targets 20% of **usable flight-campaign segments** for each
aircraft campaign. Site-day profiles are indivisible. Campaign flight/day groups
are indivisible, so adjacent segments of the same flight cannot appear in both
training and testing. The original day-based 90/10 training/validation partition,
network, optimizer, losses, pressure weighting, and global diagnostics are not
changed by this selection strategy.

## Eligibility and partition unit

Campaign files are first filtered by the existing loader for the requested
model time range and valid observations (see `load_campaign_training_files`).
Only the surviving segments enter the partition. The `raw` count in an upstream
unfiltered dataset can therefore be larger than the eligible count.

Within each campaign, a group identifier combines the **UTC calendar day** of
`day` and the `flightID`. This disambiguates reused flight identifiers across
dates. If `flightID` does not exist, the grouping defaults to day alone.
A group contains every eligible segment carrying the same identifier.

## Exact segment-count selection algorithm

For a campaign containing `N` eligible segments and a requested test fraction
`f = 0.20`, the numerical target is `T = f*N`.

1. Build the list of unique flight/day groups and count the segments in each.
2. Sort groups by a deterministic SHA-256 score of the seed (`2026` by
   default), campaign name, and flight/day identifier. This provides stable
   tie-breaking regardless of input file order.
3. Use an exact subset-sum dynamic program. At each group, update the set of
   achievable test-segment totals. Retain the first subset encountered for each
   total in the seeded ordering. This explores all achievable **counts** without
   enumerating every combination of flights.
4. Exclude the empty subset and the subset containing all eligible segments,
   ensuring both training and testing have observations whenever at least two
   distinct groups are available.
5. Choose the achievable total minimizing `abs(n_test - T)`. If totals are
   equally close to the target, prefer the larger test count; if multiple
   flight combinations reach the chosen total, use the seeded subset.
6. Mark every segment in each selected group as test; all other segments remain
   available for the original training/validation day split.

The result is reproducible for fixed eligible observations, campaign names,
group identifiers, fraction, and seed. The algorithm is exact with respect to
segment **count**, not a random 20% draw of individual segments; every distinct
achievable test-segment count is considered.

### Unequal flights and infeasible targets

If one flight contains 23 of 52 segments, holding out that flight would yield
44.2%, not 20%. The updated selector prefers an available combination of smaller
flights that gets closer to `0.2*52 = 10.4` segments.

An exact 20% split **cannot be guaranteed** if intact flights have unfavorable
sizes. In particular, when a campaign contains only one eligible flight/day
group, no independent flight-level split is possible: the selector holds out
**zero** segments rather than leak that flight across partitions. Such campaigns
must be excluded from holdout metrics or evaluated under a separately identified
protocol. With two or more groups, the closest nonempty, nonfull subset may still
be substantially different from 20%.

## Diagnostics and audit trail

`NDnet_heldout_observations.json` (schema v2) stores the requested fraction
and seed, site-day test identifiers, individual held-out campaign segments, and
a `campaign_summary` with per-campaign:

- `n_usable_segments`: eligible segments after loader/QC/time filtering;
- `target_test_segments`: floating-point `0.2*N` target;
- `n_test_segments` and `achieved_fraction`: selected held-out data;
- `n_flight_groups`: eligible distinct flight/day groups;
- `selected_flight_groups`: selected group identifiers.

This JSON manifest is generated while loading training observations. Retain it
with the checkpoint and diagnostic artifacts to document the exact partition.

The campaign diagnostic dataset is restricted to held-out segments before
plotting. Its scatter-panel `n` can be **smaller** than the selected count
because plotting additionally requires finite observation/model pairs,
positive values with logarithmic axes, a minimum Nd cutoff, and optionally a
MODIS mask. Thus always distinguish:

`N_raw >= N_eligible >= N_holdout >= N_plotted`.

There are no guarantees of statistical power for a campaign with very few
held-out segments. Report the actual `n` alongside the comparison statistics.

## Reproduction and limitations

Run `python -m pytest refinement/tests/test_holdout.py -q` after checking out
`reviewer-20pct-holdout`. The filename and fraction of an existing manifest
must not be mistaken for a valid partition if the eligible campaign inputs
changed; regenerate the manifest using the same run's data.

Training starts from the configured weights. For strict independence from the
test observations, the initial weights must not already have been refined using
those same observations. Holding out segments only during the current refinement
does not retroactively remove information learned by previously refined weights.
