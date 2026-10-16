"""Regression tests for independent observation holdout."""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from holdout import site_holdout_mask, campaign_holdout_mask


def test_site_profiles_are_reproducible_and_not_split_by_level():
    t = np.arange(20).astype("timedelta64[D]") + np.datetime64("2020-01-01")
    obs = xr.DataArray(
        np.ones((20, 3, 2)),
        dims=("time", "lev", "site"),
        coords={"time": t, "lev": [900, 800, 700], "site": ["A", "B"]},
    )
    mask = site_holdout_mask(obs, 0.2, 7)
    assert mask.shape == (20, 2)
    np.testing.assert_array_equal(mask, site_holdout_mask(obs, 0.2, 7))
    assert mask.sum(axis=0).tolist() == [4, 4]


def test_campaign_group_does_not_split_segments():
    t = np.datetime64("2020-01-01") + np.repeat(np.arange(10), 3).astype("timedelta64[D]")
    ds = xr.Dataset(
        {"flightID": ("segment", np.repeat(np.arange(10), 3)),
         "day": ("segment", t)},
        coords={"campaign": ("segment", np.repeat("C", len(t)))},
    )
    mask = campaign_holdout_mask(ds, 0.2, 42)
    assert mask.sum() == 6
    assert all(np.unique(mask[3*i:3*i+3]).size == 1 for i in range(10))
    np.testing.assert_array_equal(mask, campaign_holdout_mask(ds, 0.2, 42))


def test_unequal_flights_target_segments_not_number_of_flights():
    # One 23-segment flight and nine smaller flights: 20% of 52 = 10.4.
    # Test set should use combinations of smaller flights, not a 23-segment flight.
    sizes = [23, 5, 4, 4, 4, 4, 3, 2, 2, 1]
    assert sum(sizes) == 52
    gids = np.repeat(np.arange(len(sizes)), sizes)
    ds = xr.Dataset(
        {"flightID": ("segment", gids),
         "day": ("segment", np.datetime64("2020-01-01")
                 + gids.astype("timedelta64[D]"))},
        coords={"campaign": ("segment", np.repeat("ACCACIA", len(gids)))},
    )
    mask = campaign_holdout_mask(ds, 0.2, 2026)
    assert mask.sum() == 10  # closest feasible integer to 10.4
    for g in np.unique(gids):
        assert np.unique(mask[gids == g]).size == 1


def test_one_group_cannot_be_split():
    ds = xr.Dataset(
        {"flightID": ("segment", np.repeat(7, 54)),
         "day": ("segment", np.repeat(np.datetime64("2020-01-01"), 54))},
        coords={"campaign": ("segment", np.repeat("COPE", 54))},
    )
    assert not campaign_holdout_mask(ds, 0.2, 2026).any()
