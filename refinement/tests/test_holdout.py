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
