"""Deterministic observation-level holdout, independent of the day-based validation split.

An entire site-day profile is held out together; campaign observations are
held out by flight/day within each campaign to avoid segment-level leakage.
Indices are derived from immutable observation identifiers, not row order.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _score(*items, seed):
    message = json.dumps([seed, *[str(i) for i in items]], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "big")


def _select(keys, fraction, seed, group):
    unique = sorted(set(keys))
    if not 0 <= fraction < 1:
        raise ValueError("holdout fraction must be in [0,1)")
    if len(unique) < 2 or fraction == 0:
        return set()
    count = max(1, min(len(unique) - 1, round(len(unique) * fraction)))
    return set(sorted(unique, key=lambda k: (_score(group, k, seed=seed), k))[:count])


def site_holdout_mask(nd_sites, fraction=0.2, seed=2026):
    """Boolean [time,site] mask for complete observed daily site profiles."""
    valid = np.isfinite(np.asarray(nd_sites.transpose("time", "lev", "site").values)).any(axis=1)
    times = np.asarray(nd_sites.time.values).astype("datetime64[ns]").astype(str)
    sites = [str(x) for x in nd_sites.site.values]
    mask = np.zeros_like(valid, dtype=bool)
    for j, site in enumerate(sites):
        observed = np.flatnonzero(valid[:, j])
        keys = [times[i] for i in observed]
        chosen = _select(keys, fraction, seed, "site:" + site)
        mask[observed, j] = [times[i] in chosen for i in observed]
    return mask


def campaign_holdout_mask(ds, fraction=0.2, seed=2026):
    """Hold out complete campaign flight/day groups; never split flight segments."""
    camps = np.asarray(ds["campaign"].values).astype(str)
    days = np.asarray(ds["day"].values).astype("datetime64[ns]").astype(str)
    flights = np.asarray(ds["flightID"].values).astype(str) if "flightID" in ds else days
    mask = np.zeros(len(camps), dtype=bool)
    for camp in sorted(set(camps)):
        rows = np.flatnonzero(camps == camp)
        # Flight ID may repeat across different dates; combine it with day.
        keys = [days[i] + "|" + flights[i] for i in rows]
        chosen = _select(keys, fraction, seed, "campaign:" + camp)
        mask[rows] = [key in chosen for key in keys]
    return mask


def save_manifest(site_da, site_mask, campaign_ds, campaign_mask, path, seed, fraction):
    """Record reproducible held-out identifiers for publication audit."""
    times = np.asarray(site_da.time.values).astype("datetime64[ns]").astype(str)
    sites = [str(x) for x in site_da.site.values]
    t, s = np.where(site_mask)
    data = dict(schema="NDnet observation holdout v1", seed=int(seed),
                fraction=float(fraction), site_profiles=[
                    {"site": sites[j], "day": times[i]} for i, j in zip(t.tolist(), s.tolist())
                ], campaign_segments=[])
    if campaign_ds is not None and campaign_mask is not None:
        ds = campaign_ds.isel(segment=np.flatnonzero(campaign_mask))
        for i in range(ds.sizes["segment"]):
            data["campaign_segments"].append(dict(
                campaign=str(ds["campaign"].values[i]),
                day=str(np.asarray(ds["day"].values[i]).astype("datetime64[D]")),
                segment_gid=int(ds["segment_gid"].values[i])))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return str(path)
