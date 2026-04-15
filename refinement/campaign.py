# ============================================================
# campaign_module.py
# ============================================================
from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


# ============================================================
# Small utilities
# ============================================================

def normalize_time_coord(ds):
    if "time" not in ds.coords:
        return ds
    return ds.sortby("time")


def _print_da_info(name, obj):
    if isinstance(obj, xr.Dataset):
        print(f"[DBG] {name}: Dataset vars={list(obj.data_vars)} dims={dict(obj.sizes)}")
        if "time" in obj.coords:
            nt = obj.sizes.get("time", None)
            t0 = obj["time"].isel(time=0).item() if nt and nt > 0 else None
            t1 = obj["time"].isel(time=-1).item() if nt and nt > 0 else None
            print(f"      time: n={nt}, first={t0}, last={t1}")
        return

    if isinstance(obj, xr.DataArray):
        print(f"[DBG] {name}: DataArray dims={obj.dims} shape={obj.shape} dtype={obj.dtype}")
        if "time" in obj.coords:
            nt = obj.sizes.get("time", None)
            t0 = obj["time"].isel(time=0).item() if nt and nt > 0 else None
            t1 = obj["time"].isel(time=-1).item() if nt and nt > 0 else None
            print(f"      time: n={nt}, first={t0}, last={t1}")
        return

    print(f"[DBG] {name}: type={type(obj)}")


def campaign_name_from_path(path: Path) -> str:
    stem = path.stem
    suffixes = [
        "_segments_1deg_merraP_training",
        "_segments_1deg_merraP",
        "_training",
    ]
    for s in suffixes:
        if stem.endswith(s):
            stem = stem[:-len(s)]
    return stem


def force_datetime64_ns(vals):
    return np.asarray(vals).astype("datetime64[ns]")


def _interp_profile_logp_scalar(p_target, p_levels, y_levels):
    """
    Linear interpolation in log-pressure space.
    Returns NaN outside the pressure range or if insufficient valid data.
    """
    p_target = float(p_target)
    p_levels = np.asarray(p_levels, dtype=float)
    y_levels = np.asarray(y_levels, dtype=float)

    good = np.isfinite(p_levels) & np.isfinite(y_levels) & (p_levels > 0)
    if good.sum() < 2 or not np.isfinite(p_target) or p_target <= 0:
        return np.nan

    pp = p_levels[good]
    yy = y_levels[good]

    pmin = np.min(pp)
    pmax = np.max(pp)
    if p_target < pmin or p_target > pmax:
        return np.nan

    order = np.argsort(np.log(pp))
    x = np.log(pp[order])
    y = yy[order]

    return float(np.interp(np.log(p_target), x, y))


# ============================================================
# Tensor bundle for campaigns
# ============================================================

@dataclass
class CampaignTensorBundle:
    # segment-wise tensors, sorted by time_idx
    time_idx: torch.Tensor         # [S] long
    ilat: torch.Tensor             # [S] long
    ilon: torch.Tensor             # [S] long
    p_merra: torch.Tensor          # [S] float32
    y_mean: torch.Tensor           # [S] float32
    y_std: torch.Tensor            # [S] float32
    y_aug: torch.Tensor            # [A,S] float32
    valid_obs: torch.Tensor        # [S] bool
    aug_valid: torch.Tensor        # [A,S] bool
    flight_id: torch.Tensor        # [S] long
    campaign_id: torch.Tensor      # [S] long
    segment_gid: torch.Tensor      # [S] long  (global segment index after concat)
    day_ns: torch.Tensor           # [S] int64 datetime64[ns] viewed as int64
    time_offsets: torch.Tensor     # [T] long
    time_counts: torch.Tensor      # [T] long

    # non-tensor metadata
    campaign_names: List[str]
    time_coord_ns: np.ndarray      # [T] datetime64[ns]
    sort_index: np.ndarray         # [S] original concat order -> sorted order
    input_channels: List[str]


# ============================================================
# Loading backbone fields (same pattern as site loader)
# ============================================================

def load_backbone_inputs(
    vars_out=None,
    pth_all="",
    vars_in_2d=None,
    modis_source=None,
    prefix="",
    t_chk=64,
):
    """
    Load:
      Nin_2d    : xr.Dataset   (time, lat, lon)
      Ares_base : xr.DataArray (time, mode, lat, lon)

    This mirrors the existing site-only get_data() backbone loading.
    """
    if vars_out is None:
        vars_out = {}
    if vars_in_2d is None:
        vars_in_2d = {}

    chk2d = {"time": t_chk, "lat": -1, "lon": -1}
    chk3d = {"time": t_chk, "lev": -1, "lat": -1, "lon": -1}

    # Base residual coefficient field
    varname_base = list(vars_out.keys())[1]
    pth_base = f"{vars_out[varname_base]}/{prefix}"
    print("[LOAD 3D OUT base]", pth_base, len(sorted(glob.glob(pth_base))), "files")

    Ab = xr.open_mfdataset(pth_base, parallel=False, chunks=chk3d)[varname_base]
    Ares_base = normalize_time_coord(Ab)
    Ares_base = Ares_base.where(Ares_base > -1e9).where(Ares_base < 1e9)

    # MODIS / 2D inputs
    pth2d = f"{pth_all}dailies/{modis_source}/on_Noutgrid/{modis_source}.{prefix}.nc"
    print("[LOAD 2D IN]", pth2d)

    ds2d = xr.open_mfdataset(pth2d, parallel=False, chunks=chk2d)
    for v in ds2d.data_vars:
        mu = float(vars_in_2d[v]["mean"])
        sd = float(vars_in_2d[v]["std"])
        ds2d[v] = (ds2d[v] - mu) / sd

    Nin_2d = ds2d

    # align exactly on time
    Nin_2d, Ares_base = xr.align(Nin_2d, Ares_base, join="inner")
    Nin_2d = Nin_2d.transpose("time", "lat", "lon")
    Ares_base = Ares_base.transpose("time", "mode", "lat", "lon")

    _print_da_info("Nin_2d", Nin_2d)
    _print_da_info("Ares_base", Ares_base)
    return Nin_2d, Ares_base


# ============================================================
# Load campaign training-ready files
# ============================================================

def load_campaign_training_files(
    campaign_train_dir,
    template_time,
    campaign_pattern="*_training.nc",
    require_valid_obs=True,
):
    """
    Load all preprocessed campaign training files and align each segment day onto
    the template_time axis from the daily model/MODIS fields.

    Returns
    -------
    campaign_ds : xr.Dataset(segment, aug) sorted by time_idx
      Required vars:
        day(segment)
        time_idx(segment)
        ilat(segment), ilon(segment), p_merra(segment)
        Nd_mean(segment), Nd_std(segment)
        Nd_obs_aug(aug, segment)
        valid_obs(segment), aug_valid(aug, segment)
        flightID(segment) if available
        campaign(segment)  [string]
    """
    campaign_train_dir = Path(campaign_train_dir)
    files = sorted(campaign_train_dir.glob(campaign_pattern))
    
    print(files)
    
    if not files:
        raise FileNotFoundError(f"No files matching {campaign_pattern} in {campaign_train_dir}")

    template_time = force_datetime64_ns(template_time)
    tmap = {t: i for i, t in enumerate(template_time)}

    dsets = []
    seg_offset = 0

    for fp in files:
        with xr.open_dataset(fp, decode_times=True) as ds:
            ds = ds.load()

        campaign = ds.attrs.get("campaign_name", campaign_name_from_path(fp))

        needed = ["day", "ilat", "ilon", "p_merra", "Nd_mean", "Nd_std", "Nd_obs_aug", "valid_obs", "aug_valid"]
        for v in needed:
            if v not in ds:
                raise KeyError(f"{fp} missing required variable {v!r}")

        day_vals = force_datetime64_ns(ds["day"].values)
        time_idx = np.array([tmap.get(t, -1) for t in day_vals], dtype=np.int64)

        keep = (time_idx >= 0)
        if require_valid_obs:
            keep &= (np.asarray(ds["valid_obs"].values).astype(np.int64) > 0)

        idx = np.where(keep)[0]
        if idx.size == 0:
            print(f"[WARN] no usable segments after time/QC filtering: {fp.name}")
            continue

        ds = ds.isel(segment=idx).copy()
        time_idx = time_idx[idx]

        # attach time_idx and campaign label
        ds["time_idx"] = xr.DataArray(time_idx.astype(np.int64), dims=("segment",))
        ds["campaign"] = xr.DataArray(np.asarray([campaign] * ds.sizes["segment"], dtype="U"), dims=("segment",))

        # segment_gid after concat
        segment_gid = np.arange(seg_offset, seg_offset + ds.sizes["segment"], dtype=np.int64)
        ds["segment_gid"] = xr.DataArray(segment_gid, dims=("segment",))
        seg_offset += ds.sizes["segment"]

        dsets.append(ds)

    if not dsets:
        raise RuntimeError("No campaign segments remained after loading/filtering.")

    campaign_ds = xr.concat(dsets, dim="segment", combine_attrs="override")

    # stable sort by time_idx to make per-time offsets cheap
    order = np.argsort(campaign_ds["time_idx"].values.astype(np.int64), kind="mergesort")
    campaign_ds = campaign_ds.isel(segment=order)

    _print_da_info("campaign_ds", campaign_ds)
    print(f"[INFO] campaign files loaded: {len(dsets)}")
    print(f"[INFO] campaign segments kept: {campaign_ds.sizes['segment']}")
    return campaign_ds


def get_campaign_data(
    campaign_train_dir,
    vars_out=None,
    pth_all="",
    vars_in_2d=None,
    modis_source=None,
    prefix="",
    t_chk=64,
    campaign_pattern="*_training.nc",
):
    """
    Returns
    -------
    Nin_2d      : xr.Dataset   (time, lat, lon)
    campaign_ds : xr.Dataset   (segment, aug)
    Ares_base   : xr.DataArray (time, mode, lat, lon)
    """
    Nin_2d, Ares_base = load_backbone_inputs(
        vars_out=vars_out,
        pth_all=pth_all,
        vars_in_2d=vars_in_2d,
        modis_source=modis_source,
        prefix=prefix,
        t_chk=t_chk,
    )

    campaign_ds = load_campaign_training_files(
        campaign_train_dir=campaign_train_dir,
        template_time=Ares_base["time"].values,
        campaign_pattern=campaign_pattern,
        require_valid_obs=True,
    )

    return Nin_2d, campaign_ds, Ares_base


# ============================================================
# Tensorization
# ============================================================



def tensorize_campaign_inputs_from_existing(
    campaign_ds,
    template_time,
    pin_memory=True,
    verbose=True,
):
    """
    Convert campaign_ds into tensors indexed by the EXISTING daily time axis.

    Returns
    -------
    bundle : dict of campaign tensors
    """
    template_time = np.asarray(template_time).astype("datetime64[ns]")
    T = template_time.size

    time_idx = torch.as_tensor(campaign_ds["time_idx"].values.astype(np.int64), dtype=torch.long)
    ilat = torch.as_tensor(campaign_ds["ilat"].values.astype(np.int64), dtype=torch.long)
    ilon = torch.as_tensor(campaign_ds["ilon"].values.astype(np.int64), dtype=torch.long)
    p_merra = torch.as_tensor(campaign_ds["p_merra"].values.astype(np.float32), dtype=torch.float32)

    y_mean = torch.as_tensor(campaign_ds["Nd_mean"].values.astype(np.float32), dtype=torch.float32)
    y_std = torch.as_tensor(campaign_ds["Nd_std"].values.astype(np.float32), dtype=torch.float32)

    y_aug = torch.as_tensor(
        campaign_ds["Nd_obs_aug"].transpose("aug", "segment").values.astype(np.float32),
        dtype=torch.float32
    )

    valid_obs = torch.as_tensor(
        campaign_ds["valid_obs"].values.astype(np.int64) > 0,
        dtype=torch.bool
    )

    aug_valid = torch.as_tensor(
        campaign_ds["aug_valid"].transpose("aug", "segment").values.astype(np.int64) > 0,
        dtype=torch.bool
    )

    campaign_names_seg = np.asarray(campaign_ds["campaign"].values).astype("U")
    uniq_campaigns, inv_campaign = np.unique(campaign_names_seg, return_inverse=True)
    campaign_id = torch.as_tensor(inv_campaign.astype(np.int64), dtype=torch.long)

    segment_gid = torch.as_tensor(campaign_ds["segment_gid"].values.astype(np.int64), dtype=torch.long)
    day_ns = torch.as_tensor(
        campaign_ds["day"].values.astype("datetime64[ns]").astype(np.int64),
        dtype=torch.long
    )

    counts_np = np.bincount(time_idx.numpy(), minlength=T).astype(np.int64)
    offsets_np = np.zeros(T, dtype=np.int64)
    if T > 1:
        offsets_np[1:] = np.cumsum(counts_np[:-1])

    if "flightID" in campaign_ds:
        flight_id_np = campaign_ds["flightID"].values.astype(np.int64)
    else:
        flight_id_np = np.zeros((campaign_ds.sizes["segment"],), dtype=np.int64)

    flight_id = torch.as_tensor(flight_id_np, dtype=torch.long)

    bundle = CampaignTensorBundle(
        time_idx=time_idx,
        ilat=ilat,
        ilon=ilon,
        p_merra=p_merra,
        y_mean=y_mean,
        y_std=y_std,
        y_aug=y_aug,
        valid_obs=valid_obs,
        aug_valid=aug_valid,
        flight_id=flight_id,
        campaign_id=campaign_id,
        segment_gid=segment_gid,
        day_ns=day_ns,
        time_offsets=torch.as_tensor(offsets_np, dtype=torch.long),
        time_counts=torch.as_tensor(counts_np, dtype=torch.long),
        campaign_names=list(uniq_campaigns.tolist()),
        time_coord_ns=template_time,
        sort_index=np.arange(time_idx.numel(), dtype=np.int64),
        input_channels=[],
    )

    if pin_memory:
        for f in bundle.__dataclass_fields__:
            val = getattr(bundle, f)
            if isinstance(val, torch.Tensor):
                setattr(bundle, f, val.pin_memory())

    if verbose:
        print("[DBG] campaign bundle segments:", int(bundle.time_idx.numel()))
        print("[DBG] campaign aug shape:", tuple(bundle.y_aug.shape))
        print("[DBG] campaigns:", bundle.campaign_names)

    return bundle


class EOFNdMultiObsDatasetTorch(Dataset):
    """
    Daily dataset combining fixed-site tensors and optional campaign segment tensors.
    """
    def __init__(self, X2d_t, MU0_t, YS_t, VS_t, Nds_t, Eds_t, campaign_bundle=None, time_idx=None):
        self.X2d = X2d_t
        self.MU0 = MU0_t
        self.YS = YS_t
        self.VS = VS_t
        self.Nds = Nds_t
        self.Eds = Eds_t
        self.B = campaign_bundle
        if time_idx is None:
            time_idx = np.arange(self.X2d.shape[0], dtype=np.int64)
        self.time_idx = np.asarray(time_idx, dtype=np.int64)

    def __len__(self):
        return self.time_idx.size

    def __getitem__(self, i):
        t = int(self.time_idx[i])
        out = {
            "x2d": self.X2d[t],
            "mu0": self.MU0[t],
            "yS": self.YS[t],
            "vS": self.VS[t],
            "Nds": self.Nds[t],
            "Eds": self.Eds[t],
            "time_idx": torch.tensor(t, dtype=torch.long),
        }
        if self.B is None:
            return out

        s0 = int(self.B.time_offsets[t].item())
        n = int(self.B.time_counts[t].item())
        s1 = s0 + n
        out.update({
            "seg_ilat": self.B.ilat[s0:s1],
            "seg_ilon": self.B.ilon[s0:s1],
            "seg_p": self.B.p_merra[s0:s1],
            "seg_y_mean": self.B.y_mean[s0:s1],
            "seg_y_std": self.B.y_std[s0:s1],
            "seg_y_aug": self.B.y_aug[:, s0:s1],
            "seg_valid_obs": self.B.valid_obs[s0:s1],
            "seg_aug_valid": self.B.aug_valid[:, s0:s1],
            "seg_flight_id": self.B.flight_id[s0:s1],
            "seg_campaign_id": self.B.campaign_id[s0:s1],
            "seg_gid": self.B.segment_gid[s0:s1],
            "seg_day_ns": self.B.day_ns[s0:s1],
        })
        return out


def make_equal_site_time_sampler(VS_t, time_idx, min_sites=1, beta=1.0, eps=1.0):
    """
    Build a simple inverse-frequency sampler over days based on count of valid sites.
    """
    time_idx = np.asarray(time_idx, dtype=np.int64)
    if time_idx.size == 0:
        return time_idx, None

    if isinstance(VS_t, torch.Tensor):
        valid_counts = VS_t[time_idx].to(torch.float32).sum(dim=1).cpu().numpy()
    else:
        valid_counts = np.asarray(VS_t)[time_idx].astype(np.float32).sum(axis=1)

    keep = valid_counts >= float(min_sites)
    t_keep = time_idx[keep]
    if t_keep.size == 0:
        return time_idx, None

    c = np.maximum(valid_counts[keep], float(eps))
    weights = (1.0 / c) ** float(beta)
    weights = torch.as_tensor(weights / np.maximum(weights.mean(), 1e-12), dtype=torch.double)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(t_keep), replacement=True)
    return t_keep, sampler


def make_multiobs_collate_fn(use_campaign_aug=True, aug_mode="random_valid", seed=1234):
    rng = np.random.default_rng(seed)

    def _choose_targets(y_mean, y_aug, aug_valid):
        Ns = int(y_mean.numel())
        if Ns == 0 or (not use_campaign_aug) or aug_mode == "none" or y_aug.numel() == 0:
            return y_mean, torch.full((Ns,), -1, dtype=torch.long)

        A = y_aug.shape[0]
        if aug_mode == "fixed":
            a = 0
            y = y_aug[a].clone()
            bad = ~aug_valid[a]
            y[bad] = y_mean[bad]
            return y, torch.full((Ns,), a, dtype=torch.long)

        y = y_mean.clone()
        used = torch.full((Ns,), -1, dtype=torch.long)
        aug_valid_np = aug_valid.cpu().numpy()
        for j in range(Ns):
            vv = np.flatnonzero(aug_valid_np[:, j])
            if vv.size > 0:
                a = int(rng.choice(vv))
                y[j] = y_aug[a, j]
                used[j] = a
        return y, used

    def collate(batch):
        out = {
            "x2d": torch.stack([b["x2d"] for b in batch], dim=0),
            "mu0": torch.stack([b["mu0"] for b in batch], dim=0),
            "yS": torch.stack([b["yS"] for b in batch], dim=0),
            "vS": torch.stack([b["vS"] for b in batch], dim=0),
            "Nds": torch.stack([b["Nds"] for b in batch], dim=0),
            "Eds": torch.stack([b["Eds"] for b in batch], dim=0),
            "time_idx": torch.stack([b["time_idx"] for b in batch], dim=0),
        }

        if "seg_ilat" not in batch[0]:
            return out

        seg_batch_idx = []
        seg_ilat = []
        seg_ilon = []
        seg_p = []
        seg_y = []
        seg_y_std = []
        seg_valid_obs = []
        seg_flight_id = []
        seg_campaign_id = []

        for ib, b in enumerate(batch):
            Ns = int(b["seg_ilat"].numel())
            if Ns == 0:
                continue
            y_target, _ = _choose_targets(b["seg_y_mean"], b["seg_y_aug"], b["seg_aug_valid"])

            seg_batch_idx.append(torch.full((Ns,), ib, dtype=torch.long))
            seg_ilat.append(b["seg_ilat"])
            seg_ilon.append(b["seg_ilon"])
            seg_p.append(b["seg_p"])
            seg_y.append(y_target)
            seg_y_std.append(b["seg_y_std"])
            seg_valid_obs.append(b["seg_valid_obs"])
            seg_flight_id.append(b["seg_flight_id"])
            seg_campaign_id.append(b["seg_campaign_id"])

        if len(seg_batch_idx) == 0:
            out.update({
                "seg_batch_idx": torch.empty((0,), dtype=torch.long),
                "seg_ilat": torch.empty((0,), dtype=torch.long),
                "seg_ilon": torch.empty((0,), dtype=torch.long),
                "seg_p": torch.empty((0,), dtype=torch.float32),
                "seg_y": torch.empty((0,), dtype=torch.float32),
                "seg_y_std": torch.empty((0,), dtype=torch.float32),
                "seg_valid_obs": torch.empty((0,), dtype=torch.bool),
                "seg_flight_id": torch.empty((0,), dtype=torch.long),
                "seg_campaign_id": torch.empty((0,), dtype=torch.long),
            })
        else:
            out.update({
                "seg_batch_idx": torch.cat(seg_batch_idx, dim=0),
                "seg_ilat": torch.cat(seg_ilat, dim=0),
                "seg_ilon": torch.cat(seg_ilon, dim=0),
                "seg_p": torch.cat(seg_p, dim=0),
                "seg_y": torch.cat(seg_y, dim=0),
                "seg_y_std": torch.cat(seg_y_std, dim=0),
                "seg_valid_obs": torch.cat(seg_valid_obs, dim=0),
                "seg_flight_id": torch.cat(seg_flight_id, dim=0),
                "seg_campaign_id": torch.cat(seg_campaign_id, dim=0),
            })
        return out

    return collate

# ============================================================
# Dataset over daily times
# ============================================================

class EOFNdCampaignDatasetTorch(Dataset):
    """
    Indexable dataset over daily time indices.

    Each sample returns:
      x2d      : [C,H,W]
      mu0      : [M,H,W]
      t_idx    : scalar long
      seg_ilat : [Ns_t]
      seg_ilon : [Ns_t]
      seg_p    : [Ns_t]
      seg_ymean: [Ns_t]
      seg_ystd : [Ns_t]
      seg_yaug : [A,Ns_t]
      seg_vobs : [Ns_t] bool
      seg_vaug : [A,Ns_t] bool
      seg_fid  : [Ns_t]
      seg_cid  : [Ns_t]
      seg_gid  : [Ns_t]
      seg_day  : [Ns_t] int64 ns
    """
    def __init__(self, X2d_t, MU0_t, bundle: CampaignTensorBundle, time_idx):
        self.X2d = X2d_t
        self.MU0 = MU0_t
        self.B = bundle
        self.time_idx = np.asarray(time_idx, dtype=np.int64)

        assert self.X2d.ndim == 4, f"X2d_t must be (T,C,H,W), got {tuple(self.X2d.shape)}"
        assert self.MU0.ndim == 4, f"MU0_t must be (T,M,H,W), got {tuple(self.MU0.shape)}"

    def __len__(self):
        return self.time_idx.size

    def __getitem__(self, i):
        t = int(self.time_idx[i])
        s0 = int(self.B.time_offsets[t].item())
        n = int(self.B.time_counts[t].item())
        s1 = s0 + n

        return (
            self.X2d[t],                 # [C,H,W]
            self.MU0[t],                 # [M,H,W]
            torch.tensor(t, dtype=torch.long),
            self.B.ilat[s0:s1],          # [Ns]
            self.B.ilon[s0:s1],          # [Ns]
            self.B.p_merra[s0:s1],       # [Ns]
            self.B.y_mean[s0:s1],        # [Ns]
            self.B.y_std[s0:s1],         # [Ns]
            self.B.y_aug[:, s0:s1],      # [A,Ns]
            self.B.valid_obs[s0:s1],     # [Ns]
            self.B.aug_valid[:, s0:s1],  # [A,Ns]
            self.B.flight_id[s0:s1],     # [Ns]
            self.B.campaign_id[s0:s1],   # [Ns]
            self.B.segment_gid[s0:s1],   # [Ns]
            self.B.day_ns[s0:s1],        # [Ns]
        )


# ============================================================
# Time filtering helper
# ============================================================

def filter_times_with_campaign_segments(bundle: CampaignTensorBundle, time_idx, min_segments=1):
    time_idx = np.asarray(time_idx, dtype=np.int64)
    counts = bundle.time_counts.cpu().numpy().astype(np.int64)
    keep = counts[time_idx] >= int(min_segments)
    return time_idx[keep]


# ============================================================
# Collate: choose campaign target and flatten ragged segments
# ============================================================

def make_campaign_collate_fn(
    aug_mode="random_valid",
    fixed_aug=0,
    seed=1234,
):
    """
    aug_mode:
      - "none"         -> target = y_mean
      - "fixed"        -> target = y_aug[fixed_aug], fallback to y_mean if invalid
      - "random_valid" -> choose one valid aug member per segment, fallback to y_mean
    """
    rng = np.random.default_rng(seed)

    def _choose_targets(y_mean, y_aug, aug_valid):
        """
        y_mean   : [Ns]
        y_aug    : [A,Ns]
        aug_valid: [A,Ns]
        returns y_target [Ns], used_aug [Ns] long
        """
        Ns = y_mean.numel()
        if Ns == 0:
            return y_mean, torch.empty((0,), dtype=torch.long)

        if aug_mode == "none" or y_aug.numel() == 0:
            return y_mean, torch.full((Ns,), -1, dtype=torch.long)

        A = y_aug.shape[0]

        if aug_mode == "fixed":
            a = int(np.clip(fixed_aug, 0, A - 1))
            y = y_aug[a].clone()
            bad = ~aug_valid[a]
            y[bad] = y_mean[bad]
            return y, torch.full((Ns,), a, dtype=torch.long)

        if aug_mode == "random_valid":
            y = y_mean.clone()
            used = torch.full((Ns,), -1, dtype=torch.long)
            aug_valid_np = aug_valid.cpu().numpy()

            for j in range(Ns):
                vv = np.flatnonzero(aug_valid_np[:, j])
                if vv.size > 0:
                    a = int(rng.choice(vv))
                    y[j] = y_aug[a, j]
                    used[j] = a
            return y, used

        raise ValueError(f"Unknown aug_mode={aug_mode!r}")

    def collate(batch):
        x2d = torch.stack([b[0] for b in batch], dim=0)   # [B,C,H,W]
        mu0 = torch.stack([b[1] for b in batch], dim=0)   # [B,M,H,W]
        t_idx = torch.stack([b[2] for b in batch], dim=0) # [B]

        seg_batch_idx = []
        seg_ilat = []
        seg_ilon = []
        seg_p = []
        seg_y = []
        seg_y_mean = []
        seg_y_std = []
        seg_valid_obs = []
        seg_fid = []
        seg_cid = []
        seg_gid = []
        seg_day = []
        seg_aug_used = []

        for ib, item in enumerate(batch):
            _, _, _, ilat, ilon, p, ymean, ystd, yaug, vobs, vaug, fid, cid, gid, dayns = item
            Ns = int(ilat.numel())
            if Ns == 0:
                continue

            y_target, aug_used = _choose_targets(ymean, yaug, vaug)

            seg_batch_idx.append(torch.full((Ns,), ib, dtype=torch.long))
            seg_ilat.append(ilat)
            seg_ilon.append(ilon)
            seg_p.append(p)
            seg_y.append(y_target)
            seg_y_mean.append(ymean)
            seg_y_std.append(ystd)
            seg_valid_obs.append(vobs)
            seg_fid.append(fid)
            seg_cid.append(cid)
            seg_gid.append(gid)
            seg_day.append(dayns)
            seg_aug_used.append(aug_used)

        if len(seg_batch_idx) == 0:
            flat = {
                "seg_batch_idx": torch.empty((0,), dtype=torch.long),
                "seg_ilat": torch.empty((0,), dtype=torch.long),
                "seg_ilon": torch.empty((0,), dtype=torch.long),
                "seg_p": torch.empty((0,), dtype=torch.float32),
                "seg_y": torch.empty((0,), dtype=torch.float32),
                "seg_y_mean": torch.empty((0,), dtype=torch.float32),
                "seg_y_std": torch.empty((0,), dtype=torch.float32),
                "seg_valid_obs": torch.empty((0,), dtype=torch.bool),
                "seg_flight_id": torch.empty((0,), dtype=torch.long),
                "seg_campaign_id": torch.empty((0,), dtype=torch.long),
                "seg_gid": torch.empty((0,), dtype=torch.long),
                "seg_day_ns": torch.empty((0,), dtype=torch.long),
                "seg_aug_used": torch.empty((0,), dtype=torch.long),
            }
        else:
            flat = {
                "seg_batch_idx": torch.cat(seg_batch_idx, dim=0),
                "seg_ilat": torch.cat(seg_ilat, dim=0),
                "seg_ilon": torch.cat(seg_ilon, dim=0),
                "seg_p": torch.cat(seg_p, dim=0),
                "seg_y": torch.cat(seg_y, dim=0),
                "seg_y_mean": torch.cat(seg_y_mean, dim=0),
                "seg_y_std": torch.cat(seg_y_std, dim=0),
                "seg_valid_obs": torch.cat(seg_valid_obs, dim=0),
                "seg_flight_id": torch.cat(seg_fid, dim=0),
                "seg_campaign_id": torch.cat(seg_cid, dim=0),
                "seg_gid": torch.cat(seg_gid, dim=0),
                "seg_day_ns": torch.cat(seg_day, dim=0),
                "seg_aug_used": torch.cat(seg_aug_used, dim=0),
            }

        return {
            "x2d": x2d,
            "mu0": mu0,
            "time_idx": t_idx,
            **flat,
        }

    return collate


# ============================================================
# Dense inference on campaign loader
# ============================================================

@torch.no_grad()
def run_infer_campaign_x_mu0(device, loader, model):
    """
    Runs dense inference over a campaign loader that returns:
      batch["x2d"], batch["mu0"], plus segment metadata.

    Returns
    -------
    coeffs_resid_nn : numpy [T, M, H, W]
    sig_grid        : numpy [T, M, H, W]
    """
    was_training = model.training
    model.eval()

    coeffs_all = []
    sigma_all = []

    for batch in loader:
        x = batch["x2d"].to(device, non_blocking=True)
        mu0 = batch["mu0"].to(device, non_blocking=True)

        mu, sigma = model(x, mu0)
        coeffs_all.append(mu.detach().cpu())
        sigma_all.append(sigma.detach().cpu())

    if was_training:
        model.train()

    coeffs_resid_nn = torch.cat(coeffs_all, dim=0).numpy()
    sig_grid = torch.cat(sigma_all, dim=0).numpy()
    return coeffs_resid_nn, sig_grid


# ============================================================
# Sample dense predictions back to campaign segments
# ============================================================

def sample_campaign_from_dense_predictions(
    coeffs_resid_nn,             # [T,M,H,W]
    sigma_grid,                  # [T,M,H,W] or None
    bundle: CampaignTensorBundle,
    EOF,                         # xr.DataArray (mode, lev) or array [M,Z]
    Nd_mean,                     # xr.DataArray (lev) or array [Z]
    A_mean,                      # array or xr.DataArray [M,H,W]
):
    """
    Sample dense coefficient predictions back onto campaign segments and
    interpolate vertically to p_merra.

    Returns
    -------
    xr.Dataset with vars:
      Nd_model(segment)
      sigma_model(segment)      [if sigma_grid is given]
      Nd_obs_mean(segment)
      Nd_obs_std(segment)
      p_merra(segment)
      ilat(segment), ilon(segment)
      time_idx(segment)
      campaign_id(segment)
      flight_id(segment)
      segment_gid(segment)
      day(segment)
      A_seg_resid(segment, mode)
      A_seg_abs(segment, mode)
    """
    coeffs = np.asarray(coeffs_resid_nn, dtype=np.float32)
    sig = None if sigma_grid is None else np.asarray(sigma_grid, dtype=np.float32)

    EOF_mz = np.asarray(getattr(EOF, "values", EOF), dtype=np.float32)        # [M,Z]
    Nd_mean_z = np.asarray(getattr(Nd_mean, "values", Nd_mean), dtype=np.float32)  # [Z]
    Amean_mhw = np.asarray(getattr(A_mean, "values", A_mean), dtype=np.float32)     # [M,H,W]

    lev = np.asarray(EOF["lev"].values if isinstance(EOF, xr.DataArray) and "lev" in EOF.coords else np.arange(EOF_mz.shape[1]), dtype=np.float32)
    mode_coord = np.asarray(EOF["mode"].values if isinstance(EOF, xr.DataArray) and "mode" in EOF.coords else np.arange(EOF_mz.shape[0]), dtype=np.int64)

    # segment metadata
    t = bundle.time_idx.cpu().numpy().astype(np.int64)
    ilat = bundle.ilat.cpu().numpy().astype(np.int64)
    ilon = bundle.ilon.cpu().numpy().astype(np.int64)
    p = bundle.p_merra.cpu().numpy().astype(np.float32)
    y_mean = bundle.y_mean.cpu().numpy().astype(np.float32)
    y_std = bundle.y_std.cpu().numpy().astype(np.float32)
    cid = bundle.campaign_id.cpu().numpy().astype(np.int64)
    fid = bundle.flight_id.cpu().numpy().astype(np.int64)
    gid = bundle.segment_gid.cpu().numpy().astype(np.int64)
    day_ns = bundle.day_ns.cpu().numpy().astype(np.int64)

    # Gather residual coeffs and A_mean at each segment
    # Use transpose to [T,H,W,M] for easy advanced indexing -> [S,M]
    coeffs_thwm = np.transpose(coeffs, (0, 2, 3, 1))
    Amean_hwm = np.transpose(Amean_mhw, (1, 2, 0))

    A_seg_resid = coeffs_thwm[t, ilat, ilon, :]    # [S,M]
    A_seg_mean = Amean_hwm[ilat, ilon, :]          # [S,M]
    A_seg_abs = A_seg_mean + A_seg_resid           # [S,M]

    # Reconstruct Nd profile at segments: [S,Z]
    Nd_prof = A_seg_abs @ EOF_mz + Nd_mean_z[None, :]

    sigma_model_1d = None
    if sig is not None:
        sig_thwm = np.transpose(sig, (0, 2, 3, 1))
        sig_seg = sig_thwm[t, ilat, ilon, :]       # [S,M]
        var_prof = (sig_seg ** 2) @ (EOF_mz ** 2)  # [S,Z]
        sigma_prof = np.sqrt(np.maximum(var_prof, 0.0))

        sigma_model_1d = np.full((len(p),), np.nan, dtype=np.float32)
    else:
        sig_seg = None

    Nd_model = np.full((len(p),), np.nan, dtype=np.float32)

    # scalar log-p interpolation, same spirit as all_by_segment
    for i in range(len(p)):
        Nd_model[i] = _interp_profile_logp_scalar(p[i], lev, Nd_prof[i, :])
        if sigma_model_1d is not None:
            sigma_model_1d[i] = _interp_profile_logp_scalar(p[i], lev, sigma_prof[i, :])

    # campaign names as strings
    campaign_names = np.asarray([bundle.campaign_names[k] for k in cid], dtype="U")

    ds = xr.Dataset(
        data_vars={
            "Nd_model": xr.DataArray(Nd_model, dims=("segment",)),
            "Nd_obs_mean": xr.DataArray(y_mean, dims=("segment",)),
            "Nd_obs_std": xr.DataArray(y_std, dims=("segment",)),
            "p_merra": xr.DataArray(p, dims=("segment",)),
            "ilat": xr.DataArray(ilat.astype(np.int32), dims=("segment",)),
            "ilon": xr.DataArray(ilon.astype(np.int32), dims=("segment",)),
            "time_idx": xr.DataArray(t.astype(np.int64), dims=("segment",)),
            "campaign_id": xr.DataArray(cid.astype(np.int32), dims=("segment",)),
            "flight_id": xr.DataArray(fid.astype(np.int64), dims=("segment",)),
            "segment_gid": xr.DataArray(gid.astype(np.int64), dims=("segment",)),
            "day": xr.DataArray(day_ns.astype("datetime64[ns]"), dims=("segment",)),
            "A_seg_resid": xr.DataArray(A_seg_resid.astype(np.float32), dims=("segment", "mode")),
            "A_seg_abs": xr.DataArray(A_seg_abs.astype(np.float32), dims=("segment", "mode")),
        },
        coords={
            "segment": np.arange(len(p), dtype=np.int64),
            "mode": mode_coord,
            "campaign": xr.DataArray(campaign_names, dims=("segment",)),
        }
    )

    if sigma_model_1d is not None:
        ds["sigma_model"] = xr.DataArray(sigma_model_1d.astype(np.float32), dims=("segment",))

    return ds
