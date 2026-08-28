# quick subroutine to analyze the W data
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import xarray as xr
import time
import json
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
np.set_printoptions(precision=3)


#plotting and other utilities
###########################################################
def plot_global_level_comparison(
    Nd_base,
    Nd_nn,
    Nd_mean,
    levels,
    out_png,
    minN=0.1,
):
    """
    Global comparison using Base as reference.
    Produces maps and zonal diagnostics for Base, NN, NN-Base, RMS.
    """

    # ----------------------------
    # Masking
    # ----------------------------
    B = Nd_base.where(Nd_base >= minN)
    N = Nd_nn.where(B >= minN)
    N = N.where(N >= minN)

    # ----------------------------
    # Time means
    # ----------------------------
    avg_base = B.mean(dim="time", skipna=True)
    avg_nn   = N.mean(dim="time", skipna=True)

    avg_diff = 100.0 * (avg_nn - avg_base) / np.clip(avg_base, 0.01, None)

    # ----------------------------
    # RMS relative to Base
    # ----------------------------
    rms_3d = 100.0 * np.sqrt(
        ((N - B) ** 2).mean(dim="time", skipna=True)
    ) / B.mean(dim="time", skipna=True)

    # ----------------------------
    # Figure layout
    # ----------------------------
    proj = ccrs.PlateCarree()
    nrows = len(levels) + 1

    fig = plt.figure(figsize=(14, 3.2 * len(levels) + 2))
    gs = GridSpec(
        nrows=nrows,
        ncols=4,
        figure=fig,
        height_ratios=[1.0] * len(levels) + [0.6],
        hspace=0.15,
        wspace=0.05,
    )

    axes = np.empty((nrows, 4), dtype=object)

    for i in range(len(levels)):
        for j in range(4):
            axes[i, j] = fig.add_subplot(gs[i, j], projection=proj)

    for j in range(4):
        axes[nrows - 1, j] = fig.add_subplot(gs[nrows - 1, j])

    # ----------------------------
    # Color scales
    # ----------------------------
    Nmin, Nmax = 10, 1000
    log_norm = mcolors.LogNorm(vmin=Nmin, vmax=Nmax)
    shk = 0.4

    # ----------------------------
    # Map panels
    # ----------------------------
    for i, lev in enumerate(levels):

        base_slice = avg_base.sel(lev=lev)
        nn_slice   = avg_nn.sel(lev=lev)
        diff_slice = avg_diff.sel(lev=lev)
        rms_slice  = rms_3d.sel(lev=lev)

        panels = [
            (base_slice, "Base", "turbo", log_norm),
            (nn_slice,   "NN",   "turbo", log_norm),
            (diff_slice, "% NN-Base", "RdBu_r", None),
            (rms_slice,  "RMS Error", "RdBu_r", None),
        ]

        for ax, (da, title, cmap, norm) in zip(axes[i], panels):
            im = da.plot.pcolormesh(
                ax=ax,
                transform=proj,
                cmap=cmap,
                norm=norm,
                vmin=-100 if "NN-Base" in title else (0 if "RMS" in title else None),
                vmax=100  if "NN-Base" in title else (1 if "RMS" in title else None),
                add_colorbar=False,
            )

            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.set_title(f"{title} @ {lev} hPa")
            fig.colorbar(im, ax=ax, shrink=shk)

    # ----------------------------
    # Zonal diagnostics
    # ----------------------------
    row = len(levels)

    zonal_base = avg_base.mean(dim="lon", skipna=True)
    zonal_nn   = avg_nn.mean(dim="lon", skipna=True)
    zonal_diff = 100.0 * (zonal_nn - zonal_base) / np.clip(zonal_base, 0.1, None)

    zonal_rms = 100.0 * np.sqrt(
        ((zonal_nn - zonal_base) ** 2)
    ) / zonal_base.where(zonal_base > minN)

    ZB = np.clip(zonal_base.values, Nmin, None)
    ZN = np.clip(zonal_nn.values, Nmin, None)
    ZD = zonal_diff.values
    ZR = zonal_rms.values

    p_lev = Nd_mean.lev.values
    p_ticks = np.array([1000, 850, 700, 500, 300, 200])
    yticks = [np.argmin(np.abs(p_lev - p)) for p in p_ticks]
    ylabels = [str(int(p)) for p in p_ticks]

    zonal_panels = [
            (ZB, "Zonal Mean (Base)", "turbo", log_norm),
            (ZN, "Zonal Mean (NN)", "turbo", log_norm),
            (ZD, "% Difference (Zonal)", "RdBu_r", None),
            (ZR, "RMS Error (Zonal)", "RdBu_r", None),
        ]


    for ax, (Z, title, cmap, norm) in zip(axes[row], zonal_panels):
        im = ax.imshow(
                Z,
                aspect="auto",
                cmap=cmap,
                norm=norm,
                vmin=-100 if "Difference" in title else (0 if "RMS" in title else None),
                vmax=100  if "Difference" in title else (1 if "RMS" in title else None),
            )


    axes[row, 0].set_ylabel("Pressure (hPa)")

    # ----------------------------
    # Save
    # ----------------------------
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


#########################################################


def plot_vertical_profiles_sites(Nd_true_s, Nd_base_s, Nd_nn_s, out_png, min_samples=2, max_per_row=5):
    valid_t_site = xr.ufuncs.isfinite(Nd_true_s).any("lev")   # (time, site)
    counts = valid_t_site.sum("time").values                  # (site,)
    keep_idx = np.where(counts >= min_samples)[0]

    if keep_idx.size == 0:
        print("[WARN] No valid sites for profile plot")
        return

    n = keep_idx.size
    ncols = min(max_per_row, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.8 * ncols, 6 * nrows),
        sharey=True,
        squeeze=False
    )
    axes_flat = axes.ravel()

    lev_vals = np.asarray(Nd_true_s.lev.values)
    y_bottom, y_top = np.nanmax(lev_vals), np.nanmin(lev_vals)

    for k, sidx in enumerate(keep_idx):
        ax = axes_flat[k]
        vmask = valid_t_site[:, sidx]  # (time,)

        # Mean + std across time, masking invalid times
        Tm = Nd_true_s[:, :, sidx].where(vmask).mean("time")
        Ts = Nd_true_s[:, :, sidx].where(vmask).std("time")
        Bm = Nd_base_s[:, :, sidx].where(vmask).mean("time")
        Bs = Nd_base_s[:, :, sidx].where(vmask).std("time")
        Nm = Nd_nn_s[:, :, sidx].where(vmask).mean("time")
        Ns = Nd_nn_s[:, :, sidx].where(vmask).std("time")

        # Convert to numpy
        Tm, Ts = Tm.values, Ts.values
        Bm, Bs = Bm.values, Bs.values
        Nm, Ns = Nm.values, Ns.values

        # Lines
        ax.plot(Tm, lev_vals, "k", lw=2, label="Obs")
        ax.plot(Bm, lev_vals, "b--", lw=1.8, label="Base")
        ax.plot(Nm, lev_vals, "r", lw=2, label="NN")

        # ±1σ shading (horizontal fill vs y=lev)
        ax.fill_betweenx(lev_vals, Tm - Ts, Tm + Ts, color="k", alpha=0.15)
        ax.fill_betweenx(lev_vals, Bm - Bs, Bm + Bs, color="b", alpha=0.12)
        ax.fill_betweenx(lev_vals, Nm - Ns, Nm + Ns, color="r", alpha=0.12)

        ax.grid(True)
        ax.set_xlabel("Nd")

        name = str(Nd_true_s.site.values[sidx])
        if "geos_lat" in Nd_true_s.coords and "geos_lon" in Nd_true_s.coords:
            glat = float(Nd_true_s.geos_lat.values[sidx])
            glon = float(Nd_true_s.geos_lon.values[sidx])
            ax.set_title(f"{name}\n({glat:.1f},{glon:.1f})", fontsize=9)
        else:
            ax.set_title(name, fontsize=9)

        # Force pressure orientation
        ax.set_ylim(y_bottom, y_top)

    # Turn off unused axes
    for j in range(n, nrows * ncols):
        axes_flat[j].axis("off")

    axes_flat[0].set_ylabel("Pressure (hPa)")
    axes_flat[0].legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()


###############################

###########################################################
def plot_global_level_sigma_inference(
    sigmaNd_nn,     # xr.DataArray (time, lev, lat, lon)
    Nd_mean,        # xr.DataArray (lev)  (only used for lev ticks)
    levels,         # list of hPa levels to map
    out_png,
    minSig=1e-6,    # mask tiny/invalid sigma
):
    """
    Inference-only sigma diagnostic:
      - Maps: time-mean sigmaNd_nn at requested levels
      - Zonal diagnostic: lon-mean (time-mean) sigma as (lev x lat) heatmap

    Produces (len(levels) + 1) rows x 1 column.
    """
    # ----------------------------
    # Masking + time mean
    # ----------------------------
    S = sigmaNd_nn.where(np.isfinite(sigmaNd_nn))
    S = S.where(S >= minSig)
    avg_sig = S.mean(dim="time", skipna=True)  # (lev, lat, lon)

    # ----------------------------
    # Figure layout
    # ----------------------------
    proj = ccrs.PlateCarree()
    nrows = len(levels) + 1

    fig = plt.figure(figsize=(7.2, 3.2 * len(levels) + 2.2))
    gs = GridSpec(
        nrows=nrows,
        ncols=1,
        figure=fig,
        height_ratios=[1.0] * len(levels) + [0.8],
        hspace=0.18,
        wspace=0.05,
    )

    axes = []
    for i in range(len(levels)):
        axes.append(fig.add_subplot(gs[i, 0], projection=proj))
    ax_zonal = fig.add_subplot(gs[nrows - 1, 0])

    # ----------------------------
    # Color scale (log)
    # ----------------------------
    # Robust bounds from data
    svals = avg_sig.values
    finite = np.isfinite(svals) & (svals > 0)
    if np.any(finite):
        vmin = max(minSig, float(np.nanpercentile(svals[finite], 2)))
        vmax = float(np.nanpercentile(svals[finite], 98))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin * 10.0
    else:
        vmin, vmax = minSig, minSig * 10.0

    vmin =  10
    vmax = 1000
    log_norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    shk = 0.55

    # ----------------------------
    # Map panels
    # ----------------------------
    for ax, lev in zip(axes, levels):
        sig_slice = avg_sig.sel(lev=lev)

        im = sig_slice.plot.pcolormesh(
            ax=ax,
            transform=proj,
            cmap="viridis",
            norm=log_norm,
            add_colorbar=False,
        )
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.set_title(f"Time-mean $\\sigma_{{Nd}}$ (NN) @ {lev} hPa")
        fig.colorbar(im, ax=ax, shrink=shk, label=r"$\sigma_{Nd}$")

    # ----------------------------
    # Zonal diagnostic (lev x lat)
    # ----------------------------
    zonal_sig = avg_sig.mean(dim="lon", skipna=True)  # (lev, lat)

    Z = zonal_sig.values
    imz = ax_zonal.imshow(
        Z,
        aspect="auto",
        cmap="viridis",
        norm=log_norm,
        origin="upper",
    )
    fig.colorbar(imz, ax=ax_zonal, shrink=0.9, label=r"$\sigma_{Nd}$ (zonal)")

    # axis ticks: pressure
    p_lev = Nd_mean.lev.values
    p_ticks = np.array([1000, 850, 700, 500, 300, 200])
    yticks = [int(np.argmin(np.abs(p_lev - p))) for p in p_ticks]
    ylabels = [str(int(p)) for p in p_ticks]

    ax_zonal.set_yticks(yticks)
    ax_zonal.set_yticklabels(ylabels)
    ax_zonal.set_ylabel("Pressure (hPa)")

    # axis ticks: latitude
    lat = zonal_sig["lat"].values
    # pick a few evenly spaced ticks
    nlt = 7
    idx = np.linspace(0, len(lat) - 1, nlt).astype(int)
    ax_zonal.set_xticks(idx)
    ax_zonal.set_xticklabels([f"{lat[i]:.0f}" for i in idx])
    ax_zonal.set_xlabel("Latitude (deg)")

    ax_zonal.set_title(r"Zonal mean of time-mean $\sigma_{Nd}$ (NN)")

    # ----------------------------
    # Save
    # ----------------------------
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)

