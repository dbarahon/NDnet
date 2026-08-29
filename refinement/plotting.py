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
import math
import config as cfg

np.set_printoptions(precision=3)


# plotting and other utilities
###########################################################
def plot_global_level_comparison(
    Nd_base,
    Nd_nn,
    Nd_mean,
    levels,
    out_png,
    minN=None,
):
    """Journal-style global Nd comparison using Base as reference.

    The map rows are ordered physically from lower pressure (higher altitude)
    to higher pressure (lower altitude). The bottom row is a latitude-pressure
    zonal diagnostic plotted against the actual pressure coordinate, not level
    indices.
    """
    if minN is None:
        minN = float(getattr(cfg, "MIN_ND", 1.0))
    else:
        minN = float(minN)

    B = Nd_base.where(np.isfinite(Nd_base) & (Nd_base >= minN))
    N = Nd_nn.where(np.isfinite(Nd_nn) & (Nd_nn >= minN))
    N = N.where(np.isfinite(B))

    avg_base = B.mean(dim="time", skipna=True)
    avg_nn = N.mean(dim="time", skipna=True)

    denom = avg_base.where(avg_base >= minN)
    avg_diff = 100.0 * (avg_nn - avg_base) / denom

    rms_3d = 100.0 * np.sqrt(
        ((N - B) ** 2).mean(dim="time", skipna=True)
    ) / denom

    levels_plot = sorted(float(v) for v in levels)

    proj = ccrs.PlateCarree()
    nmap = len(levels_plot)
    nrows = nmap + 1

    fig = plt.figure(figsize=(14.0, 2.45 * nmap + 3.0))
    gs = GridSpec(
        nrows=nrows,
        ncols=4,
        figure=fig,
        height_ratios=[1.0] * nmap + [1.15],
        hspace=0.10,
        wspace=0.06,
    )

    axes = np.empty((nrows, 4), dtype=object)
    for i in range(nmap):
        for j in range(4):
            axes[i, j] = fig.add_subplot(gs[i, j], projection=proj)
    for j in range(4):
        axes[nmap, j] = fig.add_subplot(gs[nmap, j])

    Nmin = max(10.0, minN)
    Nmax = 1000.0
    nd_norm = mcolors.LogNorm(vmin=Nmin, vmax=Nmax)
    diff_norm = mcolors.TwoSlopeNorm(vmin=-100.0, vcenter=0.0, vmax=100.0)
    rms_norm = mcolors.Normalize(vmin=0.0, vmax=100.0, clip=True)

    col_titles = [
        "Base",
        "NDnet",
        "NDnet - Base",
        "RMS difference",
    ]
    cmaps = ["turbo", "turbo", "RdBu_r", "viridis"]
    norms = [nd_norm, nd_norm, diff_norm, rms_norm]
    cbar_labels = [
        r"$N_d$ (cm$^{-3}$)",
        r"$N_d$ (cm$^{-3}$)",
        "Difference (%)",
        "RMS difference (%)",
    ]

    def _global_mean(da):
        if "lat" in da.coords:
            w = np.cos(np.deg2rad(da["lat"]))
            return float(da.weighted(w).mean(("lat", "lon"), skipna=True).values)
        return float(da.mean(("lat", "lon"), skipna=True).values)

    panel_index = 0
    mappables = [None] * 4

    for i, lev in enumerate(levels_plot):
        base_slice = avg_base.sel(lev=lev, method="nearest")
        nn_slice = avg_nn.sel(lev=lev, method="nearest")
        diff_slice = avg_diff.sel(lev=lev, method="nearest")
        rms_slice = rms_3d.sel(lev=lev, method="nearest")

        actual_lev = float(base_slice["lev"].values)
        panel_data = [base_slice, nn_slice, diff_slice, rms_slice]

        for j, (ax, da) in enumerate(zip(axes[i], panel_data)):
            im = da.plot.pcolormesh(
                ax=ax,
                transform=proj,
                cmap=cmaps[j],
                norm=norms[j],
                add_colorbar=False,
            )
            mappables[j] = im

            ax.coastlines(linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linewidth=0.35)
            ax.set_global()

            xticks = [-120, -60, 0, 60, 120]
            yticks = [-60, -30, 0, 30, 60]
            ax.set_xticks(xticks, crs=proj)
            ax.set_yticks(yticks, crs=proj)
            ax.tick_params(axis="both", which="major", labelsize=7, length=2.5)
            if i < nmap - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
            ax.set_xlabel("")
            ax.set_ylabel("")

            letter = chr(ord("a") + panel_index)
            panel_index += 1
            ax.set_title(
                f"({letter}) {actual_lev:.0f} hPa",
                loc="left",
                fontsize=9,
                pad=3,
            )

            gmean = _global_mean(da)
            if j < 2:
                right = rf"$\overline{{N_d}}$={gmean:.0f}"
            else:
                right = f"mean={gmean:.1f}%"
            ax.set_title(right, loc="right", fontsize=8.5, pad=3)

            if i == 0:
                ax.set_title(
                    col_titles[j],
                    loc="center",
                    fontsize=10.5,
                    fontweight="bold",
                    pad=16,
                )

    row = nmap

    zonal_base = avg_base.mean(dim="lon", skipna=True).transpose("lev", "lat")
    zonal_nn = avg_nn.mean(dim="lon", skipna=True).transpose("lev", "lat")
    zonal_denom = zonal_base.where(zonal_base >= minN)
    zonal_diff = 100.0 * (zonal_nn - zonal_base) / zonal_denom
    zonal_rms = 100.0 * np.sqrt((zonal_nn - zonal_base) ** 2) / zonal_denom

    zonal_panels = [zonal_base, zonal_nn, zonal_diff, zonal_rms]
    lat_vals = np.asarray(zonal_base["lat"].values, dtype=float)
    lev_vals = np.asarray(zonal_base["lev"].values, dtype=float)

    p_lo = float(np.nanmin(lev_vals))
    p_hi = float(np.nanmax(lev_vals))
    standard_p_ticks = np.array([1000, 900, 800, 700, 600, 500, 400, 300, 200])
    p_ticks = standard_p_ticks[(standard_p_ticks >= p_lo) & (standard_p_ticks <= p_hi)]

    for j, (ax, da) in enumerate(zip(axes[row], zonal_panels)):
        im = ax.pcolormesh(
            lat_vals,
            lev_vals,
            da.values,
            shading="auto",
            cmap=cmaps[j],
            norm=norms[j],
        )
        mappables[j] = im

        ax.set_ylim(p_hi, p_lo)
        ax.set_yticks(p_ticks)
        ax.set_xlim(float(np.nanmin(lat_vals)), float(np.nanmax(lat_vals)))
        ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
        ax.tick_params(axis="both", labelsize=8, direction="out", length=3)
        ax.grid(False)
        ax.set_xlabel("Latitude (°)", fontsize=9)

        if j == 0:
            ax.set_ylabel("Pressure (hPa)", fontsize=9)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        letter = chr(ord("a") + panel_index)
        panel_index += 1
        ax.set_title(f"({letter}) Zonal mean", loc="left", fontsize=9, pad=3)

    for j in range(4):
        cb = fig.colorbar(
            mappables[j],
            ax=list(axes[:, j]),
            orientation="horizontal",
            fraction=0.025,
            pad=0.045,
            aspect=35,
        )
        cb.set_label(cbar_labels[j], fontsize=9)
        cb.ax.tick_params(labelsize=8)

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


#########################################################


def plot_vertical_profiles_sites(Nd_true_s, Nd_base_s, Nd_nn_s, out_png, min_samples=2, max_per_row=5):
    """Plot site profiles using exactly the same dates for Obs, Base, and NN."""
    valid_true = np.isfinite(Nd_true_s).any("lev")
    valid_base = np.isfinite(Nd_base_s).any("lev")
    valid_nn = np.isfinite(Nd_nn_s).any("lev")
    valid_t_site = valid_true & valid_base & valid_nn

    counts = valid_t_site.sum("time").values
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
        vmask = valid_t_site[:, sidx]

        Tm = Nd_true_s[:, :, sidx].where(vmask).mean("time")
        Ts = Nd_true_s[:, :, sidx].where(vmask).std("time")
        Bm = Nd_base_s[:, :, sidx].where(vmask).mean("time")
        Bs = Nd_base_s[:, :, sidx].where(vmask).std("time")
        Nm = Nd_nn_s[:, :, sidx].where(vmask).mean("time")
        Ns = Nd_nn_s[:, :, sidx].where(vmask).std("time")

        Tm, Ts = Tm.values, Ts.values
        Bm, Bs = Bm.values, Bs.values
        Nm, Ns = Nm.values, Ns.values

        ax.plot(Tm, lev_vals, "k", lw=2, label="Obs")
        ax.plot(Bm, lev_vals, "b--", lw=1.8, label="Base")
        ax.plot(Nm, lev_vals, "r", lw=2, label="NN")

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

        ax.set_ylim(y_bottom, y_top)

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
    sigmaNd_nn,
    Nd_mean,
    levels,
    out_png,
    minSig=1e-6,
):
    S = sigmaNd_nn.where(np.isfinite(sigmaNd_nn))
    S = S.where(S >= minSig)
    avg_sig = S.mean(dim="time", skipna=True)

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

    svals = avg_sig.values
    finite = np.isfinite(svals) & (svals > 0)
    if np.any(finite):
        vmin = max(minSig, float(np.nanpercentile(svals[finite], 2)))
        vmax = float(np.nanpercentile(svals[finite], 98))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin * 10.0
    else:
        vmin, vmax = minSig, minSig * 10.0

    vmin = 10
    vmax = 1000
    log_norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    shk = 0.55

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

    zonal_sig = avg_sig.mean(dim="lon", skipna=True)

    Z = zonal_sig.values
    imz = ax_zonal.imshow(
        Z,
        aspect="auto",
        cmap="viridis",
        norm=log_norm,
        origin="upper",
    )
    fig.colorbar(imz, ax=ax_zonal, shrink=0.9, label=r"$\sigma_{Nd}$ (zonal)")

    p_lev = Nd_mean.lev.values
    p_ticks = np.array([1000, 850, 700, 500, 300, 200])
    yticks = [int(np.argmin(np.abs(p_lev - p))) for p in p_ticks]
    ylabels = [str(int(p)) for p in p_ticks]

    ax_zonal.set_yticks(yticks)
    ax_zonal.set_yticklabels(ylabels)
    ax_zonal.set_ylabel("Pressure (hPa)")

    lat = zonal_sig["lat"].values
    nlt = 7
    idx = np.linspace(0, len(lat) - 1, nlt).astype(int)
    ax_zonal.set_xticks(idx)
    ax_zonal.set_xticklabels([f"{lat[i]:.0f}" for i in idx])
    ax_zonal.set_xlabel("Latitude (deg)")

    ax_zonal.set_title(r"Zonal mean of time-mean $\sigma_{Nd}$ (NN)")

    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Campaign scatter diagnostics
# ============================================================


def _campaign_layout_for_n(n):
    ncols = 2 if n <= 4 else 3
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def _campaign_compute_stats(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan, np.nan, int(m.sum())
    x = x[m]
    y = y[m]
    r = np.corrcoef(x, y)[0, 1]
    bias = np.mean(y - x)
    return float(r), float(bias), int(m.sum())


def _campaign_subset(
    df,
    obs_col,
    model_col,
    min_nd_compare=None,
    require_positive=False,
    use_modis_mask=False,
):
    m = np.isfinite(df[obs_col].values) & np.isfinite(df[model_col].values)

    if require_positive:
        m &= (df[obs_col].values > 0) & (df[model_col].values > 0)

    if min_nd_compare is not None and obs_col in df.columns:
        m &= np.isfinite(df[obs_col].values) & (df[obs_col].values >= float(min_nd_compare))

    if use_modis_mask and ("modis_ok" in df.columns):
        m &= (df["modis_ok"].values == 1)

    return df.loc[m].copy()


def _campaign_ds_to_dfs(dsC):
    if "campaign" in dsC.coords:
        campaign_vals = np.asarray(dsC["campaign"].values).astype("U")
    elif "campaign" in dsC:
        campaign_vals = np.asarray(dsC["campaign"].values).astype("U")
    else:
        raise KeyError("dsC must contain campaign as a coord or variable")

    time_vals = pd.to_datetime(dsC["time"].values) if "time" in dsC else pd.to_datetime(dsC["day"].values)

    cols = {
        "campaign": campaign_vals,
        "time": time_vals,
        "p_merra": dsC["p_merra"].values.astype(float),
        "Nd_obs": dsC["Nd_obs"].values.astype(float),
        "Nd_nn": dsC["Nd_nn"].values.astype(float),
    }

    if "Nd_base" in dsC:
        cols["Nd_base"] = dsC["Nd_base"].values.astype(float)
    if "sigma_nn" in dsC:
        cols["sigma_nn"] = dsC["sigma_nn"].values.astype(float)
    if "modis_ok" in dsC:
        cols["modis_ok"] = dsC["modis_ok"].values.astype(int)

    df = pd.DataFrame(cols)

    out = []
    for camp in np.unique(df["campaign"].values):
        dfi = df.loc[df["campaign"] == camp].copy()
        dfi = dfi.sort_values("time").reset_index(drop=True)
        out.append(dfi)
    return out


def plot_campaign_scatter_panels(
    dsC,
    out_png="NDnet_campaign_scatter.png",
    obs_col="Nd_obs",
    nn_col="Nd_nn",
    base_col="Nd_base",
    log_axes=True,
    min_nd_compare=1.0,
    use_modis_mask=False,
    xlim=None,
    ylim=None,
):
    dfs = _campaign_ds_to_dfs(dsC)
    n = len(dfs)
    if n == 0:
        print("[INFO] No campaigns found; skipping campaign scatter plot.")
        return

    vals = []
    for df in dfs:
        sub_nn = _campaign_subset(
            df, obs_col, nn_col,
            min_nd_compare=min_nd_compare,
            require_positive=log_axes,
            use_modis_mask=use_modis_mask,
        )
        if not sub_nn.empty:
            vals.append(sub_nn[obs_col].values)
            vals.append(sub_nn[nn_col].values)

        if base_col in df.columns:
            sub_b = _campaign_subset(
                df, obs_col, base_col,
                min_nd_compare=min_nd_compare,
                require_positive=log_axes,
                use_modis_mask=use_modis_mask,
            )
            if not sub_b.empty:
                vals.append(sub_b[base_col].values)

    if vals:
        vv = np.concatenate(vals)
        vv = vv[np.isfinite(vv)]
        if log_axes:
            vv = vv[vv > 0]
            lo = np.nanmin(vv) * 0.9 if vv.size else 1e-2
            hi = np.nanmax(vv) * 1.1 if vv.size else 1.0
        else:
            lo = 0.0
            hi = np.nanmax(vv) * 1.05 if vv.size else 1.0
    else:
        lo, hi = (1e-2, 1.0) if log_axes else (0.0, 1.0)

    if xlim is not None:
        lo, hi = xlim
    if ylim is None:
        ylo, yhi = lo, hi
    else:
        ylo, yhi = ylim

    p_all = []
    for df in dfs:
        sub = _campaign_subset(
            df, obs_col, nn_col,
            min_nd_compare=min_nd_compare,
            require_positive=False,
            use_modis_mask=use_modis_mask,
        )
        if not sub.empty:
            p_all.append(sub["p_merra"].values)
    p_all = np.concatenate(p_all) if p_all else np.array([1000.0])
    pmin = np.nanmin(p_all)
    pmax = np.nanmax(p_all)

    nrows, ncols = _campaign_layout_for_n(n)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.4 * ncols, 4.8 * nrows),
        sharex=True, sharey=True,
        constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()
    last_sc = None

    for ax, df in zip(axes, dfs):
        campaign = str(df["campaign"].iloc[0])
        title_bits = []

        if base_col in df.columns:
            sub_b = _campaign_subset(
                df, obs_col, base_col,
                min_nd_compare=min_nd_compare,
                require_positive=log_axes,
                use_modis_mask=use_modis_mask,
            )
            if not sub_b.empty:
                r_b, bias_b, n_b = _campaign_compute_stats(sub_b[obs_col].values, sub_b[base_col].values)
                ax.scatter(
                    sub_b[obs_col].values,
                    sub_b[base_col].values,
                    s=26,
                    facecolors="none",
                    edgecolors="0.45",
                    linewidths=1.0,
                    alpha=0.9,
                    label="Base",
                )
                title_bits.append(f"base r={r_b:.2f}, b={bias_b:.1f}, n={n_b}")

        sub_nn = _campaign_subset(
            df, obs_col, nn_col,
            min_nd_compare=min_nd_compare,
            require_positive=log_axes,
            use_modis_mask=use_modis_mask,
        )

        if not sub_nn.empty:
            r_n, bias_n, n_n = _campaign_compute_stats(sub_nn[obs_col].values, sub_nn[nn_col].values)
            last_sc = ax.scatter(
                sub_nn[obs_col].values,
                sub_nn[nn_col].values,
                c=sub_nn["p_merra"].values,
                s=30,
                cmap="viridis",
                vmin=pmin,
                vmax=pmax,
                alpha=0.85,
                label="NN",
            )
            title_bits.append(f"nn r={r_n:.2f}, b={bias_n:.1f}, n={n_n}")
        else:
            title_bits.append("no valid NN points")

        ax.plot([lo, hi], [ylo, yhi], linestyle="--", linewidth=1.0, color="k")
        ax.set_xlim(lo, hi)
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel(r"Observed $N_d$ (cm$^{-3}$)")
        ax.set_ylabel(r"Model $N_d$ (cm$^{-3}$)")
        ax.set_title(campaign + "\n" + " | ".join(title_bits), fontsize=10)
        ax.grid(True, alpha=0.3)

        if log_axes:
            ax.set_xscale("log")
            ax.set_yscale("log")

        if base_col in df.columns or not sub_nn.empty:
            ax.legend(loc="best", fontsize=8)

    for ax in axes[n:]:
        ax.remove()

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes[:n], shrink=0.95)
        cbar.set_label("Pressure (hPa)")
        cbar.ax.invert_yaxis()

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[WROTE] {out_png}")


###########################################################
def plot_global_level_sigma_rel_inference(
    sigmaNd_nn,
    Nd_nn,
    Nd_mean,
    levels,
    out_png,
    minSig=1e-6,
    minNd=1.0,
    vmin=0.0,
    vmax=100.0,
):
    S = sigmaNd_nn.where(np.isfinite(sigmaNd_nn))
    N = Nd_nn.where(np.isfinite(Nd_nn))

    valid = (
        (S >= minSig)
        & (N >= minNd)
    )

    S = S.where(valid)
    N = N.where(valid)

    rel = 100.0 * S / N
    rel = rel.where(np.isfinite(rel))

    avg_rel = rel.mean(dim="time", skipna=True)

    proj = ccrs.PlateCarree()
    nrows = len(levels) + 1

    fig = plt.figure(
        figsize=(7.2, 3.2 * len(levels) + 2.2)
    )

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
        axes.append(
            fig.add_subplot(gs[i, 0], projection=proj)
        )

    ax_zonal = fig.add_subplot(gs[nrows - 1, 0])

    norm = mcolors.Normalize(
        vmin=vmin,
        vmax=vmax,
        clip=True,
    )

    shk = 0.55

    for ax, lev in zip(axes, levels):
        rel_slice = avg_rel.sel(
            lev=lev,
            method="nearest",
        )

        actual_lev = float(rel_slice["lev"].values)

        im = rel_slice.plot.pcolormesh(
            ax=ax,
            transform=proj,
            cmap="viridis",
            norm=norm,
            add_colorbar=False,
        )

        ax.coastlines()
        ax.add_feature(
            cfeature.BORDERS,
            linewidth=0.5,
        )

        ax.set_title(
            rf"Time-mean relative $\sigma_{{N_d}}$ "
            rf"@ {actual_lev:.0f} hPa"
        )

        fig.colorbar(
            im,
            ax=ax,
            shrink=shk,
            label=r"$100\,\sigma_{N_d}/N_d$ (%)",
        )

    zonal_rel = avg_rel.mean(
        dim="lon",
        skipna=True,
    )

    Z = zonal_rel.values

    imz = ax_zonal.imshow(
        Z,
        aspect="auto",
        cmap="viridis",
        norm=norm,
        origin="upper",
    )

    fig.colorbar(
        imz,
        ax=ax_zonal,
        shrink=0.9,
        label=r"$100\,\sigma_{N_d}/N_d$ (%)",
    )

    p_lev = Nd_mean.lev.values

    p_ticks = np.array([
        1000,
        850,
        700,
        500,
        300,
        200,
    ])

    yticks = [
        int(np.argmin(np.abs(p_lev - p)))
        for p in p_ticks
    ]

    ylabels = [
        str(int(p))
        for p in p_ticks
    ]

    ax_zonal.set_yticks(yticks)
    ax_zonal.set_yticklabels(ylabels)
    ax_zonal.set_ylabel("Pressure (hPa)")

    lat = zonal_rel["lat"].values

    nlt = 7
    idx = np.linspace(
        0,
        len(lat) - 1,
        nlt,
    ).astype(int)

    ax_zonal.set_xticks(idx)

    ax_zonal.set_xticklabels([
        f"{lat[i]:.0f}"
        for i in idx
    ])

    ax_zonal.set_xlabel("Latitude (deg)")

    ax_zonal.set_title(
        r"Zonal mean relative uncertainty "
        r"$100\,\sigma_{N_d}/N_d$"
    )

    plt.savefig(
        out_png,
        dpi=120,
        bbox_inches="tight",
    )

    plt.close(fig)
