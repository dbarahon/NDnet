#!/usr/bin/env python3
# ============================================================
# Publication-quality campaign diagnostics for NDnet
#
# Produces:
#   1. Observed Nd vs predicted Nd
#      - NN prediction colored by pressure
#      - Base prediction as open gray circles
#
#   2. Observed sigma_Nd vs predicted sigma_Nd
#      - sigma_obs = Nd_std from campaign training files
#      - sigma_nn  = predicted NN sigma from campaign test file
#      - colored by pressure
#
# Example:
#
# python plot_campaign_diagnostics.py \
#     /panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/refinement/EXP
#
# or explicitly:
#
# python plot_campaign_diagnostics.py \
#     /panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/refinement/EXP \
#     --test-file NDnet_campaign_test.nc
#
# ============================================================

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import LogLocator, NullFormatter


# ============================================================
# Defaults
# ============================================================

DEFAULT_TRAINING_DIR = Path(
    "/explore/nobackup/people/dbarahon/NDnet/data/"
    "field_campaigns/training"
)

DEFAULT_TEST_FILE = "NDnet_campaign_test.nc"

ND_LIM = (1.0, 1000.0)
SIGMA_LIM = (1.0, 500.0)
# ============================================================
# Publication plotting defaults
# ============================================================

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
})


# ============================================================
# Utility functions
# ============================================================

def _campaign_layout_for_n(n):
    """
    Compact layout suitable for a journal figure.
    """
    if n <= 4:
        ncols = 2
    else:
        ncols = 3

    nrows = math.ceil(n / ncols)
    return nrows, ncols


def _campaign_compute_stats(x, y):
    """
    Compute statistics in native linear units.

    Returns
    -------
    r : Pearson correlation
    bias : mean(y - x)
    rmse : root-mean-square error
    n : sample count
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)

    if m.sum() < 2:
        return np.nan, np.nan, np.nan, int(m.sum())

    x = x[m]
    y = y[m]

    r = np.corrcoef(x, y)[0, 1]
    d = y - x

    bias = np.mean(d)
    rmse = np.sqrt(np.mean(d ** 2))

    return float(r), float(bias), float(rmse), int(m.sum())


def _campaign_subset(
    df,
    obs_col,
    model_col,
    min_compare=None,
    require_positive=False,
    use_modis_mask=False,
):
    """
    Return valid observational/model pairs.
    """
    x = df[obs_col].to_numpy(dtype=float)
    y = df[model_col].to_numpy(dtype=float)

    m = np.isfinite(x) & np.isfinite(y)

    if require_positive:
        m &= (x > 0.0) & (y > 0.0)

    if min_compare is not None:
        m &= x >= float(min_compare)

    if use_modis_mask and ("modis_ok" in df.columns):
        m &= df["modis_ok"].to_numpy() == 1

    return df.loc[m].copy()


def _panel_label(i):
    """
    0 -> (a), 1 -> (b), ...
    """
    return f"({chr(ord('a') + i)})"


def _clean_limits(values, log_axes=True):
    """
    Produce clean shared axis limits.

    For log axes, limits are rounded to powers of ten.
    For linear axes, the lower limit is zero unless negative values occur.
    """
    if len(values) == 0:
        return (1.0, 1000.0) if log_axes else (0.0, 1.0)

    v = np.concatenate([
        np.asarray(a, dtype=float).ravel()
        for a in values
        if np.size(a) > 0
    ])

    v = v[np.isfinite(v)]

    if log_axes:
        v = v[v > 0]

        if v.size == 0:
            return 1.0, 1000.0

        lo = 10.0 ** np.floor(np.log10(np.nanmin(v)))
        hi = 10.0 ** np.ceil(np.log10(np.nanmax(v)))

        if hi <= lo:
            hi = lo * 10.0

    else:
        if v.size == 0:
            return 0.0, 1.0

        vmin = np.nanmin(v)
        vmax = np.nanmax(v)

        lo = min(0.0, vmin)

        if vmax > lo:
            pad = 0.05 * (vmax - lo)
        else:
            pad = 1.0

        hi = vmax + pad

    return float(lo), float(hi)


def _set_log_ticks(ax):
    """
    Add major and minor log ticks while leaving minor labels off.
    """
    major = LogLocator(base=10.0)
    minor = LogLocator(
        base=10.0,
        subs=np.arange(2, 10) * 0.1
    )

    ax.xaxis.set_major_locator(major)
    ax.yaxis.set_major_locator(major)

    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_minor_locator(minor)

    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


# ============================================================
# Read prediction dataset
# ============================================================

def _campaign_ds_to_dataframe(dsC):
    """
    Convert NDnet campaign test dataset to one DataFrame.

    Expected variables
    ------------------
    campaign
    Nd_obs
    Nd_nn
    p_merra

    Optional
    --------
    Nd_base
    sigma_nn
    time
    time_idx
    flight_id
    segment_gid
    campaign_id
    modis_ok
    """

    if "campaign" in dsC.coords:
        campaign_vals = np.asarray(
            dsC["campaign"].values
        ).astype("U")

    elif "campaign" in dsC:
        campaign_vals = np.asarray(
            dsC["campaign"].values
        ).astype("U")

    else:
        raise KeyError(
            "Campaign test dataset must contain 'campaign'."
        )

    n = len(campaign_vals)

    data = {
        "_pred_row": np.arange(n, dtype=int),
        "campaign": campaign_vals,
        "Nd_obs": np.asarray(
            dsC["Nd_obs"].values,
            dtype=float
        ),
        "Nd_nn": np.asarray(
            dsC["Nd_nn"].values,
            dtype=float
        ),
        "p_merra": np.asarray(
            dsC["p_merra"].values,
            dtype=float
        ),
    }

    optional = [
        "Nd_base",
        "sigma_nn",
        "time_idx",
        "flight_id",
        "segment_gid",
        "campaign_id",
        "modis_ok",
    ]

    for name in optional:
        if name in dsC:
            data[name] = np.asarray(dsC[name].values)

    if "time" in dsC:
        data["time"] = np.asarray(dsC["time"].values)

    df = pd.DataFrame(data)

    return df


# ============================================================
# Attach observational sigma from campaign training files
# ============================================================

def _values_close(a, b, rtol=2.0e-5, atol=2.0e-4):
    """
    Compare two arrays while ignoring locations where both are NaN.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        return False

    m = np.isfinite(a) & np.isfinite(b)

    if not np.any(m):
        return False

    return np.allclose(
        a[m],
        b[m],
        rtol=rtol,
        atol=atol
    )


def _attach_sigma_obs(
    df,
    training_dir=DEFAULT_TRAINING_DIR,
):
    """
    Attach Nd_std from the original campaign training files.

    The returned column is called:

        sigma_obs

    Matching strategy
    -----------------
    1. Split prediction file by campaign.

    2. If prediction and training files contain the same number of
       segments, assume preserved segment order but VERIFY it using:
           Nd_obs   <-> Nd_mean
           p_merra  <-> p_merra

       sigma_obs is attached only if these checks pass.

    3. If the ordering check fails, try a secondary matching strategy
       using:
           normalized campaign day
           rounded Nd
           rounded pressure

    4. If matching is still incomplete, raise an error.

    This is deliberately strict. It is preferable to stop rather than
    silently assign the wrong observational standard deviation.
    """

    training_dir = Path(training_dir)

    out_parts = []

    campaigns = pd.unique(df["campaign"])

    for campaign in campaigns:

        pred = (
            df.loc[df["campaign"] == campaign]
            .copy()
            .reset_index(drop=True)
        )

        obs_file = (
            training_dir /
            f"{campaign}_segments_1deg_merraP_training.nc"
        )

        if not obs_file.exists():
            raise FileNotFoundError(
                f"Could not find training file for campaign "
                f"{campaign}:\n{obs_file}"
            )

        print(f"[READ] {obs_file}")

        with xr.open_dataset(
            obs_file,
            decode_times=True
        ) as dsO:

            required = [
                "Nd_mean",
                "Nd_std",
                "p_merra",
            ]

            for var in required:
                if var not in dsO:
                    raise KeyError(
                        f"{obs_file} does not contain {var}"
                    )

            obs = pd.DataFrame({
                "_obs_row": np.arange(
                    dsO.sizes["segment"],
                    dtype=int
                ),
                "Nd_mean": np.asarray(
                    dsO["Nd_mean"].values,
                    dtype=float
                ),
                "sigma_obs": np.asarray(
                    dsO["Nd_std"].values,
                    dtype=float
                ),
                "p_obsfile": np.asarray(
                    dsO["p_merra"].values,
                    dtype=float
                ),
            })

            if "time" in dsO:
                obs["segment_time"] = pd.to_datetime(
                    np.asarray(dsO["time"].values)
                )

            if "day" in dsO:
                obs["day_obs"] = pd.to_datetime(
                    np.asarray(dsO["day"].values)
                )

        # ----------------------------------------------------
        # Preferred case: same segment ordering
        # ----------------------------------------------------

        same_length = len(pred) == len(obs)

        nd_aligned = False
        p_aligned = False

        if same_length:

            nd_aligned = _values_close(
                pred["Nd_obs"].values,
                obs["Nd_mean"].values,
            )

            p_aligned = _values_close(
                pred["p_merra"].values,
                obs["p_obsfile"].values,
                rtol=2.0e-5,
                atol=2.0e-3,
            )

        if same_length and nd_aligned and p_aligned:

            pred["sigma_obs"] = obs["sigma_obs"].values
            pred["_obs_row"] = obs["_obs_row"].values

            print(
                f"[MATCH] {campaign}: "
                f"{len(pred)}/{len(pred)} segments matched "
                f"by verified segment order"
            )

            out_parts.append(pred)
            continue

        # ----------------------------------------------------
        # Secondary matching
        # ----------------------------------------------------

        print(
            f"[INFO] {campaign}: direct-order check failed; "
            f"trying value/day matching."
        )

        if "day_obs" not in obs.columns:
            raise RuntimeError(
                f"{campaign}: direct segment matching failed "
                f"and training file has no 'day' variable."
            )

        # Prediction time is day-resolution in NDnet_campaign_test.
        if "time" not in pred.columns:
            raise RuntimeError(
                f"{campaign}: prediction file has no time variable "
                f"for secondary matching."
            )

        pred["_day_key"] = pd.to_datetime(
            pred["time"]
        ).dt.normalize()

        obs["_day_key"] = pd.to_datetime(
            obs["day_obs"]
        ).dt.normalize()

        # Values originate from the same campaign file, so rounding
        # only protects against float32/float64 representation noise.
        pred["_nd_key"] = np.round(
            pred["Nd_obs"].values.astype(float),
            4
        )

        obs["_nd_key"] = np.round(
            obs["Nd_mean"].values.astype(float),
            4
        )

        pred["_p_key"] = np.round(
            pred["p_merra"].values.astype(float),
            3
        )

        obs["_p_key"] = np.round(
            obs["p_obsfile"].values.astype(float),
            3
        )

        # Protect against duplicate combinations.
        keys = [
            "_day_key",
            "_nd_key",
            "_p_key",
        ]

        pred["_dup"] = pred.groupby(
            keys,
            dropna=False,
            sort=False
        ).cumcount()

        obs["_dup"] = obs.groupby(
            keys,
            dropna=False,
            sort=False
        ).cumcount()

        merge_keys = keys + ["_dup"]

        obs_small = obs[
            merge_keys +
            [
                "_obs_row",
                "sigma_obs",
            ]
        ]

        pred = pred.merge(
            obs_small,
            on=merge_keys,
            how="left",
            validate="one_to_one",
        )

        nmatch = int(
            np.isfinite(pred["sigma_obs"]).sum()
        )

        print(
            f"[MATCH] {campaign}: "
            f"{nmatch}/{len(pred)} segments matched "
            f"by day/Nd/pressure"
        )

        if nmatch != len(pred):

            bad = pred.loc[
                ~np.isfinite(pred["sigma_obs"]),
                [
                    "Nd_obs",
                    "p_merra",
                    "_day_key",
                ],
            ]

            print()
            print("[UNMATCHED SEGMENTS]")
            print(bad.head(20).to_string(index=False))
            print()

            raise RuntimeError(
                f"{campaign}: only {nmatch}/{len(pred)} "
                f"segments could be matched to the training file. "
                f"Stopping to avoid incorrect sigma_obs assignment."
            )

        pred = pred.drop(
            columns=[
                "_day_key",
                "_nd_key",
                "_p_key",
                "_dup",
            ],
            errors="ignore",
        )

        out_parts.append(pred)

    result = pd.concat(
        out_parts,
        ignore_index=True
    )

    # Restore exact original prediction-file ordering.
    result = (
        result.sort_values("_pred_row")
        .reset_index(drop=True)
    )

    return result


# ============================================================
# Split DataFrame by campaign
# ============================================================

def _campaign_dataframe_list(df):
    """
    One DataFrame per campaign in first-occurrence order.
    """
    out = []

    for campaign in pd.unique(df["campaign"]):
        dfi = (
            df.loc[df["campaign"] == campaign]
            .copy()
            .reset_index(drop=True)
        )
        out.append(dfi)

    return out


# ============================================================
# Shared pressure normalization
# ============================================================

def _pressure_range(
    dfs,
    obs_col,
    model_col,
    min_compare=None,
    require_positive=False,
    use_modis_mask=False,
):
    p = []

    for df in dfs:

        sub = _campaign_subset(
            df,
            obs_col,
            model_col,
            min_compare=min_compare,
            require_positive=require_positive,
            use_modis_mask=use_modis_mask,
        )

        if not sub.empty:
            q = sub["p_merra"].to_numpy(dtype=float)
            q = q[np.isfinite(q)]

            if q.size:
                p.append(q)

    if not p:
        return 100.0, 1000.0

    p = np.concatenate(p)

    return float(np.nanmin(p)), float(np.nanmax(p))


# ============================================================
# Nd scatter figure
# ============================================================

def plot_campaign_nd_scatter(
    df,
    out_png,
    out_pdf=None,
    obs_col="Nd_obs",
    nn_col="Nd_nn",
    base_col="Nd_base",
    log_axes=True,
    min_nd_compare=1.0,
    use_modis_mask=False,
    xlim=ND_LIM,
    ylim=ND_LIM,
):
    """
    Publication-quality campaign Nd scatter figure.

    X: observed Nd
    Y: predicted Nd

    NN:
        filled circles, colored by MERRA pressure

    Base:
        open gray circles
    """

    dfs = _campaign_dataframe_list(df)

    n = len(dfs)

    if n == 0:
        print("[INFO] No campaigns found; skipping Nd scatter.")
        return

    # --------------------------------------------------------
    # Shared plot limits
    # --------------------------------------------------------

    values = []

    for dfi in dfs:

        sub_nn = _campaign_subset(
            dfi,
            obs_col,
            nn_col,
            min_compare=min_nd_compare,
            require_positive=log_axes,
            use_modis_mask=use_modis_mask,
        )

        if not sub_nn.empty:
            values.append(sub_nn[obs_col].values)
            values.append(sub_nn[nn_col].values)

        if base_col in dfi.columns:

            sub_b = _campaign_subset(
                dfi,
                obs_col,
                base_col,
                min_compare=min_nd_compare,
                require_positive=log_axes,
                use_modis_mask=use_modis_mask,
            )

            if not sub_b.empty:
                values.append(sub_b[base_col].values)

    lo, hi = _clean_limits(
        values,
        log_axes=log_axes
    )

    if xlim is None:
        xlo, xhi = lo, hi
    else:
        xlo, xhi = xlim

    if ylim is None:
        ylo, yhi = lo, hi
    else:
        ylo, yhi = ylim

    # --------------------------------------------------------
    # Shared pressure scale
    # --------------------------------------------------------

    pmin, pmax = _pressure_range(
        dfs,
        obs_col,
        nn_col,
        min_compare=min_nd_compare,
        require_positive=log_axes,
        use_modis_mask=use_modis_mask,
    )

    norm = Normalize(
        vmin=pmin,
        vmax=pmax
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    nrows, ncols = _campaign_layout_for_n(n)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            3.65 * ncols,
            3.25 * nrows
        ),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    axes = np.atleast_1d(axes).ravel()

    for i, (ax, dfi) in enumerate(
        zip(axes, dfs)
    ):

        campaign = str(
            dfi["campaign"].iloc[0]
        )

        # ----------------------------------------------------
        # Base
        # ----------------------------------------------------

        stats_text = []

        if base_col in dfi.columns:

            sub_b = _campaign_subset(
                dfi,
                obs_col,
                base_col,
                min_compare=min_nd_compare,
                require_positive=log_axes,
                use_modis_mask=use_modis_mask,
            )

            if not sub_b.empty:

                r_b, bias_b, rmse_b, n_b = (
                    _campaign_compute_stats(
                        sub_b[obs_col].values,
                        sub_b[base_col].values,
                    )
                )

                ax.scatter(
                    sub_b[obs_col].values,
                    sub_b[base_col].values,
                    s=24,
                    marker="o",
                    facecolors="none",
                    edgecolors="0.48",
                    linewidths=0.8,
                    alpha=0.75,
                    zorder=2,
                )

                stats_text.append(
                    "Base: "
                    rf"$r$={r_b:.2f}, "
                    rf"MB={bias_b:.1f}, "
                    rf"RMSE={rmse_b:.1f}"
                )

        # ----------------------------------------------------
        # NN
        # ----------------------------------------------------

        sub_nn = _campaign_subset(
            dfi,
            obs_col,
            nn_col,
            min_compare=min_nd_compare,
            require_positive=log_axes,
            use_modis_mask=use_modis_mask,
        )

        if not sub_nn.empty:

            r_n, bias_n, rmse_n, n_n = (
                _campaign_compute_stats(
                    sub_nn[obs_col].values,
                    sub_nn[nn_col].values,
                )
            )

            ax.scatter(
                sub_nn[obs_col].values,
                sub_nn[nn_col].values,
                c=sub_nn["p_merra"].values,
                cmap="viridis",
                norm=norm,
                s=27,
                marker="o",
                edgecolors="none",
                alpha=0.82,
                rasterized=True,
                zorder=3,
            )

            stats_text.insert(
                0,
                "NN: "
                rf"$r$={r_n:.2f}, "
                rf"MB={bias_n:.1f}, "
                rf"RMSE={rmse_n:.1f}"
            )

            stats_text.append(
                rf"$n$={n_n}"
            )

        else:
            stats_text.append(
                "No valid NN points"
            )

        # ----------------------------------------------------
        # 1:1 line
        # ----------------------------------------------------

        line_lo = max(xlo, ylo)
        line_hi = min(xhi, yhi)

        ax.plot(
            [line_lo, line_hi],
            [line_lo, line_hi],
            color="0.15",
            linestyle="--",
            linewidth=0.9,
            zorder=1,
        )

        # ----------------------------------------------------
        # Panel formatting
        # ----------------------------------------------------

        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)

        if log_axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
            _set_log_ticks(ax)

        # campaign + panel label
        ax.set_title(
            f"{_panel_label(i)}  {campaign}",
            loc="left",
            fontweight="bold",
            pad=5,
        )

        # statistics in panel
        ax.text(
            0.04,
            0.96,
            "\n".join(stats_text),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.6,
            linespacing=1.25,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="none",
                alpha=0.78,
            ),
            zorder=10,
        )

        ax.grid(
            True,
            which="major",
            linewidth=0.45,
            alpha=0.22,
        )

    # --------------------------------------------------------
    # Remove unused axes
    # --------------------------------------------------------

    for ax in axes[n:]:
        ax.remove()

    # --------------------------------------------------------
    # Only outer axis labels
    # --------------------------------------------------------

    for i, ax in enumerate(axes[:n]):

        row = i // ncols
        col = i % ncols

        if row == nrows - 1 or i + ncols >= n:
            ax.set_xlabel(
                r"Observed $N_d$ (cm$^{-3}$)"
            )
        else:
            ax.set_xlabel("")

        if col == 0:
            ax.set_ylabel(
                r"Predicted $N_d$ (cm$^{-3}$)"
            )
        else:
            ax.set_ylabel("")

    # --------------------------------------------------------
    # One shared legend
    # --------------------------------------------------------

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=5,
            markerfacecolor="0.35",
            markeredgecolor="none",
            label="NDnet",
        )
    ]

    if base_col in df.columns:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=5,
                markerfacecolor="none",
                markeredgecolor="0.48",
                label="Base",
            )
        )

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 1.015),
        handletextpad=0.4,
        columnspacing=1.2,
    )

    # --------------------------------------------------------
    # Shared pressure colorbar
    # --------------------------------------------------------

    sm = ScalarMappable(
        norm=norm,
        cmap="viridis"
    )

    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes[:n].tolist(),
        fraction=0.025,
        pad=0.025,
        aspect=30,
    )

    cbar.set_label(
        "Pressure (hPa)"
    )

    # physically intuitive:
    # lower pressure near top of colorbar
    cbar.ax.invert_yaxis()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    fig.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"[WROTE] {out_png}")

    if out_pdf is not None:
        fig.savefig(
            out_pdf,
            bbox_inches="tight"
        )
        print(f"[WROTE] {out_pdf}")

    plt.close(fig)


# ============================================================
# Sigma scatter figure
# ============================================================

def plot_campaign_sigma_scatter(
    df,
    out_png,
    out_pdf=None,
    obs_col="sigma_obs",
    nn_col="sigma_nn",
    log_axes=True,
    min_sigma_compare=None,
    use_modis_mask=False,
    xlim=SIGMA_LIM,
    ylim=SIGMA_LIM,
):
    """
    Publication-quality sigma comparison.

    X:
        observed segment spread
        sigma_obs = Nd_std

    Y:
        NN-predicted sigma
        sigma_nn

    No base points are shown because the base model does not
    provide sigma.
    """

    if obs_col not in df.columns:
        raise KeyError(
            f"{obs_col} is missing from DataFrame."
        )

    if nn_col not in df.columns:
        raise KeyError(
            f"{nn_col} is missing from campaign test file."
        )

    dfs = _campaign_dataframe_list(df)

    n = len(dfs)

    if n == 0:
        print(
            "[INFO] No campaigns found; "
            "skipping sigma scatter."
        )
        return

    # --------------------------------------------------------
    # Shared limits
    # --------------------------------------------------------

    values = []

    for dfi in dfs:

        sub = _campaign_subset(
            dfi,
            obs_col,
            nn_col,
            min_compare=min_sigma_compare,
            require_positive=log_axes,
            use_modis_mask=use_modis_mask,
        )

        if not sub.empty:
            values.append(
                sub[obs_col].values
            )
            values.append(
                sub[nn_col].values
            )

    lo, hi = _clean_limits(
        values,
        log_axes=log_axes
    )

    if xlim is None:
        xlo, xhi = lo, hi
    else:
        xlo, xhi = xlim

    if ylim is None:
        ylo, yhi = lo, hi
    else:
        ylo, yhi = ylim

    # --------------------------------------------------------
    # Pressure range
    # --------------------------------------------------------

    pmin, pmax = _pressure_range(
        dfs,
        obs_col,
        nn_col,
        min_compare=min_sigma_compare,
        require_positive=log_axes,
        use_modis_mask=use_modis_mask,
    )

    norm = Normalize(
        vmin=pmin,
        vmax=pmax
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    nrows, ncols = _campaign_layout_for_n(n)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            3.65 * ncols,
            3.25 * nrows
        ),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    axes = np.atleast_1d(axes).ravel()

    for i, (ax, dfi) in enumerate(
        zip(axes, dfs)
    ):

        campaign = str(
            dfi["campaign"].iloc[0]
        )

        sub = _campaign_subset(
            dfi,
            obs_col,
            nn_col,
            min_compare=min_sigma_compare,
            require_positive=log_axes,
            use_modis_mask=use_modis_mask,
        )

        if not sub.empty:

            r, bias, rmse, npts = (
                _campaign_compute_stats(
                    sub[obs_col].values,
                    sub[nn_col].values,
                )
            )

            ax.scatter(
                sub[obs_col].values,
                sub[nn_col].values,
                c=sub["p_merra"].values,
                cmap="viridis",
                norm=norm,
                s=28,
                edgecolors="none",
                alpha=0.82,
                rasterized=True,
                zorder=3,
            )

            stat_string = (
                rf"$r$={r:.2f}"
                "\n"
                rf"MB={bias:.1f} cm$^{{-3}}$"
                "\n"
                rf"RMSE={rmse:.1f} cm$^{{-3}}$"
                "\n"
                rf"$n$={npts}"
            )

        else:
            stat_string = "No valid points"

        # ----------------------------------------------------
        # 1:1
        # ----------------------------------------------------

        line_lo = max(xlo, ylo)
        line_hi = min(xhi, yhi)

        ax.plot(
            [line_lo, line_hi],
            [line_lo, line_hi],
            color="0.15",
            linestyle="--",
            linewidth=0.9,
            zorder=1,
        )

        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)

        if log_axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
            _set_log_ticks(ax)

        ax.set_title(
            f"{_panel_label(i)}  {campaign}",
            loc="left",
            fontweight="bold",
            pad=5,
        )

        ax.text(
            0.04,
            0.96,
            stat_string,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.8,
            linespacing=1.25,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="none",
                alpha=0.78,
            ),
            zorder=10,
        )

        ax.grid(
            True,
            which="major",
            linewidth=0.45,
            alpha=0.22,
        )

    # --------------------------------------------------------
    # Remove unused panels
    # --------------------------------------------------------

    for ax in axes[n:]:
        ax.remove()

    # --------------------------------------------------------
    # Outer labels only
    # --------------------------------------------------------

    for i, ax in enumerate(axes[:n]):

        row = i // ncols
        col = i % ncols

        if row == nrows - 1 or i + ncols >= n:
            ax.set_xlabel(
                r"Observed $\sigma_{N_d}$ "
                r"(cm$^{-3}$)"
            )
        else:
            ax.set_xlabel("")

        if col == 0:
            ax.set_ylabel(
                r"Predicted $\sigma_{N_d}$ "
                r"(cm$^{-3}$)"
            )
        else:
            ax.set_ylabel("")

    # --------------------------------------------------------
    # Shared pressure colorbar
    # --------------------------------------------------------

    sm = ScalarMappable(
        norm=norm,
        cmap="viridis"
    )

    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes[:n].tolist(),
        fraction=0.025,
        pad=0.025,
        aspect=30,
    )

    cbar.set_label(
        "Pressure (hPa)"
    )

    cbar.ax.invert_yaxis()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    fig.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"[WROTE] {out_png}")

    if out_pdf is not None:
        fig.savefig(
            out_pdf,
            bbox_inches="tight"
        )
        print(f"[WROTE] {out_pdf}")

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Plot publication-quality NDnet campaign "
            "Nd and sigma diagnostics."
        )
    )

    parser.add_argument(
        "experiment_dir",
        type=Path,
        help=(
            "Experiment directory containing "
            "NDnet_campaign_test.nc"
        ),
    )

    parser.add_argument(
        "--test-file",
        default=DEFAULT_TEST_FILE,
        help=(
            "Campaign test NetCDF filename "
            f"(default: {DEFAULT_TEST_FILE})"
        ),
    )

    parser.add_argument(
        "--training-dir",
        type=Path,
        default=DEFAULT_TRAINING_DIR,
        help=(
            "Directory containing "
            "*_segments_1deg_merraP_training.nc"
        ),
    )

    parser.add_argument(
        "--min-nd",
        type=float,
        default=1.0,
        help=(
            "Minimum observed Nd used in Nd comparison "
            "(default: 1 cm-3)"
        ),
    )

    parser.add_argument(
        "--linear-nd",
        action="store_true",
        help="Use linear axes for Nd figure.",
    )

    parser.add_argument(
        "--linear-sigma",
        action="store_true",
        help="Use linear axes for sigma figure.",
    )

    parser.add_argument(
        "--modis-mask",
        action="store_true",
        help=(
            "Require modis_ok == 1 if modis_ok exists."
        ),
    )

    args = parser.parse_args()

    exp_dir = args.experiment_dir.expanduser().resolve()

    test_path = exp_dir / args.test_file

    if not test_path.exists():
        raise FileNotFoundError(
            f"Campaign test file not found:\n{test_path}"
        )

    print(f"[READ] {test_path}")

    with xr.open_dataset(
        test_path,
        decode_times=True
    ) as dsC:

        df = _campaign_ds_to_dataframe(
            dsC
        )

    print(
        f"[INFO] Prediction file contains "
        f"{len(df)} segments across "
        f"{df['campaign'].nunique()} campaigns."
    )

    # --------------------------------------------------------
    # Add observed sigma = Nd_std
    # --------------------------------------------------------

    df = _attach_sigma_obs(
        df,
        training_dir=args.training_dir,
    )

    n_sigma = int(
        np.isfinite(df["sigma_obs"]).sum()
    )

    print(
        f"[INFO] Attached sigma_obs to "
        f"{n_sigma}/{len(df)} segments."
    )

    # --------------------------------------------------------
    # Nd figure
    # --------------------------------------------------------

    nd_png = (
        "NDnet_campaign_Nd_scatter.png"
    )

    nd_pdf = (
        "NDnet_campaign_Nd_scatter.pdf"
    )

    plot_campaign_nd_scatter(
        df,
        out_png=nd_png,
        out_pdf=nd_pdf,
        log_axes=not args.linear_nd,
        min_nd_compare=args.min_nd,
        use_modis_mask=args.modis_mask,
    )

    # --------------------------------------------------------
    # Sigma figure
    # --------------------------------------------------------

    sigma_png = (
        "NDnet_campaign_sigma_scatter.png"
    )

    sigma_pdf = (
        "NDnet_campaign_sigma_scatter.pdf"
    )

    plot_campaign_sigma_scatter(
        df,
        out_png=sigma_png,
        out_pdf=sigma_pdf,
        log_axes=not args.linear_sigma,
        min_sigma_compare=None,
        use_modis_mask=args.modis_mask,
    )


if __name__ == "__main__":
    main()
