#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt


# -----------------------------
# Input
# -----------------------------
ncfile = Path("/panfs/ccds02/nobackup/people/dbarahon/NDnet/EOF/in_cloudM2/Nd_EOF.nc")
nplot = 11   # number of EOF modes to plot

# -----------------------------
# Plot style
# -----------------------------
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

# -----------------------------
# Read data
# -----------------------------
ds = xr.open_dataset(ncfile)

lev = np.asarray(ds["lev"].values)
nd_mean = np.asarray(ds["Nd_mean"].values)
expl_var = np.asarray(ds["expl_var"].values)
eofs = np.asarray(ds["EOFs"].values)   # (mode, lev)
modes = np.asarray(ds["mode"].values)

# Convert explained variance to percent if needed
if np.nanmax(expl_var) <= 1.0:
    expl_var_plot = 100.0 * expl_var
    ev_label = "Explained variance (%)"
else:
    expl_var_plot = expl_var
    ev_label = "Explained variance"

# Sort pressure consistently
isort = np.argsort(lev)
lev = lev[isort]
nd_mean = nd_mean[isort]
eofs = eofs[:, isort]

# Choose readable pressure ticks
pressure_ticks = lev
if len(lev) > 10:
    idx = np.linspace(0, len(lev) - 1, 8).round().astype(int)
    idx = np.unique(idx)
    pressure_ticks = lev[idx]

# -----------------------------
# Figure
# -----------------------------
fig = plt.figure(figsize=(14, 5.5), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 0.9])

ax0 = fig.add_subplot(gs[0, 0])               # mean Nd
ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)   # EOFs in middle
ax2 = fig.add_subplot(gs[0, 2])               # explained variance

# -----------------------------
# Panel 1: mean Nd
# -----------------------------
ax0.plot(nd_mean, lev, lw=2.2)
ax0.set_xlabel(r"Mean $N_d$ (cm$^{-3})$")
ax0.set_ylabel("Pressure (hPa)")
ax0.set_title(r"(a)", loc="left")
ax0.grid(True, alpha=0.25, linewidth=0.6)
ax0.set_yticks(pressure_ticks)

# -----------------------------
# Panel 2: EOFs
# -----------------------------

mode_idx = range(11) #[0, 5, 11]

for i in mode_idx:
    ax1.plot(eofs[i, :], lev, label=f"{int(modes[i])}")
    

ax1.set_xlabel(r"EOF amplitude (cm$^{-3})$")
ax1.set_ylabel("Pressure (hPa)")
ax1.set_title("(b)", loc="left")
ax1.grid(True, alpha=0.25, linewidth=0.6)
ax1.set_yticks(pressure_ticks)
ax1.tick_params(labelleft=True)

ax1.legend(
    title="Mode",
    loc="upper right",
    frameon=True,
    borderpad=0.6,
    handlelength=2.0,
)

# invert shared pressure axis ONCE
ax0.set_ylim(np.nanmax(lev), np.nanmin(lev))

# show pressure labels on the EOF panel too
ax1.tick_params(labelleft=True)

ax1.legend(
    title="Mode",
    loc="upper right",
    frameon=True,
    borderpad=0.6,
    handlelength=2.0,
)

# -----------------------------
# Panel 3: explained variance
# -----------------------------
nplot=12
ax2.plot(expl_var_plot[:nplot], modes[:nplot], marker="o", lw=1.8)
ax2.set_xlabel(ev_label)
ax2.set_ylabel("Mode")
ax2.set_title("(c)", loc="left")
ax2.grid(True, alpha=0.25, linewidth=0.6)
ax2.set_yticks(modes[:nplot])
#ax2.invert_yaxis()

#fig.suptitle("GiOcean EOF summary", y=1.02)

# Save
outpng = "Nd_EOF_summary.png"
outpdf = "Nd_EOF_summary.pdf"
fig.savefig(outpng, bbox_inches="tight")
fig.savefig(outpdf, bbox_inches="tight")
plt.show()

print(f"Saved: {outpng}")
print(f"Saved: {outpdf}")
