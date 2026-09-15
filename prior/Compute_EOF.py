import xarray as xr
import numpy as np
from sklearn.decomposition import PCA
import xesmf as xe

# -----------------------------------------
# User params
# -----------------------------------------
minP = 250
K = 12
outfile = "Nd_EOF.nc"

pth_all = '/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/dailies/mkincloudNd/'
prefix = '*20[0-1][0-3]*'
varname = "NDinc_cf01p"

# -----------------------------------------
# Load data
# -----------------------------------------
pth = f'{pth_all}{prefix}'
print("Loading:", pth)

N = xr.open_mfdataset(pth, parallel=True)
N = N.sel(lev=N.lev.where(N.lev >= minP, drop=True))

Nd = N[varname]
Nd["time"] = Nd["time"].dt.floor("D")
Nd = Nd.where(Nd >= 0.0)

print(Nd)

# ============================================================
# SAFE PER-LEVEL STD
# ============================================================
def safe_level_std(X, smooth_win=5, floor_frac=0.15):
    """
    X: (Nsamples, Nz) centered anomalies
    Returns σ̂(p) with vertical smoothing + global floor
    """
    sigma = np.nanstd(X, axis=0)

    # Smooth in log-space for vertical stability
    log_sigma = np.log(np.maximum(sigma, 1e-12))
    kernel = np.ones(smooth_win) / smooth_win
    log_sigma_smooth = np.convolve(log_sigma, kernel, mode="same")
    sigma_smooth = np.exp(log_sigma_smooth)

    # Global floor to prevent σ → 0 aloft
    floor = floor_frac * np.nanmedian(sigma_smooth)
    sigma_safe = np.maximum(sigma_smooth, floor)

    return sigma_safe.astype("float32")

# ============================================================
# GLOBAL EOF COMPUTATION
# ============================================================

# -------------------------------------------------
# 1. Physical climatological mean μ(p)
# -------------------------------------------------
xr_mean = Nd.mean(dim=("time", "lat", "lon"), skipna=True)
mean_vec = xr_mean.values.astype("float32")

print("[GLOBAL] physical mean:", mean_vec)

# -------------------------------------------------
# 2. Build sample × level matrix
# -------------------------------------------------
Nd2 = Nd.stack(sample=("time", "lat", "lon"))     # (lev, sample)
Nd_mat = Nd2.transpose("sample", "lev").values    # (sample, lev)

# keep samples with at least one valid level
valid = ~np.all(np.isnan(Nd_mat), axis=1)
Nd_mat = Nd_mat[valid]

# fill NaNs with physical mean (EOF convention)
Nd_filled = np.where(np.isnan(Nd_mat), mean_vec, Nd_mat)

# -------------------------------------------------
# 3. Center
# -------------------------------------------------
Nd_centered = Nd_filled - mean_vec[None, :]       # (Ns, Nz)

# -------------------------------------------------
# 4. SAFE per-level normalization
# -------------------------------------------------
sigma_vec = safe_level_std(
    Nd_centered,
    smooth_win=5,
    floor_frac=0.15
)

print("[GLOBAL] σ min / max:",
      sigma_vec.min(), sigma_vec.max())

Nd_norm = Nd_centered / sigma_vec[None, :]

# -------------------------------------------------
# 5. PCA on normalized anomalies
# -------------------------------------------------
print(f"[GLOBAL] fitting PCA on {Nd_norm.shape[0]} samples")

pca = PCA(n_components=K)
pca.fit(Nd_norm)

EOFs_norm = pca.components_.astype("float32")     # (K, Nz)
expl_var = pca.explained_variance_ratio_.astype("float32")

# -------------------------------------------------
# 6. Convert EOFs back to physical units
# -------------------------------------------------
EOFs_phys = EOFs_norm * sigma_vec[None, :]

# ============================================================
# SAVE OUTPUT (FULL PHYSICAL RECONSTRUCTION METADATA)
# ============================================================

ds_out = xr.Dataset(
    {
        "EOFs":        (("mode", "lev"), EOFs_phys),
        "Nd_mean":     (("lev",), mean_vec),
        "Nd_sigma":    (("lev",), sigma_vec),
        "expl_var":    (("mode",), expl_var),
    },
    coords={
        "mode": np.arange(K),
        "lev": Nd.lev.values,
    }
)

ds_out.to_netcdf(outfile)
print("Saved:", outfile)

