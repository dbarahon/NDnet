# ============================================================
# construct_v1.py
#
# v1 uncertainty-control changes:
#   1) mode-weighted fixed-site sigma regularization
#   2) explicit hierarchical campaign sigma regularization
#   3) optional detach of sigma branch from shared MODIS features
# ============================================================

import numpy as np
import xarray as xr
import dask
import dask.array as da
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset



from torch.utils.data import Dataset
import numpy as np
import torch

from torch.utils.data import Dataset
import numpy as np

class EOFNdDatasetSitesTorch(Dataset):
    """
    Indexable dataset over time indices.

    Returns (per sample):
      x2d  : [C,H,W]
      mu0  : [M,H,W]
      ys   : [M,S]
      vs   : [S] (bool)
      nds  : [Z,S]
      Eds  : [Z,S]
    """
    def __init__(self, X2d_t, MU0_t, YS_t, VS_t, Nds_t, Eds_t, time_idx):
        self.X2d = X2d_t
        self.MU0 = MU0_t
        self.YS  = YS_t
        self.VS  = VS_t
        self.Nds = Nds_t
        self.Eds = Eds_t
        self.time_idx = np.asarray(time_idx, dtype=np.int64)

        # hard sanity checks (fail fast instead of silent mixing)
        assert self.X2d.ndim == 4, f"X2d_t must be (T,C,H,W), got {tuple(self.X2d.shape)}"
        assert self.MU0.ndim == 4, f"MU0_t must be (T,M,H,W), got {tuple(self.MU0.shape)}"
        assert self.YS.ndim  == 3, f"YS_t must be (T,M,S), got {tuple(self.YS.shape)}"
        assert self.VS.ndim  == 2, f"VS_t must be (T,S), got {tuple(self.VS.shape)}"
        assert self.Nds.ndim == 3, f"Nds_t must be (T,Z,S), got {tuple(self.Nds.shape)}"
        assert self.Eds.ndim == 3, f"Eds_t must be (T,Z,S), got {tuple(self.Nds.shape)}"

    def __len__(self):
        return self.time_idx.size

    def __getitem__(self, i):
        t = int(self.time_idx[i])
        return (
            self.X2d[t],   # [C,H,W]
            self.MU0[t],   # [M,H,W]
            self.YS[t],    # [M,S]
            self.VS[t],    # [S]
            self.Nds[t],   # [Z,S]
            self.Eds[t],   # [Z,S]
        )



# ============================================================
# Model: refinement heads operating on cached base coeffs mu0
# ============================================================

class ProbRefineFromBase(nn.Module):
    """
    Input:  mu0   [B, M, H, W]  (cached base residual coeffs)
    Output: mu    [B, M, H, W]  (corrected)
            sigma [B, M, H, W]  (uncertainty)
    """
    def __init__(self, num_modes: int, head_ch: int = 32, sigma_min: float = 1e-3):
        super().__init__()
        self.sigma_min = float(sigma_min)

        self.delta_head = nn.Sequential(
            nn.Conv2d(num_modes, head_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_ch, head_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_ch, num_modes, 1),
        )
        self.sigma_head = nn.Sequential(
            nn.Conv2d(num_modes, head_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_ch, head_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_ch, num_modes, 1),
        )

    def forward(self, mu0):
        delta = self.delta_head(mu0)
        mu = mu0 + delta
        rho = self.sigma_head(mu0)
        sigma = F.softplus(rho) + self.sigma_min
        return mu, sigma


class ProbRefineFromBaseAndX(nn.Module):
    """
    Inputs:
      x2d : [B, Cin, H, W]   (MODIS)
      mu0 : [B, M,   H, W]   (cached base residual coeffs)

    Outputs:
      mu    : [B, M, H, W]
      sigma : [B, M, H, W]

    v2 changes:
      detach_sigma_features=True prevents sigma-loss gradients from flowing
      back into the shared MODIS encoder. The sigma head remains trainable.
    """
    def __init__(
        self,
        in_channels,
        num_modes,
        feat_ch=32,
        head_ch=64,
        sigma_min=1e-3,
        detach_sigma_features=True,
    ):
        super().__init__()
        self.sigma_min = float(sigma_min)
        self.detach_sigma_features = bool(detach_sigma_features)

        # Lightweight encoder for x2d (FAST)
        self.x_enc = nn.Sequential(
            nn.Conv2d(in_channels, feat_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Fuse (concat) then predict delta and sigma
        fuse_in = num_modes + feat_ch

        self.delta_head = nn.Sequential(
            nn.Conv2d(fuse_in, head_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_ch, head_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_ch, num_modes, 1),
        )

        self.sigma_head = nn.Sequential(
            nn.Conv2d(fuse_in, head_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_ch, head_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_ch, num_modes, 1),
        )

    def forward(self, x2d, mu0):
        fx = self.x_enc(x2d)                # [B, feat_ch, H, W]
        h  = torch.cat([mu0, fx], dim=1)    # [B, M+feat_ch, H, W]

        # Mean branch uses the shared representation normally.
        delta = self.delta_head(h)
        mu = mu0 + delta

        # Uncertainty branch can be isolated from the shared encoder.
        # Detaching h does NOT freeze sigma_head; it only blocks gradients
        # from sigma losses into x_enc (and the input mu0 pathway).
        h_sigma = h.detach() if self.detach_sigma_features else h
        rho = self.sigma_head(h_sigma)
        sigma = F.softplus(rho) + self.sigma_min
        return mu, sigma

########################LOSS##########################

class SparseSiteCampaignNLLRefineLoss(nn.Module):
    """
    Joint loss for:
      - fixed sites (existing path)
      - moving campaign segments sampled from the same dense dA/sigma fields

    Expected campaign tensors (all flattened across the batch):
      seg_batch_idx   [Nseg] long
      seg_ilat        [Nseg] long
      seg_ilon        [Nseg] long
      seg_p           [Nseg] float32   pressure in hPa
      seg_y           [Nseg] float32   training target (Nd_mean or one Nd_obs_aug)
      seg_y_std       [Nseg] float32   observed Nd_std
      seg_valid_obs   [Nseg] bool
      seg_campaign_id [Nseg] long
      seg_group_id    [Nseg] long      use flightID for now; later trajectory_id if available
    """
    def __init__(
        self,
        lat_grid,
        site_meta,
        mode_weights,
        EOF,
        Nd_mean,
        A_mean,
        lambda_grad=0.001,
        lambda_anchor=0.01,
        lambda_sigma=0.01,
        lambda_campaign_sigma=0.01,
        sigma_mode_var_floor=0.05,
        lambda_sites=100.0,
        lambda_nd=1.0,
        lambda_Ndgauss=0.1,
        lambda_campaign=1.0,
        lambda_campaign_gauss=1.0,
        min_sigma=1e-3,
        min_valid_points=1,
        normalize_site_loss=True,
        normalize_nd_loss=True,
        nd_require_vS=True,
        err_mix=0.25,
        err_power=1.0,
        err_w_clip=(0.5, 1.5),
        campaign_obs_std_floor=1e-3,
        max_abs_coeff=500.0,
        max_abs_nd=5000.0,
        max_abs_err=5000.0,
        max_sigma_stable=1000.0,
        max_z=100.0,
    ):
        super().__init__()

        # -------------------------
        # fixed-site metadata
        # -------------------------
        ilat = getattr(site_meta["ilat"], "values", site_meta["ilat"])
        ilon = getattr(site_meta["ilon"], "values", site_meta["ilon"])
        self.register_buffer("ilat", torch.as_tensor(np.asarray(ilat, dtype=np.int64), dtype=torch.long))
        self.register_buffer("ilon", torch.as_tensor(np.asarray(ilon, dtype=np.int64), dtype=torch.long))

        # -------------------------
        # latitude and mode weights
        # -------------------------
        lat_grid = torch.as_tensor(lat_grid, dtype=torch.float32)
        lat_w_grid = torch.cos(torch.deg2rad(lat_grid))
        lat_w_grid = torch.nan_to_num(lat_w_grid, nan=0.0, posinf=0.0, neginf=0.0)
        self.register_buffer("lat_w_grid", lat_w_grid / lat_w_grid.mean().clamp(min=1e-12))

        mw = torch.as_tensor(mode_weights, dtype=torch.float32)
        mode_w = mw / mw.mean().clamp(min=1e-12)

        # Preserve the original EOF-mode weighting in all existing likelihoods.
        self.register_buffer("mode_w", mode_w)

        # Sigma regularization uses the same supplied EOF-derived quantity as
        # a RELATIVE mode variance: sigma_m^2 / V_m(rel).  Only this sigma
        # normalization gets a floor, so tiny high-order EOF energies cannot
        # create an arbitrarily large regularization gradient.
        self.sigma_mode_var_floor = float(sigma_mode_var_floor)
        mode_var_rel = torch.clamp(mode_w, min=self.sigma_mode_var_floor)
        self.register_buffer("mode_var_rel", mode_var_rel)
        self.register_buffer("mode_std_rel", torch.sqrt(mode_var_rel))

        # -------------------------
        # EOF / Nd / Amean
        # -------------------------
        EOF_mz = torch.as_tensor(getattr(EOF, "values", EOF), dtype=torch.float32)   # [M,Z]
        Nd_mean_z = torch.as_tensor(getattr(Nd_mean, "values", Nd_mean), dtype=torch.float32)
        Amean_mhw = torch.as_tensor(getattr(A_mean, "values", A_mean), dtype=torch.float32)

        self.register_buffer("EOF_mz", EOF_mz)
        self.register_buffer("EOF2_mz", EOF_mz * EOF_mz)
        self.register_buffer("Nd_mean_z", Nd_mean_z)
        self.register_buffer("Amean_mhw", Amean_mhw)
        self.register_buffer("Amean_hwm", Amean_mhw.permute(1, 2, 0).contiguous())  # [H,W,M]

        # pressure coordinate from EOF lev
        lev_vals = getattr(getattr(EOF, "lev", None), "values", None)
        if lev_vals is None:
            raise ValueError("EOF must carry a lev coordinate for campaign pressure interpolation.")
        lev_vals = np.asarray(lev_vals, dtype=np.float32)
        if lev_vals.ndim != 1 or lev_vals.size < 2:
            raise ValueError(f"EOF lev must be 1D with at least 2 levels. Got shape={lev_vals.shape}")

        if np.any(~np.isfinite(lev_vals)) or np.any(lev_vals <= 0):
            raise ValueError("EOF lev contains non-finite or non-positive values; cannot use log-pressure interpolation.")

        z_sort_idx = np.argsort(np.log(lev_vals))
        lev_log_asc = np.log(lev_vals[z_sort_idx]).astype(np.float32)

        self.register_buffer("z_sort_idx", torch.as_tensor(z_sort_idx, dtype=torch.long))
        self.register_buffer("lev_log_asc", torch.as_tensor(lev_log_asc, dtype=torch.float32))

        # -------------------------
        # lambdas and knobs
        # -------------------------
        self.lambda_grad = float(lambda_grad)
        self.lambda_anchor = float(lambda_anchor)
        self.lambda_sigma = float(lambda_sigma)
        self.lambda_campaign_sigma = float(lambda_campaign_sigma)
        self.lambda_sites = float(lambda_sites)
        self.lambda_nd = float(lambda_nd)
        self.lambda_Ndgauss = float(lambda_Ndgauss)
        self.lambda_campaign = float(lambda_campaign)
        self.lambda_campaign_gauss = float(lambda_campaign_gauss)

        self.sigma_min = float(min_sigma)
        self.min_valid_points = int(min_valid_points)
        self.normalize_site_loss = bool(normalize_site_loss)
        self.normalize_nd_loss = bool(normalize_nd_loss)
        self.nd_require_vS = bool(nd_require_vS)

        self.err_mix = float(err_mix)
        self.err_power = float(err_power)
        self.err_w_min = float(err_w_clip[0])
        self.err_w_max = float(err_w_clip[1])

        self.campaign_obs_std_floor = float(campaign_obs_std_floor)

        self.max_abs_coeff = float(max_abs_coeff)
        self.max_abs_nd = float(max_abs_nd)
        self.max_abs_err = float(max_abs_err)
        self.max_sigma_stable = float(max_sigma_stable)
        self.max_z = float(max_z)

    # =========================================================
    # helpers
    # =========================================================
    @staticmethod
    def _nan0(x):
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _f32(self, x):
        return self._nan0(x).float()

    def _zero_like_sum(self, x):
        return x.sum() * 0.0

    def _clip_abs(self, x, vmax):
        vmax = float(vmax)
        return torch.clamp(x, min=-vmax, max=vmax)

    def _safe_sigma(self, sig):
        sig = self._f32(sig)
        return torch.clamp(sig, min=self.sigma_min)

    def _safe_sigma_stable(self, sig):
        sig = self._safe_sigma(sig)
        return torch.clamp(sig, min=self.sigma_min, max=self.max_sigma_stable)

    def _count_finite_global(self, x):
        return torch.isfinite(x).sum().to(dtype=x.dtype).clamp(min=1.0)

    def _has_obs(self, x):
        return (x is not None) and isinstance(x, torch.Tensor) and (x.numel() > 0)

    def _nd_error_weights(self, Eds, finite):
        if (Eds is None) or (self.err_mix <= 0.0):
            return None

        E = self._f32(Eds).abs()
        E = torch.clamp(E, min=0.0, max=self.max_abs_err)
        valid = finite
        if int(valid.sum().detach().cpu().item()) < 1:
            return None

        E_ref = torch.median(E[valid]).clamp(min=1e-6)
        w_raw = 1.0 / (1.0 + (E / E_ref) ** self.err_power)
        w_mean = w_raw[valid].mean().clamp(min=1e-6)
        w = w_raw / w_mean
        w = torch.clamp(w, self.err_w_min, self.err_w_max)
        w = torch.where(finite, w, torch.zeros_like(w))
        return w

    # =========================================================
    # dense regularizers
    # =========================================================
    def weighted_mse_dense_normed(self, x):
        x = self._clip_abs(self._f32(x), self.max_abs_coeff)
        lat_w = self.lat_w_grid.view(1, 1, -1, 1).float()
        mode_w = self.mode_w.view(1, -1, 1, 1).float()
        w = lat_w * mode_w

        num = (w * (x ** 2)).sum()
        den = self._count_finite_global(x)
        return num / den

    def grad_l1_normed(self, x):
        x = self._clip_abs(self._f32(x), self.max_abs_coeff)
        dx = (x[..., :, 1:] - x[..., :, :-1]).abs().sum()
        dy = (x[..., 1:, :] - x[..., :-1, :]).abs().sum()
        den = self._count_finite_global(x)
        return (dx + dy) / den

    # =========================================================
    # site branch (same logic as before)
    # =========================================================
    def site_gaussian_nll_equal_sites(self, pred_s, sig_s, yS, vS):
        vS = vS.bool()
        finite = torch.isfinite(yS) & torch.isfinite(pred_s) & torch.isfinite(sig_s)
        finite = finite & vS[:, None, :]

        cnt_s = finite.sum(dim=(0, 1))
        valid_site = cnt_s >= self.min_valid_points
        if int(valid_site.sum().detach().cpu().item()) < 1:
            return None

        pred0 = self._clip_abs(self._f32(pred_s), self.max_abs_coeff)
        y0 = self._clip_abs(self._f32(yS), self.max_abs_coeff)
        sig = self._safe_sigma_stable(sig_s)

        diff = torch.clamp(pred0 - y0, min=-2.0 * self.max_abs_coeff, max=2.0 * self.max_abs_coeff)
        z = torch.clamp(diff / sig, -self.max_z, self.max_z)

        nll = torch.log(sig) + 0.5 * (z * z)
        nll = torch.where(finite, nll, torch.zeros_like(nll))

        mode_w = self.mode_w.view(1, -1, 1).float()
        num_s = (mode_w * nll).sum(dim=(0, 1))

        if self.normalize_site_loss:
            den_s = (mode_w * finite.float()).sum(dim=(0, 1)).clamp(min=1.0)
            loss_s = num_s / den_s
        else:
            loss_s = num_s

        return loss_s[valid_site].mean()

    def masked_sigma_mean_equal_sites(self, sig_s, vS):
        """
        Variance-relative sigma penalty at fixed sites.

        The EOF-derived mode_weights passed to this loss are interpreted as
        relative mode variances V_m / mean(V).  The regularizer is therefore

            L_sigma = mean[ sigma_m^2 / V_m(rel) ]

        over valid batch/mode entries, followed by equal weighting across
        sites.  This penalizes uncertainty relative to the characteristic
        variance of each EOF mode instead of penalizing absolute sigma.

        sig_s : [B,M,S]
        vS    : [B,S]
        """
        vS = vS.bool()
        finite = torch.isfinite(sig_s) & vS[:, None, :]
        cnt_s = finite.sum(dim=(0, 1))
        valid_site = cnt_s >= 1

        if int(valid_site.sum().detach().cpu().item()) < 1:
            return self._zero_like_sum(sig_s)

        sig = self._safe_sigma_stable(sig_s)
        mode_var = self.mode_var_rel.view(1, -1, 1).float()

        rel_var = (sig * sig) / mode_var
        rel_var = torch.where(finite, rel_var, torch.zeros_like(rel_var))

        num_s = rel_var.sum(dim=(0, 1))
        den_s = finite.float().sum(dim=(0, 1)).clamp(min=1.0)
        loss_s = num_s / den_s

        # Equal weighting between sites.
        return loss_s[valid_site].mean()

    def masked_sigma_raw_mean_equal_sites(self, sig_s, vS):
        """Raw mean sigma diagnostic only; not used in the loss."""
        vS = vS.bool()
        finite = torch.isfinite(sig_s) & vS[:, None, :]
        cnt_s = finite.sum(dim=(0, 1))
        valid_site = cnt_s >= 1
        if int(valid_site.sum().detach().cpu().item()) < 1:
            return self._zero_like_sum(sig_s)

        sig = self._safe_sigma_stable(sig_s)
        sig = torch.where(finite, sig, torch.zeros_like(sig))
        num_s = sig.sum(dim=(0, 1))
        den_s = cnt_s.to(num_s.dtype).clamp(min=1.0)
        return (num_s / den_s)[valid_site].mean()

    def predict_Nd_sites_from_abs_coeffs(self, A_abs_s):
        A = self._clip_abs(self._f32(A_abs_s), self.max_abs_coeff)
        Nd_pred = torch.einsum("bms,mz->bzs", A, self.EOF_mz.float()) + self.Nd_mean_z.view(1, -1, 1)
        Nd_pred = self._f32(Nd_pred)
        return torch.clamp(Nd_pred, min=0.0, max=self.max_abs_nd)

    def predict_sigma_Nd_sites_from_sigma_A(self, sig_A_s):
        sigA = self._safe_sigma_stable(sig_A_s)
        varA = sigA * sigA
        var_Nd = torch.einsum("bms,mz->bzs", varA, self.EOF2_mz.float())
        var_Nd = self._f32(var_Nd)
        var_Nd = torch.clamp(var_Nd, min=0.0, max=self.max_abs_nd ** 2)
        sigma_Nd = torch.sqrt(var_Nd)
        return torch.clamp(sigma_Nd, min=self.sigma_min, max=self.max_abs_nd)

    def masked_mse_Nd_equal_sites(self, Nd_pred, Nd_obs, vS, Eds=None):
        vS = vS.bool()
        finite = torch.isfinite(Nd_obs) & torch.isfinite(Nd_pred)
        if self.nd_require_vS:
            finite = finite & vS[:, None, :]

        cnt_s = finite.sum(dim=(0, 1))
        valid_site = cnt_s >= self.min_valid_points
        if int(valid_site.sum().detach().cpu().item()) < 1:
            return None

        pred0 = torch.clamp(self._f32(Nd_pred), min=0.0, max=self.max_abs_nd)
        obs0 = torch.clamp(self._f32(Nd_obs), min=0.0, max=self.max_abs_nd)
        diff = torch.clamp(pred0 - obs0, min=-self.max_abs_nd, max=self.max_abs_nd)
        diff2 = torch.where(finite, diff * diff, torch.zeros_like(diff))

        num_s_u = diff2.sum(dim=(0, 1))
        if self.normalize_nd_loss:
            den_s_u = cnt_s.to(num_s_u.dtype).clamp(min=1.0)
            loss_s_u = num_s_u / den_s_u
        else:
            loss_s_u = num_s_u

        w = self._nd_error_weights(Eds, finite)
        if (w is None) or (self.err_mix <= 0.0):
            loss_s = loss_s_u
        else:
            num_s_w = (w * diff2).sum(dim=(0, 1))
            if self.normalize_nd_loss:
                den_s_w = (w * finite.float()).sum(dim=(0, 1)).clamp(min=1.0)
                loss_s_w = num_s_w / den_s_w
            else:
                loss_s_w = num_s_w

            a = self.err_mix
            loss_s = (1.0 - a) * loss_s_u + a * loss_s_w

        return loss_s[valid_site].mean()

    def nd_gaussian_nll_equal_sites(self, Nd_pred, sigma_A_s, Nd_obs, vS, Eds=None):
        vS = vS.bool()
        finite = torch.isfinite(Nd_obs) & torch.isfinite(Nd_pred)
        if self.nd_require_vS:
            finite = finite & vS[:, None, :]

        cnt_s = finite.sum(dim=(0, 1))
        valid_site = cnt_s >= self.min_valid_points
        if int(valid_site.sum().detach().cpu().item()) < 1:
            return None

        pred0 = torch.clamp(self._f32(Nd_pred), min=0.0, max=self.max_abs_nd)
        obs0 = torch.clamp(self._f32(Nd_obs), min=0.0, max=self.max_abs_nd)
        sigma_model = self.predict_sigma_Nd_sites_from_sigma_A(sigma_A_s)

        if Eds is None:
            sigma_eff = sigma_model
        else:
            E = torch.clamp(self._f32(Eds).abs(), min=0.0, max=self.max_abs_err)
            sigma_eff = torch.sqrt(sigma_model * sigma_model + E * E)

        sigma_eff = torch.clamp(sigma_eff, min=self.sigma_min, max=self.max_abs_nd)

        diff = torch.clamp(pred0 - obs0, min=-self.max_abs_nd, max=self.max_abs_nd)
        z = torch.clamp(diff / sigma_eff, -self.max_z, self.max_z)

        nll = torch.log(sigma_eff) + 0.5 * (z * z)
        nll = torch.where(finite, nll, torch.zeros_like(nll))

        num_s = nll.sum(dim=(0, 1))
        if self.normalize_nd_loss:
            den_s = cnt_s.to(num_s.dtype).clamp(min=1.0)
            loss_s = num_s / den_s
        else:
            loss_s = num_s

        return loss_s[valid_site].mean()

    # =========================================================
    # campaign helpers
    # =========================================================
    def _gather_campaign_coeffs(self, dA, sigma, seg_batch_idx, seg_ilat, seg_ilon):
        dA_hwm = dA.permute(0, 2, 3, 1).contiguous()       # [B,H,W,M]
        sig_hwm = sigma.permute(0, 2, 3, 1).contiguous()   # [B,H,W,M]

        dA_seg = dA_hwm[seg_batch_idx, seg_ilat, seg_ilon, :]      # [N,M]
        sig_seg = sig_hwm[seg_batch_idx, seg_ilat, seg_ilon, :]    # [N,M]
        Amean_seg = self.Amean_hwm[seg_ilat, seg_ilon, :]          # [N,M]
        Aabs_seg = Amean_seg + dA_seg
        return dA_seg, sig_seg, Aabs_seg

    def _interp_logp_profiles(self, prof_nz, p_n):
        """
        prof_nz : [N,Z] on EOF lev grid
        p_n     : [N] hPa
        returns:
          y_n   : [N]
          valid : [N] bool
        """
        p = torch.clamp(self._f32(p_n), min=1e-6)
        x = torch.log(p)

        prof_asc = prof_nz.index_select(1, self.z_sort_idx)  # [N,Z] ascending in log-p
        lev = self.lev_log_asc                               # [Z]

        idx_hi = torch.searchsorted(lev, x)
        valid = torch.isfinite(x) & (idx_hi > 0) & (idx_hi < lev.numel())

        idx_hi = idx_hi.clamp(min=1, max=lev.numel() - 1)
        idx_lo = idx_hi - 1

        lev_lo = lev[idx_lo]
        lev_hi = lev[idx_hi]
        denom = (lev_hi - lev_lo).clamp(min=1e-6)
        w = ((x - lev_lo) / denom).clamp(min=0.0, max=1.0)

        y_lo = prof_asc.gather(1, idx_lo.view(-1, 1)).squeeze(1)
        y_hi = prof_asc.gather(1, idx_hi.view(-1, 1)).squeeze(1)

        y = (1.0 - w) * y_lo + w * y_hi
        y = torch.where(valid, y, torch.zeros_like(y))
        return y, valid

    def predict_campaign_Nd_and_sigma(self, Aabs_seg, sigA_seg, seg_p):
        """
        Aabs_seg : [N,M]
        sigA_seg : [N,M]
        seg_p    : [N]
        returns:
          Nd_pred_seg        [N]
          sigma_Nd_model_seg [N]
          valid_p            [N] bool
        """
        Aabs_seg = self._clip_abs(self._f32(Aabs_seg), self.max_abs_coeff)
        sigA_seg = self._safe_sigma_stable(sigA_seg)

        # Nd profile
        Nd_prof = torch.einsum("nm,mz->nz", Aabs_seg, self.EOF_mz.float()) + self.Nd_mean_z.view(1, -1)
        Nd_prof = torch.clamp(self._f32(Nd_prof), min=0.0, max=self.max_abs_nd)

        # variance profile
        varA = sigA_seg * sigA_seg
        var_prof = torch.einsum("nm,mz->nz", varA, self.EOF2_mz.float())
        var_prof = torch.clamp(self._f32(var_prof), min=0.0, max=self.max_abs_nd ** 2)

        Nd_pred_seg, valid_p = self._interp_logp_profiles(Nd_prof, seg_p)
        var_seg, valid_p2 = self._interp_logp_profiles(var_prof, seg_p)
        valid_p = valid_p & valid_p2

        sigma_seg = torch.sqrt(torch.clamp(var_seg, min=0.0))
        sigma_seg = torch.clamp(sigma_seg, min=self.sigma_min, max=self.max_abs_nd)

        Nd_pred_seg = torch.clamp(Nd_pred_seg, min=0.0, max=self.max_abs_nd)
        return Nd_pred_seg, sigma_seg, valid_p

    def _hierarchical_campaign_reduce(self, point_loss, valid, campaign_id, group_id=None):
        """
        Equal-campaign weighting, and equal-group weighting within campaign.
        Use flightID as group_id for now. If later you have true trajectory_id,
        pass it as seg_group_id instead.
        """
        valid = valid.bool() & torch.isfinite(point_loss)
        if int(valid.sum().detach().cpu().item()) < 1:
            return None

        loss_v = point_loss[valid]
        cid_v = campaign_id[valid].long()

        if group_id is None:
            gid_v = torch.zeros_like(cid_v)
        else:
            gid_v = group_id[valid].long()

        camp_vals = []
        for c in torch.unique(cid_v):
            mc = (cid_v == c)
            lc = loss_v[mc]
            gc = gid_v[mc]

            group_vals = []
            for g in torch.unique(gc):
                mg = (gc == g)
                group_vals.append(lc[mg].mean())

            camp_vals.append(torch.stack(group_vals).mean())

        return torch.stack(camp_vals).mean()

    def campaign_sigma_penalty(self, sigA_seg, valid_obs, campaign_id, group_id=None):
        """
        Variance-relative coefficient-sigma penalty at campaign locations.

        For each campaign segment:

            point_loss = mean_m[ sigma_m^2 / V_m(rel) ]

        where V_m(rel) is the normalized EOF-derived mode variance supplied
        through mode_weights.  The existing hierarchical reduction then gives
        equal weight to groups within campaigns and equal weight to campaigns.
        """
        finite = valid_obs.bool() & torch.isfinite(sigA_seg).all(dim=1)
        if int(finite.sum().detach().cpu().item()) < 1:
            return None

        sig = self._safe_sigma_stable(sigA_seg)
        mode_var = self.mode_var_rel.view(1, -1).float()

        point_rel_var = ((sig * sig) / mode_var).mean(dim=1)

        return self._hierarchical_campaign_reduce(
            point_loss=point_rel_var,
            valid=finite,
            campaign_id=campaign_id,
            group_id=group_id,
        )

    def campaign_sigma_raw_mean(self, sigA_seg, valid_obs, campaign_id, group_id=None):
        """Raw mean sigma diagnostic only; not used in the loss."""
        finite = valid_obs.bool() & torch.isfinite(sigA_seg).all(dim=1)
        if int(finite.sum().detach().cpu().item()) < 1:
            return None

        sig = self._safe_sigma_stable(sigA_seg)
        point_sigma = sig.mean(dim=1)
        return self._hierarchical_campaign_reduce(
            point_loss=point_sigma,
            valid=finite,
            campaign_id=campaign_id,
            group_id=group_id,
        )

    def campaign_mse(self, Nd_pred_seg, Nd_obs_seg, valid_obs, campaign_id, group_id=None):
        finite = valid_obs.bool() & torch.isfinite(Nd_pred_seg) & torch.isfinite(Nd_obs_seg)
        if int(finite.sum().detach().cpu().item()) < 1:
            return None

        pred0 = torch.clamp(self._f32(Nd_pred_seg), min=0.0, max=self.max_abs_nd)
        obs0 = torch.clamp(self._f32(Nd_obs_seg), min=0.0, max=self.max_abs_nd)
        diff2 = torch.clamp(pred0 - obs0, min=-self.max_abs_nd, max=self.max_abs_nd) ** 2

        return self._hierarchical_campaign_reduce(diff2, finite, campaign_id, group_id)

    def campaign_gaussian_nll(self, Nd_pred_seg, sigma_model_seg, Nd_obs_seg, Nd_obs_std_seg, valid_obs, campaign_id, group_id=None):
        finite = valid_obs.bool() & torch.isfinite(Nd_pred_seg) & torch.isfinite(Nd_obs_seg) & torch.isfinite(sigma_model_seg)
        if int(finite.sum().detach().cpu().item()) < 1:
            return None

        pred0 = torch.clamp(self._f32(Nd_pred_seg), min=0.0, max=self.max_abs_nd)
        obs0 = torch.clamp(self._f32(Nd_obs_seg), min=0.0, max=self.max_abs_nd)
        sigm = torch.clamp(self._safe_sigma_stable(sigma_model_seg), min=self.sigma_min, max=self.max_abs_nd)

        if Nd_obs_std_seg is None:
            sigma_eff = sigm
        else:
            E = torch.clamp(self._f32(Nd_obs_std_seg).abs(), min=self.campaign_obs_std_floor, max=self.max_abs_err)
            sigma_eff = torch.sqrt(sigm * sigm + E * E)

        sigma_eff = torch.clamp(sigma_eff, min=self.sigma_min, max=self.max_abs_nd)
        diff = torch.clamp(pred0 - obs0, min=-self.max_abs_nd, max=self.max_abs_nd)
        z = torch.clamp(diff / sigma_eff, -self.max_z, self.max_z)
        nll = torch.log(sigma_eff) + 0.5 * (z * z)

        return self._hierarchical_campaign_reduce(nll, finite, campaign_id, group_id)

    # =========================================================
    # forward
    # =========================================================
    def forward(
        self,
        dA,
        sigma,
        yS=None,
        vS=None,
        mu0=None,
        Nd_obs_sites=None,
        Eds=None,
        seg_batch_idx=None,
        seg_ilat=None,
        seg_ilon=None,
        seg_p=None,
        seg_y=None,
        seg_y_std=None,
        seg_valid_obs=None,
        seg_campaign_id=None,
        seg_group_id=None,
        return_parts=False,
    ):
        dA0 = self._f32(dA)
        sigma0 = self._f32(sigma)
        mu00 = self._f32(mu0)

        # -------------------------
        # dense regularizers always available
        # -------------------------
        corr = dA0 - mu00
        L_anchor = self.weighted_mse_dense_normed(corr) if self.lambda_anchor > 0 else self._zero_like_sum(dA0)
        L_grad = self.grad_l1_normed(corr) if self.lambda_grad > 0 else self._zero_like_sum(dA0)

        # -------------------------
        # site branch
        # -------------------------
        L_site = self._zero_like_sum(dA0)
        L_nd = self._zero_like_sum(dA0)
        L_nd_gauss = self._zero_like_sum(dA0)
        L_sig = self._zero_like_sum(dA0)
        sig_mean_sites_scalar = 0.0
        vS_frac_scalar = 0.0

        if self._has_obs(yS) and self._has_obs(vS):
            vS = vS.bool()

            dA_s = dA0[:, :, self.ilat, self.ilon]       # [B,M,S]
            sig_s = sigma0[:, :, self.ilat, self.ilon]   # [B,M,S]
            Amean_s = self.Amean_mhw[:, self.ilat, self.ilon].unsqueeze(0).float()
            A_abs_s = Amean_s + dA_s

            Ls = self.site_gaussian_nll_equal_sites(dA_s, sig_s, yS, vS)
            if Ls is not None:
                L_site = Ls

            if self.lambda_sigma > 0:
                L_sig = self.masked_sigma_mean_equal_sites(sig_s, vS)
                sig_raw = self.masked_sigma_raw_mean_equal_sites(sig_s, vS)
                sig_mean_sites_scalar = float(sig_raw.detach().cpu())
            vS_frac_scalar = float(vS.float().mean().detach().cpu())

            if self._has_obs(Nd_obs_sites):
                Nd_pred_sites = self.predict_Nd_sites_from_abs_coeffs(A_abs_s)

                if self.lambda_nd > 0:
                    Ltmp = self.masked_mse_Nd_equal_sites(Nd_pred_sites, Nd_obs_sites, vS, Eds=Eds)
                    if Ltmp is not None:
                        L_nd = Ltmp

                if self.lambda_Ndgauss > 0:
                    Ltmp = self.nd_gaussian_nll_equal_sites(Nd_pred_sites, sig_s, Nd_obs_sites, vS, Eds=Eds)
                    if Ltmp is not None:
                        L_nd_gauss = Ltmp

        # -------------------------
        # campaign branch
        # -------------------------
        L_campaign = self._zero_like_sum(dA0)
        L_campaign_gauss = self._zero_like_sum(dA0)
        L_campaign_sig = self._zero_like_sum(dA0)
        campaign_sig_mean_scalar = 0.0
        n_campaign_valid = 0.0

        have_campaign = (
            self._has_obs(seg_batch_idx) and
            self._has_obs(seg_ilat) and
            self._has_obs(seg_ilon) and
            self._has_obs(seg_p) and
            self._has_obs(seg_y) and
            self._has_obs(seg_valid_obs) and
            self._has_obs(seg_campaign_id)
        )

        if have_campaign and (
            self.lambda_campaign > 0
            or self.lambda_campaign_gauss > 0
            or self.lambda_campaign_sigma > 0
        ):
            seg_batch_idx = seg_batch_idx.long()
            seg_ilat = seg_ilat.long()
            seg_ilon = seg_ilon.long()
            seg_valid_obs = seg_valid_obs.bool()
            seg_campaign_id = seg_campaign_id.long()
            if seg_group_id is not None:
                seg_group_id = seg_group_id.long()

            _, sigA_seg, Aabs_seg = self._gather_campaign_coeffs(
                dA0, sigma0, seg_batch_idx, seg_ilat, seg_ilon
            )

            Nd_pred_seg, sigma_model_seg, valid_p = self.predict_campaign_Nd_and_sigma(
                Aabs_seg, sigA_seg, seg_p
            )

            valid_campaign = seg_valid_obs & valid_p & torch.isfinite(seg_y)
            n_campaign_valid = float(valid_campaign.float().sum().detach().cpu())

            # Explicit sigma control at campaign locations. This closes the
            # asymmetry where campaign Gaussian NLL could reward larger sigma
            # without an equivalent campaign-local sigma penalty.
            if self.lambda_campaign_sigma > 0:
                Ltmp = self.campaign_sigma_penalty(
                    sigA_seg=sigA_seg,
                    valid_obs=valid_campaign,
                    campaign_id=seg_campaign_id,
                    group_id=seg_group_id,
                )
                if Ltmp is not None:
                    L_campaign_sig = Ltmp
                    sig_raw = self.campaign_sigma_raw_mean(
                        sigA_seg=sigA_seg,
                        valid_obs=valid_campaign,
                        campaign_id=seg_campaign_id,
                        group_id=seg_group_id,
                    )
                    if sig_raw is not None:
                        campaign_sig_mean_scalar = float(sig_raw.detach().cpu())

            if self.lambda_campaign > 0:
                Ltmp = self.campaign_mse(
                    Nd_pred_seg=Nd_pred_seg,
                    Nd_obs_seg=seg_y,
                    valid_obs=valid_campaign,
                    campaign_id=seg_campaign_id,
                    group_id=seg_group_id,
                )
                if Ltmp is not None:
                    L_campaign = Ltmp

            if self.lambda_campaign_gauss > 0:
                Ltmp = self.campaign_gaussian_nll(
                    Nd_pred_seg=Nd_pred_seg,
                    sigma_model_seg=sigma_model_seg,
                    Nd_obs_seg=seg_y,
                    Nd_obs_std_seg=seg_y_std,
                    valid_obs=valid_campaign,
                    campaign_id=seg_campaign_id,
                    group_id=seg_group_id,
                )
                if Ltmp is not None:
                    L_campaign_gauss = Ltmp

        # -------------------------
        # weighted sum
        # -------------------------
        L_site_w = self.lambda_sites * L_site
        L_anchor_w = self.lambda_anchor * L_anchor
        L_grad_w = self.lambda_grad * L_grad
        L_sig_w = self.lambda_sigma * L_sig
        L_campaign_sig_w = self.lambda_campaign_sigma * L_campaign_sig
        L_nd_w = self.lambda_nd * L_nd
        L_nd_gauss_w = self.lambda_Ndgauss * L_nd_gauss
        L_campaign_w = self.lambda_campaign * L_campaign
        L_campaign_gauss_w = self.lambda_campaign_gauss * L_campaign_gauss

        loss = (
            L_site_w + L_anchor_w + L_grad_w +
            L_sig_w + L_campaign_sig_w +
            L_nd_w + L_nd_gauss_w +
            L_campaign_w + L_campaign_gauss_w
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss detected: "
                f"L_site={float(L_site_w.detach().cpu())} "
                f"L_anchor={float(L_anchor_w.detach().cpu())} "
                f"L_grad={float(L_grad_w.detach().cpu())} "
                f"L_sig={float(L_sig_w.detach().cpu())} "
                f"L_campaign_sig={float(L_campaign_sig_w.detach().cpu())} "
                f"L_nd={float(L_nd_w.detach().cpu())} "
                f"L_nd_gauss={float(L_nd_gauss_w.detach().cpu())} "
                f"L_campaign={float(L_campaign_w.detach().cpu())} "
                f"L_campaign_gauss={float(L_campaign_gauss_w.detach().cpu())}"
            )

        if not return_parts:
            return loss

        parts = {
            "loss": float(loss.detach().cpu()),
            "L_site": float(L_site_w.detach().cpu()),
            "L_anchor": float(L_anchor_w.detach().cpu()),
            "L_grad": float(L_grad_w.detach().cpu()),
            "L_sig": float(L_sig_w.detach().cpu()),
            "L_campaign_sig": float(L_campaign_sig_w.detach().cpu()),
            "L_nd": float(L_nd_w.detach().cpu()),
            "L_nd_gauss": float(L_nd_gauss_w.detach().cpu()),
            "L_campaign": float(L_campaign_w.detach().cpu()),
            "L_campaign_gauss": float(L_campaign_gauss_w.detach().cpu()),
            "sig_mean_sites": float(sig_mean_sites_scalar),
            "sig_mean_campaign": float(campaign_sig_mean_scalar),
            "sigvar_rel_sites": float(L_sig.detach().cpu()),
            "sigvar_rel_campaign": float(L_campaign_sig.detach().cpu()),
            "vS_frac": float(vS_frac_scalar),
            "n_campaign_valid": float(n_campaign_valid),
        }
        return loss, parts        
