"""Dimensionless loss terms for NDnet refinement.

The refinement objective mixes observations in Nd space, coefficient-space
uncertainties, and prior regularization.  This module puts those terms on
explicit, fixed reference scales so their lambda coefficients are meaningful
tuning parameters rather than compensating for units.

The normalizations are fixed during a run:

* Nd-space terms use ``nd_loss_scale`` in cm-3.
* coefficient-space terms use the area-weighted RMS of each base residual
  coefficient over the training period.
* the dense prior anchor is evaluated as a whole-column Nd departure using
  the same ``nd_loss_scale`` as the observational MSE terms.

No running/batch-dependent loss normalization is used.
"""

from __future__ import annotations

import numpy as np
import torch

from NN_construct_v2 import SparseSiteCampaignNLLRefineLoss as _V2Loss


def logp_cell_weights(pressure_hpa: np.ndarray) -> np.ndarray:
    """Return normalized trapezoidal cell weights in ln(p).

    Parameters
    ----------
    pressure_hpa
        One-dimensional positive pressure coordinate. Ordering may be either
        increasing or decreasing.

    Returns
    -------
    np.ndarray
        Positive weights in the original pressure ordering, summing to one.
    """
    p = np.asarray(pressure_hpa, dtype=np.float64)
    if p.ndim != 1 or p.size < 2:
        raise ValueError("pressure_hpa must be a 1D array with at least two levels")
    if np.any(~np.isfinite(p)) or np.any(p <= 0.0):
        raise ValueError("pressure_hpa must contain finite positive values")

    x = np.log(p)
    order = np.argsort(x)
    xs = x[order]
    if np.any(np.diff(xs) <= 0.0):
        raise ValueError("pressure_hpa must contain distinct levels")

    ws = np.empty_like(xs)
    ws[0] = 0.5 * (xs[1] - xs[0])
    ws[-1] = 0.5 * (xs[-1] - xs[-2])
    if xs.size > 2:
        ws[1:-1] = 0.5 * (xs[2:] - xs[:-2])

    w = np.empty_like(ws)
    w[order] = ws
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("invalid ln(p) integration weights")
    return w / total


def linear_logp_relax_weight(
    pressure_hpa: np.ndarray,
    p_start: float,
    p_top: float,
) -> np.ndarray:
    """Return a 0--1 upper-relaxation weight linear in ln(p)."""
    p = np.asarray(pressure_hpa, dtype=np.float64)
    p_start = float(p_start)
    p_top = float(p_top)
    if not p_start > p_top:
        raise ValueError(
            "p_start must exceed p_top for an upper-atmosphere relaxation; "
            f"got {p_start:g} and {p_top:g}"
        )
    if np.any(~np.isfinite(p)) or np.any(p <= 0.0):
        raise ValueError("pressure_hpa must contain finite positive values")

    denom = np.log(p_start) - np.log(p_top)
    w = (np.log(p_start) - np.log(p)) / denom
    return np.clip(w, 0.0, 1.0)


def build_nd_gram(
    eof_mz: np.ndarray,
    pressure_hpa: np.ndarray,
    nd_scale: float,
    level_modifier: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a dimensionless mode-space metric for mean squared Nd departure.

    The metric represents a log-pressure-weighted vertical mean of

        (sum_m c_m EOF_m(p) / nd_scale)^2.

    ``level_modifier`` can impose an additional pressure-dependent weighting,
    e.g. the upper-relaxation ramp. The resulting vertical weights are always
    renormalized to sum to one over levels with nonzero support.
    """
    eof = np.asarray(eof_mz, dtype=np.float64)
    p = np.asarray(pressure_hpa, dtype=np.float64)
    nd_scale = float(nd_scale)

    if eof.ndim != 2 or eof.shape[1] != p.size:
        raise ValueError(
            f"Expected EOF shape (mode, lev); got {eof.shape} for {p.size} levels"
        )
    if nd_scale <= 0.0 or not np.isfinite(nd_scale):
        raise ValueError("nd_scale must be finite and positive")

    w = logp_cell_weights(p)
    if level_modifier is not None:
        mod = np.asarray(level_modifier, dtype=np.float64)
        if mod.shape != p.shape:
            raise ValueError(
                f"level_modifier shape {mod.shape} does not match pressure shape {p.shape}"
            )
        mod = np.where(np.isfinite(mod), np.clip(mod, 0.0, None), 0.0)
        w = w * mod
        total = float(w.sum())
        if total <= 0.0:
            raise ValueError("level_modifier leaves no vertical support")
        w = w / total

    e = np.nan_to_num(eof, nan=0.0, posinf=0.0, neginf=0.0)
    gram = (e * w[None, :]) @ e.T
    gram /= nd_scale**2
    gram = 0.5 * (gram + gram.T)
    return gram, w


def estimate_base_coefficient_scales(
    mu0_t: torch.Tensor,
    lat_grid: np.ndarray,
    *,
    n_time: int | None = None,
    floor: float = 0.25,
    chunk_size: int = 16,
) -> np.ndarray:
    """Estimate fixed per-mode coefficient scales from the training prior.

    The returned scale is the latitude-area-weighted RMS of each base residual
    coefficient over training time and longitude. Computation is chunked in
    time to avoid allocating another full ``(time, mode, lat, lon)`` tensor.
    """
    if not isinstance(mu0_t, torch.Tensor) or mu0_t.ndim != 4:
        raise ValueError("mu0_t must be a torch.Tensor with shape (time, mode, lat, lon)")

    nt, nm, ny, _ = mu0_t.shape
    n_use = nt if n_time is None else int(n_time)
    if n_use < 1 or n_use > nt:
        raise ValueError(f"n_time must be in [1, {nt}], got {n_use}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    lat = np.asarray(lat_grid, dtype=np.float64)
    if lat.ndim != 1 or lat.size != ny:
        raise ValueError(f"lat_grid must have length {ny}; got shape {lat.shape}")

    lat_w = np.cos(np.deg2rad(lat))
    lat_w = np.nan_to_num(lat_w, nan=0.0, posinf=0.0, neginf=0.0)
    if np.all(lat_w <= 0.0):
        raise ValueError("latitude weights are all zero")
    lat_w = lat_w / lat_w.mean()
    w = torch.as_tensor(lat_w, dtype=torch.float64).view(1, 1, ny, 1)

    sum_sq = torch.zeros(nm, dtype=torch.float64)
    sum_w = torch.zeros(nm, dtype=torch.float64)

    for i0 in range(0, n_use, int(chunk_size)):
        x = mu0_t[i0 : min(i0 + int(chunk_size), n_use)].detach().cpu().to(torch.float64)
        finite = torch.isfinite(x)
        x0 = torch.where(finite, x, torch.zeros_like(x))
        ww = w.expand(x.shape[0], 1, ny, x.shape[3])
        sum_sq += (x0.square() * ww).sum(dim=(0, 2, 3))
        sum_w += (finite.to(torch.float64) * ww).sum(dim=(0, 2, 3))

    rms = torch.sqrt(sum_sq / sum_w.clamp(min=1e-12))
    floor = float(floor)
    if floor <= 0.0:
        raise ValueError("floor must be positive")
    rms = torch.clamp(rms, min=floor)
    return rms.to(torch.float32).numpy()


class DimensionlessSparseSiteCampaignNLLRefineLoss(_V2Loss):
    """V2 observation loss with fixed, explicit dimensionless normalizations."""

    def __init__(self, *args, **kwargs):
        coeff_scales = kwargs.pop("coeff_scales", None)
        nd_loss_scale = float(kwargs.pop("nd_loss_scale", 100.0))
        coeff_scale_floor = float(kwargs.pop("coeff_scale_floor", 0.25))

        EOF = kwargs.get("EOF", args[3] if len(args) > 3 else None)
        if EOF is None:
            raise ValueError("EOF is required for dimensionless Nd-space regularization")

        super().__init__(*args, **kwargs)

        if nd_loss_scale <= 0.0:
            raise ValueError("nd_loss_scale must be positive")
        if coeff_scale_floor <= 0.0:
            raise ValueError("coeff_scale_floor must be positive")
        self.nd_loss_scale = nd_loss_scale
        self.coeff_scale_floor = coeff_scale_floor

        nm = int(self.EOF_mz.shape[0])
        if coeff_scales is None:
            raise ValueError(
                "coeff_scales are required; derive them once from training-prior residual coefficients"
            )
        cs = np.asarray(coeff_scales, dtype=np.float32)
        if cs.shape != (nm,):
            raise ValueError(f"coeff_scales must have shape ({nm},), got {cs.shape}")
        if np.any(~np.isfinite(cs)) or np.any(cs <= 0.0):
            raise ValueError("coeff_scales must be finite and positive")
        cs = np.maximum(cs, coeff_scale_floor)
        self.register_buffer("coeff_scale", torch.as_tensor(cs, dtype=torch.float32))

        lev = np.asarray(getattr(EOF.lev, "values", EOF.lev), dtype=np.float64)
        gram, vertical_w = build_nd_gram(
            np.asarray(getattr(EOF, "values", EOF), dtype=np.float64),
            lev,
            nd_loss_scale,
        )
        self.register_buffer("anchor_G", torch.as_tensor(gram, dtype=torch.float32))
        self.register_buffer(
            "anchor_vertical_weight_lev",
            torch.as_tensor(vertical_w, dtype=torch.float32),
        )

    def weighted_mse_dense_normed(self, x):
        """Whole-column Nd-space prior departure on the common Nd scale."""
        corr = self._clip_abs(self._f32(x), self.max_abs_coeff)
        q = torch.einsum(
            "bmhw,mn,bnhw->bhw",
            corr,
            self.anchor_G.float(),
            corr,
        )
        q = torch.clamp(self._f32(q), min=0.0)
        lat_w = self.lat_w_grid.view(1, -1, 1).float()
        return (q * lat_w).sum() / lat_w.expand_as(q).sum().clamp(min=1e-12)

    def grad_l1_normed(self, x):
        """Dimensionless horizontal coefficient-gradient regularizer."""
        x = self._clip_abs(self._f32(x), self.max_abs_coeff)
        xn = x / self.coeff_scale.view(1, -1, 1, 1).float()
        dx = (xn[..., :, 1:] - xn[..., :, :-1]).abs().mean()
        dy = (xn[..., 1:, :] - xn[..., :-1, :]).abs().mean()
        return 0.5 * (dx + dy)

    def site_gaussian_nll_equal_sites(self, pred_s, sig_s, yS, vS):
        if self.lambda_sites <= 0.0:
            return self._zero_like_sum(pred_s)
        vS = vS.bool()
        finite = torch.isfinite(yS) & torch.isfinite(pred_s) & torch.isfinite(sig_s)
        finite = finite & vS[:, None, :]
        cnt_s = finite.sum(dim=(0, 1))
        valid_site = cnt_s >= self.min_valid_points
        if int(valid_site.sum().detach().cpu().item()) < 1:
            return None

        pred = self._clip_abs(self._f32(pred_s), self.max_abs_coeff)
        target = self._clip_abs(self._f32(yS), self.max_abs_coeff)
        sig = self._safe_sigma_stable(sig_s)
        scale = self.coeff_scale.view(1, -1, 1).float()
        diff = torch.clamp(
            pred - target,
            min=-2.0 * self.max_abs_coeff,
            max=2.0 * self.max_abs_coeff,
        )
        z = torch.clamp(diff / sig, -self.max_z, self.max_z)
        nll = torch.log((sig / scale).clamp(min=1e-12)) + 0.5 * z.square()
        nll = torch.where(finite, nll, torch.zeros_like(nll))
        num_s = nll.sum(dim=(0, 1))
        den_s = cnt_s.to(num_s.dtype).clamp(min=1.0)
        return (num_s / den_s)[valid_site].mean()

    def masked_sigma_mean_equal_sites(self, sig_s, vS):
        """Mean squared coefficient uncertainty relative to prior RMS scale."""
        vS = vS.bool()
        finite = torch.isfinite(sig_s) & vS[:, None, :]
        cnt_s = finite.sum(dim=(0, 1))
        valid_site = cnt_s >= 1
        if int(valid_site.sum().detach().cpu().item()) < 1:
            return self._zero_like_sum(sig_s)

        sig = self._safe_sigma_stable(sig_s)
        scale = self.coeff_scale.view(1, -1, 1).float()
        rel_var = (sig / scale).square()
        rel_var = torch.where(finite, rel_var, torch.zeros_like(rel_var))
        num_s = rel_var.sum(dim=(0, 1))
        den_s = cnt_s.to(num_s.dtype).clamp(min=1.0)
        return (num_s / den_s)[valid_site].mean()

    def masked_mse_Nd_equal_sites(self, Nd_pred, Nd_obs, vS, Eds=None):
        vS = vS.bool()
        finite = torch.isfinite(Nd_obs) & torch.isfinite(Nd_pred)
        if self.nd_require_vS:
            finite = finite & vS[:, None, :]

        cnt_s = finite.sum(dim=(0, 1))
        valid_site = cnt_s >= self.min_valid_points
        if int(valid_site.sum().detach().cpu().item()) < 1:
            return None

        pred = torch.clamp(self._f32(Nd_pred), min=0.0, max=self.max_abs_nd)
        obs = torch.clamp(self._f32(Nd_obs), min=0.0, max=self.max_abs_nd)
        diff = torch.clamp(pred - obs, min=-self.max_abs_nd, max=self.max_abs_nd)
        diff2 = (diff / self.nd_loss_scale).square()
        diff2 = torch.where(finite, diff2, torch.zeros_like(diff2))

        num_u = diff2.sum(dim=(0, 1))
        if self.normalize_nd_loss:
            den_u = cnt_s.to(num_u.dtype).clamp(min=1.0)
            loss_u = num_u / den_u
        else:
            loss_u = num_u

        w = self._nd_error_weights(Eds, finite)
        if w is None or self.err_mix <= 0.0:
            loss_s = loss_u
        else:
            num_w = (w * diff2).sum(dim=(0, 1))
            if self.normalize_nd_loss:
                den_w = (w * finite.float()).sum(dim=(0, 1)).clamp(min=1.0)
                loss_w = num_w / den_w
            else:
                loss_w = num_w
            a = self.err_mix
            loss_s = (1.0 - a) * loss_u + a * loss_w

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

        pred = torch.clamp(self._f32(Nd_pred), min=0.0, max=self.max_abs_nd)
        obs = torch.clamp(self._f32(Nd_obs), min=0.0, max=self.max_abs_nd)
        sigma_model = self.predict_sigma_Nd_sites_from_sigma_A(sigma_A_s)
        if Eds is None:
            sigma_eff = sigma_model
        else:
            err = torch.clamp(self._f32(Eds).abs(), min=0.0, max=self.max_abs_err)
            sigma_eff = torch.sqrt(sigma_model.square() + err.square())

        sigma_eff = torch.clamp(sigma_eff, min=self.sigma_min, max=self.max_abs_nd)
        diff = torch.clamp(pred - obs, min=-self.max_abs_nd, max=self.max_abs_nd)
        z = torch.clamp(diff / sigma_eff, -self.max_z, self.max_z)
        nll = (
            torch.log((sigma_eff / self.nd_loss_scale).clamp(min=1e-12))
            + 0.5 * z.square()
        )
        nll = torch.where(finite, nll, torch.zeros_like(nll))
        num_s = nll.sum(dim=(0, 1))
        if self.normalize_nd_loss:
            den_s = cnt_s.to(num_s.dtype).clamp(min=1.0)
            loss_s = num_s / den_s
        else:
            loss_s = num_s
        return loss_s[valid_site].mean()

    def campaign_sigma_penalty(self, sigA_seg, valid_obs, campaign_id, group_id=None):
        finite = valid_obs.bool() & torch.isfinite(sigA_seg).all(dim=1)
        if int(finite.sum().detach().cpu().item()) < 1:
            return None
        sig = self._safe_sigma_stable(sigA_seg)
        scale = self.coeff_scale.view(1, -1).float()
        point_loss = (sig / scale).square().mean(dim=1)
        return self._hierarchical_campaign_reduce(
            point_loss=point_loss,
            valid=finite,
            campaign_id=campaign_id,
            group_id=group_id,
        )

    def campaign_mse(self, Nd_pred_seg, Nd_obs_seg, valid_obs, campaign_id, group_id=None):
        finite = valid_obs.bool() & torch.isfinite(Nd_pred_seg) & torch.isfinite(Nd_obs_seg)
        if int(finite.sum().detach().cpu().item()) < 1:
            return None
        pred = torch.clamp(self._f32(Nd_pred_seg), min=0.0, max=self.max_abs_nd)
        obs = torch.clamp(self._f32(Nd_obs_seg), min=0.0, max=self.max_abs_nd)
        diff = torch.clamp(pred - obs, min=-self.max_abs_nd, max=self.max_abs_nd)
        point_loss = (diff / self.nd_loss_scale).square()
        return self._hierarchical_campaign_reduce(point_loss, finite, campaign_id, group_id)

    def campaign_gaussian_nll(
        self,
        Nd_pred_seg,
        sigma_model_seg,
        Nd_obs_seg,
        Nd_obs_std_seg,
        valid_obs,
        campaign_id,
        group_id=None,
    ):
        finite = (
            valid_obs.bool()
            & torch.isfinite(Nd_pred_seg)
            & torch.isfinite(Nd_obs_seg)
            & torch.isfinite(sigma_model_seg)
        )
        if int(finite.sum().detach().cpu().item()) < 1:
            return None

        pred = torch.clamp(self._f32(Nd_pred_seg), min=0.0, max=self.max_abs_nd)
        obs = torch.clamp(self._f32(Nd_obs_seg), min=0.0, max=self.max_abs_nd)
        sigma_model = torch.clamp(
            self._safe_sigma_stable(sigma_model_seg),
            min=self.sigma_min,
            max=self.max_abs_nd,
        )
        if Nd_obs_std_seg is None:
            sigma_eff = sigma_model
        else:
            err = torch.clamp(
                self._f32(Nd_obs_std_seg).abs(),
                min=self.campaign_obs_std_floor,
                max=self.max_abs_err,
            )
            sigma_eff = torch.sqrt(sigma_model.square() + err.square())

        sigma_eff = torch.clamp(sigma_eff, min=self.sigma_min, max=self.max_abs_nd)
        diff = torch.clamp(pred - obs, min=-self.max_abs_nd, max=self.max_abs_nd)
        z = torch.clamp(diff / sigma_eff, -self.max_z, self.max_z)
        nll = (
            torch.log((sigma_eff / self.nd_loss_scale).clamp(min=1e-12))
            + 0.5 * z.square()
        )
        return self._hierarchical_campaign_reduce(nll, finite, campaign_id, group_id)
