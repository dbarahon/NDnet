"""Scale-separated upper-level relaxation for NDnet refinement.

The regularizer acts only during training and never modifies inference output.
It constrains the poorly observed large-scale upper-tropospheric correction
while allowing local departures needed to fit high-reaching observations.

For correction coefficients ``c = dA - mu0`` the Nd correction is

    dNd(p) = sum_m c_m EOF_m(p).

The correction is separated into a longitude mean and a longitude anomaly.
The zonal-mean component is strongly regularized; the anomaly component can be
regularized weakly or disabled. Both terms use the same fixed Nd reference
scale and upper-pressure Gram matrix as the dimensionless objective.

The direct site Nd MSE can additionally be reweighted smoothly with pressure.
This gives observed upper-level site values more leverage against the zonal
constraint without introducing a step at a model level. The pressure weights
are normalized within each site, so the equal-site hierarchy is preserved.
"""

from __future__ import annotations

import numpy as np
import torch

import config as cfg
from loss_normalization import (
    DimensionlessSparseSiteCampaignNLLRefineLoss as _DimensionlessLoss,
    build_nd_gram,
    linear_logp_relax_weight,
)
from pressure_weighting import linear_upper_pressure_weight


class SparseSiteCampaignNLLRefineLossRelax(_DimensionlessLoss):
    """Dimensionless refinement loss with scale-separated upper relaxation."""

    def __init__(self, *args, **kwargs):
        EOF = kwargs.get("EOF", args[3] if len(args) > 3 else None)
        if EOF is None:
            raise ValueError("EOF is required to build the upper-relaxation metric")

        super().__init__(*args, **kwargs)

        self.lambda_upper_zonal = float(
            getattr(cfg, "UPPER_RELAX_ZONAL_LAMBDA", 10.0)
        )
        self.lambda_upper_anomaly = float(
            getattr(cfg, "UPPER_RELAX_ANOMALY_LAMBDA", 0.0)
        )
        self.upper_relax_start_hpa = float(
            getattr(cfg, "UPPER_RELAX_START_HPA", 700.0)
        )
        self.upper_relax_top_hpa = float(
            getattr(cfg, "UPPER_RELAX_TOP_HPA", 500.0)
        )

        self.site_nd_weight_start_hpa = float(
            getattr(cfg, "SITE_ND_PRESSURE_WEIGHT_START_HPA", 500.0)
        )
        self.site_nd_weight_top_hpa = float(
            getattr(cfg, "SITE_ND_PRESSURE_WEIGHT_TOP_HPA", 350.0)
        )
        self.site_nd_weight_top_factor = float(
            getattr(cfg, "SITE_ND_PRESSURE_WEIGHT_TOP_FACTOR", 3.0)
        )

        lev = np.asarray(getattr(EOF.lev, "values", EOF.lev), dtype=np.float64)
        eof = np.asarray(getattr(EOF, "values", EOF), dtype=np.float64)
        relax_w = linear_logp_relax_weight(
            lev,
            self.upper_relax_start_hpa,
            self.upper_relax_top_hpa,
        )
        gram, integrated_w = build_nd_gram(
            eof,
            lev,
            self.nd_loss_scale,
            level_modifier=relax_w,
        )
        site_nd_w = linear_upper_pressure_weight(
            lev,
            self.site_nd_weight_start_hpa,
            self.site_nd_weight_top_hpa,
            self.site_nd_weight_top_factor,
        )

        self.register_buffer(
            "upper_relax_G",
            torch.as_tensor(gram, dtype=torch.float32),
        )
        self.register_buffer(
            "upper_relax_weight_lev",
            torch.as_tensor(relax_w, dtype=torch.float32),
        )
        self.register_buffer(
            "upper_relax_integrated_weight_lev",
            torch.as_tensor(integrated_w, dtype=torch.float32),
        )
        self.register_buffer(
            "site_nd_pressure_weight_lev",
            torch.as_tensor(site_nd_w, dtype=torch.float32),
        )

    def masked_mse_Nd_equal_sites(self, Nd_pred, Nd_obs, vS, Eds=None):
        """Pressure-weighted, dimensionless site Nd MSE with equal site weight.

        The pressure multiplier is applied only to the direct deterministic
        site Nd MSE. Each site's numerator is divided by its own weighted
        valid-observation sum, so a site with upper-level data does not gain
        extra total weight merely because its pressure weights are larger.
        The Gaussian site likelihood is intentionally left unchanged.
        """
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

        pweight = self.site_nd_pressure_weight_lev.view(1, -1, 1).float()
        weighted_valid = pweight * finite.float()
        num_u = (pweight * diff2).sum(dim=(0, 1))
        if self.normalize_nd_loss:
            den_u = weighted_valid.sum(dim=(0, 1)).clamp(min=1e-12)
            loss_u = num_u / den_u
        else:
            loss_u = num_u

        w = self._nd_error_weights(Eds, finite)
        if w is None or self.err_mix <= 0.0:
            loss_s = loss_u
        else:
            combined_w = w * pweight
            num_w = (combined_w * diff2).sum(dim=(0, 1))
            if self.normalize_nd_loss:
                den_w = (combined_w * finite.float()).sum(dim=(0, 1)).clamp(min=1e-12)
                loss_w = num_w / den_w
            else:
                loss_w = num_w
            a = self.err_mix
            loss_s = (1.0 - a) * loss_u + a * loss_w

        return loss_s[valid_site].mean()

    def _upper_quadratic_zonal(self, corr):
        """Area-weighted upper Nd penalty for the longitude-mean correction."""
        # corr: (batch, mode, lat, lon). Longitude is averaged first so a
        # localized observational correction is diluted rather than penalized
        # as though it were a latitude-circle-wide departure from the prior.
        corr_zonal = corr.mean(dim=-1)  # (batch, mode, lat)
        q = torch.einsum(
            "bmh,mn,bnh->bh",
            corr_zonal,
            self.upper_relax_G.float(),
            corr_zonal,
        )
        q = torch.clamp(self._f32(q), min=0.0)
        lat_w = self.lat_w_grid.view(1, -1).float()
        return (q * lat_w).sum() / lat_w.expand_as(q).sum().clamp(min=1e-12)

    def _upper_quadratic_anomaly(self, corr):
        """Area-weighted upper Nd penalty for deviations from the zonal mean."""
        corr_zonal = corr.mean(dim=-1, keepdim=True)
        corr_anom = corr - corr_zonal
        q = torch.einsum(
            "bmhw,mn,bnhw->bhw",
            corr_anom,
            self.upper_relax_G.float(),
            corr_anom,
        )
        q = torch.clamp(self._f32(q), min=0.0)
        lat_w = self.lat_w_grid.view(1, -1, 1).float()
        return (q * lat_w).sum() / lat_w.expand_as(q).sum().clamp(min=1e-12)

    def upper_relax_components(self, dA, mu0):
        """Return raw zonal-mean and anomaly upper-relaxation losses."""
        if mu0 is None:
            zero = self._zero_like_sum(dA)
            return zero, zero

        corr = self._clip_abs(
            self._f32(dA) - self._f32(mu0),
            self.max_abs_coeff,
        )
        zonal = self._upper_quadratic_zonal(corr)
        anomaly = self._upper_quadratic_anomaly(corr)
        return zonal, anomaly

    def upper_relax_loss(self, dA, mu0):
        """Return weighted scale-separated upper-relaxation loss."""
        zonal, anomaly = self.upper_relax_components(dA, mu0)
        return (
            self.lambda_upper_zonal * zonal
            + self.lambda_upper_anomaly * anomaly
        )

    def forward(self, *args, return_parts=False, **kwargs):
        dA = args[0] if len(args) > 0 else kwargs.get("dA")
        mu0 = kwargs.get("mu0", args[4] if len(args) > 4 else None)
        if dA is None:
            raise ValueError("dA is required by the refinement loss")

        base_loss, parts = super().forward(
            *args,
            return_parts=True,
            **kwargs,
        )

        if (
            (self.lambda_upper_zonal > 0.0 or self.lambda_upper_anomaly > 0.0)
            and mu0 is not None
        ):
            raw_zonal, raw_anomaly = self.upper_relax_components(dA, mu0)
        else:
            raw_zonal = self._zero_like_sum(dA)
            raw_anomaly = self._zero_like_sum(dA)

        weighted_zonal = self.lambda_upper_zonal * raw_zonal
        weighted_anomaly = self.lambda_upper_anomaly * raw_anomaly
        weighted = weighted_zonal + weighted_anomaly
        loss = base_loss + weighted

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite loss after upper relaxation: "
                f"base={float(base_loss.detach().cpu())} "
                f"upper_zonal={float(weighted_zonal.detach().cpu())} "
                f"upper_anomaly={float(weighted_anomaly.detach().cpu())}"
            )

        if not return_parts:
            return loss

        parts = dict(parts)
        # Retain total keys used by the existing training logger, and expose
        # both components explicitly for reproducibility and diagnostics.
        parts["L_upper_relax"] = float(weighted.detach().cpu())
        parts["L_upper_relax_raw"] = float(
            (raw_zonal + raw_anomaly).detach().cpu()
        )
        parts["L_upper_zonal"] = float(weighted_zonal.detach().cpu())
        parts["L_upper_zonal_raw"] = float(raw_zonal.detach().cpu())
        parts["L_upper_anomaly"] = float(weighted_anomaly.detach().cpu())
        parts["L_upper_anomaly_raw"] = float(raw_anomaly.detach().cpu())
        parts["loss"] = float(loss.detach().cpu())
        return loss, parts
