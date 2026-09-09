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
            getattr(cfg, "UPPER_RELAX_ANOMALY_LAMBDA", 0.1)
        )
        self.upper_relax_start_hpa = float(
            getattr(cfg, "UPPER_RELAX_START_HPA", 700.0)
        )
        self.upper_relax_top_hpa = float(
            getattr(cfg, "UPPER_RELAX_TOP_HPA", 500.0)
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
