"""Dimensionless learned upper-level relaxation for NDnet refinement.

The relaxation acts only during training. It penalizes the raw Nd departure
from the base reconstruction at low pressure; no taper is applied to inference
or diagnostic products.

This module builds on the dimensionless V2 loss so the whole-column anchor,
site/campaign Nd MSE terms, and upper relaxation all use the same fixed Nd
reference scale.
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
    """Dimensionless refinement loss plus pressure-dependent prior relaxation.

    For ``c = dA - mu0`` the Nd correction is

        dNd(p) = sum_m c_m EOF_m(p).

    The regularizer is the log-pressure-weighted vertical mean of
    ``(dNd / nd_loss_scale)^2`` after applying a 0--1 relaxation weight. The
    weight is zero at and below ``UPPER_RELAX_START_HPA`` and reaches one at
    and above ``UPPER_RELAX_TOP_HPA``.
    """

    def __init__(self, *args, **kwargs):
        EOF = kwargs.get("EOF", args[3] if len(args) > 3 else None)
        if EOF is None:
            raise ValueError("EOF is required to build the upper-relaxation metric")

        super().__init__(*args, **kwargs)

        self.lambda_upper_relax = float(getattr(cfg, "UPPER_RELAX_LAMBDA", 1.0))
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

    def upper_relax_loss(self, dA, mu0):
        """Area-weighted dimensionless upper-level Nd departure from base."""
        if mu0 is None:
            return self._zero_like_sum(dA)

        corr = self._clip_abs(
            self._f32(dA) - self._f32(mu0),
            self.max_abs_coeff,
        )
        q = torch.einsum(
            "bmhw,mn,bnhw->bhw",
            corr,
            self.upper_relax_G.float(),
            corr,
        )
        q = torch.clamp(self._f32(q), min=0.0)

        lat_w = self.lat_w_grid.view(1, -1, 1).float()
        return (q * lat_w).sum() / lat_w.expand_as(q).sum().clamp(min=1e-12)

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

        if self.lambda_upper_relax > 0.0 and mu0 is not None:
            raw = self.upper_relax_loss(dA, mu0)
        else:
            raw = self._zero_like_sum(dA)

        weighted = self.lambda_upper_relax * raw
        loss = base_loss + weighted
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite loss after upper relaxation: "
                f"base={float(base_loss.detach().cpu())} "
                f"upper_relax={float(weighted.detach().cpu())}"
            )

        if not return_parts:
            return loss

        parts = dict(parts)
        parts["L_upper_relax"] = float(weighted.detach().cpu())
        parts["L_upper_relax_raw"] = float(raw.detach().cpu())
        parts["loss"] = float(loss.detach().cpu())
        return loss, parts
