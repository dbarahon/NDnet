"""Learned upper-level relaxation for the V2 refinement loss.

This module does not taper predictions.  It adds a pressure-aware Nd-space
penalty that teaches the raw V2 refinement to converge toward the base
(reanalysis-emulation) reconstruction as EOF support weakens aloft.

Site and campaign observation losses remain unchanged and therefore retain
all valid high-altitude observations.  Production inference is unchanged:
the network output is reconstructed directly, with no post-processing taper.
"""

from __future__ import annotations

import numpy as np
import torch

import config as cfg
from NN_construct_v2 import SparseSiteCampaignNLLRefineLoss as _V2Loss


def _linear_logp_relax_weight(pressure_hpa, p_start, p_top):
    """Return W(p): 0 below p_start, 1 above p_top, linear in log-pressure."""
    p = np.asarray(pressure_hpa, dtype=np.float64)
    p_start = float(p_start)
    p_top = float(p_top)
    if not p_start > p_top:
        raise ValueError(
            f"UPPER_RELAX_START_HPA must exceed UPPER_RELAX_TOP_HPA; "
            f"got {p_start:g} and {p_top:g}."
        )
    if np.any(~np.isfinite(p)) or np.any(p <= 0.0):
        raise ValueError(
            "Pressure levels must be finite and positive for log-pressure relaxation."
        )

    # W=0 at and below the start pressure, W=1 at and above the top pressure.
    # Between them, the preference for the base increases linearly with log(p),
    # which spreads the relaxation force more evenly over atmospheric height.
    denom = np.log(p_start) - np.log(p_top)
    w = (np.log(p_start) - np.log(p)) / denom
    return np.clip(w, 0.0, 1.0)


class SparseSiteCampaignNLLRefineLossRelax(_V2Loss):
    """Exact V2 loss plus a learned upper-atmosphere base-relaxation term.

    Let c = dA - mu0 be the refinement in residual-EOF coefficient space.
    The corresponding Nd departure from the base is

        dNd(p) = sum_m c_m EOF_m(p).

    We penalize this departure with a weight W(p) that is zero below
    ``UPPER_RELAX_START_HPA``, rises linearly in log-pressure through the
    transition, and reaches one by ``UPPER_RELAX_TOP_HPA``.  Crucially, the
    prediction itself is never multiplied by W.  The network must learn the
    relaxation in its raw output.

    The vertical quadratic is precomputed in mode space:

        G_mn = mean_z[ W_z EOF_mz EOF_nz ] / Nd_scale^2
        Lrelax = mean_{b,lat,lon}[ c^T G c ].

    This includes cross-mode interactions, unlike the ordinary diagonal
    coefficient-space anchor.
    """

    def __init__(self, *args, **kwargs):
        EOF = kwargs.get("EOF", args[3] if len(args) > 3 else None)
        if EOF is None:
            raise ValueError("EOF is required to build the upper-relaxation metric.")

        super().__init__(*args, **kwargs)

        self.lambda_upper_relax = float(
            getattr(cfg, "UPPER_RELAX_LAMBDA", 0.1)
        )
        self.upper_relax_start_hpa = float(
            getattr(cfg, "UPPER_RELAX_START_HPA", 600.0)
        )
        self.upper_relax_top_hpa = float(
            getattr(cfg, "UPPER_RELAX_TOP_HPA", 350.0)
        )
        self.upper_relax_nd_scale = float(
            getattr(cfg, "UPPER_RELAX_ND_SCALE", 100.0)
        )
        if self.upper_relax_nd_scale <= 0.0:
            raise ValueError("UPPER_RELAX_ND_SCALE must be positive.")

        lev = getattr(getattr(EOF, "lev", None), "values", None)
        if lev is None:
            raise ValueError("EOF must carry a lev coordinate for upper relaxation.")
        lev = np.asarray(lev, dtype=np.float64)
        eof = np.asarray(getattr(EOF, "values", EOF), dtype=np.float64)
        if eof.ndim != 2 or eof.shape[1] != lev.size:
            raise ValueError(
                f"Expected EOF shape (mode,lev), got {eof.shape} for {lev.size} levels."
            )

        relax_w = _linear_logp_relax_weight(
            lev,
            self.upper_relax_start_hpa,
            self.upper_relax_top_hpa,
        )
        valid = np.isfinite(lev) & np.isfinite(relax_w) & (lev > 0.0)
        relax_w = np.where(valid, relax_w, 0.0)
        wsum = float(relax_w.sum())
        if wsum <= 0.0:
            raise ValueError("Upper-relaxation weights are all zero on the EOF levels.")

        # G is positive semidefinite apart from roundoff.  Normalizing by the
        # vertical weight sum and a configurable Nd scale keeps lambda_relax
        # dimensionless and easier to tune against the existing V2 losses.
        eof_clean = np.nan_to_num(eof, nan=0.0, posinf=0.0, neginf=0.0)
        G = (eof_clean * relax_w[None, :]) @ eof_clean.T
        G /= wsum * (self.upper_relax_nd_scale ** 2)
        G = 0.5 * (G + G.T)
        self.register_buffer(
            "upper_relax_G",
            torch.as_tensor(G, dtype=torch.float32),
        )
        self.register_buffer(
            "upper_relax_weight_lev",
            torch.as_tensor(relax_w, dtype=torch.float32),
        )

    def upper_relax_loss(self, dA, mu0):
        """Area-weighted mean squared upper-level Nd departure from the base."""
        if mu0 is None:
            return self._zero_like_sum(dA)

        corr = self._clip_abs(
            self._f32(dA) - self._f32(mu0),
            self.max_abs_coeff,
        )

        # q[b,h,w] = c^T G c.  This is algebraically equivalent to
        # reconstructing dNd on all EOF levels and applying the vertical
        # relaxation weights, without allocating a BxZxHxW tensor.
        q = torch.einsum(
            "bmhw,mn,bnhw->bhw",
            corr,
            self.upper_relax_G.float(),
            corr,
        )
        q = torch.clamp(self._f32(q), min=0.0)

        lat_w = self.lat_w_grid.view(1, -1, 1).float()
        num = (q * lat_w).sum()
        den = lat_w.expand_as(q).sum().clamp(min=1e-12)
        return num / den

    def forward(self, *args, return_parts=False, **kwargs):
        dA = args[0] if len(args) > 0 else kwargs.get("dA", None)
        mu0 = kwargs.get("mu0", args[4] if len(args) > 4 else None)
        if dA is None:
            raise ValueError("dA is required by the refinement loss.")

        base_loss, parts = super().forward(
            *args,
            return_parts=True,
            **kwargs,
        )

        if self.lambda_upper_relax > 0.0 and mu0 is not None:
            L_upper_relax_raw = self.upper_relax_loss(dA, mu0)
        else:
            L_upper_relax_raw = self._zero_like_sum(dA)

        L_upper_relax = self.lambda_upper_relax * L_upper_relax_raw
        loss = base_loss + L_upper_relax

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite loss after upper relaxation: "
                f"base={float(base_loss.detach().cpu())} "
                f"upper_relax={float(L_upper_relax.detach().cpu())}"
            )

        if not return_parts:
            return loss

        parts = dict(parts)
        parts["L_upper_relax"] = float(L_upper_relax.detach().cpu())
        parts["L_upper_relax_raw"] = float(L_upper_relax_raw.detach().cpu())
        parts["loss"] = float(loss.detach().cpu())
        return loss, parts
