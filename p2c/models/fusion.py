"""Fusion of per-view features into one observation embedding.

The choice of fusion decides whether capacity matching — harness contract C2
(docs/harness_contract.md) — holds by construction or has to be patched afterwards:

``mean``   permutation-invariant average. Parameter count is **independent of the number
           of views**, so a 1-view and a 3-view model are exactly the same size. This is
           the default, and it is what makes the headline comparison fair.
``attn``   single learned query attending over views. Also **view-count independent**, and
           unlike ``mean`` it can select rather than average, so complementary evidence is
           not washed out by a view that happens to be uninformative.
``concat`` concatenates view features. Parameter count **grows with the number of views**,
           so it violates C2. Supported for completeness, but it warns, and the
           training script records the violation in the run metadata.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn


class MeanFusion(nn.Module):
    """Average over views; ignores dropped views via the keep mask."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.out_dim = feat_dim

    def forward(self, feats: torch.Tensor, mask: torch.Tensor | None = None):
        # feats [B, V, D]; mask [B, V] of bools (True = keep)
        if mask is None:
            return feats.mean(dim=1)
        m = mask.to(feats.dtype).unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1.0)
        return (feats * m).sum(dim=1) / denom


class AttnFusion(nn.Module):
    """One learned query attends over the view axis. View-count independent."""

    def __init__(self, feat_dim: int, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, feat_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            feat_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(feat_dim)
        self.feat_dim = feat_dim
        self.out_dim = feat_dim

    def forward(self, feats: torch.Tensor, mask: torch.Tensor | None = None):
        B = feats.shape[0]
        q = self.query.expand(B, 1, self.feat_dim)
        key_padding_mask = None
        if mask is not None:
            # MultiheadAttention wants True where a position should be *ignored*.
            key_padding_mask = ~mask
            # A row with every view dropped would produce NaNs; keep view 0 in that case.
            all_dropped = key_padding_mask.all(dim=1)
            if all_dropped.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_dropped, 0] = False
        out, _ = self.attn(q, feats, feats, key_padding_mask=key_padding_mask)
        return self.norm(out.squeeze(1))


class ConcatFusion(nn.Module):
    """Concatenate view features, then project. Parameter count scales with views."""

    def __init__(self, feat_dim: int, num_views: int, out_dim: int | None = None):
        super().__init__()
        warnings.warn(
            "ConcatFusion makes the parameter count depend on the number of views, "
            "which violates harness contract C2 (capacity matching). Results "
            "from it cannot separate complementary information from extra capacity.",
            stacklevel=2,
        )
        out_dim = out_dim or feat_dim
        self.proj = nn.Sequential(
            nn.Linear(feat_dim * num_views, out_dim), nn.LayerNorm(out_dim)
        )
        self.num_views = num_views
        self.out_dim = out_dim

    def forward(self, feats: torch.Tensor, mask: torch.Tensor | None = None):
        if mask is not None:
            feats = feats * mask.to(feats.dtype).unsqueeze(-1)
        return self.proj(feats.flatten(1))


def build_fusion(name: str, feat_dim: int, num_views: int) -> nn.Module:
    if name == "mean":
        return MeanFusion(feat_dim)
    if name == "attn":
        return AttnFusion(feat_dim)
    if name == "concat":
        return ConcatFusion(feat_dim, num_views)
    raise ValueError(f"unknown fusion '{name}' (mean | attn | concat)")


def fusion_is_capacity_matched(name: str) -> bool:
    """Whether this fusion keeps the parameter count independent of the view count."""
    return name in ("mean", "attn")
