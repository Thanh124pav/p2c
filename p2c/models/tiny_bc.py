"""Tiny behaviour-cloning policy: the phenomenon detector of PLAN.md Stage 2.

This is deliberately small. It is not meant to be a strong policy and its numbers are not
a paper result — it exists to falsify the P2C hypothesis cheaply on a 4 GB GPU.

Shape contract::

    images [B, V, 3, H, W]           (or [B, V, T, 3, H, W] with history)
    state  [B, S]
    ->
    action [B, A]                    (or [B, Ha, A] with an action horizon)

The encoder is shared across the ``V`` axis and the default fusion is view-count
independent, so ``num_parameters()`` is identical for the 1-, 2- and 3-view conditions.
:meth:`capacity_report` makes that claim checkable rather than asserted.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from p2c.models.fusion import build_fusion, fusion_is_capacity_matched
from p2c.models.view_encoder import build_encoder


class TinyBCPolicy(nn.Module):
    def __init__(
        self,
        action_dim: int,
        state_dim: int = 0,
        num_views: int = 1,
        n_obs_steps: int = 1,
        action_horizon: int = 1,
        encoder: str = "tiny_cnn",
        fusion: str = "mean",
        feat_dim: int = 256,
        width: int = 32,
        hidden_dim: int = 512,
        input_res: int = 84,
        use_state: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim if use_state else 0
        self.num_views = num_views
        self.n_obs_steps = n_obs_steps
        self.action_horizon = action_horizon
        self.use_state = use_state and state_dim > 0
        self.fusion_name = fusion
        self.encoder_name = encoder

        # Frame history enters as extra channels, which keeps one shared encoder for all
        # views (and so keeps the parameter count independent of V).
        self.encoder = build_encoder(
            encoder,
            in_channels=3 * n_obs_steps,
            feat_dim=feat_dim,
            width=width,
            input_res=input_res,
        )
        self.fusion = build_fusion(fusion, feat_dim, num_views)

        head_in = self.fusion.out_dim + (self.state_dim if self.use_state else 0)
        layers: list[nn.Module] = [nn.Linear(head_in, hidden_dim), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, action_dim * action_horizon))
        self.head = nn.Sequential(*layers)

    # ---------------- forward ----------------

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor | None = None,
        view_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if images.dim() == 6:  # [B, V, T, 3, H, W] -> fold history into channels
            B, V, T, C, H, W = images.shape
            x = images.reshape(B, V, T * C, H, W)
        elif images.dim() == 5:  # [B, V, 3, H, W]
            B, V = images.shape[:2]
            x = images
        else:
            raise ValueError(f"expected images with 5 or 6 dims, got {images.shape}")

        # One shared encoder call over the flattened view axis.
        feats = self.encoder(x.reshape(B * V, *x.shape[2:])).reshape(B, V, -1)
        h = self.fusion(feats, view_mask)

        if self.use_state:
            if state is None:
                raise ValueError("policy was built with use_state=True but state is None")
            h = torch.cat([h, state], dim=-1)

        out = self.head(h)
        if self.action_horizon > 1:
            return out.reshape(-1, self.action_horizon, self.action_dim)
        return out

    # ---------------- capacity accounting (harness contract (docs/harness_contract.md)
    # C2) ----------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        ps = self.parameters()
        return sum(p.numel() for p in ps if p.requires_grad or not trainable_only)

    def capacity_report(self) -> dict:
        """Parameter counts per component, plus whether capacity matching (C2) holds.

        ``view_count_independent`` is the claim that matters: when True, swapping the
        view condition cannot change the model size, so a multi-view gain cannot be
        explained by extra capacity.
        """
        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        return {
            "params_encoder": count(self.encoder),
            "params_fusion": count(self.fusion),
            "params_head": count(self.head),
            "total_trainable": self.num_parameters(True),
            "total_all": self.num_parameters(False),
            "encoder_name": self.encoder_name,
            "encoder_shared_across_views": True,
            "fusion_name": self.fusion_name,
            "view_count_independent": fusion_is_capacity_matched(self.fusion_name),
            "num_views": self.num_views,
        }


# ---------------- losses and metrics ----------------


def bc_loss(
    pred: torch.Tensor, target: torch.Tensor, kind: str = "l2"
) -> torch.Tensor:
    if kind == "l2":
        return nn.functional.mse_loss(pred, target)
    if kind == "l1":
        return nn.functional.l1_loss(pred, target)
    if kind == "huber":
        return nn.functional.smooth_l1_loss(pred, target)
    raise ValueError(f"unknown loss '{kind}' (l2 | l1 | huber)")


@torch.no_grad()
def action_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    action_groups: dict[str, tuple[int, int]] | None = None,
) -> dict[str, float]:
    """Overall and per-group action errors.

    ``action_groups`` comes from the dataset's own ``modality.json`` (via the cache
    index), so the rotation and gripper metrics PLAN.md Stage 2 asks for are computed
    on the real slices rather than guessed offsets.

    Errors are reported in the *normalised* action space the model is trained in, which is
    what makes them comparable across view conditions (the statistics are shared).
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    p = pred.reshape(-1, pred.shape[-1]).float()
    t = target.reshape(-1, target.shape[-1]).float()

    out = {
        "mse": torch.mean((p - t) ** 2).item(),
        "l1": torch.mean(torch.abs(p - t)).item(),
    }
    for name, (s, e) in (action_groups or {}).items():
        if e > p.shape[-1]:
            continue
        d = p[:, s:e] - t[:, s:e]
        out[f"mse/{name}"] = torch.mean(d**2).item()
        out[f"l1/{name}"] = torch.mean(torch.abs(d)).item()
    return out


@torch.no_grad()
def per_sample_squared_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error per sample, [B].

    The complementarity metric of PLAN.md section 1 and the stage breakdown of
    the per-stage breakdown both need error attributed to individual frames, not just a
    batch average.
    """
    p = pred.reshape(pred.shape[0], -1).float()
    t = target.reshape(target.shape[0], -1).float()
    return ((p - t) ** 2).mean(dim=1)
