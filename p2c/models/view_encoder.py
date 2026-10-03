"""Per-view visual encoder: small enough for a 4 GB GTX 1650.

One encoder instance is *shared* across all cameras. That is the main lever for
SETUP.md section 12 Control B: because the same weights process every view, the trainable
parameter count of the encoder does not grow with the number of views, so a 3-view model
is not trivially stronger than a 1-view model simply by having more capacity.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyCNNEncoder(nn.Module):
    """A 4-layer strided CNN in the DrQ/Impala mould, ~0.5 M parameters.

    GroupNorm rather than BatchNorm: batch statistics would be computed over the flattened
    ``B * V`` view axis, which would make a 1-view batch and a 3-view batch normalise
    differently and quietly confound every comparison in this study.
    """

    def __init__(
        self,
        in_channels: int = 3,
        feat_dim: int = 256,
        width: int = 32,
        input_res: int = 84,
    ):
        super().__init__()
        w = width
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, w, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, w), w),
            nn.ReLU(inplace=True),
            nn.Conv2d(w, w * 2, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, w * 2), w * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(w * 2, w * 4, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, w * 4), w * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(w * 4, w * 4, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, w * 4), w * 4),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_res, input_res)
            n_flat = self.conv(dummy).flatten(1).shape[1]
        self.proj = nn.Sequential(
            nn.Flatten(1), nn.Linear(n_flat, feat_dim), nn.LayerNorm(feat_dim)
        )
        self.feat_dim = feat_dim
        self.input_res = input_res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[N, C, H, W] -> [N, feat_dim]."""
        return self.proj(self.conv(x))


class FrozenResNetEncoder(nn.Module):
    """Optional frozen pretrained encoder (SETUP.md section 8 allows either).

    Frozen on purpose: it keeps VRAM low and keeps trainable capacity identical across
    view conditions. Loaded lazily so torchvision stays an optional dependency.
    """

    def __init__(self, feat_dim: int = 256, input_res: int = 84, arch: str = "resnet18"):
        super().__init__()
        from torchvision import models

        net = getattr(models, arch)(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(net.children())[:-1])  # drop fc
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        out_dim = net.fc.in_features
        self.proj = nn.Sequential(nn.Linear(out_dim, feat_dim), nn.LayerNorm(feat_dim))
        self.feat_dim = feat_dim
        self.input_res = input_res
        # ImageNet statistics; inputs arrive already scaled to [0, 1].
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):  # keep the frozen trunk in eval mode
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        with torch.no_grad():
            h = self.backbone(x).flatten(1)
        return self.proj(h)


def build_encoder(
    name: str = "tiny_cnn",
    in_channels: int = 3,
    feat_dim: int = 256,
    width: int = 32,
    input_res: int = 84,
) -> nn.Module:
    if name == "tiny_cnn":
        return TinyCNNEncoder(in_channels, feat_dim, width, input_res)
    if name.startswith("resnet"):
        return FrozenResNetEncoder(feat_dim, input_res, arch=name)
    raise ValueError(f"unknown encoder '{name}' (tiny_cnn | resnet18 | resnet34)")
