"""Training-time image augmentation.

Only one transform, and it is applied in the training loop rather than in the Dataset.
That placement is deliberate:

* the Dataset stays **deterministic**, so validation frames are byte-identical across view
  conditions and their per-sample errors can be joined frame by frame
  (SETUP.md sections 7 and 13);
* augmentation therefore touches training only, and never perturbs the numbers the
  comparison is read from.

Random shift is the standard regulariser for pixel-based control (DrQ, Kostrikov et al.
2020). It is needed here because the tiny BC model memorises the training split within a
couple of epochs otherwise — the first composite run reached train MSE 0.14 while
validation MSE rose, which would reduce the ablation to "which arm happened to be best at
epoch 1".

The same shift magnitude is applied to every view in every condition, so it cannot favour
one arm over another.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def random_shift(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Pad by ``pad`` pixels (replicate) and take a random crop back to the input size.

    Accepts ``[B, V, C, H, W]`` or ``[B, V, T, C, H, W]``. Each sample-and-view gets its
    own shift, which is what makes this a regulariser rather than a global jitter.
    """
    if pad <= 0:
        return x

    orig_shape = x.shape
    if x.dim() == 6:  # fold history into channels for the shift, then restore
        B, V, T, C, H, W = orig_shape
        flat = x.reshape(B * V, T * C, H, W)
    elif x.dim() == 5:
        B, V, C, H, W = orig_shape
        flat = x.reshape(B * V, C, H, W)
    else:
        raise ValueError(f"expected 5 or 6 dims, got {orig_shape}")

    n, _, h, w = flat.shape
    padded = F.pad(flat, (pad, pad, pad, pad), mode="replicate")

    # One (dx, dy) per image, drawn on the input's device so this adds no host sync.
    offs = torch.randint(0, 2 * pad + 1, (n, 2), device=x.device)

    rows = torch.arange(h, device=x.device)
    cols = torch.arange(w, device=x.device)
    r = rows.view(1, h, 1) + offs[:, 0].view(n, 1, 1)
    c = cols.view(1, 1, w) + offs[:, 1].view(n, 1, 1)
    idx_n = torch.arange(n, device=x.device).view(n, 1, 1)

    out = padded[idx_n, :, r, c]          # [n, h, w, C]
    out = out.permute(0, 3, 1, 2).contiguous()
    return out.reshape(orig_shape)


def augment_batch(images: torch.Tensor, pad: int = 4, enabled: bool = True) -> torch.Tensor:
    """Apply training augmentation. A no-op when disabled, so eval paths stay exact."""
    if not enabled:
        return images
    return random_shift(images, pad)
