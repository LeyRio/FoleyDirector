"""Tiny RoPE helper used by FoleyDirector.

Computes per-head rotary positional embeddings (cos / sin) once and lets
attention modules apply them at runtime.
"""

import torch


def compute_rope_rotations(
    length: int,
    dim: int,
    theta: float = 10_000,
    freq_scaling: float = 1.0,
    device=None,
):
    """Return rotary embedding tensor of shape (1, length, dim/2, 2, 2)."""
    half = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    freqs = freqs.to(device) * freq_scaling

    pos = torch.arange(length, device=device, dtype=torch.float32)
    angle = torch.einsum("i,j->ij", pos, freqs)  # (L, half)

    cos = torch.cos(angle)
    sin = torch.sin(angle)

    rot = torch.stack(
        [
            torch.stack([cos, -sin], dim=-1),
            torch.stack([sin, cos], dim=-1),
        ],
        dim=-2,
    )  # (L, half, 2, 2)
    return rot.unsqueeze(0)  # (1, L, half, 2, 2)
