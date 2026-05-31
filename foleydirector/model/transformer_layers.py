"""Transformer building blocks for the FoleyDirector backbone.

The blocks are deliberately compact placeholders. Replace ``JointBlock`` /
``MMDitSingleBlock`` / ``FinalBlock`` with your full implementations from the
research codebase if you need bit-exact reproduction. They expose the same
interface used by ``foleydirector.model.networks.FoleyDirectorNet``.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ModulatedLayerNorm(nn.Module):
    """LayerNorm without affine; affine is provided by adaLN modulation."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale, shift = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class JointBlock(nn.Module):
    """Joint multimodal DiT block with optional separate FGC stream.

    Skeleton implementation: replace with the full Script-Guided Temporal
    Fusion variant (Temporal Script Attention + Interleaved RoPE) when
    open-sourcing the actual checkpoints.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        sep_fgc: bool = True,
        delay_fusion: bool = False,
        joint_fgc: bool = False,
        interleaved_rope: bool = True,
        fusion_fgc_clip: bool = False,
        pre_only: bool = False,
    ):
        super().__init__()
        self.sep_fgc = sep_fgc
        self.joint_fgc = joint_fgc
        self.delay_fusion = delay_fusion
        self.interleaved_rope = interleaved_rope
        self.fusion_fgc_clip = fusion_fgc_clip
        self.pre_only = pre_only

        self.norm_latent = _ModulatedLayerNorm(hidden_dim)
        self.norm_clip = _ModulatedLayerNorm(hidden_dim)
        self.norm_text = _ModulatedLayerNorm(hidden_dim)

        self.attn_latent = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.attn_clip = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.attn_text = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        if sep_fgc:
            self.norm_fgc = _ModulatedLayerNorm(hidden_dim)
            self.attn_fgc = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        conds,
        global_cond: torch.Tensor,
        *,
        latent_rot: Optional[torch.Tensor] = None,
        clip_rot: Optional[torch.Tensor] = None,
        fgc_rot: Optional[torch.Tensor] = None,
        inter_rot: Optional[torch.Tensor] = None,
        inter_fusion_rot: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, "PreprocessedConditions"]:  # noqa: F821
        # Latent self-attention
        h = self.norm_latent(x, global_cond)
        x = x + self.attn_latent(h, h, h, need_weights=False)[0]

        # Cross-modal interactions
        if self.sep_fgc:
            f = self.norm_fgc(conds.fgc_f, global_cond)
            x = x + self.attn_fgc(x, f, f, need_weights=False)[0]

        c = self.norm_clip(conds.clip_f, global_cond)
        x = x + self.attn_clip(x, c, c, need_weights=False)[0]

        t = self.norm_text(conds.text_f, global_cond)
        x = x + self.attn_text(x, t, t, need_weights=False)[0]

        # Feed-forward
        x = x + self.mlp(x)
        return x, conds


class MMDitSingleBlock(nn.Module):
    """Final fused DiT block (single-stream)."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.norm1 = _ModulatedLayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = _ModulatedLayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x, c)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x, c))
        return x


class FinalBlock(nn.Module):
    """Final DiT projection back to latent_dim."""

    def __init__(self, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        self.conv = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale, shift = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.conv(x.transpose(1, 2)).transpose(1, 2)
