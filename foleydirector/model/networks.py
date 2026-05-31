"""FoleyDirector backbone network.

This is a faithful, cleaned-up sketch of the public model used in the paper.
It implements:
    * Audio / video / sync / text input projections (shared with MMAudio-style models)
    * **Script-Guided Temporal Fusion Module** via separate FGC tokens
      (when ``sep_fgc=True``)
    * **Interleaved RoPE** between audio and visual / script tokens
    * **Bi-Frame** in-frame / out-of-frame audio synthesis (handled at the
      Runner level by feeding two parallel streams)

Sub-modules (transformer blocks, embeddings, low_level convs) are kept compact
on purpose - replace them with your own implementations as needed.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from foleydirector.model.embeddings import TimestepEmbedder
from foleydirector.model.low_level import ChannelLastConv1d, ConvMLP, MLP
from foleydirector.model.transformer_layers import (
    FinalBlock,
    JointBlock,
    MMDitSingleBlock,
)
from foleydirector.utils.rope import compute_rope_rotations

log = logging.getLogger(__name__)


@dataclass
class PreprocessedConditions:
    """Conditioning features after projection / pooling."""

    clip_f: torch.Tensor       # visual frame tokens
    sync_f: torch.Tensor       # synchformer tokens
    text_f: torch.Tensor       # global caption tokens
    clip_f_c: torch.Tensor     # pooled visual context
    text_f_c: torch.Tensor     # pooled text context
    fgc_f: torch.Tensor        # Structured Temporal Script tokens


class FoleyDirectorNet(nn.Module):
    """DiT-based V2A backbone with Structured Temporal Scripts (STS)."""

    def __init__(
        self,
        *,
        latent_dim: int,
        clip_dim: int,
        sync_dim: int,
        text_dim: int,
        hidden_dim: int,
        depth: int,
        fused_depth: int,
        num_heads: int,
        latent_seq_len: int,
        clip_seq_len: int,
        sync_seq_len: int,
        text_seq_len: int = 77,
        mlp_ratio: float = 4.0,
        latent_mean: Optional[torch.Tensor] = None,
        latent_std: Optional[torch.Tensor] = None,
        empty_string_feat: Optional[torch.Tensor] = None,
        sep_fgc: bool = True,
        joint_fgc: bool = False,
        delay_fusion: bool = False,
        interleaved_rope: bool = True,
        fusion_fgc_clip: bool = False,
        fgc_pt: int = 10,
        fgc_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.fgc_pt = fgc_pt
        self.fgc_dim = fgc_dim if fgc_dim is not None else text_dim
        self.sep_fgc = sep_fgc or joint_fgc
        self.joint_fgc = joint_fgc
        self.delay_fusion = delay_fusion
        self.interleaved_rope = interleaved_rope
        self.fusion_fgc_clip = fusion_fgc_clip

        self._latent_seq_len = latent_seq_len
        self._clip_seq_len = clip_seq_len
        self._sync_seq_len = sync_seq_len
        self._text_seq_len = text_seq_len
        self._fgc_seq_len = 8 * fgc_pt

        # ------------------------------------------------------------------
        # Input projections
        # ------------------------------------------------------------------
        self.audio_input_proj = nn.Sequential(
            ChannelLastConv1d(latent_dim, hidden_dim, kernel_size=7, padding=3),
            nn.SELU(),
            ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
        )
        self.clip_input_proj = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
        )
        self.sync_input_proj = nn.Sequential(
            ChannelLastConv1d(sync_dim, hidden_dim, kernel_size=7, padding=3),
            nn.SELU(),
            ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
        )
        self.text_input_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            MLP(hidden_dim, hidden_dim * 4),
        )
        self.fgc_text_input_proj = nn.Sequential(
            nn.Linear(self.fgc_dim, hidden_dim),
            MLP(hidden_dim, hidden_dim * 4),
        )

        self.clip_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.global_cond_mlp = MLP(hidden_dim, hidden_dim * 4)

        # 8 sync feature frames per synchformer segment
        self.sync_pos_emb = nn.Parameter(torch.zeros((1, 1, 8, sync_dim)))

        # ------------------------------------------------------------------
        # Backbone blocks
        # ------------------------------------------------------------------
        self.t_embed = TimestepEmbedder(hidden_dim, frequency_embedding_size=256, max_period=10_000)

        self.joint_blocks = nn.ModuleList(
            [
                JointBlock(
                    hidden_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    sep_fgc=self.sep_fgc,
                    delay_fusion=self.delay_fusion,
                    joint_fgc=self.joint_fgc,
                    interleaved_rope=self.interleaved_rope,
                    fusion_fgc_clip=self.fusion_fgc_clip,
                    pre_only=(i == depth - fused_depth - 1),
                )
                for i in range(depth - fused_depth)
            ]
        )
        self.fused_blocks = nn.ModuleList(
            [
                MMDitSingleBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, kernel_size=3, padding=1)
                for _ in range(fused_depth)
            ]
        )
        self.final_layer = FinalBlock(hidden_dim, latent_dim)

        # ------------------------------------------------------------------
        # Mean/std + empty-feat buffers (for classifier-free guidance)
        # ------------------------------------------------------------------
        if latent_mean is None or latent_std is None:
            latent_mean = torch.full((1, 1, latent_dim), float("nan"))
            latent_std = torch.full((1, 1, latent_dim), float("nan"))
        if empty_string_feat is None:
            empty_string_feat = torch.zeros((text_seq_len, text_dim))
        self.latent_mean = nn.Parameter(latent_mean.view(1, 1, -1), requires_grad=False)
        self.latent_std = nn.Parameter(latent_std.view(1, 1, -1), requires_grad=False)
        self.empty_string_feat = nn.Parameter(empty_string_feat, requires_grad=False)
        self.empty_clip_feat = nn.Parameter(torch.zeros(1, clip_dim))
        self.empty_sync_feat = nn.Parameter(torch.zeros(1, sync_dim))
        self.empty_fgc_feat = nn.Parameter(torch.zeros(1, self.fgc_dim), requires_grad=False)

        self.initialize_rotations()

    # --------------------------------------------------------------------
    # RoPE precomputation
    # --------------------------------------------------------------------
    def initialize_rotations(self) -> None:
        head_dim = self.hidden_dim // self.num_heads
        base_freq = 1.0
        device = self.device

        self.latent_rot = nn.Buffer(
            compute_rope_rotations(self._latent_seq_len, head_dim, 10_000,
                                   freq_scaling=base_freq, device=device),
            persistent=False,
        )
        self.clip_rot = nn.Buffer(
            compute_rope_rotations(
                self._clip_seq_len, head_dim, 10_000,
                freq_scaling=base_freq * self._latent_seq_len / self._clip_seq_len,
                device=device,
            ),
            persistent=False,
        )
        self.fgc_rot = nn.Buffer(
            compute_rope_rotations(
                self._fgc_seq_len, head_dim, 10_000,
                freq_scaling=base_freq * self._latent_seq_len / self._fgc_seq_len,
                device=device,
            ),
            persistent=False,
        )
        self.inter_rot = nn.Buffer(
            compute_rope_rotations(self._latent_seq_len * 2, head_dim, 10_000,
                                   freq_scaling=base_freq, device=device),
            persistent=False,
        )
        self.inter_fusion_rot = nn.Buffer(
            compute_rope_rotations(self._clip_seq_len * 2, head_dim, 10_000,
                                   freq_scaling=base_freq, device=device),
            persistent=False,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # --------------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------------
    def preprocess_conditions(
        self,
        clip_f: torch.Tensor,
        sync_f: torch.Tensor,
        text_f: torch.Tensor,
        fgc_f: torch.Tensor,
    ) -> PreprocessedConditions:
        clip_f = self.clip_input_proj(clip_f)
        sync_f = sync_f + self.sync_pos_emb
        sync_f = self.sync_input_proj(sync_f.flatten(1, 2))
        text_f = self.text_input_proj(text_f)
        fgc_f = self.fgc_text_input_proj(fgc_f)

        clip_f_c = self.clip_cond_proj(clip_f.mean(dim=1))
        text_f_c = self.text_cond_proj(text_f.mean(dim=1))
        return PreprocessedConditions(
            clip_f=clip_f,
            sync_f=sync_f,
            text_f=text_f,
            clip_f_c=clip_f_c,
            text_f_c=text_f_c,
            fgc_f=fgc_f,
        )

    def forward(
        self,
        latent: torch.Tensor,
        t: torch.Tensor,
        conds: PreprocessedConditions,
    ) -> torch.Tensor:
        """One denoising step.

        Args:
            latent: (B, T, C) noisy audio latent.
            t:      (B,) timestep.
            conds:  preprocessed multimodal features.

        Returns:
            Predicted velocity / score with shape == latent.shape.
        """
        x = self.audio_input_proj(latent)
        global_cond = self.global_cond_mlp(self.t_embed(t) + conds.clip_f_c + conds.text_f_c)

        for block in self.joint_blocks:
            x, conds = block(
                x,
                conds,
                global_cond,
                latent_rot=self.latent_rot,
                clip_rot=self.clip_rot,
                fgc_rot=self.fgc_rot,
                inter_rot=self.inter_rot,
                inter_fusion_rot=self.inter_fusion_rot,
            )

        for block in self.fused_blocks:
            x = block(x, global_cond)

        return self.final_layer(x, global_cond)
