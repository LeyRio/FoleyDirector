"""Minimal building blocks for the FoleyDirector backbone."""

import math

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding -> MLP."""

    def __init__(self, hidden_dim: int, frequency_embedding_size: int = 256, max_period: int = 10_000):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.timestep_embedding(t, self.frequency_embedding_size, self.max_period)
        return self.mlp(emb)
