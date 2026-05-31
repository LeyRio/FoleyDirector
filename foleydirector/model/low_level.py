"""Lightweight low-level building blocks used across FoleyDirector."""

import torch
import torch.nn as nn


class ChannelLastConv1d(nn.Conv1d):
    """1D conv that operates on (B, T, C) tensors (channel-last)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = super().forward(x)
        return x.transpose(1, 2)


class MLP(nn.Module):
    """Two-layer MLP with SiLU activation."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvMLP(nn.Module):
    """1D ConvMLP for per-token mixing along the temporal axis (channel-last)."""

    def __init__(self, dim: int, hidden_dim: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.fc1 = ChannelLastConv1d(dim, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.act = nn.SiLU()
        self.fc2 = ChannelLastConv1d(hidden_dim, dim, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
