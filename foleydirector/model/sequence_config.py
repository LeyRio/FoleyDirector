"""Sequence-length presets for 16 kHz / 44 kHz Foley audio.

These constants are consumed by ``train.py`` to populate ``cfg.data_dim``.
Adjust to match your own audio VAE / vocoder.
"""

from dataclasses import dataclass


@dataclass
class SequenceConfig:
    sample_rate: int
    duration_sec: float
    latent_seq_len: int
    clip_seq_len: int
    sync_seq_len: int


# 16 kHz audio, 8s clip, latent stride ~16 -> 250 latent frames
CONFIG_16K = SequenceConfig(
    sample_rate=16_000,
    duration_sec=8.0,
    latent_seq_len=250,
    clip_seq_len=64,
    sync_seq_len=192,
)

# 44.1 kHz audio, 8s clip, latent stride matches MMAudio-44k -> 345 latent frames
CONFIG_44K = SequenceConfig(
    sample_rate=44_100,
    duration_sec=8.0,
    latent_seq_len=345,
    clip_seq_len=64,
    sync_seq_len=192,
)
