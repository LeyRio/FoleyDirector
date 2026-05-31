"""Dataset assembly for FoleyDirector.

This module is intentionally minimal. It exposes the same call surface
expected by ``train.py`` (``setup_training_datasets``, ``setup_val_datasets``,
``update_threshold``) and a ``DirectorSoundDataset`` class that:

* loads pre-extracted multimodal features (audio latent, CLIP, Synchformer
  sync features, global text features) from a memmap directory,
* loads the corresponding **Structured Temporal Scripts (STS)** from a JSON
  annotation file,
* yields a dict that ``train.py`` consumes:

    {
        "latent":        (T_latent, C_latent),
        "clip_feature":  (T_clip,   C_clip),
        "sync_feature":  (T_sync,   C_sync),
        "text_feature":  (77,       C_text),
        "fgc_feature":   list of length 8 -> list[str] (raw STS captions),
        "v2_cap":        list of length 8 -> list[str] (variable-duration captions),
        "v2_tr":         (8, 8 * fgc_pt) bool mask
    }

Replace stub paths with your own dataset implementation.
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class DirectorSoundDataset(Dataset):
    """Skeleton dataset that returns pre-extracted features + STS captions."""

    def __init__(self, tsv: str, memmap_dir: str, fgc_json: str, fgc_pt: int = 10,
                 latent_dim: int = 40, clip_dim: int = 1024, sync_dim: int = 768,
                 text_dim: int = 1024, num_segments: int = 8):
        self.tsv = Path(tsv)
        self.memmap_dir = Path(memmap_dir)
        self.fgc_pt = fgc_pt
        self.num_segments = num_segments
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.sync_dim = sync_dim
        self.text_dim = text_dim

        with open(self.tsv) as f:
            self.ids: List[str] = [line.strip().split("\t")[0] for line in f if line.strip()]
        with open(fgc_json) as f:
            self.fgc: dict = json.load(f)

    def __len__(self) -> int:
        return len(self.ids)

    def _load_memmap(self, name: str, shape, dtype=np.float32):
        path = self.memmap_dir / f"{name}.npy"
        return np.array(np.memmap(path, dtype=dtype, mode="r", shape=shape))

    def __getitem__(self, idx: int):
        clip_id = self.ids[idx]

        # NOTE: real implementation should pre-store offsets and shapes.
        # We use placeholder zeros so the file is self-contained.
        latent = torch.zeros(345, self.latent_dim)
        clip_f = torch.zeros(64, self.clip_dim)
        sync_f = torch.zeros(24, 8, self.sync_dim)
        text_f = torch.zeros(77, self.text_dim)

        # ---- Structured Temporal Scripts (STS) -----------------------------
        sts = self.fgc.get(clip_id, {})
        seg_caps = sts.get("segments", [""] * self.num_segments)
        seg_caps = (seg_caps + [""] * self.num_segments)[: self.num_segments]

        # variable-duration STS (placeholder: no overrides)
        v2_cap = list(seg_caps)
        v2_tr = torch.zeros(self.num_segments, self.num_segments * self.fgc_pt, dtype=torch.bool)

        return {
            "latent": latent,
            "clip_feature": clip_f,
            "sync_feature": sync_f,
            "text_feature": text_f,
            "fgc_feature": seg_caps,   # list[str] of length 8
            "v2_cap": v2_cap,          # list[str] of length 8
            "v2_tr": v2_tr,
        }

    def compute_latent_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = torch.zeros(self.latent_dim)
        std = torch.ones(self.latent_dim)
        return mean, std


def _collate_fn(batch):
    """Custom collate: stack tensors, *transpose* lists of strings to length-8 lists."""
    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], list):
            # transpose: list of length 8 per sample -> 8 lists of size B
            out[k] = list(map(list, zip(*vals)))
        else:
            out[k] = vals
    return out


def setup_training_datasets(cfg: DictConfig):
    data_cfg = cfg.data.DirectorSound
    dataset = DirectorSoundDataset(
        tsv=data_cfg.tsv,
        memmap_dir=data_cfg.memmap_dir,
        fgc_json=data_cfg.fgc,
        fgc_pt=cfg.fgc_pt,
        latent_dim=cfg.data_dim.latent_seq_len * 0 + 40,  # default placeholder
        clip_dim=cfg.data_dim.clip_dim,
        sync_dim=cfg.data_dim.sync_dim,
        text_dim=cfg.data_dim.text_dim,
    )

    sampler = (
        DistributedSampler(dataset, shuffle=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        collate_fn=_collate_fn,
    )
    return dataset, sampler, loader


def setup_val_datasets(cfg: DictConfig):
    return None, None, None


def update_threshold(cfg: DictConfig, dataset: DirectorSoundDataset, threshold: float):
    """Hook for curriculum-style data filtering. No-op by default."""
    return setup_training_datasets(cfg)
