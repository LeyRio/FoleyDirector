"""Runner: training / validation / inference orchestration.

The Runner intentionally exposes the same surface that ``train.py`` consumes
(`enter_train`, `train_pass`, `validation_pass`, `inference_pass`,
`load_checkpoint`, `load_weights`, `get_latest_checkpoint_path`, `eval`,
`rng`, `log`, `network`).

This is a clean skeleton; plug in your full implementation (flow-matching loss,
EMA, classifier-free dropout, BigVGAN vocoder, etc.) when releasing weights.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig

from foleydirector.model.networks import FoleyDirectorNet, PreprocessedConditions

log = logging.getLogger(__name__)


_MODEL_PRESETS = {
    "small_16k":  dict(hidden_dim=384,  depth=12, fused_depth=2,  num_heads=6,  latent_dim=20),
    "medium_44k": dict(hidden_dim=896,  depth=24, fused_depth=4,  num_heads=14, latent_dim=40),
    "large_44k":  dict(hidden_dim=1152, depth=36, fused_depth=6,  num_heads=18, latent_dim=40),
}


class _RNGState:
    """Thin wrapper to mimic torch.Generator state save/restore."""

    def __init__(self, device="cuda"):
        self.gen = torch.Generator(device=device)

    def graphsafe_get_state(self):
        return self.gen.get_state()

    def graphsafe_set_state(self, state):
        self.gen.set_state(state)


class Runner:
    def __init__(
        self,
        cfg: DictConfig,
        log,  # TensorboardLogger
        run_path: str,
        for_training: bool,
        latent_mean: Optional[torch.Tensor] = None,
        latent_std: Optional[torch.Tensor] = None,
    ):
        self.cfg = cfg
        self.log = log
        self.run_path = run_path
        self.for_training = for_training

        preset = _MODEL_PRESETS[cfg.model]
        self.network = FoleyDirectorNet(
            **preset,
            clip_dim=cfg.data_dim.clip_dim,
            sync_dim=cfg.data_dim.sync_dim,
            text_dim=cfg.data_dim.text_dim,
            latent_seq_len=cfg.data_dim.latent_seq_len,
            clip_seq_len=cfg.data_dim.clip_seq_len,
            sync_seq_len=cfg.data_dim.sync_seq_len,
            text_seq_len=cfg.data_dim.text_seq_len,
            latent_mean=latent_mean,
            latent_std=latent_std,
            sep_fgc=cfg.sep_fgc,
            joint_fgc=cfg.joint_fgc,
            delay_fusion=cfg.delay_fusion,
            interleaved_rope=cfg.interleaved_rope,
            fusion_fgc_clip=cfg.fusion_fgc_clip,
            fgc_pt=cfg.fgc_pt,
            fgc_dim=cfg.fgc_dim,
        ).cuda()

        if dist.is_available() and dist.is_initialized():
            self.network = nn.parallel.DistributedDataParallel(
                self.network, device_ids=[torch.cuda.current_device()],
                find_unused_parameters=False,
            )

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
        )
        self.rng = _RNGState()

    # --------------------------------------------------------------------
    # Lifecycle hooks (no-ops in this skeleton)
    # --------------------------------------------------------------------
    def enter_train(self):
        self.network.train()
        return self

    def enter_val(self):
        self.network.eval()
        return self

    # --------------------------------------------------------------------
    # Training step
    # --------------------------------------------------------------------
    def train_pass(self, data: dict, curr_iter: int) -> None:
        """Single training step (placeholder - flow matching loss).

        Replace with the full loss (e.g., flow-matching with classifier-free
        guidance dropout) used in the paper.
        """
        latent = data["latent"].cuda(non_blocking=True)
        clip_f = data["clip_feature"].cuda(non_blocking=True)
        sync_f = data["sync_feature"].cuda(non_blocking=True)
        text_f = data["text_feature"].cuda(non_blocking=True)
        fgc_f = data["fgc_feature"].cuda(non_blocking=True)

        bs = latent.shape[0]
        t = torch.rand(bs, device=latent.device)
        noise = torch.randn_like(latent)
        latent_noisy = (1 - t.view(-1, 1, 1)) * latent + t.view(-1, 1, 1) * noise
        target = noise - latent  # flow-matching velocity

        net = self.network.module if isinstance(self.network, nn.parallel.DistributedDataParallel) else self.network
        conds = net.preprocess_conditions(clip_f, sync_f, text_f, fgc_f)
        pred = self.network(latent_noisy, t, conds)

        loss = torch.nn.functional.mse_loss(pred, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.cfg.clip_grad_norm)
        self.optimizer.step()

        if curr_iter % self.cfg.log_text_interval == 0:
            self.log.add_scalar("train/loss", loss.item(), curr_iter)
            self.log.info(f"iter {curr_iter}: loss={loss.item():.4f}")

        if curr_iter and curr_iter % self.cfg.save_checkpoint_interval == 0:
            self.save_checkpoint(curr_iter)

    @torch.inference_mode()
    def validation_pass(self, data: dict, curr_iter: int):
        return None

    @torch.inference_mode()
    def inference_pass(self, data: dict, curr_iter: int, val_cfg, save_eval: bool = False):
        return None

    def eval(self, audio_path, curr_iter, val_cfg):  # noqa: D401
        return None

    # --------------------------------------------------------------------
    # Checkpointing
    # --------------------------------------------------------------------
    def get_latest_checkpoint_path(self) -> Optional[str]:
        ckpt_dir = Path(self.run_path) / "ckpts"
        if not ckpt_dir.exists():
            return None
        ckpts = sorted(ckpt_dir.glob("ckpt_*.pth"))
        return str(ckpts[-1]) if ckpts else None

    def save_checkpoint(self, curr_iter: int) -> None:
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        ckpt_dir = Path(self.run_path) / "ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"ckpt_{curr_iter:09d}.pth"
        net = self.network.module if isinstance(self.network, nn.parallel.DistributedDataParallel) else self.network
        torch.save(
            {
                "iter": curr_iter,
                "model": net.state_dict(),
                "optim": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> int:
        sd = torch.load(path, map_location="cpu")
        net = self.network.module if isinstance(self.network, nn.parallel.DistributedDataParallel) else self.network
        net.load_state_dict(sd["model"], strict=False)
        if "optim" in sd:
            self.optimizer.load_state_dict(sd["optim"])
        return int(sd.get("iter", 0))

    def load_weights(self, path: str) -> None:
        sd = torch.load(path, map_location="cpu")
        if "model" in sd:
            sd = sd["model"]
        net = self.network.module if isinstance(self.network, nn.parallel.DistributedDataParallel) else self.network
        missing, unexpected = net.load_state_dict(sd, strict=False)
        if missing or unexpected:
            log.info(f"Loaded weights with missing={len(missing)}, unexpected={len(unexpected)}")
