"""FoleyDirector training entry point.

This script trains the FoleyDirector model (DiT-based V2A backbone with
Structured Temporal Scripts (STS) conditioning + Script-Guided Temporal
Fusion Module). It is built on top of an MMAudio-style runner.

Quick start
-----------
    # Single-node, 8 GPUs:
    bash train.sh

    # Or directly via torchrun:
    torchrun --standalone --nproc_per_node=8 train.py \
        exp_id=foleydirector_run0 \
        model=medium_44k

Environment variables
---------------------
    DATA_PREFIX   Root directory containing datasets / pretrained CLIP / etc.
                  Defaults to "./data_root". All paths in the YAML configs
                  are resolved relative to this prefix.

The actual model code lives in the ``foleydirector`` package (see README).
"""

import logging
import math
import os
import random
from datetime import timedelta
from pathlib import Path

import hydra
import numpy as np
import open_clip
import torch
import torch.distributed as distributed
import torch.nn.functional as F
from hydra import compose
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from torch.distributed.elastic.multiprocessing.errors import record
from torchvision.transforms import Normalize

# NOTE: rename ``mmaudio`` to your local package if you keep that name.
from foleydirector.data.data_setup import (
    setup_training_datasets,
    setup_val_datasets,
    update_threshold,
)
from foleydirector.model.sequence_config import CONFIG_16K, CONFIG_44K
from foleydirector.runner import Runner
from foleydirector.utils.dist_utils import info_if_rank_zero, local_rank, world_size
from foleydirector.utils.logger import TensorboardLogger

DATA_PREFIX = os.environ.get("DATA_PREFIX", "./data_root")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()


# ----------------------------------------------------------------------------
# CLIP utilities (used for encoding STS scripts)
# ----------------------------------------------------------------------------
def patch_clip(clip_model):
    """Make CLIP text encoder return per-token last-hidden-state.

    Reference:
        https://github.com/mlfoundations/open_clip
    """

    def new_encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)
        return F.normalize(x, dim=-1) if normalize else x

    clip_model.encode_text = new_encode_text.__get__(clip_model)
    return clip_model


@torch.inference_mode()
def encode_text(text, tokenizer, clip_model):
    """Per-token text encoding -> (B, T, D)."""
    tokens = tokenizer(text).cuda()
    return clip_model.encode_text(tokens, normalize=True)


@torch.inference_mode()
def encode_text_pool(text, tokenizer, clip_model):
    """Mask-aware mean pooling -> (B, 1, D)."""
    tokens = tokenizer(text).cuda()
    sequence_output = clip_model.encode_text(tokens, normalize=True)
    attention_mask = (tokens != 0).float().unsqueeze(-1)
    sum_embeddings = torch.sum(sequence_output * attention_mask, dim=1)
    sum_mask = torch.clamp(torch.sum(attention_mask, dim=1), min=1e-9)
    return (sum_embeddings / sum_mask).unsqueeze(1)


# ----------------------------------------------------------------------------
# Structured Temporal Scripts (STS) feature builders
# ----------------------------------------------------------------------------
def process_fgc_feature(fgc_feature, tokenizer, clip_model, fgc_pool=False, fgc_pt=10):
    """Default STS pipeline: 8 segments x ``fgc_pt`` tokens each.

    Args:
        fgc_feature: list of length 8, each element a list[str] of size B
                     (i.e., a transposed batch of 8-segment scripts).
        tokenizer: open_clip tokenizer.
        clip_model: patched CLIP text encoder.
        fgc_pool:  If True, use mean-pooled token (broadcast to fgc_pt slots).
        fgc_pt:    Number of script tokens allocated per segment.

    Returns:
        Tensor of shape (B, 8 * fgc_pt, D).
    """
    batch_size = len(fgc_feature[0])
    all_batch_feats = []

    for bi in range(batch_size):
        segment_texts = [fgc_feature[t][bi] for t in range(8)]
        feats = encode_text(segment_texts, tokenizer, clip_model)  # (8, T, D) or (8, D)
        if feats.ndim == 2:
            feats = feats.unsqueeze(1)

        if fgc_pool:
            pooled = feats.mean(dim=1)
            seg_tokens = pooled.unsqueeze(1).expand(-1, fgc_pt, -1)  # (8, fgc_pt, D)
        else:
            seg_tokens = feats[:, :fgc_pt, :]  # (8, fgc_pt, D)

        all_batch_feats.append(seg_tokens.reshape(8 * fgc_pt, -1))

    return torch.stack(all_batch_feats, dim=0)


def process_fgc_feature_v2(v2_cap, v2_tr, tokenizer, clip_model, fgc_pt=10):
    """Variable-duration STS pipeline (DirectorBench-style annotations).

    Each caption is associated with a time-range mask ``v2_tr``.
    A blank caption is encoded once and used as the default background.
    """
    total_tokens = 8 * fgc_pt
    batch_size = v2_tr.shape[0]
    all_batch_feats = []
    valid = [1] * batch_size

    for b in range(batch_size):
        categories = [cap[b] for cap in v2_cap]
        tr_mask = v2_tr[b]
        all_empty = all(all(not cap.strip() for cap in cap_tuple) for cap_tuple in categories)
        valid[b] = int(not all_empty)

        all_categories = list(categories) + [""]
        category_feats = encode_text_pool(all_categories, tokenizer, clip_model)
        category_embeddings = {c: category_feats[i] for i, c in enumerate(all_categories)}

        token_features = torch.zeros(total_tokens, category_feats.shape[-1]).to(
            category_feats[0].device
        )
        token_features[:] = category_embeddings[""]  # background

        for i, text in enumerate(categories):
            mask = tr_mask[i]
            if text and mask.any():
                token_features[mask] = category_embeddings[text]

        all_batch_feats.append(token_features)

    return torch.stack(all_batch_feats, dim=0), valid


# ----------------------------------------------------------------------------
# Distributed helper
# ----------------------------------------------------------------------------
def distributed_setup():
    distributed.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    log.info(f"Initialized: local_rank={local_rank}, world_size={world_size}")
    return local_rank, world_size


# ----------------------------------------------------------------------------
# Main training loop
# ----------------------------------------------------------------------------
@record
@hydra.main(version_base="1.3.2", config_path="configs", config_name="train.yaml")
def train(cfg: DictConfig):
    # ---- distributed init ------------------------------------------------
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    distributed_setup()
    num_gpus = world_size
    run_dir = HydraConfig.get().run.dir

    eval_cfg = compose("eval", overrides=[f"exp_id={cfg.exp_id}"])

    # ---- patch sequence dims based on sample rate ------------------------
    if cfg.model.endswith("16k"):
        seq_cfg = CONFIG_16K
    elif cfg.model.endswith("44k"):
        seq_cfg = CONFIG_44K
    else:
        raise ValueError(f"Unknown model: {cfg.model}")

    with open_dict(cfg):
        cfg.data_dim.latent_seq_len = seq_cfg.latent_seq_len
        cfg.data_dim.clip_seq_len = seq_cfg.clip_seq_len
        cfg.data_dim.sync_seq_len = seq_cfg.sync_seq_len

    # ---- logger ----------------------------------------------------------
    log_obj = TensorboardLogger(
        cfg.exp_id,
        run_dir,
        logging.getLogger(),
        is_rank0=(local_rank == 0),
        enable_email=cfg.enable_email and not cfg.debug,
    )
    info_if_rank_zero(log_obj, f"All configuration: {cfg}")
    info_if_rank_zero(log_obj, f"Number of GPUs detected: {num_gpus}")

    # ---- seeds -----------------------------------------------------------
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    cfg.batch_size //= num_gpus
    info_if_rank_zero(log_obj, f"Batch size (per GPU): {cfg.batch_size}")

    total_iterations = cfg["num_iterations"]

    # ---- datasets --------------------------------------------------------
    dataset, sampler, loader = setup_training_datasets(cfg)
    info_if_rank_zero(log_obj, f"Number of training samples: {len(dataset)}")
    info_if_rank_zero(log_obj, f"Number of training batches: {len(loader)}")

    latent_mean, latent_std = dataset.compute_latent_stats()

    # ---- runner ----------------------------------------------------------
    trainer = Runner(
        cfg,
        log=log_obj,
        run_path=run_dir,
        for_training=True,
        latent_mean=latent_mean,
        latent_std=latent_std,
    ).enter_train()
    eval_rng_clone = trainer.rng.graphsafe_get_state()

    # ---- resume / load weights ------------------------------------------
    curr_iter = 0
    if cfg["checkpoint"] is not None:
        curr_iter = trainer.load_checkpoint(cfg["checkpoint"])
        cfg["checkpoint"] = None
        info_if_rank_zero(log_obj, "Model checkpoint loaded.")
    else:
        latest = trainer.get_latest_checkpoint_path()
        if latest is not None:
            curr_iter = trainer.load_checkpoint(latest)
            info_if_rank_zero(log_obj, "Latest checkpoint loaded.")
        elif cfg["weights"] is not None:
            info_if_rank_zero(log_obj, f"Loading initial weights from {cfg['weights']}")
            trainer.load_weights(cfg["weights"])
            cfg["weights"] = None

    total_epoch = math.ceil(total_iterations / len(loader))
    current_epoch = curr_iter // len(loader)
    info_if_rank_zero(log_obj, f"We will use approximately {total_epoch} epochs.")

    # ---- selectively unfreeze parameters when adapter-tuning -------------
    if not cfg.train_full:
        for name, param in trainer.network.named_parameters():
            requires_grad = (
                ("fgc" in name)
                or ("clip" in name and (not cfg.sep_fgc or cfg.clip_train))
                or ("fusion" in name and not cfg.sep_fgc)
                or ("delay" in name)
            )
            param.requires_grad = requires_grad

    # ---- CLIP encoder for STS encoding -----------------------------------
    clip_root = os.path.join(DATA_PREFIX, cfg.clip.weights_dir)
    clip_model = open_clip.create_model_from_pretrained(
        f"local-dir:{clip_root}", return_transform=False
    ).cuda()
    _ = Normalize(  # exposed for downstream image preprocessing if needed
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    clip_model = patch_clip(clip_model)
    tokenizer = open_clip.get_tokenizer(os.path.join(DATA_PREFIX, cfg.clip.tokenizer_dir))

    # ---- training loop ---------------------------------------------------
    dynamic_iters = cfg.dynamic_iterations
    np.random.seed(np.random.randint(2**30 - 1) + local_rank * 1000)

    while curr_iter < total_iterations:
        sampler.set_epoch(current_epoch)
        current_epoch += 1
        log_obj.debug(f"Current epoch: {current_epoch}")

        if cfg.dynamic_threshold:
            new_threshold = cfg.dt_list[
                min(current_epoch % dynamic_iters, len(cfg.dt_list) - 1)
            ]
            dataset, sampler, loader = update_threshold(cfg, dataset, new_threshold)

        trainer.enter_train()
        trainer.log.data_timer.start()

        for data in loader:
            # ---- build STS feature on the fly ---------------------------
            fgc_feature = process_fgc_feature(
                data["fgc_feature"], tokenizer, clip_model, cfg.fgc_pool, fgc_pt=cfg.fgc_pt
            ).detach().cpu()
            fgc_feature_v2, valid = process_fgc_feature_v2(
                data["v2_cap"], data["v2_tr"], tokenizer, clip_model, fgc_pt=cfg.fgc_pt
            )
            fgc_feature_v2 = fgc_feature_v2.detach().cpu()
            valid_tensor = torch.tensor(valid, dtype=torch.bool, device=fgc_feature.device)

            fgc_feature_input = fgc_feature
            if random.random() < cfg.v2_ratio:
                fgc_feature_input[valid_tensor] = fgc_feature_v2[valid_tensor]
            data["fgc_feature"] = fgc_feature_input

            trainer.train_pass(data, curr_iter)
            curr_iter += 1
            if curr_iter >= total_iterations:
                break

    # ---- finalise --------------------------------------------------------
    del trainer
    torch.cuda.empty_cache()
    distributed.barrier()

    log_obj.complete()
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    train()
