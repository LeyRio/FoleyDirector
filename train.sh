#!/usr/bin/env bash
# Default 8-GPU single-node training entry point for FoleyDirector.
#
# Usage:
#   bash train.sh                     # uses configs/train.yaml as is
#   bash train.sh exp_id=run0 model=medium_44k batch_size=16
#
# All extra arguments are forwarded to Hydra.

set -euo pipefail

export DATA_PREFIX="${DATA_PREFIX:-./data_root}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"

torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    train.py "$@"
