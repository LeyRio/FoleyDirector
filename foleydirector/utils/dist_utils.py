"""Distributed training helpers."""

import os

local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))


def info_if_rank_zero(logger, msg: str) -> None:
    if local_rank == 0:
        logger.info(msg)
