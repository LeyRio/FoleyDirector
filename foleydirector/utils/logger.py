"""Minimal Tensorboard wrapper used by ``train.py``."""

import logging
import os
import time
from typing import Optional


class _Timer:
    def __init__(self):
        self._t0 = None

    def start(self):
        self._t0 = time.time()

    def end(self):
        if self._t0 is None:
            return 0.0
        dt = time.time() - self._t0
        self._t0 = None
        return dt


class TensorboardLogger:
    """Lightweight rank-aware logger.

    Wraps the standard Python logger and (optionally) a Tensorboard
    SummaryWriter. The Runner calls ``data_timer.start()`` to track data-loading
    time, mirroring the original MMAudio interface.
    """

    def __init__(
        self,
        exp_id: str,
        run_dir: str,
        py_logger: logging.Logger,
        is_rank0: bool = True,
        enable_email: bool = False,
        enable_tb: bool = True,
    ):
        self.exp_id = exp_id
        self.run_dir = run_dir
        self.py_logger = py_logger
        self.is_rank0 = is_rank0
        self.enable_email = enable_email
        self.data_timer = _Timer()

        self.writer = None
        if enable_tb and is_rank0:
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(run_dir, exist_ok=True)
                self.writer = SummaryWriter(run_dir)
            except ImportError:
                self.py_logger.warning("Tensorboard not available; disable TB logging.")

    # ---- pass-through logging API -------------------------------------------
    def info(self, msg: str) -> None:
        if self.is_rank0:
            self.py_logger.info(msg)

    def debug(self, msg: str) -> None:
        if self.is_rank0:
            self.py_logger.debug(msg)

    def error(self, msg: str) -> None:
        self.py_logger.error(msg)

    def critical(self, msg: str) -> None:
        self.py_logger.critical(msg)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def complete(self) -> None:
        if self.writer is not None:
            self.writer.close()
