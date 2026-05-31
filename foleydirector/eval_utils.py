"""Evaluation utilities (FD, KL, IB, DeSync ...) - stub.

Implement / wire up with av-benchmark or your in-house metrics package.
"""

from typing import Dict


def compute_metrics(generated_dir: str, reference_dir: str) -> Dict[str, float]:
    """Compute all metrics reported in the FoleyDirector paper.

    Returns a dict with keys:
        FD_VGG, FD_PANN, FD_PaSST, KL_PANN, KL_PaSST, ISC_PANN, IB, DeSync
    """
    raise NotImplementedError(
        "Stub: hook this up to your evaluation pipeline."
    )
