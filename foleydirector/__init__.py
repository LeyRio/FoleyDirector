"""FoleyDirector - Fine-Grained Temporal Steering for Video-to-Audio Generation.

The package layout mirrors a typical DiT-based generative pipeline:

    foleydirector/
      data/        : datasets, memmap loaders, STS annotation parsers
      ext/         : external feature extractors (CLIP, Synchformer, BigVGAN, VAE)
      model/       : the FoleyDirector network, transformer blocks, embeddings
      utils/       : distributed helpers, logger, EMA, etc.
      runner.py    : Runner class that encapsulates train / val / inference loops
      sample.py    : standalone sampling entry-point
      eval_utils.py: evaluation metric helpers (FD, KL, IB, DeSync, ...)
"""

__version__ = "0.1.0"
