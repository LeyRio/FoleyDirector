"""Standalone sampling / inference entry-point.

Usage:
    python -m foleydirector.sample exp_id=run0 \
        sampling.num_steps=25 cfg_strength=4.5
"""

from omegaconf import DictConfig


def sample(cfg: DictConfig):
    """Run inference for the configured eval set.

    Replace this stub with your actual sampling routine - typically:
        1. Build the model + load EMA weights
        2. Iterate over the eval dataloader
        3. Encode video / sync / text / STS conditions
        4. Run the flow-matching solver (Euler / Heun) for cfg.sampling.num_steps
        5. Decode latent -> waveform via the audio VAE + vocoder
        6. Save .wav files to ``cfg.output_dir``
    """
    raise NotImplementedError(
        "Stub: plug in your inference loop. See README for sampler signature."
    )


if __name__ == "__main__":
    import hydra

    @hydra.main(version_base="1.3.2", config_path="../configs", config_name="eval")
    def _main(cfg: DictConfig):
        sample(cfg)

    _main()
