
# [ CVPR2026 Highlight] FoleyDirector: Fine-Grained Temporal Steering for Video-to-Audio Generation via Structured Scripts

### [[Paper]](https://arxiv.org/pdf/2603.19857) [[Project Page]](https://leyrio.github.io/FoleyDirector_ProjectPage/)

## 🔥 News
- **2026-02-21**: Our paper has been accepted by **CVPR 2026**.


## ✅ To Do List
- [x] Project Page
- [x] Training scaffolding
- [ ] Inference Code (Coming Soon)
- [ ] DirectorBench Benchmark (Coming Soon)
- [ ] Data Pipeline (Coming Soon)


## 📦 Repository Layout

```text
FoleyDirector/
├── train.py                      # multi-GPU training entry-point
├── train.sh                      # torchrun launcher (8 GPUs by default)
├── requirements.txt
├── configs/
│   ├── train.yaml                # main training config
│   ├── eval.yaml                 # inference / evaluation config
│   ├── data/
│   │   └── vggsound_director.yaml
│   └── eval_data/
│       └── directorbench.yaml
└── foleydirector/
    ├── data/                     # datasets + STS annotation loaders
    ├── ext/                      # external feature extractors (CLIP / VAE / vocoder)
    ├── model/
    │   ├── networks.py           # FoleyDirectorNet (DiT backbone + STS adapter)
    │   ├── transformer_layers.py # Joint / Single / Final blocks (SG-TFM hooks)
    │   ├── embeddings.py         # timestep / positional embeddings
    │   ├── low_level.py          # ChannelLastConv1d / MLP / ConvMLP
    │   └── sequence_config.py    # 16k / 44k sequence-length presets
    ├── utils/
    │   ├── rope.py               # rotary positional embeddings (incl. interleaved RoPE)
    │   ├── dist_utils.py         # local_rank / world_size helpers
    │   └── logger.py             # rank-aware Tensorboard logger
    ├── runner.py                 # train / val / inference orchestration
    ├── sample.py                 # standalone sampler entry-point
    └── eval_utils.py             # FD / KL / IB / DeSync metric stubs
```

## 🛠 Installation

```bash
git clone https://github.com/<your-account>/FoleyDirector.git
cd FoleyDirector

conda create -n foleydirector python=3.10 -y
conda activate foleydirector

pip install -r requirements.txt
pip install -e .                  # optional: register the foleydirector package
```

## 📁 Data & checkpoint layout

All paths in `configs/*.yaml` are resolved relative to the environment variable
`DATA_PREFIX` (default: `./data_root`):

```text
$DATA_PREFIX/
├── checkpoints/
│   ├── DFN5B-CLIP-ViT-H-14-384/          # open_clip text encoder
│   ├── ViT-H-14-378-quickgelu__dfn5b/    # tokenizer
│   ├── v1-44.pth                         # 44 kHz audio VAE
│   ├── best_netG.pt                      # BigVGAN vocoder
│   ├── synchformer_state_dict.pth        # video sync encoder
│   └── mmaudio_medium_44k.pth            # (optional) initialization weights
├── datasets/
│   ├── director_sound/
│   │   ├── train.tsv
│   │   ├── memmap/                       # extracted audio / video / text features
│   │   └── sts_annotations.json          # Structured Temporal Scripts
│   ├── VGGSound/
│   ├── VGGSound-Director/
│   └── directorbench/
└── train_output/                         # Hydra run dir (logs + checkpoints)
```

## 🚂 Training

### Quick start

```bash
export DATA_PREFIX=/path/to/your/data_root
bash train.sh exp_id=foleydirector_run0
```

This launches 8-GPU `torchrun` training with `configs/train.yaml`. All Hydra
overrides are forwarded:

```bash
bash train.sh \
    exp_id=foleydirector_large \
    model=large_44k \
    batch_size=32 \
    num_iterations=400_000 \
    interleaved_rope=True \
    fgc_pt=10 \
    v2_ratio=0.0
```

To run on fewer GPUs, set `NUM_GPUS`:

```bash
NUM_GPUS=4 bash train.sh exp_id=four_gpu_run
```

### Key training-time knobs

| Config key                | Description                                                                |
| ------------------------- | -------------------------------------------------------------------------- |
| `model`                   | `small_16k`, `medium_44k`, `large_44k` backbone preset                     |
| `weights`                 | path to a pretrained MMAudio / FoleyDirector checkpoint (resume-from-init) |
| `checkpoint`              | path to a full `(model, optim)` checkpoint (resume-from-state)             |
| `train_full`              | `False` = freeze base DiT, train only STS adapter + fusion modules         |
| `sep_fgc`                 | enable the separate **Script-Guided Temporal Fusion** stream               |
| `interleaved_rope`        | enable **Interleaved RoPE** between audio and visual / script tokens       |
| `fgc_pt`                  | number of script tokens per segment (8 segments → `8 * fgc_pt` total)      |
| `v2_ratio`                | probability of using variable-duration STS during training                 |
| `cfg_strength`            | classifier-free guidance scale used at sampling time                       |
| `sampling.num_steps`      | denoising steps (default 25, Euler)                                        |


## 🏫 About us
Thank you for your interest in this project. The project is supervised by the
ReLER Lab at Zhejiang University’s College of Computer Science and Technology
and [ByteDance](https://www.bytedance.com/en/). ReLER was established by
[Yang Yi](https://scholar.google.com/citations?user=RMSuNFwAAAAJ&hl=en), a Qiu
Shi Distinguished Professor at Zhejiang University. Our dedicated team of
contributors includes
[You Li](https://scholar.google.com/citations?user=2lRNus0AAAAJ&hl=en&oi=sra),
[Dewei Zhou](https://scholar.google.com/citations?hl=en&user=4C_OwWMAAAAJ),
[Fan Ma](https://scholar.google.com/citations?hl=en&user=FyglsaAAAAAJ),
[Yi Yang](https://scholar.google.com/citations?user=RMSuNFwAAAAJ&hl=en).

## 📮 Contact
If you have any questions, feel free to contact us via email
**uli2000@zju.edu.cn**.

## 📚 Citation
If you find this repository useful, please cite:

```bibtex
@misc{li2026foleydirectorfinegrainedtemporalsteering,
      title={FoleyDirector: Fine-Grained Temporal Steering for Video-to-Audio Generation via Structured Scripts},
      author={You Li and Dewei Zhou and Fan Ma and Fu Li and Dongliang He and Yi Yang},
      year={2026},
      eprint={2603.19857},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2603.19857},
}
```
