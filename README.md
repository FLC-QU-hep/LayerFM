# LayerFM
[![Python Version](https://img.shields.io/badge/Python_3.13-306998?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch_2.8-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_weights-mmozzani%2FLayerFM-yellow)](https://huggingface.co/mmozzani/LayerFM)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/Mmozzanica5/LayerFM/pre_commit.yaml?label=pre-commit&logo=github)](https://github.com/Mmozzanica5/LayerFM/actions/workflows/pre_commit.yaml)
[![Tests](https://img.shields.io/github/actions/workflow/status/Mmozzanica5/LayerFM/test.yaml?label=tests&logo=github)](https://github.com/Mmozzanica5/LayerFM/actions/workflows/test.yaml)

LayerFM generates the **per-layer structure of a hadronic shower** in a highly granular
calorimeter: for each of the 78 layers, the number of points (hits) and, optionally, the
earliest hit time in that layer, conditioned on the incident energy. It is the first
stage of a fast point-cloud shower generator, where these per-layer quantities fix how
many points the later stages produce and condition how they are placed.

The model is a conditional flow-matching network (CNF) trained on Geant4 showers, which
is then **distilled into a one-step student**: generation costs a single network
evaluation instead of tens of ODE steps.

Generating the per-layer point counts as a separate first stage was introduced in
[CaloHadronic](https://doi.org/10.1088/1748-0221/21/01/P01042) (Buss et al., JINST 21
(2026) P01042). LayerFM is built on [PointCountFM](https://github.com/FLC-QU-hep/PointCountFM)
by Thorsten Buss, the point-count model of [AllShowers](https://arxiv.org/abs/2601.11716).
It adds the per-layer minimum time as a second output channel, time-aware preprocessing,
and a working distillation path.

## Table of Contents <!-- omit in toc -->
- [Setup](#setup)
- [Pretrained weights](#pretrained-weights)
- [Data](#data)
- [Pipeline](#pipeline)
  - [1. Train the CNF](#1-train-the-cnf)
  - [2. Distill to one step](#2-distill-to-one-step)
  - [3. Generate](#3-generate)
  - [4. Validation plots](#4-validation-plots)
- [Configuration](#configuration)
- [Development](#development)
- [Citation](#citation)

## Setup
```bash
git clone git@github.com:FLC-QU-hep/LayerFM.git
cd LayerFM
```
Install the dependencies with uv:
```bash
uv sync --all-groups
source .venv/bin/activate
```
All commands below are run from the repository root with the environment active.

## Pretrained weights
The trained models are on Hugging Face: [huggingface.co/mmozzani/LayerFM](https://huggingface.co/mmozzani/LayerFM).

| Folder | Output per layer |
|---|---|
| `points_time/` | number of points + minimum hit time |
| `points_only/` | number of points |

## Data
Training data is an HDF5 file with one row per shower:

| key | shape | dtype | description |
|---|---|---|---|
| `energies` (or `energy`) | (n, 1) | float32 | incident energy (GeV) |
| `num_points` | (n, 78) | int32 | number of points per layer |
| `min_time_per_layer` (or `min_time`) | (n, 78) | float32 | earliest hit time per layer (ns); only for time models |
| `energy_per_layer` | (n, 78) | float32 | optional third output channel |
| `labels` | (n,) | int32 | optional particle label (with `use_labels: true`) |
| `directions` | (n, 3) | float32 | optional incident direction (with `use_directions: true`) |

The number of layers must match `num_layers` and `dim_input` in the configuration
(`dim_input` = 78 per output channel).

## Pipeline
The full chain is train, distill, generate. Every run writes to
`results/<YYYYmmdd_HHMMSS>_<name>/`, where `name` comes from the configuration.

### 1. Train the CNF
```bash
python pointcountfm/trainer.py config/config_T.yaml
```
`config/config_T.yaml` trains the points + time model; `config/config.yaml` trains the
points-only model. Set `data.data_file` to your HDF5 file first.

The run directory then contains:

| file | content |
|---|---|
| `conf.yaml` | the configuration, with `result_path` added |
| `checkpoint.pt` | model, optimizer and scheduler state (used to resume) |
| `compiled.pt` | self-contained TorchScript sampler (preprocessing built in) |
| `losses.csv`, `losses.pdf` | training and validation loss |
| `new_samples.h5` | 50 000 samples on the validation conditions (`training.num_new_samples`) |

**Resuming:** pass the run's own `conf.yaml`. It carries `result_path`, so training
continues from `checkpoint.pt` instead of starting a new run:
```bash
python pointcountfm/trainer.py results/<run>/conf.yaml
```

Trainer options:

| option | description |
|---|---|
| `-d`, `--device` | `cpu`, `cuda` or `mps` (default: automatic) |
| `-t`, `--time` | time the sampling speed |
| `--fast-dev-run` | tiny end-to-end run for testing |
| `--distill` | distill the trained model (step 2) |

### 2. Distill to one step
```bash
python pointcountfm/trainer.py results/<run>/conf.yaml --distill
```
Distillation runs on a finished run and happens in two phases:
1. **Teacher samples.** The CNF is integrated with 200 Heun steps on the full training +
   validation conditioning and the result is stored in `new_samples.h5`, together with
   the noise that produced each sample and the samples in model space. This is the most
   expensive part (about 2 h for 1.9 M showers on one A100).
2. **Student training.** A copy of the network learns the direct noise-to-sample map
   for `training.epochs` epochs.

It writes `checkpoint_distill.pt`, `compiled_distill.pt`, `losses_distill.{csv,pdf}` and
`new_samples_distilled.h5`. Re-running the same command resumes from
`checkpoint_distill.pt` and reuses `new_samples.h5`. Only one GPU is used.

### 3. Generate
Every run directory contains `compiled.pt` (CNF teacher) and, after distillation,
`compiled_distill.pt` (one-step student). They are self-contained TorchScript files with
the fitted preprocessing built in, so they run without this repository or the training
data:
```python
import torch

model = torch.jit.load("results/<run>/compiled_distill.pt", map_location="cpu")
energy = torch.full((1000, 1), 50.0)        # incident energy in GeV, shape (N, 1)
with torch.no_grad():
    num_points, min_time = model(energy)    # each (N, 78); points-only models return one tensor
```
The number of points comes out as a float per layer; round it to the nearest integer
and clip at 0 before use.

`pointcountfm/inference_cond_file.py` is the generation script used by the validation
plots (step 4). It loads a checkpoint (`--from-checkpoint`, plus `--distill` for the
student or `--steps N` for the teacher), takes the incident energies of the first N
showers of `data.data_file` in `config/config.yaml`, with N the length of the first
dataset in the given input file, and writes `num_points_per_layer` (and
`min_time_per_layer`) to the `-o` file.

### 4. Validation plots
```bash
python pointcountfm/plotting_withTandE.py results/<run> --distill   # student
python pointcountfm/plotting_withTandE.py results/<run> --steps 30  # teacher, 30 steps
```
This generates 20 000 showers and writes to `results/<run>/plots/`: the total number of
points, the number of points at 20, 50 and 80 GeV, the longitudinal profile with a
Generated/Geant4 ratio panel and, for time models, the minimum time per layer. The
Geant4 comparison file is set inside the script. Delete an existing
`output_FM_*.h5` before re-plotting the same setting, otherwise the old file is plotted.

## Configuration
A configuration file has four top-level keys: `name`, `data`, `model` and `training`.
See `config/config_T.yaml` for a complete example.

### data
| key | description |
|---|---|
| `data_file` | path to the training HDF5 file |
| `batch_size`, `batch_size_val` | training and validation batch size |
| `num_val` | number of showers held out for validation |
| `num_layers` | number of calorimeter layers (78) |
| `num_classes` | number of particle classes (0 = no labels) |
| `use_labels`, `use_directions` | add particle label / direction to the condition |
| `transform_num_points` | preprocessing for the points per layer, e.g. `Dequantize`, `Log`, `StandardScaler` |
| `transform_min_time` | preprocessing for the minimum time, e.g. `QuantileTransformer` |
| `transform_en_perl` | preprocessing for the energy per layer |
| `transform_inc` | preprocessing for the incident energy |

### model
| key | description |
|---|---|
| `name` | network class: `FullyConnected` or `ConcatSquash` |
| `dim_input` | 78 x number of output channels (78, 156 or 234) |
| `dim_condition` | size of the condition: 1 (energy) + labels + 3 (directions) |
| `dim_time` | size of the flow-time embedding |
| `n_layers`, `dim_embedding` | number of layers and size of the per-layer embedding |
| `hidden_dims` | widths of the hidden blocks |

### training
| key | description |
|---|---|
| `epochs` | number of epochs (also used for the distillation student) |
| `optimizer` | `name` (e.g. `Adam`) and its arguments |
| `scheduler` | `name` (e.g. `OneCycleLR`, `CosineAnnealing`) and its arguments; the total number of steps is computed automatically |
| `steps` | Heun steps used when sampling the teacher (default 50) |
| `num_new_samples` | samples written after training (default 50 000) |

## Development
Every push runs two GitHub Actions workflows (`.github/workflows/`): `pre-commit`, and
`tests`, which runs the unit tests and a fast training + distillation run on a small
public sample. Locally, checks run with [pre-commit](https://pre-commit.com):
```bash
pre-commit install
pre-commit run --all-files
```
Unit tests:
```bash
python -m unittest discover -s test -p "*_test.py" -v
```

## Citation
If you use LayerFM, please cite the paper that introduces it:
```bibtex
@article{LayerFM,
    author = {Mozzanica, Martina and Buss, Thorsten},
    title  = {<paper title>},
    year   = {2026}
}
```
and the works it builds on, CaloHadronic and PointCountFM / AllShowers:
```bibtex
@article{Buss:2025cyw,
    author  = {Buss, Thorsten and Gaede, Frank and Kasieczka, Gregor and Korol, Anatolii and Kr{\"u}ger, Katja and McKeown, Peter and Mozzanica, Martina},
    title   = {{CaloHadronic: a diffusion model for the generation of hadronic showers}},
    eprint  = {2506.21720},
    archivePrefix = {arXiv},
    primaryClass  = {physics.ins-det},
    doi     = {10.1088/1748-0221/21/01/P01042},
    journal = {JINST},
    volume  = {21},
    number  = {01},
    pages   = {P01042},
    year    = {2026}
}
```
```bibtex
@article{Buss:2026yrf,
    author  = {Buss, Thorsten and Day-Hall, Henry and Gaede, Frank and Kasieczka, Gregor and Kr{\"u}ger, Katja},
    title   = {{AllShowers: One model for all calorimeter showers}},
    eprint  = {2601.11716},
    archivePrefix = {arXiv},
    primaryClass  = {physics.ins-det},
    year    = {2026}
}
```

---
Questions and comments: [martina.mozzanica@uni-hamburg.de](mailto:martina.mozzanica@uni-hamburg.de).
