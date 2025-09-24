# regs-small-vits
Registers in Small Vision Transformers: A Reproducibility Study

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/OpenReview-Paper-blue)](https://openreview.net/pdf?id=5JflRlCt3Q)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

This repository contains code and experiments for the paper:
**“Registers in Small Vision Transformers: A Reproducibility Study of Vision Transformers Need Registers.”**

- Paper (OpenReview PDF): https://openreview.net/pdf?id=5JflRlCt3Q

---

## Quick start

```bash
# 1) Clone
git clone https://github.com/SnorrenanxD/regs-small-vits.git
cd regs-small-vits

# 2) Create a fresh environment (example with venv)
python3 -m venv .venv
source .venv/bin/activate 

# 3) Install dependencies
pip install -r requirements.txt

# 4) Prepare data and checkpoints (see sections below)
#    datasets/imagenette2/
#    checkpoints/

# 5) Open and run the notebook
jupyter notebook main.ipynb
```

> Note: This repository’s main entry point is the **`main.ipynb`** notebook. There is no `main.py` in this repo at the moment.

---

## Repository structure

```
.
├── LOST/             # Scripts/notebooks for LOST object localization experiments
├── attn_maps/        # Attention map generation and visualization utilities
├── comparison/       # Utilities to compare models/settings (with vs without registers)
├── cos_sim/          # Cosine-similarity analyses for tokens/embeddings
├── model/            # Model components and modifications (e.g., DeiT-III + registers)
├── patch_recon/      # Patch reconstruction experiments and helpers
├── main.ipynb        # End-to-end analysis and figure generation
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Data and checkpoints

### Checkpoints
Download the checkpoints archive and place its contents into a folder named **`checkpoints/`** in the repository root.

- Google Drive: https://drive.google.com/file/d/1iaYoDliQixlOYfVQkYKtyp1Z9ygcLAYn/view

Your tree should look like:

```
regs-small-vits/
├── checkpoints/
│   ├── ... (model files)
└── ...
```

### Datasets
Most heavy training was done on ImageNet-1k, but a lot of analysis and examples use **Imagenette** for speed.

- Imagenette (tarball): https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz

After extracting, place the `imagenette2/` directory under `datasets/` in the repo root:

```
regs-small-vits/
├── datasets/
│   └── imagenette2/
│       ├── train/
│       └── val/
└── ...
```

> Depending on your experiments, you may also use other datasets (e.g., VOC2007, Caltech101, CIFAR10, Flowers102). Make sure to adapt paths in your scripts/notebooks if needed.

---

## How to reproduce the main analyses

1) **Token norms and artifacts**
- Use `main.ipynb` to compute per-layer token norms for DeiT-III Small, with and without registers.
- Inspect distributions and identify high-norm “artifact” tokens.

2) **Attention maps**
- Generate and compare attention maps for CLS, patch tokens, and register tokens (if present).
- See `attn_maps/` for supportive utilities.

3) **Patch reconstruction**
- Reconstruct image patches from embeddings to probe local-detail retention.
- See `patch_recon/` for helpers.

4) **Object localization via LOST**
- Run the LOST pipeline on images (e.g., VOC2007) to compute CorLoc and visualize qualitative results.
- See `LOST/` for scripts and references.

> Many of these steps are wired through the `main.ipynb` notebook to ease end-to-end execution and figure generation.

---

## Requirements and environment notes

- This code assumes a GPU-enabled environment for most experiments.
- Some components were developed across different Python versions (e.g., 3.7 vs 3.12). If you encounter version conflicts, use the versions pinned in `requirements.txt` and a clean virtual environment.
- Cluster-specific bits (e.g., Slurm) are not required for running the notebook locally, but were used during training runs.

---

## Troubleshooting

- **Missing checkpoints**: Ensure the `checkpoints/` folder exists and contains the required files from the Google Drive archive.
- **Dataset paths**: Verify that `datasets/imagenette2/` is present and that your code points to the correct root.
- **Kernel issues in Jupyter**: If the notebook fails to run, confirm the active kernel uses the environment where you installed the requirements.
- **Out-of-memory errors**: Reduce batch size, image size, or number of samples when running locally.

---

## Citation

If you use this repository in your research, please cite the paper:

```
@article{bach2025regs,
  title   = {Registers in Small Vision Transformers: A Reproducibility Study of Vision Transformers Need Registers},
  author  = {Bach, Linus and Bakker, Emma and van Dijk, Rénan and de Vries, Jip and Szewczyk, Konrad},
  journal = {Transactions on Machine Learning Research},
  year    = {2025},
  url     = {https://openreview.net/pdf?id=5JflRlCt3Q}
}
```
---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
