# Weighted-Normative Embeddings

**A plug-and-play pipeline for brain–AI alignment.**

This repository implements a robust pipeline to evaluate how well computer vision models align with human brain representations. Specifically, it compares 89 pretrained **timm** vision encoders against **Natural Object Dataset (NOD) fMRI** data using **Representation-Weighted Grouping (RWG)**.

The core contribution is a robust weighting scheme (RWG) that iteratively down-weights noisy or idiosyncratic fMRI subjects to create a better "group normative" target, without requiring per-subject normalization. We then report the percent change in Spearman alignment when moving from a traditional Simple Mean to the RWG target.

## 1. Overview

### The Problem
Traditionally, brain–AI alignment benchmarks average fMRI subjects into a single group Representational Dissimilarity Matrix (RDM) using a simple mean. This treats every subject as equally "typical," allowing noisy subjects to distort the group geometry and affect model rankings.

### The Solution: RWG
We implement **Representation-Weighted Grouping (RWG)**. For each Region of Interest (ROI):
1.  We iteratively reweight subjects based on how well their RDM agrees with the current group estimate.
2.  We use a **Huber-style M-estimator** on residuals (without z-score normalization).
3.  We compare the alignment of 89 vision models against the **Simple Mean** vs. the **RWG Mean**.

## 2. Repository Layout

```text
.
├── configs/
│   └── default.yaml               # Main configuration (paths, batch size)
├── imagenet_weighted_embeddings.py  # Core logic: RWG + Alignment + Ridge Regression
├── train.py                       # CLI entrypoint
├── ckpts/                         # Checkpoint root
├── outputs/                       # Results: NPZs, ROI CSVs, logs
└── README.md
```

## 3. Installation

This pipeline requires Python 3.9+.

```bash
pip install numpy scipy scikit-learn matplotlib pandas torch torchvision timm
```

*Note: Ensure you install the version of PyTorch compatible with your CUDA/cuDNN version.*

## 4. Data Preparation

### 4.1 NOD Images (Stimuli)
You need the NOD stimuli images organized in an ImageNet-1k style structure (compatible with `torchvision.datasets.ImageFolder`).

```text
/path/to/NOD_images/
├── n01440764/
├── n01443537/
└── ...
```

### 4.2 fMRI ROI RDMs
You need pre-computed RDM vectors for your subjects. The pipeline expects a directory with one subfolder per subject and `.npy` files for each ROI.

```text
/path/to/fmri_rdms/
├── sub-01/
│   ├── V1.npy
│   ├── V2.npy
│   └── ...
├── sub-02/
│   └── ...
└── ...
```
* **Format:** Each `.npy` file must be a 1D vector containing the upper-triangle entries of the RDM.
* **Consistency:** All subjects for a given ROI must have vectors of the same length.

## 5. Configuration

Edit `configs/default.yaml` to point to your data paths:

```yaml
run_mode: full
experiment_name: default
data:
  nod_root: /path/to/NOD_images
  fmri_rdm_file: /path/to/fmri_rdms
training:
  batch_size: 64
  device: cuda
  architectures: [] # Not used in pretrained mode, kept for consistency
paths:
  checkpoint_root: ckpts/default_run
  output_root: outputs/default_run
misc:
  num_workers: 8
```

## 6. Usage

The pipeline runs in two stages via `train.py`.

### Stage 1: Extract Model RDMs
Run ~89 `timm` models on the stimulus images to extract class-level activations and build per-layer RDMs. This caches a large `.npz` file so you don't have to run inference again.

```bash
python train.py --mode pretrained --config configs/default.yaml
```
**Outputs:**
* `outputs/default_run/timm_pretrained_nod_rdms.npz` (Model RDM vectors)
* `outputs/default_run/timm_pretrained_nod_rdms.json` (Metadata)

### Stage 2: fMRI RWG & Alignment Calculation
This stage performs the RWG algorithm on the fMRI data, fits layer-wise Ridge Regressions for the models, and computes Spearman correlations.

```bash
python train.py \
  --mode inference \
  --config configs/default.yaml \
  --fmri-rdm-file /path/to/fmri_rdms
```

**What happens in this stage:**
1.  **RWG:** Calculates both Simple Mean and Huber-weighted RWG targets for each ROI.
2.  **Ridge Regression:** Fits model layers to both targets (cross-validated).
3.  **Scoring:** Computes Spearman correlation for Model $\to$ Simple Mean vs. Model $\to$ RWG Mean.

## 7. Outputs

Results are saved in `outputs/default_run/`. The most important files are:

* **`roi_results/<ROI>.npy`**: The computed group RDM vectors (Simple and RWG).
* **`ridge_layer_weights_long.csv`**: Detailed layer weights for every architecture and ROI.
* **`roi_alignments_per_seed_<ROI>_ridgeSimple.csv`**: Alignment scores using the standard mean.
* **`roi_alignments_per_seed_<ROI>_ridgeRew.csv`**: Alignment scores using the RWG mean.

These CSVs contain the `perc_change` column, quantifying the improvement RWG provides over the standard approach.

## 8. Method Details

The **RWG algorithm** used here is specifically designed for RDM vectors:
* **No Normalization:** We work on raw RDM vectors (no per-subject z-scoring).
* **Initialization:** Starts with the element-wise median.
* **Iterative Reweighting:**
    * Computes correlation between subject and current mean.
    * Calculates residuals ($1 - \text{correlation}$).
    * Applies **Huber weights** to down-weight outliers based on Median Absolute Deviation (MAD).
    * Updates the weighted mean until convergence.

## Citation

If you use this pipeline or methodology, please describe it as follows:

> “We computed ROI-wise RDMs for NOD fMRI across subjects, estimated both a simple mean and a Huber-based RWG group mean per ROI (with no per-subject normalization), and compared Spearman alignment of 89 pretrained timm vision encoders to these two targets, reporting the percent change in alignment induced by RWG.”
