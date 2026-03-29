# MedShapeNet — AI-Based 3D Medical Model Synthesis

> **System for Synthesizing 3D Medical Models from 2D Images Using Artificial Intelligence**  
> *Система синтезу 3D медичних моделей із зображень із застосуванням штучного інтелекту*

---

## Overview

This repository provides an end-to-end pipeline for **reconstructing 3D anatomical shapes** (organs, tumors, bones) from 2D medical images (CT slices, X-rays, MRI projections) using deep learning, built on top of the **MedShapeNet** dataset — a collection of 106 000+ STL meshes covering 80+ anatomical structures.

---

## Proposed Methodology

### Problem Statement

Given one or more 2D projections (axial / sagittal / coronal CT slices, or multi-view X-rays) of a specific anatomical structure, reconstruct its full 3D surface mesh or volumetric representation.

### Recommended Architecture: **Pix2Vox++** (adapted for medical shapes)

```
2D input image(s)
      │
      ▼
┌─────────────┐
│  2D Encoder │  ResNet-50 / EfficientNet-B4 backbone
│  (per view) │  → feature vector per view
└─────────────┘
      │  (multi-view context fusion via attention)
      ▼
┌─────────────────────┐
│  3D Voxel Decoder   │  transposed 3D convolutions
│  64³ or 128³ output │  → binary occupancy grid
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Mesh Extraction    │  Marching Cubes → STL mesh
└─────────────────────┘
```

**Why this approach for MedShapeNet?**

| Criterion | Pix2Vox++ (voxels) | NeRF / IM-NET |
|---|---|---|
| Training stability | ✅ High | ⚠️ Moderate |
| Multi-organ generalization | ✅ Excellent | ⚠️ Per-scene |
| Inference speed | ✅ < 1 s | ❌ Minutes |
| Output directly usable as STL | ✅ Marching Cubes | ✅ |
| Handles missing views | ✅ Attention fusion | ❌ |

### Training Strategy

1. **Data preparation** — render 2D projections from each STL file along 3–6 standard views (axial, sagittal, coronal ± 45°).
2. **Input** — 1–6 rendered 128×128 grayscale images per sample.
3. **Target** — 64³ binary voxel grid extracted from the STL ground truth.
4. **Loss function** — Binary Cross-Entropy + IoU loss.
5. **Augmentation** — random rotation, intensity jitter, random view dropout.
6. **Optimizer** — AdamW, learning-rate 1e-4, cosine annealing.

### Alternative Approaches (for future work)

- **Point-cloud decoder** (FoldingNet / AtlasNet) — lighter, good for thin structures (vertebrae).
- **Neural Radiance Fields (NeRF / MedNeRF)** — photorealistic but slower.
- **Diffusion-based 3D generation** (Shap-E, Point-E) — state-of-the-art, requires more data/compute.

---

## Dataset

The file `MedShapeNetDataset.txt` contains **106 101 download URLs** for STL files organised as:

```
<index>_<category>.stl
```

**Available categories (80+):**
`tumoredbrain`, `tumoredKidney`, `vertebrae`, `liver`, `spleen`, `kidney`, `heart`, `lung*`, `skull`, `femur*`, `hip*`, `aorta`, `pancreas`, `stomach`, `colon`, `bladder`, `brain`, `spinalcanal`, `sacrum`, `sternum`, `ribs`, `trachea`, `esophagus`, `thyroid*`, `prostate`, `uterus`, and many more.

---

## Project Structure

```
MedShapeNetDataset/
├── MedShapeNetDataset.txt      # 106 K download URLs
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── configs/
│   └── default.yaml            # Training & model hyperparameters
├── scripts/
│   ├── download_dataset.py     # Bulk dataset downloader
│   └── demo.py                 # End-to-end reconstruction demo
└── src/
    ├── medshapenet/
    │   ├── __init__.py
    │   ├── dataset.py          # PyTorch Dataset + downloader
    │   ├── preprocess.py       # STL → voxels / 2D projections
    │   ├── model.py            # Pix2Vox++ encoder–decoder
    │   ├── train.py            # Training loop
    │   └── inference.py        # 3D reconstruction from 2D images
    └── utils/
        ├── __init__.py
        └── stl_utils.py        # STL / mesh utilities
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download a subset of the dataset

```bash
# Download 100 liver shapes
python scripts/download_dataset.py \
    --url-file MedShapeNetDataset.txt \
    --category liver \
    --max-samples 100 \
    --output-dir data/stl
```

### 3. Preprocess (STL → voxels + 2D projections)

```bash
python -c "
from src.medshapenet.preprocess import batch_preprocess
batch_preprocess('data/stl', 'data/processed', voxel_size=64)
"
```

### 4. Train the model

```bash
python src/medshapenet/train.py \
    --config configs/default.yaml \
    --data-dir data/processed \
    --category liver
```

### 5. Reconstruct a 3D model from 2D images

```bash
python scripts/demo.py \
    --checkpoint checkpoints/liver_best.pth \
    --input-images view_axial.png view_sagittal.png view_coronal.png \
    --output result.stl
```

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA (recommended, CPU also supported)

See `requirements.txt` for the full list.

---

## Citation

If you use this work, please cite the original MedShapeNet paper:

```bibtex
@article{li2023medshapenet,
  title     = {MedShapeNet -- A Large-Scale Dataset of 3D Medical Shapes for Computer Vision},
  author    = {Jianning Li et al.},
  journal   = {arXiv preprint arXiv:2308.16139},
  year      = {2023}
}
```

---

## License

This project is released under the **MIT License**.  
The underlying MedShapeNet dataset follows its own licence terms — please review them before use.
