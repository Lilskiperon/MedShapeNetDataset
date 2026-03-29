#!/usr/bin/env python3
"""
End-to-end demo: reconstruct a 3D medical shape from 2D projections.

The demo can work in two modes:

1. **Synthetic mode** (no data needed) — generates a synthetic sphere voxel
   grid, renders projections from it, and reconstructs back, verifying the
   whole pipeline end-to-end without downloading any data.

2. **File mode** — loads a real checkpoint and input PNG/NPY images.

Examples
--------
# Synthetic self-test (no data, no trained model required)
python scripts/demo.py --synthetic

# Reconstruct from a trained model
python scripts/demo.py \\
    --checkpoint checkpoints/liver/best.pth \\
    --input-images view_axial.png view_sagittal.png view_coronal.png \\
    --output result.stl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sphere_voxels(size: int = 64, radius: float = 0.4) -> np.ndarray:
    """Create a binary voxel grid containing a sphere."""
    coords = np.linspace(-0.5, 0.5, size)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    return (x ** 2 + y ** 2 + z ** 2) <= radius ** 2


def _render_orthographic(voxels: np.ndarray, axis: int = 0) -> np.ndarray:
    """Max-project a voxel grid along *axis* → uint8 image."""
    proj = voxels.max(axis=axis).astype(np.float32)
    proj = (proj * 255).astype(np.uint8)
    return proj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MedShapeNet 3D reconstruction demo")
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Run synthetic self-test without a trained model.",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a trained .pth checkpoint.",
    )
    p.add_argument(
        "--input-images",
        nargs="+",
        default=None,
        help="Paths to 2D input view images (PNG / NPY).",
    )
    p.add_argument(
        "--output",
        default="output.stl",
        help="Output STL file path (default: %(default)s)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Voxel occupancy threshold (default: %(default)s)",
    )
    return p.parse_args()


def run_synthetic_demo(output_path: str) -> None:
    """Validate the preprocessing → model → inference pipeline synthetically."""
    print("=== Synthetic Demo ===")
    print("Creating a synthetic sphere voxel grid …")
    voxels_gt = _make_sphere_voxels(64)
    print(f"  Ground-truth voxels: {voxels_gt.sum():,} occupied cells "
          f"out of {voxels_gt.size:,}")

    print("Rendering orthographic projections …")
    views = [_render_orthographic(voxels_gt, axis=i) for i in range(3)]
    images = np.stack(views, axis=0)  # (3, 64, 64)
    print(f"  Images shape: {images.shape}, dtype: {images.dtype}")

    # Instantiate an untrained model and run a forward pass
    print("Running untrained Pix2VoxMed forward pass (no checkpoint) …")
    import torch
    from src.medshapenet.model import Pix2VoxMed

    model = Pix2VoxMed(voxel_size=64, encoder_pretrained=False)
    model.eval()
    tensor = torch.from_numpy(
        images.astype(np.float32) / 255.0
    ).unsqueeze(0).unsqueeze(2)   # (1, 3, 1, 64, 64)

    with torch.no_grad():
        logits = model(tensor)

    print(f"  Output logits shape: {logits.shape}")

    # Save the ground-truth sphere as STL to verify the exporter
    print(f"Exporting ground-truth sphere to {output_path} …")
    from src.medshapenet.inference import Reconstructor
    # Use a temporary Reconstructor instance just for the STL writer
    Reconstructor._write_stl(
        *_voxels_to_mesh(voxels_gt), Path(output_path)
    )
    print("=== Synthetic demo complete ===")


def _voxels_to_mesh(voxels: np.ndarray):
    """Return (verts, faces, normals) for the given voxel grid."""
    from skimage.measure import marching_cubes
    return marching_cubes(voxels.astype(np.float32), level=0.5,
                          allow_degenerate=False)[:3]


def run_file_demo(
    checkpoint: str,
    input_images: list[str],
    output: str,
    threshold: float,
) -> None:
    """Reconstruct a 3D shape from real views using a trained model."""
    from src.medshapenet.inference import Reconstructor

    print(f"Loading checkpoint: {checkpoint}")
    rec = Reconstructor(checkpoint, threshold=threshold)

    print(f"Reconstructing from {len(input_images)} view(s): {input_images}")
    voxels = rec.reconstruct_from_files(input_images)
    print(f"  Predicted voxels: {voxels.sum():,} occupied cells")

    rec.save_stl(voxels, output)


def main() -> None:
    args = _parse_args()

    if args.synthetic:
        run_synthetic_demo(args.output)
        return

    if args.checkpoint is None or args.input_images is None:
        print(
            "Error: provide --checkpoint and --input-images, "
            "or use --synthetic for a self-test.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_file_demo(
        checkpoint=args.checkpoint,
        input_images=args.input_images,
        output=args.output,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
