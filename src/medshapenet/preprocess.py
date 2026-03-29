"""
Preprocessing pipeline: STL → voxel grid + 2D rendered projections.

Typical usage
-------------
from src.medshapenet.preprocess import batch_preprocess

batch_preprocess(
    stl_dir="data/stl",
    output_dir="data/processed",
    voxel_size=64,
    image_size=128,
    num_views=6,
)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from scipy.ndimage import binary_fill_holes
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Projection directions (unit vectors for orthographic projection)
# ---------------------------------------------------------------------------

_DEFAULT_VIEW_DIRECTIONS: list[tuple[str, np.ndarray]] = [
    ("axial",      np.array([0.0, 0.0, 1.0])),   # top-down
    ("sagittal",   np.array([1.0, 0.0, 0.0])),   # left-right
    ("coronal",    np.array([0.0, 1.0, 0.0])),   # front-back
    ("oblique_1",  np.array([1.0, 1.0, 0.0]) / np.sqrt(2)),
    ("oblique_2",  np.array([0.0, 1.0, 1.0]) / np.sqrt(2)),
    ("oblique_3",  np.array([1.0, 0.0, 1.0]) / np.sqrt(2)),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def batch_preprocess(
    stl_dir: str | Path,
    output_dir: str | Path,
    voxel_size: int = 64,
    image_size: int = 128,
    num_views: int = 3,
    fill_voxels: bool = True,
    overwrite: bool = False,
) -> None:
    """Process all STL files in *stl_dir* and save results to *output_dir*.

    For every ``<idx>_<category>.stl`` file the function creates::

        output_dir/<category>/<idx>/
            voxels.npy          # bool array (D, D, D)
            view_axial.npy      # uint8 array (H, W)
            view_sagittal.npy
            view_coronal.npy
            …

    Parameters
    ----------
    stl_dir:
        Directory that contains the downloaded STL files.
    output_dir:
        Root directory for preprocessed samples.
    voxel_size:
        Spatial resolution of the cubic voxel grid.
    image_size:
        Pixel resolution for rendered 2D projections.
    num_views:
        Number of orthographic projection views to render.
    fill_voxels:
        If True and scipy is available, fill interior voxels using
        binary morphological hole-filling.
    overwrite:
        Reprocess files even if the output directory already exists.
    """
    stl_dir = Path(stl_dir)
    output_dir = Path(output_dir)
    stl_files = sorted(stl_dir.rglob("*.stl"))

    if not stl_files:
        raise FileNotFoundError(f"No STL files found in {stl_dir}")

    preprocessor = STLPreprocessor(
        voxel_size=voxel_size,
        image_size=image_size,
        num_views=num_views,
        fill_voxels=fill_voxels,
    )

    for stl_path in stl_files:
        stem = stl_path.stem  # e.g. "000042_liver"
        parts = stem.split("_", 1)
        if len(parts) == 2:
            idx, category = parts
        else:
            idx, category = stem, "unknown"

        sample_dir = output_dir / category / idx
        if not overwrite and (sample_dir / "voxels.npy").exists():
            continue

        sample_dir.mkdir(parents=True, exist_ok=True)
        try:
            preprocessor.process(stl_path, sample_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to preprocess {stl_path} ({type(exc).__name__}): {exc}")


# ---------------------------------------------------------------------------
# STLPreprocessor
# ---------------------------------------------------------------------------


class STLPreprocessor:
    """Convert a single STL file into a voxel grid and 2D projections.

    Parameters
    ----------
    voxel_size:
        Side length of the cubic output voxel grid.
    image_size:
        Pixel size of each rendered 2D projection.
    num_views:
        Number of orthographic views to render.
    fill_voxels:
        Use morphological hole-filling to produce solid voxel objects
        (requires scipy).
    """

    def __init__(
        self,
        voxel_size: int = 64,
        image_size: int = 128,
        num_views: int = 3,
        fill_voxels: bool = True,
    ) -> None:
        self.voxel_size = voxel_size
        self.image_size = image_size
        self.num_views = num_views
        self.fill_voxels = fill_voxels and _HAS_SCIPY
        self._view_dirs = _DEFAULT_VIEW_DIRECTIONS[:num_views]

    # ------------------------------------------------------------------

    def process(self, stl_path: str | Path, output_dir: str | Path) -> None:
        """Read *stl_path*, compute voxels and projections, save to *output_dir*."""
        from src.utils.stl_utils import (
            normalise_vertices,
            read_stl_vertices,
            vertices_to_voxels,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        vertices = read_stl_vertices(stl_path)
        vertices = normalise_vertices(vertices)

        # ---- Voxels -------------------------------------------------------
        voxels = vertices_to_voxels(vertices, self.voxel_size)
        if self.fill_voxels:
            voxels = binary_fill_holes(voxels)
        np.save(output_dir / "voxels.npy", voxels.astype(bool))

        # ---- 2D projections -----------------------------------------------
        for name, direction in self._view_dirs:
            img = self._render_projection(vertices, direction)
            np.save(output_dir / f"view_{name}.npy", img)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render_projection(
        self, vertices: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """Render an orthographic depth projection along *direction*.

        Returns a uint8 image of shape ``(image_size, image_size)``.
        """
        # Build an orthonormal basis aligned with *direction*
        up = np.array([0.0, 1.0, 0.0])
        if np.abs(np.dot(direction, up)) > 0.9:
            up = np.array([1.0, 0.0, 0.0])
        right = np.cross(direction, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, direction)
        up /= np.linalg.norm(up)

        # Project vertices onto the image plane
        u = vertices @ right          # (N,)
        v = vertices @ up             # (N,)
        d = vertices @ direction      # (N,) depth

        # Normalise u, v to [0, image_size)
        s = self.image_size
        u_norm = ((u - u.min()) / max(u.max() - u.min(), 1e-8) * (s - 1)).astype(int)
        v_norm = ((v - v.min()) / max(v.max() - v.min(), 1e-8) * (s - 1)).astype(int)
        u_norm = np.clip(u_norm, 0, s - 1)
        v_norm = np.clip(v_norm, 0, s - 1)

        # Depth image (min-depth = foreground, rendered as intensity)
        depth_img = np.full((s, s), np.inf, dtype=np.float32)
        np.minimum.at(depth_img, (v_norm, u_norm), d)

        # Convert to uint8 (foreground = bright, background = 0)
        mask = np.isfinite(depth_img)
        if mask.any():
            d_min = depth_img[mask].min()
            d_max = depth_img[mask].max()
            img = np.zeros((s, s), dtype=np.float32)
            if d_max > d_min:
                img[mask] = (depth_img[mask] - d_min) / (d_max - d_min)
            else:
                img[mask] = 1.0
            img_uint8 = (img * 255).astype(np.uint8)
        else:
            img_uint8 = np.zeros((s, s), dtype=np.uint8)

        return img_uint8
