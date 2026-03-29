"""
3D reconstruction from 2D medical images using a trained Pix2VoxMed model.

Usage
-----
from src.medshapenet.inference import Reconstructor

rec = Reconstructor("checkpoints/liver_best.pth")

# From file paths
voxels = rec.reconstruct_from_files(
    ["view_axial.png", "view_sagittal.png", "view_coronal.png"]
)
rec.save_stl(voxels, "output.stl")

# From numpy arrays  (H x W, uint8)
voxels = rec.reconstruct(images_np)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .model import Pix2VoxMed


# ---------------------------------------------------------------------------
# Reconstructor
# ---------------------------------------------------------------------------


class Reconstructor:
    """Load a trained checkpoint and reconstruct 3D voxels from 2D views.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.pth`` checkpoint saved by ``train.py``.
    device:
        Target device.  Defaults to GPU if available.
    threshold:
        Probability threshold for voxel occupancy (default 0.5).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: Optional[torch.device] = None,
        threshold: float = 0.5,
    ) -> None:
        self.threshold = threshold
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model, self.config = self._load_checkpoint(
            Path(checkpoint_path), self.device
        )
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconstruct(self, images: np.ndarray) -> np.ndarray:
        """Reconstruct a 3D voxel grid from 2D images.

        Parameters
        ----------
        images:
            Numpy array of shape ``(V, H, W)`` or ``(V, 1, H, W)`` with
            pixel values in ``[0, 255]`` (uint8) or ``[0, 1]`` (float32).

        Returns
        -------
        np.ndarray
            Boolean voxel grid of shape ``(D, D, D)``.
        """
        tensor = self._preprocess(images)          # (1, V, 1, H, W)
        tensor = tensor.to(self.device)
        binary = self.model.predict(tensor, threshold=self.threshold)
        return binary[0, 0].cpu().numpy()          # (D, D, D)

    def reconstruct_from_files(self, image_paths: list[str | Path]) -> np.ndarray:
        """Load PNG/JPEG/NumPy files and reconstruct 3D voxels.

        Parameters
        ----------
        image_paths:
            Ordered list of paths to 2D view images.  Accepted formats:
            ``.npy``, ``.png``, ``.jpg``, ``.jpeg``, ``.tif``.

        Returns
        -------
        np.ndarray
            Boolean voxel grid ``(D, D, D)``.
        """
        images = np.stack([self._load_image(p) for p in image_paths], axis=0)
        return self.reconstruct(images)

    def save_stl(self, voxels: np.ndarray, output_path: str | Path) -> None:
        """Export a boolean voxel grid to an STL surface mesh.

        Uses the Marching Cubes algorithm (scikit-image) to extract an
        isosurface from the voxel occupancy grid.

        Parameters
        ----------
        voxels:
            Boolean or float array of shape ``(D, D, D)``.
        output_path:
            Destination ``.stl`` file path.
        """
        try:
            from skimage.measure import marching_cubes
        except ImportError as exc:
            raise ImportError(
                "scikit-image is required for STL export.  "
                "Install with: pip install scikit-image"
            ) from exc

        voxels_float = voxels.astype(np.float32)
        if voxels_float.max() <= 0:
            raise ValueError("Voxel grid is empty — nothing to export.")

        verts, faces, normals, _ = marching_cubes(
            voxels_float, level=0.5, allow_degenerate=False
        )
        self._write_stl(verts, faces, normals, Path(output_path))
        print(f"Saved STL mesh to {output_path}  "
              f"({len(faces):,} triangles, {len(verts):,} vertices)")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_checkpoint(
        path: Path, device: torch.device
    ) -> tuple[Pix2VoxMed, dict]:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        cfg = checkpoint.get("config", {})
        ds_cfg = cfg.get("dataset", {})
        mdl_cfg = cfg.get("model", {})

        model = Pix2VoxMed(
            voxel_size=ds_cfg.get("voxel_size", 64),
            feat_dim=mdl_cfg.get("feature_dim", 512),
            decoder_channels=tuple(
                mdl_cfg.get("decoder_channels", [512, 256, 128, 64, 32])
            ),
            encoder_pretrained=False,  # weights come from checkpoint
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model, cfg

    @staticmethod
    def _load_image(path: str | Path) -> np.ndarray:
        """Load a single 2D image as a grayscale (H, W) uint8 array."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".npy":
            img = np.load(path)
            if img.ndim == 3:
                img = img.mean(axis=2)
            return img.astype(np.uint8)
        # PIL for common raster formats
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "Pillow is required to load image files.  "
                "Install with: pip install Pillow"
            ) from exc
        img = Image.open(path).convert("L")   # grayscale
        return np.array(img, dtype=np.uint8)

    @staticmethod
    def _preprocess(images: np.ndarray) -> torch.Tensor:
        """Convert (V, H, W) or (V, 1, H, W) array to (1, V, 1, H, W) tensor."""
        if images.ndim == 3:
            images = images[:, np.newaxis, :, :]   # (V, 1, H, W)
        arr = images.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        t = torch.from_numpy(arr).unsqueeze(0)     # (1, V, 1, H, W)
        return t

    @staticmethod
    def _write_stl(
        verts: np.ndarray,
        faces: np.ndarray,
        vertex_normals: np.ndarray,
        path: Path,
    ) -> None:
        """Write a binary STL file from verts/faces arrays.

        Face normals are computed from vertices (cross product) rather than
        from the per-vertex normal array returned by marching_cubes, which
        has one entry per *vertex*, not per *face*.
        """
        import struct

        path.parent.mkdir(parents=True, exist_ok=True)
        # Compute face normals from triangle vertices
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normals = np.cross(edge1, edge2)
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        face_normals = face_normals / norms

        with open(path, "wb") as fh:
            fh.write(b" " * 80)                          # header
            fh.write(struct.pack("<I", len(faces)))      # num triangles
            for i, face in enumerate(faces):
                n = face_normals[i]
                fh.write(struct.pack("<fff", *n))        # normal
                for vi in face:
                    fh.write(struct.pack("<fff", *verts[vi]))  # vertex
                fh.write(struct.pack("<H", 0))           # attr byte count
