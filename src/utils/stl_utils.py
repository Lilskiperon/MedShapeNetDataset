"""
Utilities for STL mesh handling.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def read_stl_vertices(path: str | Path) -> np.ndarray:
    """Return the (N, 3) array of triangle vertices read from a binary STL file.

    Both ASCII and binary STL variants are supported.  For binary files the
    fast struct-based reader is used; ASCII files fall back to a line parser.

    Parameters
    ----------
    path:
        Path to the STL file.

    Returns
    -------
    np.ndarray
        Float32 array of shape (3*num_triangles, 3) containing all vertex
        coordinates.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        header = fh.read(80)
    # ASCII STL files start with "solid"
    if header[:5] == b"solid":
        return _read_ascii_stl(path)
    return _read_binary_stl(path)


def _read_binary_stl(path: Path) -> np.ndarray:
    with open(path, "rb") as fh:
        fh.read(80)  # skip header
        num_triangles = struct.unpack("<I", fh.read(4))[0]
        # Each triangle: 12-byte normal + 3 * 12-byte vertex + 2-byte attr
        data = np.frombuffer(
            fh.read(num_triangles * 50),
            dtype=np.dtype(
                [
                    ("normal", np.float32, (3,)),
                    ("v0", np.float32, (3,)),
                    ("v1", np.float32, (3,)),
                    ("v2", np.float32, (3,)),
                    ("attr", np.uint16),
                ]
            ),
        )
    vertices = np.vstack([data["v0"], data["v1"], data["v2"]])
    return vertices.astype(np.float32)


def _read_ascii_stl(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    with open(path, encoding="ascii", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("vertex"):
                coords = line.split()[1:]
                vertices.append([float(c) for c in coords])
    return np.array(vertices, dtype=np.float32)


def normalise_vertices(vertices: np.ndarray) -> np.ndarray:
    """Translate and scale vertices to the unit cube [-0.5, 0.5]^3.

    Parameters
    ----------
    vertices:
        Float array of shape (N, 3).

    Returns
    -------
    np.ndarray
        Normalised float32 array of the same shape.
    """
    centroid = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    vertices = vertices - centroid
    scale = np.abs(vertices).max()
    if scale > 0:
        vertices = vertices / scale / 2.0
    return vertices.astype(np.float32)


def vertices_to_voxels(
    vertices: np.ndarray, voxel_size: int = 64
) -> np.ndarray:
    """Convert a set of surface vertices to a binary voxel grid.

    Vertices are assumed to lie inside [-0.5, 0.5]^3.  Each vertex is mapped
    to the nearest voxel cell; only voxels that contain at least one vertex are
    marked as occupied.

    Parameters
    ----------
    vertices:
        Normalised float32 array of shape (N, 3) in [-0.5, 0.5].
    voxel_size:
        Side length of the cubic voxel grid.

    Returns
    -------
    np.ndarray
        Boolean array of shape (voxel_size, voxel_size, voxel_size).
    """
    grid = np.zeros((voxel_size, voxel_size, voxel_size), dtype=bool)
    idx = np.floor((vertices + 0.5) * voxel_size).astype(int)
    idx = np.clip(idx, 0, voxel_size - 1)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid
