"""
MedShapeNet dataset downloader and PyTorch Dataset class.

Usage
-----
# Download a subset
downloader = MedShapeNetDownloader("MedShapeNetDataset.txt")
downloader.download(category="liver", max_samples=200, output_dir="data/stl")

# Use in a training loop
from torch.utils.data import DataLoader
ds = MedShapeNetDataset("data/processed", category="liver", split="train")
loader = DataLoader(ds, batch_size=8, shuffle=True)
for images, voxels in loader:
    ...  # images: (B, V, 1, H, W), voxels: (B, 1, D, D, D)
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class MedShapeNetDownloader:
    """Download STL files listed in ``MedShapeNetDataset.txt``.

    Parameters
    ----------
    url_file:
        Path to the text file containing one download URL per line.
    """

    def __init__(self, url_file: str | Path = "MedShapeNetDataset.txt") -> None:
        self.url_file = Path(url_file)
        if not self.url_file.exists():
            raise FileNotFoundError(f"URL file not found: {self.url_file}")

        with open(self.url_file) as fh:
            self._urls: list[str] = [line.strip() for line in fh if line.strip()]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_categories(self) -> list[str]:
        """Return the sorted list of all unique anatomical categories."""
        categories: set[str] = set()
        for url in self._urls:
            name = Path(url).stem  # e.g. "000042_liver"
            parts = name.split("_", 1)
            if len(parts) == 2:
                categories.add(parts[1])
        return sorted(categories)

    def download(
        self,
        category: Optional[str] = None,
        max_samples: Optional[int] = None,
        output_dir: str | Path = "data/stl",
        skip_existing: bool = True,
        timeout: int = 30,
    ) -> list[Path]:
        """Download STL files, optionally filtered by *category*.

        Parameters
        ----------
        category:
            Anatomical category string, e.g. ``"liver"``.  Pass ``None`` to
            download all categories.
        max_samples:
            Upper bound on the number of files to download.
        output_dir:
            Directory to save files into.  Created if necessary.
        skip_existing:
            Skip files that already exist locally.
        timeout:
            HTTP request timeout in seconds.

        Returns
        -------
        list[Path]
            Paths of all successfully downloaded files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        urls = self._filter_urls(category)
        if max_samples is not None:
            urls = urls[:max_samples]

        downloaded: list[Path] = []
        for i, url in enumerate(urls, 1):
            filename = Path(url).name
            dest = output_dir / filename
            if skip_existing and dest.exists():
                downloaded.append(dest)
                continue
            try:
                print(f"[{i}/{len(urls)}] Downloading {filename} …", end="\r")
                urllib.request.urlretrieve(url, dest)  # noqa: S310
                downloaded.append(dest)
            except Exception as exc:  # noqa: BLE001
                print(f"\nWarning: could not download {url} ({type(exc).__name__}): {exc}")

        print(f"\nDone. {len(downloaded)} files in {output_dir}")
        return downloaded

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _filter_urls(self, category: Optional[str]) -> list[str]:
        if category is None:
            return list(self._urls)
        suffix = f"_{category}.stl"
        return [u for u in self._urls if Path(u).name.endswith(suffix)]


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class MedShapeNetDataset(Dataset):
    """PyTorch Dataset that loads pre-processed samples produced by
    :func:`src.medshapenet.preprocess.batch_preprocess`.

    Each sample is a pair ``(images, voxels)`` where:

    * ``images`` — float32 tensor of shape ``(num_views, 1, H, W)`` containing
      the rendered 2D projections (normalised to ``[0, 1]``).
    * ``voxels`` — float32 tensor of shape ``(1, D, D, D)`` containing the
      binary occupancy grid (0 or 1).

    Parameters
    ----------
    data_dir:
        Root directory produced by ``batch_preprocess``, containing
        ``<category>/`` sub-directories.
    category:
        Anatomical category to load, e.g. ``"liver"``.
    split:
        One of ``"train"``, ``"val"``, ``"test"``.
    train_ratio / val_ratio:
        Dataset split fractions (test = remainder).
    num_views:
        How many of the available rendered views to use.
    transform:
        Optional callable applied to the image tensor.
    seed:
        Random seed used to produce reproducible splits.
    """

    def __init__(
        self,
        data_dir: str | Path,
        category: str,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        num_views: int = 3,
        transform: Optional[Callable] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir) / category
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Processed data directory not found: {self.data_dir}\n"
                "Run batch_preprocess() first."
            )
        self.split = split
        self.num_views = num_views
        self.transform = transform

        all_samples = sorted(self.data_dir.glob("*/"))  # one dir per sample
        # Deterministic shuffle using stable hash of category name + seed
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(all_samples))
        all_samples = [all_samples[i] for i in indices]

        n = len(all_samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        if split == "train":
            self._samples = all_samples[:train_end]
        elif split == "val":
            self._samples = all_samples[train_end:val_end]
        elif split == "test":
            self._samples = all_samples[val_end:]
        else:
            raise ValueError(f"Unknown split '{split}'. Use 'train', 'val', or 'test'.")

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        sample_dir = self._samples[idx]

        # Load voxels (saved as .npy boolean array)
        voxel_path = sample_dir / "voxels.npy"
        voxels = np.load(voxel_path).astype(np.float32)  # (D, D, D)
        voxels = torch.from_numpy(voxels).unsqueeze(0)   # (1, D, D, D)

        # Load rendered views (saved as .npy float32 arrays, shape H×W)
        view_paths = sorted(sample_dir.glob("view_*.npy"))
        if len(view_paths) == 0:
            raise FileNotFoundError(f"No view files found in {sample_dir}")

        # Cycle / truncate views to match requested num_views
        views = []
        for i in range(self.num_views):
            vp = view_paths[i % len(view_paths)]
            img = np.load(vp).astype(np.float32)          # (H, W)
            views.append(img)

        images = np.stack(views, axis=0)                   # (V, H, W)
        images = torch.from_numpy(images).unsqueeze(1)     # (V, 1, H, W)
        images = images / 255.0                            # normalise to [0,1]

        if self.transform is not None:
            images = self.transform(images)

        return images, voxels
