#!/usr/bin/env python3
"""
Download a subset of the MedShapeNet dataset.

Examples
--------
# Download 200 liver STL files
python scripts/download_dataset.py \\
    --url-file MedShapeNetDataset.txt \\
    --category liver \\
    --max-samples 200 \\
    --output-dir data/stl

# List all available categories
python scripts/download_dataset.py --url-file MedShapeNetDataset.txt --list-categories

# Download everything (warning: very large)
python scripts/download_dataset.py --url-file MedShapeNetDataset.txt --output-dir data/stl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.medshapenet.dataset import MedShapeNetDownloader


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MedShapeNet dataset downloader")
    p.add_argument(
        "--url-file",
        default="MedShapeNetDataset.txt",
        help="Path to MedShapeNetDataset.txt (default: %(default)s)",
    )
    p.add_argument(
        "--category",
        default=None,
        help="Anatomical category to download, e.g. 'liver'.  "
             "Omit to download all categories.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of files to download.",
    )
    p.add_argument(
        "--output-dir",
        default="data/stl",
        help="Directory to save STL files (default: %(default)s)",
    )
    p.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-download files that already exist locally.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP request timeout in seconds (default: %(default)s)",
    )
    p.add_argument(
        "--list-categories",
        action="store_true",
        help="Print all available categories and exit.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    downloader = MedShapeNetDownloader(args.url_file)

    if args.list_categories:
        categories = downloader.list_categories()
        print(f"Available categories ({len(categories)}):")
        for cat in categories:
            print(f"  {cat}")
        return

    downloader.download(
        category=args.category,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        skip_existing=not args.no_skip,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
