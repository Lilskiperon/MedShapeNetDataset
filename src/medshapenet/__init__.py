"""
MedShapeNet — AI-Based 3D Medical Model Synthesis System
=========================================================

A pipeline for reconstructing 3D anatomical shapes from 2D medical images
using the MedShapeNet dataset.
"""

from .dataset import MedShapeNetDataset, MedShapeNetDownloader
from .model import Pix2VoxMed
from .preprocess import STLPreprocessor, batch_preprocess

__all__ = [
    "MedShapeNetDataset",
    "MedShapeNetDownloader",
    "Pix2VoxMed",
    "STLPreprocessor",
    "batch_preprocess",
]
