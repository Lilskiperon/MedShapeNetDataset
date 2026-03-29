"""
Training script for the Pix2VoxMed 3D synthesis model.

Usage
-----
python src/medshapenet/train.py \\
    --config configs/default.yaml \\
    --data-dir data/processed \\
    --category liver \\
    --checkpoint-dir checkpoints/liver

The script supports CPU and single / multi-GPU training via
``torch.cuda.is_available()``.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from .dataset import MedShapeNetDataset
from .model import Pix2VoxMed, VoxelLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str,
    num_epochs: int,
    warmup_epochs: int,
) -> optim.lr_scheduler.LRScheduler:
    if scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(num_epochs - warmup_epochs, 1)
        )
    if scheduler_type == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)


def _iou_metric(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Binary IoU for logging (uses 0.5 threshold)."""
    preds = (torch.sigmoid(logits) >= 0.5).float()
    intersection = (preds * targets).sum().item()
    union = (preds + targets - preds * targets).sum().item()
    return (intersection + 1.0) / (union + 1.0)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(config: dict) -> None:
    """Run the full training procedure using the provided *config* dict."""
    ds_cfg = config["dataset"]
    mdl_cfg = config["model"]
    tr_cfg = config["training"]
    aug_cfg = config.get("augmentation", {})
    ckpt_cfg = config.get("checkpointing", {})
    log_cfg = config.get("logging", {})

    _seed_everything(tr_cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Datasets ----------------------------------------------------------
    common_kwargs = dict(
        data_dir=ds_cfg["data_dir"],
        category=ds_cfg["category"],
        num_views=ds_cfg.get("num_views", 3),
        train_ratio=ds_cfg.get("train_split", 0.8),
        val_ratio=ds_cfg.get("val_split", 0.1),
        seed=tr_cfg.get("seed", 42),
    )
    train_ds = MedShapeNetDataset(split="train", **common_kwargs)
    val_ds = MedShapeNetDataset(split="val", **common_kwargs)
    print(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=tr_cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=tr_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tr_cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=tr_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    # ---- Model -------------------------------------------------------------
    model = Pix2VoxMed(
        voxel_size=ds_cfg.get("voxel_size", 64),
        feat_dim=mdl_cfg.get("feature_dim", 512),
        decoder_channels=tuple(mdl_cfg.get("decoder_channels", [512, 256, 128, 64, 32])),
        encoder_pretrained=mdl_cfg.get("encoder_pretrained", True),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # ---- Optimiser & scheduler --------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=tr_cfg.get("learning_rate", 1e-4),
        weight_decay=tr_cfg.get("weight_decay", 1e-5),
    )
    num_epochs = tr_cfg.get("num_epochs", 100)
    warmup_epochs = tr_cfg.get("warmup_epochs", 5)
    scheduler = _build_scheduler(
        optimizer,
        tr_cfg.get("lr_scheduler", "cosine"),
        num_epochs,
        warmup_epochs,
    )
    criterion = VoxelLoss(
        bce_weight=tr_cfg.get("bce_weight", 1.0),
        iou_weight=tr_cfg.get("iou_weight", 1.0),
    )

    # ---- Checkpointing -----------------------------------------------------
    ckpt_dir = Path(ckpt_cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_every = ckpt_cfg.get("save_every_n_epochs", 10)
    best_val_iou = 0.0
    log_interval = log_cfg.get("log_interval", 10)

    # ---- Training ----------------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        # Warmup: linearly scale LR
        if epoch <= warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = tr_cfg.get("learning_rate", 1e-4) * epoch / warmup_epochs

        # ---- Train epoch ---------------------------------------------------
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        for batch_idx, (images, voxels) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)   # (B, V, 1, H, W)
            voxels = voxels.to(device, non_blocking=True)   # (B, 1, D, D, D)

            # Random view dropout augmentation
            if aug_cfg.get("random_view_dropout", False):
                drop_prob = aug_cfg.get("dropout_prob", 0.2)
                if random.random() < drop_prob and images.size(1) > 1:
                    keep = random.randint(1, images.size(1))
                    idx = torch.randperm(images.size(1))[:keep]
                    images = images[:, idx]

            optimizer.zero_grad()
            logits = model(images)
            loss, loss_dict = criterion(logits, voxels)
            loss.backward()

            grad_clip = tr_cfg.get("grad_clip", 1.0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            train_loss += loss_dict["total"]
            train_iou += _iou_metric(logits.detach(), voxels)

            if batch_idx % log_interval == 0:
                print(
                    f"  Epoch {epoch}/{num_epochs}  "
                    f"Batch {batch_idx}/{len(train_loader)}  "
                    f"Loss: {loss_dict['total']:.4f}  "
                    f"BCE: {loss_dict['bce']:.4f}  "
                    f"IoU loss: {loss_dict['iou']:.4f}"
                )

        train_loss /= len(train_loader)
        train_iou /= len(train_loader)

        # Update LR (after warmup)
        if epoch > warmup_epochs:
            scheduler.step()

        # ---- Validation ----------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for images, voxels in val_loader:
                images = images.to(device, non_blocking=True)
                voxels = voxels.to(device, non_blocking=True)
                logits = model(images)
                _, loss_dict = criterion(logits, voxels)
                val_loss += loss_dict["total"]
                val_iou += _iou_metric(logits, voxels)

        val_loss /= max(len(val_loader), 1)
        val_iou /= max(len(val_loader), 1)

        print(
            f"Epoch {epoch}/{num_epochs}  "
            f"Train loss: {train_loss:.4f}  Train IoU: {train_iou:.4f}  "
            f"Val loss: {val_loss:.4f}  Val IoU: {val_iou:.4f}  "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # ---- Save checkpoints ----------------------------------------------
        if epoch % save_every == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_iou": val_iou,
                    "config": config,
                },
                ckpt_path,
            )
            print(f"  Checkpoint saved: {ckpt_path}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_path = ckpt_dir / "best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_iou": val_iou,
                    "config": config,
                },
                best_path,
            )
            print(f"  ★ New best Val IoU {val_iou:.4f} → {best_path}")

    print(f"\nTraining complete.  Best Val IoU: {best_val_iou:.4f}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Pix2VoxMed on the MedShapeNet dataset"
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument("--data-dir", default=None, help="Override dataset.data_dir")
    parser.add_argument("--category", default=None, help="Override dataset.category")
    parser.add_argument(
        "--checkpoint-dir", default=None, help="Override checkpointing.checkpoint_dir"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    # CLI overrides
    if args.data_dir is not None:
        config["dataset"]["data_dir"] = args.data_dir
    if args.category is not None:
        config["dataset"]["category"] = args.category
    if args.checkpoint_dir is not None:
        config["checkpointing"]["checkpoint_dir"] = args.checkpoint_dir

    train(config)
