"""
Pix2Vox-style encoder–decoder model for 2D → 3D medical shape synthesis.

Architecture overview
---------------------

    Input:  (B, V, 1, H, W)  — B samples, V views each, 1 channel, H×W pixels
    Output: (B, 1, D, D, D)  — binary occupancy logits (D³ voxel grid)

The encoder is a per-view CNN (ResNet backbone) that produces a feature
vector for each view.  The multi-view features are fused with an attention
mechanism and then decoded into a 3D voxel grid through a series of 3D
transposed convolutions.

Reference: Xie et al., "Pix2Vox++: Multi-scale Context-aware 3D Object
Reconstruction from Single and Multiple Images" (IJCV 2020).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 2D Encoder (per-view)
# ---------------------------------------------------------------------------


class _ResNetEncoder(nn.Module):
    """Lightweight ResNet-18 encoder adapted for single-channel input.

    Parameters
    ----------
    out_dim:
        Dimension of the output feature vector per view.
    pretrained:
        Load ImageNet-pretrained weights (channel 1 → channel 3 duplication).
    """

    def __init__(self, out_dim: int = 512, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            backbone = resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
        except ImportError as exc:
            raise ImportError(
                "torchvision is required.  Install with: pip install torchvision"
            ) from exc

        # Adapt first conv to accept 1-channel images
        orig_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            1,
            orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False,
        )
        if pretrained:
            # Average the weights across the original 3 input channels
            with torch.no_grad():
                backbone.conv1.weight.copy_(
                    orig_conv.weight.mean(dim=1, keepdim=True)
                )

        # Remove the classification head
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Tensor of shape ``(B, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Feature vector of shape ``(B, out_dim)``.
        """
        feat = self.features(x)          # (B, 512, h, w)
        feat = self.pool(feat).flatten(1) # (B, 512)
        return self.proj(feat)            # (B, out_dim)


# ---------------------------------------------------------------------------
# Multi-view attention fusion
# ---------------------------------------------------------------------------


class _MultiViewFusion(nn.Module):
    """Fuse variable-length view features using scaled dot-product attention.

    Parameters
    ----------
    feat_dim:
        Dimension of each per-view feature vector.
    """

    def __init__(self, feat_dim: int = 512) -> None:
        super().__init__()
        self.query = nn.Linear(feat_dim, feat_dim)
        self.key = nn.Linear(feat_dim, feat_dim)
        self.value = nn.Linear(feat_dim, feat_dim)
        self.scale = feat_dim ** -0.5

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features:
            Tensor of shape ``(B, V, feat_dim)``.

        Returns
        -------
        torch.Tensor
            Fused tensor of shape ``(B, feat_dim)`` (mean over attended views).
        """
        Q = self.query(features)   # (B, V, D)
        K = self.key(features)
        V = self.value(features)

        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (B, V, V)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, V)   # (B, V, D)
        return out.mean(dim=1)     # (B, D)


# ---------------------------------------------------------------------------
# 3D Voxel Decoder
# ---------------------------------------------------------------------------


class _VoxelDecoder(nn.Module):
    """Decode a latent vector into a 3D voxel grid via transposed 3D convs.

    Parameters
    ----------
    feat_dim:
        Input latent vector dimension.
    channels:
        Channel sizes for each deconvolution stage.
    voxel_size:
        Target spatial resolution (must be a power of 2).
    """

    def __init__(
        self,
        feat_dim: int = 512,
        channels: tuple[int, ...] = (512, 256, 128, 64, 32),
        voxel_size: int = 64,
    ) -> None:
        super().__init__()

        # Determine the number of upsampling steps needed
        # Starting spatial: 2^? → target voxel_size
        # Each ConvTranspose3d with stride=2 doubles spatial size.
        start_spatial = 2
        num_upsample = 0
        s = start_spatial
        while s < voxel_size:
            s *= 2
            num_upsample += 1

        in_ch = feat_dim
        self.fc = nn.Linear(feat_dim, channels[0] * start_spatial ** 3)
        self.start_spatial = start_spatial
        self.channels_in = channels[0]

        layers: list[nn.Module] = []
        ch_list = list(channels)
        # Pad / truncate channel list to num_upsample
        while len(ch_list) < num_upsample + 1:
            ch_list.append(ch_list[-1])
        ch_list = ch_list[: num_upsample + 1]

        for i in range(num_upsample):
            in_c = ch_list[i]
            out_c = ch_list[i + 1]
            layers += [
                nn.ConvTranspose3d(in_c, out_c, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
            ]

        layers.append(nn.Conv3d(ch_list[-1], 1, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z:
            Latent vector ``(B, feat_dim)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, 1, D, D, D)``.
        """
        B = z.size(0)
        s = self.start_spatial
        x = self.fc(z).view(B, self.channels_in, s, s, s)
        return self.decoder(x)


# ---------------------------------------------------------------------------
# Full Pix2VoxMed model
# ---------------------------------------------------------------------------


class Pix2VoxMed(nn.Module):
    """Pix2Vox++ inspired model for medical 3D shape reconstruction.

    Accepts one or more 2D views of an anatomical structure and predicts a
    binary 3D voxel occupancy grid.

    Parameters
    ----------
    voxel_size:
        Side length of the output voxel grid.
    feat_dim:
        Internal feature dimension.
    decoder_channels:
        Channel sizes for the 3D decoder stages.
    encoder_pretrained:
        Use ImageNet-pretrained ResNet-18 weights for the 2D encoder.
    """

    def __init__(
        self,
        voxel_size: int = 64,
        feat_dim: int = 512,
        decoder_channels: tuple[int, ...] = (512, 256, 128, 64, 32),
        encoder_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = _ResNetEncoder(out_dim=feat_dim, pretrained=encoder_pretrained)
        self.fusion = _MultiViewFusion(feat_dim=feat_dim)
        self.decoder = _VoxelDecoder(
            feat_dim=feat_dim,
            channels=decoder_channels,
            voxel_size=voxel_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        images:
            Float tensor of shape ``(B, V, 1, H, W)`` in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Voxel logits of shape ``(B, 1, D, D, D)``.
        """
        B, V, C, H, W = images.shape
        # Encode each view independently
        imgs_flat = images.view(B * V, C, H, W)   # (B*V, 1, H, W)
        feats_flat = self.encoder(imgs_flat)       # (B*V, feat_dim)
        feats = feats_flat.view(B, V, -1)          # (B, V, feat_dim)

        # Fuse views
        z = self.fusion(feats)                     # (B, feat_dim)

        # Decode to voxels
        logits = self.decoder(z)                   # (B, 1, D, D, D)
        return logits

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def predict(self, images: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Run forward pass and threshold logits to binary voxels.

        Parameters
        ----------
        images:
            Float tensor ``(B, V, 1, H, W)`` in ``[0, 1]``.
        threshold:
            Probability threshold for occupancy.

        Returns
        -------
        torch.Tensor
            Bool tensor ``(B, 1, D, D, D)``.
        """
        with torch.no_grad():
            logits = self(images)
        probs = torch.sigmoid(logits)
        return probs >= threshold


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


class VoxelLoss(nn.Module):
    """Combined Binary Cross-Entropy + IoU loss for voxel prediction.

    Parameters
    ----------
    bce_weight:
        Weight for the BCE term.
    iou_weight:
        Weight for the soft IoU term.
    """

    def __init__(self, bce_weight: float = 1.0, iou_weight: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss.

        Parameters
        ----------
        logits:
            Raw model output ``(B, 1, D, D, D)``.
        targets:
            Ground-truth occupancy ``(B, 1, D, D, D)`` with values in ``{0, 1}``.

        Returns
        -------
        tuple[Tensor, dict]
            Total loss tensor and a metrics dict with ``bce``, ``iou``,
            ``total`` keys.
        """
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        union = (probs + targets - probs * targets).sum()
        iou_loss = 1.0 - (intersection + 1.0) / (union + 1.0)

        total = self.bce_weight * bce_loss + self.iou_weight * iou_loss
        metrics = {
            "bce": bce_loss.item(),
            "iou": iou_loss.item(),
            "total": total.item(),
        }
        return total, metrics
