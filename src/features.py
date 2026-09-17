"""Backbone construction and frozen-feature extraction.

Three ImageNet-pretrained torchvision backbones are used as feature extractors:
ResNet-50 (IMAGENET1K_V2), ConvNeXt-Tiny (IMAGENET1K_V1) and ViT-B/16
(IMAGENET1K_V1, supervised). Weights come from the official torchvision URLs and
are downloaded once, then cached on the stand.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import set_seed

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_backbone(name: str) -> tuple[torch.nn.Module, int, int]:
    """Return (feature extractor, feature dim, input resolution)."""
    from torchvision import models

    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Identity()
        dim, size = 2048, 224
    elif name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[2] = torch.nn.Identity()
        dim, size = 768, 224
    elif name == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = torch.nn.Identity()
        dim, size = 768, 224
    else:
        raise ValueError(f"unknown backbone {name!r}")
    model.eval()
    return model, dim, size


class ImageList(Dataset):
    def __init__(self, paths: list[Path], labels: list[int], size: int):
        from torchvision import transforms

        self.paths, self.labels = paths, labels
        self.tf = transforms.Compose(
            [
                transforms.Resize(int(size * 256 / 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        from PIL import Image

        with Image.open(self.paths[index]) as im:
            tensor = self.tf(im.convert("RGB"))
        return tensor, self.labels[index]


def extract_features(
    backbone: str,
    paths: list[Path],
    labels: list[int],
    device: str = "cuda",
    precision: str = "fp16",
    batch_size: int = 64,
    workers: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run the frozen backbone once over a split and return L2-normalised features."""
    set_seed(0)
    model, dim, size = build_backbone(backbone)
    model = model.to(device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    if device == "cuda" and precision != "fp32":
        model = model.to(dtype)
    loader = DataLoader(
        ImageList(paths, labels, size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device == "cuda"),
    )
    feats, targets = [], []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            if device == "cuda" and precision != "fp32":
                images = images.to(dtype)
            out = model(images)
            out = out.reshape(out.shape[0], -1).float()
            out = torch.nn.functional.normalize(out, dim=-1)
            feats.append(out.cpu().numpy().astype(np.float32))
            targets.append(target.numpy())
    elapsed = time.perf_counter() - t0
    features = np.concatenate(feats)
    targets_arr = np.concatenate(targets)
    stats = {
        "backbone": backbone,
        "device": device,
        "precision": precision,
        "images": len(paths),
        "feature_dim": dim,
        "resolution": size,
        "extract_seconds": round(elapsed, 3),
        "images_per_second": round(len(paths) / elapsed, 2),
    }
    return features, targets_arr, stats
