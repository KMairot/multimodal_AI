from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="The verbose parameter is deprecated")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def set_seed(seed: int = 1998) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_volume_layout(arr: np.ndarray, n_slices: int = 19) -> np.ndarray:
    a = arr if isinstance(arr, np.ndarray) else np.array(arr)
    a = np.squeeze(a)

    if a.ndim == 2:
        a = np.repeat(a[None, ...], n_slices, axis=0)
        return a.astype(np.float32)

    if a.ndim == 3:
        if a.shape[0] == n_slices:
            return a.astype(np.float32)
        if a.shape[-1] == n_slices:
            return np.moveaxis(a, -1, 0).astype(np.float32)
        if a.shape[0] < a.shape[-1] and a.shape[0] <= 64:
            return a.astype(np.float32)
        raise ValueError(f"Unrecognized 3D layout {a.shape}. Expected (19,H,W) or (H,W,19).")

    if a.ndim == 4:
        if a.shape[0] == n_slices and a.shape[1] in (1, 3):
            return a[:, 0, ...].astype(np.float32)
        if a.shape[1] == n_slices and a.shape[0] in (1, 3):
            return a[0, :, ...].astype(np.float32)
        if a.shape[0] == n_slices and a.shape[-1] in (1, 3):
            return a[..., 0].astype(np.float32)
        if a.shape[-1] == n_slices and a.shape[0] in (1, 3):
            return np.moveaxis(a[0, ...], -1, 0).astype(np.float32)
        raise ValueError(f"Unrecognized 4D layout {a.shape}.")

    raise ValueError(f"Unsupported ndarray ndim={a.ndim}, shape={a.shape}")


def robust_normalize(vol_shw: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    v = vol_shw.astype(np.float32, copy=False)
    lo = float(v.min())
    hi = float(v.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < eps:
        lo = float(np.min(v))
        hi = float(np.max(v))
        if (hi - lo) < eps:
            return np.zeros_like(v, dtype=np.float32)
    v = np.clip(v, lo, hi)
    v = (v - lo) / (hi - lo + eps)
    return v.astype(np.float32)


def imagenet_normalize_3ch(x_bshw: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x_bshw.device)[None, None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=x_bshw.device)[None, None, :, None, None]
    return (x_bshw - mean) / std


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def metrics_from_cm(cm: np.ndarray) -> Dict[str, Any]:
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred_sum = cm.sum(axis=0).astype(np.float64)

    acc = tp.sum() / max(cm.sum(), 1)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, pred_sum, out=np.zeros_like(tp), where=pred_sum > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)

    mask = support > 0
    bacc = float(np.mean(recall[mask])) if mask.any() else 0.0
    macro_f1 = float(np.mean(f1[mask])) if mask.any() else 0.0

    return {
        "acc": float(acc),
        "bacc": bacc,
        "macro_f1": macro_f1,
        "per_class_recall": recall.tolist(),
    }


def compute_soft_class_weights(dataset, device: torch.device):
    ys = np.array([s["y"] for s in dataset.samples], dtype=np.int64)
    counts = np.bincount(ys, minlength=len(dataset.labels)).astype(np.float32)
    w = counts.sum() / np.maximum(counts, 1.0)
    w = w / w.mean()
    w = np.sqrt(w)
    return torch.tensor(w, dtype=torch.float32, device=device), counts


def safe_json_dump(obj: Any, path: Path):
    def _convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.float32, np.float64)):
            return float(x)
        if isinstance(x, (np.int32, np.int64)):
            return int(x)
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, dict):
            return {k: _convert(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_convert(v) for v in x]
        return x

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_convert(obj), f, indent=2, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Augmentations / slice dropout
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Augmentations / slice dropout
# -----------------------------------------------------------------------------
def _random_apply(p: float) -> bool:
    return random.random() < p


def _apply_brightness_contrast(x: torch.Tensor, brightness: float, contrast: float) -> torch.Tensor:
    c = 1.0 + random.uniform(-contrast, contrast)
    b = random.uniform(-brightness, brightness)
    x = (x - 0.5) * c + 0.5 + b
    return x


def _apply_gaussian_noise(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    return x + torch.randn_like(x) * noise_std


def _apply_speckle_noise(x: torch.Tensor, speckle_std: float) -> torch.Tensor:
    # bruit multiplicatif plus réaliste pour OCT
    noise = torch.randn_like(x) * speckle_std
    return x * (1.0 + noise)


def _apply_shift(x: torch.Tensor, max_shift_h: int, max_shift_w: int) -> torch.Tensor:
    sh = random.randint(-max_shift_h, max_shift_h)
    sw = random.randint(-max_shift_w, max_shift_w)
    return torch.roll(x, shifts=(sh, sw), dims=(-2, -1))


def _apply_hflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=(-1,))


def _apply_small_rotation(x: torch.Tensor, max_angle_deg: float = 5.0) -> torch.Tensor:
    # x: [S,1,H,W]
    angle = random.uniform(-max_angle_deg, max_angle_deg)
    theta = math.radians(angle)
    c = math.cos(theta)
    s = math.sin(theta)

    S, C, H, W = x.shape
    affine = torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0]],
        dtype=x.dtype,
        device=x.device
    ).unsqueeze(0).repeat(S, 1, 1)

    grid = F.affine_grid(affine, size=x.size(), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return x


def _apply_elastic_like_warp(x: torch.Tensor, alpha: float = 3.0, sigma: float = 8.0) -> torch.Tensor:
    """
    Déformation légère, simplifiée, cohérente slice par slice.
    On crée un petit champ lissé basse fréquence puis on l'applique à toutes les slices.
    """
    S, C, H, W = x.shape
    device = x.device
    dtype = x.dtype

    # petit champ basse résolution puis upsample
    low_h = max(8, H // 16)
    low_w = max(8, W // 16)

    disp = torch.randn(1, 2, low_h, low_w, device=device, dtype=dtype)
    disp = F.interpolate(disp, size=(H, W), mode="bilinear", align_corners=False)
    disp = disp / (disp.abs().amax() + 1e-6)
    disp = disp * (alpha / max(H, W))  # amplitude légère en coordonnées normalisées approx

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing="ij"
    )
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # [1,H,W,2]
    flow = disp.permute(0, 2, 3, 1)  # [1,H,W,2]
    grid = base_grid + flow
    grid = grid.repeat(S, 1, 1, 1)

    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return x


def _apply_coarse_dropout(x: torch.Tensor, p: float = 0.5, max_h_frac: float = 0.12, max_w_frac: float = 0.12) -> torch.Tensor:
    if not _random_apply(p):
        return x
    S, C, H, W = x.shape
    hole_h = max(4, int(H * random.uniform(0.04, max_h_frac)))
    hole_w = max(4, int(W * random.uniform(0.04, max_w_frac)))
    top = random.randint(0, max(0, H - hole_h))
    left = random.randint(0, max(0, W - hole_w))
    x = x.clone()
    x[:, :, top:top + hole_h, left:left + hole_w] = 0.0
    return x


def augment_volume_tensor(
    x_s1hw: torch.Tensor,
    policy: str = "basic",
) -> torch.Tensor:
    """
    Politiques :
      - basic
      - oct_light
      - oct_realistic
      - oct_realistic_noflip
      - oct_occlusion
    """
    x = x_s1hw.clone()
    policy = str(policy).lower()

    if policy == "basic":
        if _random_apply(0.8):
            x = _apply_brightness_contrast(x, brightness=0.10, contrast=0.10)
        if _random_apply(0.5):
            x = _apply_gaussian_noise(x, noise_std=0.015)
        if _random_apply(0.5):
            x = _apply_shift(x, max_shift_h=8, max_shift_w=8)

    elif policy == "oct_light":
        if _random_apply(0.8):
            x = _apply_brightness_contrast(x, brightness=0.08, contrast=0.08)
        if _random_apply(0.5):
            x = _apply_gaussian_noise(x, noise_std=0.010)
        if _random_apply(0.5):
            x = _apply_shift(x, max_shift_h=6, max_shift_w=6)
        if _random_apply(0.4):
            x = _apply_small_rotation(x, max_angle_deg=4.0)

    elif policy == "oct_realistic":
        if _random_apply(0.8):
            x = _apply_brightness_contrast(x, brightness=0.08, contrast=0.10)
        if _random_apply(0.6):
            x = _apply_speckle_noise(x, speckle_std=0.06)
        if _random_apply(0.5):
            x = _apply_shift(x, max_shift_h=6, max_shift_w=6)
        if _random_apply(0.4):
            x = _apply_small_rotation(x, max_angle_deg=4.0)
        if _random_apply(0.25):
            x = _apply_elastic_like_warp(x, alpha=2.5, sigma=8.0)
        if _random_apply(0.5):
            x = _apply_hflip(x)

    elif policy == "oct_realistic_noflip":
        if _random_apply(0.8):
            x = _apply_brightness_contrast(x, brightness=0.08, contrast=0.10)
        if _random_apply(0.6):
            x = _apply_speckle_noise(x, speckle_std=0.06)
        if _random_apply(0.5):
            x = _apply_shift(x, max_shift_h=6, max_shift_w=6)
        if _random_apply(0.4):
            x = _apply_small_rotation(x, max_angle_deg=4.0)
        if _random_apply(0.25):
            x = _apply_elastic_like_warp(x, alpha=2.5, sigma=8.0)

    elif policy == "oct_occlusion":
        if _random_apply(0.8):
            x = _apply_brightness_contrast(x, brightness=0.08, contrast=0.10)
        if _random_apply(0.6):
            x = _apply_speckle_noise(x, speckle_std=0.06)
        if _random_apply(0.5):
            x = _apply_shift(x, max_shift_h=6, max_shift_w=6)
        if _random_apply(0.4):
            x = _apply_small_rotation(x, max_angle_deg=4.0)
        if _random_apply(0.25):
            x = _apply_elastic_like_warp(x, alpha=2.5, sigma=8.0)
        x = _apply_coarse_dropout(x, p=0.5, max_h_frac=0.10, max_w_frac=0.10)

    else:
        raise ValueError(f"Unknown augment policy: {policy}")

    x = x.clamp(0.0, 1.0)
    return x


def apply_slice_dropout(x_schw: torch.Tensor, p_drop: float, min_keep: int = 12) -> torch.Tensor:
    if p_drop <= 0:
        return x_schw
    s = x_schw.shape[0]
    keep_mask = torch.rand(s) > p_drop
    if int(keep_mask.sum().item()) < min_keep:
        idx = torch.randperm(s)[:min_keep]
        keep_mask[:] = False
        keep_mask[idx] = True
    x = x_schw.clone()
    x[~keep_mask] = 0.0
    return x


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class OCTNPYVolumeDataset(Dataset):
    def __init__(
        self,
        split_root: Path,
        image_h: int = 256,
        image_w: int = 512,
        n_slices: int = 19,
        augment: bool = False,
        augment_policy: str = "basic",
        slice_dropout_p: float = 0.0,
        min_keep_slices: int = 12,
    ):
        self.split_root = Path(split_root)
        if not self.split_root.exists():
            raise FileNotFoundError(self.split_root)

        self.image_h = int(image_h)
        self.image_w = int(image_w)
        self.n_slices = int(n_slices)
        self.augment = bool(augment)
        self.slice_dropout_p = float(slice_dropout_p)
        self.min_keep_slices = int(min_keep_slices)
        self.augment_policy = str(augment_policy)

        self.labels = sorted([d.name for d in self.split_root.iterdir() if d.is_dir()])
        if not self.labels:
            raise RuntimeError(f"No class folders found under: {self.split_root}")

        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.labels)}
        self.samples: List[Dict[str, Any]] = []

        for lbl in self.labels:
            for p in (self.split_root / lbl).rglob("oct19.npy"):
                self.samples.append({"path": p, "label": lbl, "y": self.label_to_idx[lbl]})

        if not self.samples:
            raise RuntimeError(f"No oct19.npy files found under {self.split_root}/<label>/**/oct19.npy")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        arr = np.load(s["path"], allow_pickle=False, mmap_mode=None)
        vol = infer_volume_layout(arr, n_slices=self.n_slices)
        vol = robust_normalize(vol)

        x = torch.from_numpy(vol).unsqueeze(1)  # [S,1,H,W]
        x = F.interpolate(x, size=(self.image_h, self.image_w), mode="bilinear", align_corners=False)

        if self.augment:
            x = augment_volume_tensor(x, policy=self.augment_policy)

        x = x.repeat(1, 3, 1, 1)  # [S,3,H,W]

        if self.augment and self.slice_dropout_p > 0:
            x = apply_slice_dropout(x, p_drop=self.slice_dropout_p, min_keep=self.min_keep_slices)

        y = torch.tensor(s["y"], dtype=torch.long)
        meta = {"path": str(s["path"]), "label": s["label"]}
        return x, y, meta


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class OCTVolumeNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        embed_dim: int = 512,
        dropout: float = 0.3,
        pooling: str = "meanmax",
    ):
        super().__init__()
        pooling = pooling.lower()
        assert pooling in ("mean", "meanmax", "attn"), "pooling must be one of: mean, meanmax, attn"
        self.pooling = pooling

        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.convnext_tiny(weights=weights)
        self.encoder = nn.Sequential(base.features, base.avgpool)
        self.enc_dim = 768

        if self.pooling == "attn":
            self.attn = nn.Sequential(
                nn.Linear(self.enc_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
            )

        pooled_dim = self.enc_dim if self.pooling in ("mean", "attn") else self.enc_dim * 2
        self.proj = nn.Sequential(
            nn.Linear(pooled_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        feat = self.encoder(x).flatten(1)
        feat = feat.view(b, s, self.enc_dim)

        if self.pooling == "mean":
            f = feat.mean(dim=1)
        elif self.pooling == "meanmax":
            f = torch.cat([feat.mean(dim=1), feat.max(dim=1).values], dim=1)
        else:
            a = self.attn(feat).squeeze(-1)
            w = torch.softmax(a, dim=1)
            f = (feat * w.unsqueeze(-1)).sum(dim=1)

        emb = self.proj(f)
        logits = self.head(emb)
        return logits, emb


# -----------------------------------------------------------------------------
# Fine-tuning policy
# -----------------------------------------------------------------------------
def set_trainable_backbone(model: nn.Module, mode: str):
    mode = mode.lower()
    valid = {"head_only", "last_block", "last_stage", "last_two_stages"}
    if mode not in valid:
        raise ValueError(f"unfreeze_mode must be in {sorted(valid)}")

    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.proj.parameters():
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
    if hasattr(model, "attn"):
        for p in model.attn.parameters():
            p.requires_grad = True

    if mode == "head_only":
        return

    feats = model.encoder[0]  # ConvNeXt features
    last_stage = feats[-1]

    if mode == "last_block":
        for p in last_stage[-1].parameters():
            p.requires_grad = True
    elif mode == "last_stage":
        for p in last_stage.parameters():
            p.requires_grad = True
    elif mode == "last_two_stages":
        for p in feats[-2].parameters():
            p.requires_grad = True
        for p in feats[-1].parameters():
            p.requires_grad = True


def build_optimizer(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float):
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("proj.") or name.startswith("head.") or name.startswith("attn."):
            head_params.append(p)
        else:
            backbone_params.append(p)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


# -----------------------------------------------------------------------------
# Eval / train
# -----------------------------------------------------------------------------
@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool = True) -> Dict[str, Any]:
    model.eval()
    criterion = getattr(model, "_criterion", None) or nn.CrossEntropyLoss()

    all_true, all_pred = [], []
    total_loss, total = 0.0, 0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x = imagenet_normalize_3ch(x)

        if device.type == "cuda" and use_amp:
            with torch.amp.autocast("cuda", enabled=True):
                logits, _ = model(x)
                loss = criterion(logits, y)
        else:
            logits, _ = model(x)
            loss = criterion(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        total += x.size(0)
        pred = logits.argmax(dim=1)
        all_true.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
    n_classes = int(model.head.out_features)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    m = metrics_from_cm(cm)

    return {
        "loss": total_loss / max(total, 1),
        "acc": m["acc"],
        "bacc": m["bacc"],
        "macro_f1": m["macro_f1"],
        "cm": cm,
        "per_class_recall": m["per_class_recall"],
    }


def _train_one_epoch(model, loader, optimizer, criterion, device, use_amp=True):
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    all_true, all_pred = [], []
    running_loss, total = 0.0, 0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x = imagenet_normalize_3ch(x)
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda" and use_amp:
            with torch.amp.autocast("cuda", enabled=True):
                logits, _ = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item()) * x.size(0)
        total += x.size(0)
        pred = logits.argmax(dim=1)
        all_true.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, model.head.out_features)
    m = metrics_from_cm(cm)
    return {
        "loss": running_loss / max(total, 1),
        "acc": m["acc"],
        "bacc": m["bacc"],
        "macro_f1": m["macro_f1"],
        "cm": cm,
    }


def default_search_space() -> Dict[str, List[Any]]:
    return {
        "seed": [1998, 2024, 7],
        "pooling": ["meanmax", "attn"],
        "dropout": [0.1, 0.2, 0.3],
        "weight_decay": [1e-5, 1e-4, 5e-4],
        "augment": [False, True],
        "augment_policy": ["basic", "oct_light", "oct_realistic", "oct_realistic_noflip"],
        "slice_dropout_p": [0.0, 0.15],
        "unfreeze_mode": ["last_block", "last_stage", "last_two_stages"],
        "lr_head": [5e-4, 1e-3],
        "lr_backbone": [1e-5, 3e-5, 1e-4],
    }


def iter_search_configs(search_space: Dict[str, List[Any]], skip_invalid: bool = True):
    keys = list(search_space.keys())

    for values in itertools.product(*(search_space[k] for k in keys)):
        cfg = dict(zip(keys, values))

        if skip_invalid:
            # 1) Si slice dropout > 0 mais augment=False, on ignore
            if cfg.get("slice_dropout_p", 0.0) > 0 and not cfg.get("augment", False):
                continue

            # 2) Si augment=False, la policy d'augmentation n'a pas de sens :
            #    on ne garde que "basic" pour éviter les doublons inutiles
            if not cfg.get("augment", False):
                if cfg.get("augment_policy", "basic") != "basic":
                    continue

            # 3) Règles de sécurité pour les LR backbone selon le mode de dégel
            mode = cfg.get("unfreeze_mode", "last_block")
            lr_backbone = float(cfg.get("lr_backbone", 0.0))

            if mode == "last_two_stages" and lr_backbone > 3e-5:
                continue

            if mode == "last_stage" and lr_backbone > 1e-4:
                continue

        yield cfg


def make_run_name(cfg: Dict[str, Any], idx: int) -> str:
    parts = [
        f"r{idx:04d}",
        f"seed{cfg['seed']}",
        f"pool-{cfg['pooling']}",
        f"do-{cfg['dropout']}",
        f"wd-{cfg['weight_decay']}",
        f"aug-{int(bool(cfg['augment']))}",
        f"ap-{cfg['augment_policy']}",
        f"sd-{cfg['slice_dropout_p']}",
        f"uf-{cfg['unfreeze_mode']}",
        f"lrb-{cfg['lr_backbone']}",
        f"lrh-{cfg['lr_head']}",
    ]
    return "__".join(str(x).replace("/", "-") for x in parts)


def train_one_experiment(
    root: str | Path,
    outdir: str | Path,
    *,
    image_h: int = 256,
    image_w: int = 512,
    n_slices: int = 19,
    pooling: str = "meanmax",
    embed_dim: int = 512,
    batch: int = 4,
    dropout: float = 0.1,
    seed: int = 1998,
    num_workers: int = 0,
    use_amp: bool = True,
    pretrained: bool = True,
    eval_val: bool = True,
    lr_backbone: float = 3e-5,
    lr_head: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs_a: int = 8,
    epochs_b: int = 12,
    unfreeze_mode: str = "last_stage",
    augment: bool = False,
    augment_policy: str = "basic",
    slice_dropout_p: float = 0.0,
    min_keep_slices: int = 12,
    early_stop_patience: int = 6,
    run_name: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    root = Path(root)
    outdir = Path(outdir)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = outdir / (run_name or f"run_seed{seed}_{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = OCTNPYVolumeDataset(
        root / "train",
        image_h=image_h,
        image_w=image_w,
        n_slices=n_slices,
        augment=augment,
        augment_policy=augment_policy,
        slice_dropout_p=slice_dropout_p,
        min_keep_slices=min_keep_slices,
    )
    val_ds = OCTNPYVolumeDataset(
        root / "val",
        image_h=image_h,
        image_w=image_w,
        n_slices=n_slices,
        augment=False,
        slice_dropout_p=0.0,
        min_keep_slices=min_keep_slices,
    )

    if train_ds.labels != val_ds.labels:
        raise RuntimeError(f"Label folders differ between train and val. train={train_ds.labels}, val={val_ds.labels}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    model = OCTVolumeNet(
        num_classes=len(train_ds.labels),
        pretrained=pretrained,
        embed_dim=embed_dim,
        dropout=dropout,
        pooling=pooling,
    ).to(device)

    class_weights, counts = compute_soft_class_weights(train_ds, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    model._criterion = criterion

    best_path = run_dir / "best.pt"
    history = []
    best_val_bacc = -1.0
    best_epoch = -1
    best_stage = None
    best_metrics = None
    no_improve = 0
    start_time = time.time()

    config = {
        "root": str(root),
        "run_dir": str(run_dir),
        "labels": train_ds.labels,
        "class_counts": counts.tolist(),
        "image_h": image_h,
        "image_w": image_w,
        "n_slices": n_slices,
        "pooling": pooling,
        "embed_dim": embed_dim,
        "batch": batch,
        "dropout": dropout,
        "seed": seed,
        "num_workers": num_workers,
        "use_amp": use_amp,
        "pretrained": pretrained,
        "lr_backbone": lr_backbone,
        "lr_head": lr_head,
        "weight_decay": weight_decay,
        "epochs_a": epochs_a,
        "epochs_b": epochs_b,
        "unfreeze_mode": unfreeze_mode,
        "augment": augment,
        "augment_policy": augment_policy,
        "slice_dropout_p": slice_dropout_p,
        "min_keep_slices": min_keep_slices,
        "early_stop_patience": early_stop_patience,
    }
    safe_json_dump(config, run_dir / "config.json")

    # Phase A
    set_trainable_backbone(model, "head_only")
    optimizer = build_optimizer(model, lr_backbone=0.0, lr_head=lr_head, weight_decay=weight_decay)

    if verbose:
        print(f"\n=== RUN: {run_dir.name} ===", flush=True)
        print(f"device={device} | train={len(train_ds)} | val={len(val_ds)} | labels={train_ds.labels}", flush=True)
        print(f"class_counts={counts.tolist()} | class_weights={np.round(class_weights.detach().cpu().numpy(), 3).tolist()}", flush=True)
        print(f"Phase A: head_only for {epochs_a} epochs", flush=True)

    for epoch in range(1, epochs_a + 1):
        tr = _train_one_epoch(model, train_loader, optimizer, criterion, device, use_amp=use_amp)
        val = run_eval(model, val_loader, device, use_amp=False) if eval_val else {"loss": math.nan, "acc": math.nan, "bacc": math.nan, "macro_f1": math.nan, "cm": None, "per_class_recall": []}

        row = {
            "phase": "A",
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "train_bacc": tr["bacc"],
            "train_macro_f1": tr["macro_f1"],
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_bacc": val["bacc"],
            "val_macro_f1": val["macro_f1"],
            "lr_backbone": 0.0,
            "lr_head": lr_head,
        }
        history.append(row)

        improved = eval_val and (val["bacc"] > best_val_bacc)
        if improved:
            best_val_bacc = float(val["bacc"])
            best_epoch = epoch
            best_stage = "A"
            best_metrics = deepcopy(val)
            torch.save({"model": model.state_dict(), "labels": train_ds.labels, "config": config}, best_path)
            no_improve = 0
        else:
            no_improve += 1

        if verbose:
            print(
                f"[A {epoch:02d}/{epochs_a}] tr_loss={tr['loss']:.4f} tr_bacc={tr['bacc']:.3f} | "
                f"val_loss={val['loss']:.4f} val_bacc={val['bacc']:.3f} | best={best_val_bacc:.3f}",
                flush=True,
            )

    # Phase B
    set_trainable_backbone(model, unfreeze_mode)
    optimizer = build_optimizer(model, lr_backbone=lr_backbone, lr_head=lr_head, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    if verbose:
        print(f"Phase B: {unfreeze_mode} for {epochs_b} epochs", flush=True)

    for epoch in range(1, epochs_b + 1):
        tr = _train_one_epoch(model, train_loader, optimizer, criterion, device, use_amp=use_amp)
        val = run_eval(model, val_loader, device, use_amp=False) if eval_val else {"loss": math.nan, "acc": math.nan, "bacc": math.nan, "macro_f1": math.nan, "cm": None, "per_class_recall": []}
        scheduler.step(val["loss"])
        lrs = [pg["lr"] for pg in optimizer.param_groups]

        row = {
            "phase": "B",
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "train_bacc": tr["bacc"],
            "train_macro_f1": tr["macro_f1"],
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_bacc": val["bacc"],
            "val_macro_f1": val["macro_f1"],
            "lr_backbone": lrs[0] if lrs else math.nan,
            "lr_head": lrs[-1] if lrs else math.nan,
        }
        history.append(row)

        improved = eval_val and (val["bacc"] > best_val_bacc)
        if improved:
            best_val_bacc = float(val["bacc"])
            best_epoch = epoch
            best_stage = "B"
            best_metrics = deepcopy(val)
            torch.save({"model": model.state_dict(), "labels": train_ds.labels, "config": config}, best_path)
            no_improve = 0
        else:
            no_improve += 1

        if verbose:
            print(
                f"[B {epoch:02d}/{epochs_b}] tr_loss={tr['loss']:.4f} tr_bacc={tr['bacc']:.3f} | "
                f"val_loss={val['loss']:.4f} val_bacc={val['bacc']:.3f} | best={best_val_bacc:.3f} | "
                f"lrs={[f'{x:.1e}' for x in lrs]}",
                flush=True,
            )

        if no_improve >= early_stop_patience:
            if verbose:
                print(f"Early stopping triggered after {no_improve} epochs without val_bacc improvement.", flush=True)
            break

    hist_path = run_dir / "history.csv"
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "best_path": str(best_path),
        "best_val_bacc": best_val_bacc,
        "best_stage": best_stage,
        "best_epoch_in_stage": best_epoch,
        "duration_sec": time.time() - start_time,
        "config": config,
        "best_metrics": best_metrics,
    }
    safe_json_dump(summary, run_dir / "summary.json")

    if verbose and best_metrics is not None:
        print("[best] confusion matrix:")
        print(np.array(best_metrics["cm"]))
        print("[best] per-class recall:", np.round(np.array(best_metrics["per_class_recall"]), 3).tolist())
        print(f"[done] best val_bacc={best_val_bacc:.4f} | saved={best_path}", flush=True)

    return summary


def append_summary_row(csv_path: Path, summary: Dict[str, Any]):
    row = {
        "run_name": summary["run_name"],
        "run_dir": summary["run_dir"],
        "best_path": summary["best_path"],
        "best_val_bacc": summary["best_val_bacc"],
        "best_stage": summary["best_stage"],
        "best_epoch_in_stage": summary["best_epoch_in_stage"],
        "duration_sec": summary["duration_sec"],
        "seed": summary["config"]["seed"],
        "pooling": summary["config"]["pooling"],
        "dropout": summary["config"]["dropout"],
        "weight_decay": summary["config"]["weight_decay"],
        "augment": summary["config"]["augment"],
        "slice_dropout_p": summary["config"]["slice_dropout_p"],
        "unfreeze_mode": summary["config"]["unfreeze_mode"],
        "lr_backbone": summary["config"]["lr_backbone"],
        "lr_head": summary["config"]["lr_head"],
        "epochs_a": summary["config"]["epochs_a"],
        "epochs_b": summary["config"]["epochs_b"],
    }
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_grid_search(
    root: str | Path,
    outdir: str | Path,
    *,
    search_space: Optional[Dict[str, List[Any]]] = None,
    shared_kwargs: Optional[Dict[str, Any]] = None,
    max_runs: Optional[int] = None,
    skip_invalid: bool = True,
    sort_by_bacc: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summaries_csv = outdir / "all_runs_summary.csv"

    if search_space is None:
        search_space = default_search_space()
    if shared_kwargs is None:
        shared_kwargs = {}

    configs = list(iter_search_configs(search_space, skip_invalid=skip_invalid))
    if max_runs is not None:
        configs = configs[:max_runs]

    if verbose:
        print(f"Preparing {len(configs)} runs", flush=True)

    results = []
    for i, cfg in enumerate(configs, start=1):
        run_name = make_run_name(cfg, i)
        summary = train_one_experiment(
            root=root,
            outdir=outdir,
            run_name=run_name,
            verbose=verbose,
            **shared_kwargs,
            **cfg,
        )
        results.append(summary)
        append_summary_row(summaries_csv, summary)

    if sort_by_bacc:
        results.sort(key=lambda x: x["best_val_bacc"], reverse=True)
        safe_json_dump(results, outdir / "ranked_results.json")
    return results


# -----------------------------------------------------------------------------
# CLI / notebook helpers
# -----------------------------------------------------------------------------
def load_search_space_from_json(path: str | Path) -> Dict[str, List[Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def notebook_example_small() -> Tuple[Dict[str, List[Any]], Dict[str, Any]]:
    search_space = {
        "seed": [1998, 2024],
        "pooling": ["meanmax", "attn"],
        "dropout": [0.1, 0.2],
        "weight_decay": [1e-4],
        "augment": [False, True],
        "slice_dropout_p": [0.0, 0.15],
        "unfreeze_mode": ["last_block", "last_stage"],
        "lr_head": [1e-3],
        "lr_backbone": [3e-5, 1e-4],
    }
    shared_kwargs = {
        "batch": 4,
        "epochs_a": 8,
        "epochs_b": 12,
        "num_workers": 0,
        "image_h": 256,
        "image_w": 512,
    }
    return search_space, shared_kwargs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Dataset root containing train/val/test")
    ap.add_argument("--outdir", type=str, default="runs/oct_volume_search")
    ap.add_argument("--search", action="store_true", help="Run grid search instead of a single experiment")
    ap.add_argument("--search-json", type=str, default="", help="Optional JSON file describing the search space")
    ap.add_argument("--max-runs", type=int, default=0)

    # single/shared arguments
    ap.add_argument("--image-h", type=int, default=256)
    ap.add_argument("--image-w", type=int, default=512)
    ap.add_argument("--n-slices", type=int, default=19)
    ap.add_argument("--pooling", type=str, default="meanmax", choices=["mean", "meanmax", "attn"])
    ap.add_argument("--embed-dim", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1998)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--lr-backbone", type=float, default=3e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--epochs-a", type=int, default=8)
    ap.add_argument("--epochs-b", type=int, default=12)
    ap.add_argument("--unfreeze-mode", type=str, default="last_stage", choices=["head_only", "last_block", "last_stage", "last_two_stages"])
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--slice-dropout-p", type=float, default=0.0)
    ap.add_argument("--min-keep-slices", type=int, default=12)
    ap.add_argument("--early-stop-patience", type=int, default=6)
    args = ap.parse_args()

    shared_kwargs = {
        "image_h": args.image_h,
        "image_w": args.image_w,
        "n_slices": args.n_slices,
        "embed_dim": args.embed_dim,
        "batch": args.batch,
        "num_workers": args.num_workers,
        "use_amp": (not args.no_amp),
        "pretrained": (not args.no_pretrained),
        "epochs_a": args.epochs_a,
        "epochs_b": args.epochs_b,
        "min_keep_slices": args.min_keep_slices,
        "early_stop_patience": args.early_stop_patience,
    }

    if args.search:
        search_space = load_search_space_from_json(args.search_json) if args.search_json else default_search_space()
        run_grid_search(
            root=args.root,
            outdir=args.outdir,
            search_space=search_space,
            shared_kwargs=shared_kwargs,
            max_runs=(args.max_runs if args.max_runs > 0 else None),
            verbose=True,
        )
    else:
        train_one_experiment(
            root=args.root,
            outdir=args.outdir,
            pooling=args.pooling,
            dropout=args.dropout,
            seed=args.seed,
            lr_backbone=args.lr_backbone,
            lr_head=args.lr_head,
            weight_decay=args.weight_decay,
            unfreeze_mode=args.unfreeze_mode,
            augment=args.augment,
            slice_dropout_p=args.slice_dropout_p,
            **shared_kwargs,
        )


if __name__ == "__main__":
    main()
