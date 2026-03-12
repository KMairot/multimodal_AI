from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def set_seed(seed: int = 1998) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_volume_layout(arr: np.ndarray, n_slices: int = 19) -> np.ndarray:
    """Convert many common OCT npy layouts to [S,H,W]."""
    a = np.squeeze(arr)

    if a.ndim == 2:
        return np.repeat(a[None, ...], n_slices, axis=0).astype(np.float32)

    if a.ndim == 3:
        if a.shape[0] == n_slices:
            return a.astype(np.float32)
        if a.shape[-1] == n_slices:
            return np.moveaxis(a, -1, 0).astype(np.float32)
        if a.shape[0] <= 64 and a.shape[0] < a.shape[-1]:
            return a.astype(np.float32)
        raise ValueError(f"Unrecognized 3D OCT layout: {a.shape}")

    if a.ndim == 4:
        # [S,C,H,W]
        if a.shape[0] == n_slices and a.shape[1] in (1, 3):
            return a[:, 0, ...].astype(np.float32)
        # [C,S,H,W]
        if a.shape[1] == n_slices and a.shape[0] in (1, 3):
            return a[0, :, ...].astype(np.float32)
        # [S,H,W,C]
        if a.shape[0] == n_slices and a.shape[-1] in (1, 3):
            return a[..., 0].astype(np.float32)
        # [C,H,W,S]
        if a.shape[-1] == n_slices and a.shape[0] in (1, 3):
            return np.moveaxis(a[0, ...], -1, 0).astype(np.float32)
        raise ValueError(f"Unrecognized 4D OCT layout: {a.shape}")

    raise ValueError(f"Unsupported OCT ndim={a.ndim}, shape={a.shape}")


def robust_normalize(vol_shw: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    v = vol_shw.astype(np.float32, copy=False)
    lo = float(np.min(v))
    hi = float(np.max(v))
    if (hi - lo) < eps:
        return np.zeros_like(v, dtype=np.float32)
    v = np.clip(v, lo, hi)
    return ((v - lo) / (hi - lo + eps)).astype(np.float32)


def imagenet_normalize_2d(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype)[None, :, None, None]
    return (x - mean) / std


def imagenet_normalize_oct(x: torch.Tensor) -> torch.Tensor:
    # x: [B,S,3,H,W]
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype)[None, None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype)[None, None, :, None, None]
    return (x - mean) / std


def save_json(data: Dict[str, Any], path: Path) -> None:
    def _conv(x: Any) -> Any:
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.float32, np.float64)):
            return float(x)
        if isinstance(x, (np.int32, np.int64)):
            return int(x)
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, dict):
            return {k: _conv(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_conv(v) for v in x]
        return x

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_conv(data), f, indent=2, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Dataset + Augmentations
# -----------------------------------------------------------------------------
def _random_apply(p: float) -> bool:
    return random.random() < p


def augment_oct_tensor(x_s3hw: torch.Tensor) -> torch.Tensor:
    """Moderate OCT augmentations on [S,3,H,W] in [0,1]."""
    x = x_s3hw.clone()

    if _random_apply(0.7):
        c = 1.0 + random.uniform(-0.08, 0.08)
        b = random.uniform(-0.05, 0.05)
        x = (x - 0.5) * c + 0.5 + b

    if _random_apply(0.5):
        noise_std = random.uniform(0.005, 0.015)
        x = x + torch.randn_like(x) * noise_std

    if _random_apply(0.4):
        sh = random.randint(-6, 6)
        sw = random.randint(-6, 6)
        x = torch.roll(x, shifts=(sh, sw), dims=(-2, -1))

    return x.clamp(0.0, 1.0)


class OCTIRContrastiveDataset(Dataset):
    """
    Dataset for OCT↔IR contrastive pretraining.
    Filters rows with split match + has_oct=1 + has_ir=1 + valid paths.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        split: str,
        image_h: int = 256,
        image_w: int = 512,
        fundus_h: int = 256,
        fundus_w: int = 256,
        n_slices: int = 19,
        augment_oct: bool = False,
        augment_ir: bool = False,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        if not self.manifest_csv.exists():
            raise FileNotFoundError(self.manifest_csv)

        df = pd.read_csv(self.manifest_csv, dtype=str).fillna("")

        required = [
            "gene", "patient_key", "laterality", "oct_path", "ir_path",
            "has_oct", "has_ir", "split",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in manifest: {missing}")

        df = df[df["split"] == str(split)].copy()
        df = df[(df["has_oct"] == "1") & (df["has_ir"] == "1")].copy()
        df = df[(df["oct_path"].str.len() > 0) & (df["ir_path"].str.len() > 0)].copy()

        # Keep only rows with existing files
        keep_idx = []
        for i, r in df.iterrows():
            if Path(r["oct_path"]).exists() and Path(r["ir_path"]).exists():
                keep_idx.append(i)
        df = df.loc[keep_idx].reset_index(drop=True)

        self.df = df
        self.image_h = int(image_h)
        self.image_w = int(image_w)
        self.fundus_h = int(fundus_h)
        self.fundus_w = int(fundus_w)
        self.n_slices = int(n_slices)
        self.augment_oct = bool(augment_oct)
        self.augment_ir = bool(augment_ir)

        self.ir_tf_train = T.Compose([
            T.Resize((self.fundus_h, self.fundus_w)),
            # Conservative IR augmentation for cross-modal geometric consistency.
            T.ColorJitter(brightness=0.05, contrast=0.10),
            T.ToTensor(),
        ])
        self.ir_tf_eval = T.Compose([
            T.Resize((self.fundus_h, self.fundus_w)),
            T.ToTensor(),
        ])

        if len(self.df) == 0:
            raise RuntimeError(f"No valid OCT-IR pairs found for split={split} in {manifest_csv}")

    def __len__(self) -> int:
        return len(self.df)

    def _load_oct(self, oct_path: str) -> torch.Tensor:
        arr = np.load(oct_path, allow_pickle=False)
        vol = infer_volume_layout(arr, n_slices=self.n_slices)
        vol = robust_normalize(vol)

        x = torch.from_numpy(vol).float().unsqueeze(1)  # [S,1,H,W]
        x = F.interpolate(x, size=(self.image_h, self.image_w), mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)  # [S,3,H,W]

        if self.augment_oct:
            x = augment_oct_tensor(x)

        return x

    def _load_ir(self, ir_path: str) -> torch.Tensor:
        img = Image.open(ir_path).convert("RGB")
        tf = self.ir_tf_train if self.augment_ir else self.ir_tf_eval
        return tf(img)  # [3,H,W], in [0,1]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.df.iloc[idx]
        oct_x = self._load_oct(r["oct_path"])
        ir_x = self._load_ir(r["ir_path"])

        meta = {
            "gene": str(r["gene"]),
            "patient_key": str(r["patient_key"]),
            "laterality": str(r["laterality"]),
            "oct_path": str(r["oct_path"]),
            "ir_path": str(r["ir_path"]),
        }
        return {"oct": oct_x, "ir": ir_x, "meta": meta}


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class OCTVolumeEncoder(nn.Module):
    """
    Slice-wise ConvNeXt-Tiny + volume aggregation (meanmax), then projection.
    forward_oct returns:
      - z_global: non-projected global embedding
      - z_proj: projected embedding for contrastive learning
    """

    def __init__(self, global_dim: int = 512, embed_dim: int = 256, dropout: float = 0.2, pretrained: bool = True):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.convnext_tiny(weights=weights)
        self.encoder = nn.Sequential(base.features, base.avgpool)
        self.enc_dim = 768
        pooled_dim = self.enc_dim * 2  # meanmax

        self.global_head = nn.Sequential(
            nn.Linear(pooled_dim, global_dim),
            nn.LayerNorm(global_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_head = nn.Sequential(
            nn.Linear(global_dim, global_dim),
            nn.GELU(),
            nn.Linear(global_dim, embed_dim),
        )

    def forward_oct(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B,S,3,H,W] (already normalized)
        b, s, c, h, w = x.shape
        feat = self.encoder(x.view(b * s, c, h, w)).flatten(1)  # [B*S,768]
        feat = feat.view(b, s, self.enc_dim)

        pooled = torch.cat([feat.mean(dim=1), feat.max(dim=1).values], dim=1)
        z_global = self.global_head(pooled)
        z_proj = self.proj_head(z_global)
        return z_global, z_proj


class IREncoder(nn.Module):
    """2D ConvNeXt-Tiny IR encoder + projection head."""

    def __init__(self, global_dim: int = 512, embed_dim: int = 256, dropout: float = 0.2, pretrained: bool = True):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.convnext_tiny(weights=weights)
        self.encoder = nn.Sequential(base.features, base.avgpool)
        self.enc_dim = 768

        self.global_head = nn.Sequential(
            nn.Linear(self.enc_dim, global_dim),
            nn.LayerNorm(global_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_head = nn.Sequential(
            nn.Linear(global_dim, global_dim),
            nn.GELU(),
            nn.Linear(global_dim, embed_dim),
        )

    def forward_ir(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B,3,H,W] (already normalized)
        feat = self.encoder(x).flatten(1)
        z_global = self.global_head(feat)
        z_proj = self.proj_head(z_global)
        return z_global, z_proj


class OCTIRContrastiveModel(nn.Module):
    def __init__(self, embed_dim: int = 256, oct_global_dim: int = 512, ir_global_dim: int = 512, pretrained_backbones: bool = True):
        super().__init__()
        self.oct_encoder = OCTVolumeEncoder(global_dim=oct_global_dim, embed_dim=embed_dim, pretrained=pretrained_backbones)
        self.ir_encoder = IREncoder(global_dim=ir_global_dim, embed_dim=embed_dim, pretrained=pretrained_backbones)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), dtype=torch.float32))

    def forward(self, oct_x: torch.Tensor, ir_x: torch.Tensor) -> Dict[str, torch.Tensor]:
        oct_global, oct_proj = self.oct_encoder.forward_oct(oct_x)
        ir_global, ir_proj = self.ir_encoder.forward_ir(ir_x)

        oct_norm = F.normalize(oct_proj, dim=1)
        ir_norm = F.normalize(ir_proj, dim=1)

        scale = self.logit_scale.exp().clamp(max=100.0)
        logits_o2i = scale * oct_norm @ ir_norm.t()
        logits_i2o = logits_o2i.t()

        return {
            "oct_global": oct_global,
            "ir_global": ir_global,
            "oct_embed": oct_norm,
            "ir_embed": ir_norm,
            "logits_o2i": logits_o2i,
            "logits_i2o": logits_i2o,
            "logit_scale": scale,
        }


# -----------------------------------------------------------------------------
# Loss / metrics
# -----------------------------------------------------------------------------
def clip_contrastive_loss(logits_o2i: torch.Tensor, logits_i2o: torch.Tensor) -> torch.Tensor:
    b = logits_o2i.size(0)
    target = torch.arange(b, device=logits_o2i.device)
    loss_o2i = F.cross_entropy(logits_o2i, target)
    loss_i2o = F.cross_entropy(logits_i2o, target)
    return 0.5 * (loss_o2i + loss_i2o)


def recall_at_1(logits: torch.Tensor) -> float:
    target = torch.arange(logits.size(0), device=logits.device)
    pred = logits.argmax(dim=1)
    return float((pred == target).float().mean().item())


# -----------------------------------------------------------------------------
# Train / Eval
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    manifest: str
    outdir: str
    run_name: str
    split_train: str = "train"
    split_val: str = "val"
    image_h: int = 256
    image_w: int = 512
    fundus_h: int = 256
    fundus_w: int = 256
    n_slices: int = 19
    batch: int = 8
    epochs: int = 30
    lr_enc: float = 1e-4
    lr_proj: float = 1e-3
    weight_decay: float = 1e-4
    embed_dim: int = 256
    oct_global_dim: int = 512
    ir_global_dim: int = 512
    num_workers: int = 4
    no_amp: bool = False
    seed: int = 1998
    patience: int = 8
    scheduler: str = "cosine"  # cosine|none
    pretrained_backbones: bool = True


def build_optimizer(model: OCTIRContrastiveModel, lr_enc: float, lr_proj: float, weight_decay: float) -> torch.optim.Optimizer:
    enc_params: List[nn.Parameter] = []
    proj_params: List[nn.Parameter] = []
    other_params: List[nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("oct_encoder.encoder") or name.startswith("ir_encoder.encoder"):
            enc_params.append(p)
        elif "global_head" in name or "proj_head" in name:
            proj_params.append(p)
        else:
            other_params.append(p)

    groups = []
    if enc_params:
        groups.append({"params": enc_params, "lr": lr_enc})
    if proj_params:
        groups.append({"params": proj_params, "lr": lr_proj})
    if other_params:
        groups.append({"params": other_params, "lr": lr_proj})

    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def make_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    return None


def _prepare_batch(batch: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    oct_x = batch["oct"].to(device, non_blocking=True)  # [B,S,3,H,W] in [0,1]
    ir_x = batch["ir"].to(device, non_blocking=True)    # [B,3,H,W] in [0,1]

    oct_x = imagenet_normalize_oct(oct_x)
    ir_x = imagenet_normalize_2d(ir_x)
    return oct_x, ir_x


def train_one_epoch(
    model: OCTIRContrastiveModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    losses = []
    r1_o2i, r1_i2o = [], []

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        oct_x, ir_x = _prepare_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            out = model(oct_x, ir_x)
            loss = clip_contrastive_loss(out["logits_o2i"], out["logits_i2o"])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.item()))
        r1_o2i.append(recall_at_1(out["logits_o2i"]))
        r1_i2o.append(recall_at_1(out["logits_i2o"]))
        pbar.set_postfix(loss=f"{np.mean(losses):.4f}", r1=f"{np.mean(r1_o2i):.3f}")

    return {
        "loss": float(np.mean(losses) if losses else 0.0),
        "r1_o2i": float(np.mean(r1_o2i) if r1_o2i else 0.0),
        "r1_i2o": float(np.mean(r1_i2o) if r1_i2o else 0.0),
    }


@torch.no_grad()
def evaluate(
    model: OCTIRContrastiveModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()

    losses = []
    r1_o2i, r1_i2o = [], []

    pbar = tqdm(loader, desc="val", leave=False)
    for batch in pbar:
        oct_x, ir_x = _prepare_batch(batch, device)
        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            out = model(oct_x, ir_x)
            loss = clip_contrastive_loss(out["logits_o2i"], out["logits_i2o"])

        losses.append(float(loss.item()))
        r1_o2i.append(recall_at_1(out["logits_o2i"]))
        r1_i2o.append(recall_at_1(out["logits_i2o"]))

    return {
        "loss": float(np.mean(losses) if losses else 0.0),
        "r1_o2i": float(np.mean(r1_o2i) if r1_o2i else 0.0),
        "r1_i2o": float(np.mean(r1_i2o) if r1_i2o else 0.0),
    }


def save_checkpoint(path: Path, model: OCTIRContrastiveModel, cfg: TrainConfig, epoch: int, val_metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "oct_encoder": model.oct_encoder.state_dict(),
            "ir_encoder": model.ir_encoder.state_dict(),
            "config": asdict(cfg),
            "epoch": epoch,
            "val": val_metrics,
        },
        path,
    )


def load_pretrained_oct_encoder(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> OCTVolumeEncoder:
    """
    Load OCTVolumeEncoder from best.pt / best_oct_encoder.pt.
    Returns full module: backbone + meanmax aggregation + global/projection heads.
    """
    ckpt = torch.load(Path(checkpoint_path), map_location="cpu")

    if isinstance(ckpt, dict) and "config" in ckpt:
        cfg = ckpt["config"]
        model = OCTVolumeEncoder(
            global_dim=int(cfg.get("oct_global_dim", 512)),
            embed_dim=int(cfg.get("embed_dim", 256)),
            pretrained=False,
        )
        if "oct_encoder" in ckpt:
            state = ckpt["oct_encoder"]
        elif "model" in ckpt:
            # fallback: extract oct encoder keys from full model
            full = ckpt["model"]
            state = {k.replace("oct_encoder.", "", 1): v for k, v in full.items() if k.startswith("oct_encoder.")}
        else:
            raise ValueError("No oct_encoder or model state found in checkpoint")
        result = model.load_state_dict(state, strict=strict)
        if not strict:
            print(f"[load_pretrained_oct_encoder] missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")
        return model.to(device)

    # if directly saved encoder state_dict
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        model = OCTVolumeEncoder(pretrained=False)
        result = model.load_state_dict(ckpt, strict=strict)
        if not strict:
            print(f"[load_pretrained_oct_encoder] missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")
        return model.to(device)

    raise ValueError("Unsupported checkpoint format for OCT encoder loading")


def run_training(cfg: TrainConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not cfg.no_amp) and (device.type == "cuda")

    run_dir = Path(cfg.outdir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    save_json(asdict(cfg), run_dir / "config.json")

    train_ds = OCTIRContrastiveDataset(
        manifest_csv=cfg.manifest,
        split=cfg.split_train,
        image_h=cfg.image_h,
        image_w=cfg.image_w,
        fundus_h=cfg.fundus_h,
        fundus_w=cfg.fundus_w,
        n_slices=cfg.n_slices,
        augment_oct=True,
        augment_ir=True,
    )
    val_ds = OCTIRContrastiveDataset(
        manifest_csv=cfg.manifest,
        split=cfg.split_val,
        image_h=cfg.image_h,
        image_w=cfg.image_w,
        fundus_h=cfg.fundus_h,
        fundus_w=cfg.fundus_w,
        n_slices=cfg.n_slices,
        augment_oct=False,
        augment_ir=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = OCTIRContrastiveModel(
        embed_dim=cfg.embed_dim,
        oct_global_dim=cfg.oct_global_dim,
        ir_global_dim=cfg.ir_global_dim,
        pretrained_backbones=cfg.pretrained_backbones,
    ).to(device)

    optimizer = build_optimizer(model, cfg.lr_enc, cfg.lr_proj, cfg.weight_decay)
    scheduler = make_scheduler(optimizer, cfg)

    history_rows: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_val_r1_o2i = 0.0
    best_val_r1_i2o = 0.0
    patience_count = 0

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f_hist:
        writer = csv.DictWriter(
            f_hist,
            fieldnames=[
                "epoch", "train_loss", "train_r1_o2i", "train_r1_i2o",
                "val_loss", "val_r1_o2i", "val_r1_i2o", "lr",
            ],
        )
        writer.writeheader()

        print(f"[INFO] device={device} | amp={use_amp}")
        print(f"[INFO] train pairs={len(train_ds)} | val pairs={len(val_ds)}")

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            train_m = train_one_epoch(model, train_loader, optimizer, device, use_amp)
            val_m = evaluate(model, val_loader, device, use_amp)

            if scheduler is not None:
                scheduler.step()

            lr_now = float(optimizer.param_groups[0]["lr"])
            row = {
                "epoch": epoch,
                "train_loss": train_m["loss"],
                "train_r1_o2i": train_m["r1_o2i"],
                "train_r1_i2o": train_m["r1_i2o"],
                "val_loss": val_m["loss"],
                "val_r1_o2i": val_m["r1_o2i"],
                "val_r1_i2o": val_m["r1_i2o"],
                "lr": lr_now,
            }
            history_rows.append(row)
            writer.writerow(row)
            f_hist.flush()

            dt = time.time() - t0
            print(
                f"[ep {epoch:03d}] "
                f"train_loss={train_m['loss']:.4f} val_loss={val_m['loss']:.4f} "
                f"val_r1(o2i/i2o)=({val_m['r1_o2i']:.3f}/{val_m['r1_i2o']:.3f}) "
                f"lr={lr_now:.2e} time={dt:.1f}s"
            )

            improved = val_m["loss"] < best_val_loss
            if improved:
                best_val_loss = val_m["loss"]
                best_epoch = epoch
                best_val_r1_o2i = val_m["r1_o2i"]
                best_val_r1_i2o = val_m["r1_i2o"]
                patience_count = 0
                save_checkpoint(run_dir / "best.pt", model, cfg, epoch, val_m)
                torch.save(model.oct_encoder.state_dict(), run_dir / "best_oct_encoder.pt")
            else:
                patience_count += 1

            if patience_count >= cfg.patience:
                print(f"[EARLY STOP] no val_loss improvement for {cfg.patience} epochs")
                break

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_r1_o2i": best_val_r1_o2i,
        "best_val_r1_i2o": best_val_r1_i2o,
        "final_epoch": history_rows[-1]["epoch"] if history_rows else 0,
        "num_train_pairs": len(train_ds),
        "num_val_pairs": len(val_ds),
        "device": str(device),
        "amp": use_amp,
        "run_dir": str(run_dir),
    }
    save_json(summary, run_dir / "summary.json")

    return {
        "config": asdict(cfg),
        "history": history_rows,
        "summary": summary,
        "run_dir": str(run_dir),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OCT↔IR contrastive pretraining (CLIP-style)")
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--run-name", type=str, default="")

    p.add_argument("--split-train", type=str, default="train")
    p.add_argument("--split-val", type=str, default="val")

    p.add_argument("--image-h", type=int, default=256)
    p.add_argument("--image-w", type=int, default=512)
    p.add_argument("--fundus-h", type=int, default=256)
    p.add_argument("--fundus-w", type=int, default=256)
    p.add_argument("--n-slices", type=int, default=19)

    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr-enc", type=float, default=1e-4)
    p.add_argument("--lr-proj", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--oct-global-dim", type=int, default=512)
    p.add_argument("--ir-global-dim", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=1998)
    p.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained ConvNeXt backbones")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_name = args.run_name.strip() or time.strftime("oct_ir_clip_%Y%m%d_%H%M%S")

    cfg = TrainConfig(
        manifest=args.manifest,
        outdir=args.outdir,
        run_name=run_name,
        split_train=args.split_train,
        split_val=args.split_val,
        image_h=args.image_h,
        image_w=args.image_w,
        fundus_h=args.fundus_h,
        fundus_w=args.fundus_w,
        n_slices=args.n_slices,
        batch=args.batch,
        epochs=args.epochs,
        lr_enc=args.lr_enc,
        lr_proj=args.lr_proj,
        weight_decay=args.weight_decay,
        embed_dim=args.embed_dim,
        oct_global_dim=args.oct_global_dim,
        ir_global_dim=args.ir_global_dim,
        num_workers=args.num_workers,
        no_amp=args.no_amp,
        seed=args.seed,
        patience=args.patience,
        scheduler=args.scheduler,
        pretrained_backbones=not args.no_pretrained,
    )

    result = run_training(cfg)
    print("[DONE]", result["summary"])


if __name__ == "__main__":
    main()
