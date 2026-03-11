from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # optional dependency
    roc_auc_score = None

from oct_volume_search import (
    OCTVolumeNet,
    confusion_matrix,
    infer_volume_layout,
    metrics_from_cm,
    robust_normalize,
)


# -----------------------------
# Repro / utils
# -----------------------------
def set_seed(seed: int = 1998) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_pil_gray_as_rgb(path: str) -> Image.Image:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return Image.open(p).convert("RGB")


def compute_soft_class_weights_from_df(df: pd.DataFrame, classes: List[str], device: torch.device) -> torch.Tensor:
    counts = df["gene"].value_counts().reindex(classes).fillna(0).values.astype(np.float32)
    w = counts.sum() / np.maximum(counts, 1.0)
    w = w / max(w.mean(), 1e-6)
    w = np.sqrt(w)
    return torch.tensor(w, dtype=torch.float32, device=device)


def confusion_matrix_figure(cm: np.ndarray, out_path: Path, labels: Optional[List[str]] = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    n = cm.shape[0]
    if labels and len(labels) == n:
        ax.set_xticks(range(n), labels=labels, rotation=45, ha="right")
        ax.set_yticks(range(n), labels=labels)
    else:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def resolve_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    for key in ["model", "state_dict", "model_state_dict", "net"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise ValueError("Checkpoint format unsupported: no state dict found")


def load_state_dict_report(model: nn.Module, state: Dict[str, torch.Tensor], strict: bool, model_name: str) -> None:
    result = model.load_state_dict(state, strict=strict)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    print(f"[{model_name}] strict={strict} | missing={len(missing)} | unexpected={len(unexpected)}")
    if missing:
        print(f"[{model_name}] missing keys (first 20): {missing[:20]}")
    if unexpected:
        print(f"[{model_name}] unexpected keys (first 20): {unexpected[:20]}")


# -----------------------------
# Fundus model (conforme train_fundus_only.py)
# -----------------------------
class DinoBackbone(nn.Module):
    def __init__(self, model_name="vit_small_patch16_dinov3", out_dim=384, pretrained=True, drop_path_rate=0.0):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            attn_drop_rate=0.0,
        )
        feat_dim = self.backbone.num_features
        self.proj = nn.Linear(feat_dim, out_dim) if feat_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


class FundusOnlyNet(nn.Module):
    def __init__(self, genes: List[str], d=384, dinov3_name="vit_small_patch16_dinov3", drop_path_rate=0.05, head_dropout=0.1):
        super().__init__()
        self.genes = list(genes)
        self.n_classes = len(self.genes)
        self.ir = DinoBackbone(dinov3_name, out_dim=d, pretrained=True, drop_path_rate=drop_path_rate)
        self.faf = DinoBackbone(dinov3_name, out_dim=d, pretrained=True, drop_path_rate=drop_path_rate)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, 2 * d),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(2 * d, self.n_classes),
        )

    def extract_embedding(self, ir: torch.Tensor, faf: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.ir(ir), self.faf(faf)], dim=1)

    def forward(self, ir: torch.Tensor, faf: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_embedding(ir, faf))


def unfreeze_last_blocks(vit_model: nn.Module, n_last_blocks=2):
    for p in vit_model.parameters():
        p.requires_grad = False
    if hasattr(vit_model, "norm"):
        for p in vit_model.norm.parameters():
            p.requires_grad = True
    if hasattr(vit_model, "blocks"):
        for blk in vit_model.blocks[-n_last_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True


# -----------------------------
# Dataset
# -----------------------------
class MultimodalDataset(Dataset):
    """
    Colonnes manifest minimales:
      split,(sgene|gene),(case_id optionnel),has_oct,has_ir,has_faf,oct_path,ir_path,faf_path
      + patient_key/laterality optionnels pour reconstruire case_id.

    Convention fundus stricte (défaut): has_fundus=1 seulement si IR+FAF.
    """

    OCT_MEAN = 0.3688
    OCT_STD = 0.0108
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        df: pd.DataFrame,
        classes: List[str],
        split: str,
        label_col: str,
        oct_h: int = 256,
        oct_w: int = 512,
        n_slices: int = 19,
        fundus_size: int = 256,
        strict_fundus_pair: bool = True,
        augment: bool = False,
    ):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.classes = list(classes)
        self.cls_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.label_col = str(label_col)
        self.oct_h, self.oct_w, self.n_slices = oct_h, oct_w, n_slices
        self.strict_fundus_pair = strict_fundus_pair
        self.augment = augment
        self.fundus_size = int(fundus_size)

        self.tf_train = T.Compose([
            T.RandomResizedCrop(self.fundus_size, scale=(0.85, 1.00), ratio=(0.95, 1.05)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=5),
            T.ColorJitter(brightness=0.10, contrast=0.15),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])
        self.tf_val = T.Compose([
            T.Resize((self.fundus_size, self.fundus_size)),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.df)

    def _load_oct(self, p: str) -> torch.Tensor:
        arr = np.load(p, allow_pickle=False)
        vol = infer_volume_layout(arr, n_slices=self.n_slices)
        vol = robust_normalize(vol)
        x = torch.from_numpy(vol).float().unsqueeze(1)  # [S,1,H,W]
        x = F.interpolate(x, size=(self.oct_h, self.oct_w), mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)
        mean = torch.tensor(self.OCT_MEAN).view(1, 1, 1, 1)
        std = torch.tensor(self.OCT_STD).view(1, 1, 1, 1)
        x = (x - mean) / (std + 1e-6)
        return x

    def __getitem__(self, i: int) -> Dict[str, Any]:
        r = self.df.iloc[i]
        y = self.cls_to_idx[r[self.label_col]]
        has_oct = int(str(r.get("has_oct", "0")) == "1" and str(r.get("oct_path", "")) != "")
        has_ir = int(str(r.get("has_ir", "0")) == "1" and str(r.get("ir_path", "")) != "")
        has_faf = int(str(r.get("has_faf", "0")) == "1" and str(r.get("faf_path", "")) != "")
        has_fundus = int(has_ir and has_faf) if self.strict_fundus_pair else int(has_ir or has_faf)

        oct_x = self._load_oct(r["oct_path"]) if has_oct else torch.zeros(self.n_slices, 3, self.oct_h, self.oct_w)
        tf = self.tf_train if self.augment else self.tf_val
        ir_x = tf(load_pil_gray_as_rgb(r["ir_path"])) if has_ir else torch.zeros(3, self.fundus_size, self.fundus_size)
        faf_x = tf(load_pil_gray_as_rgb(r["faf_path"])) if has_faf else torch.zeros(3, self.fundus_size, self.fundus_size)

        case_id = str(r.get("case_id", ""))
        if not case_id:
            pk = str(r.get("patient_key", ""))
            lat = str(r.get("laterality", ""))
            if pk and lat:
                case_id = f"{pk}_{lat}"
            elif pk:
                case_id = pk
            else:
                case_id = f"idx_{i}"

        return {
            "oct": oct_x,
            "ir": ir_x,
            "faf": faf_x,
            "has_oct": torch.tensor(has_oct, dtype=torch.float32),
            "has_fundus": torch.tensor(has_fundus, dtype=torch.float32),
            "label": torch.tensor(y, dtype=torch.long),
            "case_id": case_id,
        }


# -----------------------------
# Extractors / Loaders
# -----------------------------
class OCTWrapper(nn.Module):
    """Wrapper explicite pour API OCTVolumeNet réelle (forward -> logits, emb)."""

    def __init__(self, model: OCTVolumeNet):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(x)
        return logits

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        _, emb = self.model(x)
        return emb

    @property
    def embedding_dim(self) -> int:
        return int(self.model.head.in_features)


class FundusWrapper(nn.Module):
    def __init__(self, model: FundusOnlyNet):
        super().__init__()
        self.model = model

    def forward(self, ir: torch.Tensor, faf: torch.Tensor) -> torch.Tensor:
        return self.model(ir, faf)

    def extract_embedding(self, ir: torch.Tensor, faf: torch.Tensor) -> torch.Tensor:
        return self.model.extract_embedding(ir, faf)

    def infer_embedding_dim(self, device: torch.device, fundus_size: int = 256) -> int:
        with torch.no_grad():
            ir = torch.zeros(1, 3, fundus_size, fundus_size, device=device)
            faf = torch.zeros(1, 3, fundus_size, fundus_size, device=device)
            z = self.extract_embedding(ir, faf)
        return int(z.shape[1])


def load_pretrained_oct_model(
    ckpt_path: str,
    num_classes: int,
    device: torch.device,
    pooling: str = "meanmax",
    dropout: float = 0.2,
    strict_load: bool = False,
) -> OCTWrapper:
    model = OCTVolumeNet(num_classes=num_classes, pretrained=True, pooling=pooling, dropout=dropout)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = resolve_state_dict(ckpt)
    load_state_dict_report(model, state, strict=strict_load, model_name="OCT")
    return OCTWrapper(model.to(device))


def load_pretrained_fundus_model(
    ckpt_path: str,
    classes: List[str],
    device: torch.device,
    strict_load: bool = False,
) -> FundusWrapper:
    model = FundusOnlyNet(classes, d=384, dinov3_name="vit_small_patch16_dinov3", drop_path_rate=0.05, head_dropout=0.1)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = resolve_state_dict(ckpt)
    load_state_dict_report(model, state, strict=strict_load, model_name="Fundus")
    return FundusWrapper(model.to(device))


# -----------------------------
# Fusion models
# -----------------------------
class ProbabilityFusionModel(nn.Module):
    """
    Stratégie: calcul conditionnel par sous-batch.
    Si modalité absente pour un échantillon, on n'exécute pas ce sample dans la branche correspondante.
    Cela évite de "simuler" une modalité par des zéros.
    """

    def __init__(self, oct_model: OCTWrapper, fundus_model: FundusWrapper, num_classes: int, w_oct: float = 0.5, w_fundus: float = 0.5):
        super().__init__()
        self.oct_model = oct_model
        self.fundus_model = fundus_model
        self.num_classes = int(num_classes)
        self.w_oct = float(w_oct)
        self.w_fundus = float(w_fundus)

    @torch.no_grad()
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = batch["label"].device
        b = batch["label"].shape[0]
        probs_oct = torch.zeros(b, self.num_classes, device=device)
        probs_fun = torch.zeros(b, self.num_classes, device=device)

        has_oct = batch["has_oct"] > 0
        has_fun = batch["has_fundus"] > 0

        if has_oct.any():
            idx = torch.where(has_oct)[0]
            logits = self.oct_model(batch["oct"][idx])
            probs_oct[idx] = torch.softmax(logits, dim=1)

        if has_fun.any():
            idx = torch.where(has_fun)[0]
            logits = self.fundus_model(batch["ir"][idx], batch["faf"][idx])
            probs_fun[idx] = torch.softmax(logits, dim=1)

        w_oct = self.w_oct * batch["has_oct"]
        w_fun = self.w_fundus * batch["has_fundus"]
        denom = (w_oct + w_fun).clamp_min(1e-6).unsqueeze(1)
        probs = (w_oct.unsqueeze(1) * probs_oct + w_fun.unsqueeze(1) * probs_fun) / denom

        both_missing = ((batch["has_oct"] + batch["has_fundus"]) == 0).unsqueeze(1)
        probs = torch.where(both_missing, torch.full_like(probs, 1.0 / self.num_classes), probs)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return {"probs": probs, "p_oct": probs_oct, "p_fundus": probs_fun}


class MultimodalLateFusionModel(nn.Module):
    def __init__(self, oct_model: OCTWrapper, fundus_model: FundusWrapper, num_classes: int, d_model: int = 512, fundus_size: int = 256, head_dropout: float = 0.2):
        super().__init__()
        self.oct_model = oct_model
        self.fundus_model = fundus_model
        self.num_classes = int(num_classes)

        oct_dim = oct_model.embedding_dim
        fundus_dim = fundus_model.infer_embedding_dim(next(oct_model.parameters()).device, fundus_size=fundus_size)
        self.oct_proj = nn.Linear(oct_dim, d_model)
        self.fundus_proj = nn.Linear(fundus_dim, d_model)

        self.gate = nn.Sequential(
            nn.Linear(2 * d_model + 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(d_model, num_classes),
        )

    def extract_oct_embedding(self, oct_x: torch.Tensor) -> torch.Tensor:
        return self.oct_model.extract_embedding(oct_x)

    def extract_fundus_embedding(self, ir_x: torch.Tensor, faf_x: torch.Tensor) -> torch.Tensor:
        return self.fundus_model.extract_embedding(ir_x, faf_x)

    def apply_modality_dropout(self, m_oct: torch.Tensor, m_fundus: torch.Tensor, p_oct: float, p_fundus: float):
        if p_oct <= 0 and p_fundus <= 0:
            return m_oct, m_fundus
        m_oct_new = m_oct.clone()
        m_fun_new = m_fundus.clone()
        both = (m_oct > 0) & (m_fundus > 0)
        if both.any():
            drop_oct = (torch.rand_like(m_oct) < p_oct) & both
            drop_fun = (torch.rand_like(m_fundus) < p_fundus) & both
            collide = drop_oct & drop_fun
            drop_fun[collide] = False  # ne jamais masquer les 2 modalités ensemble
            m_oct_new[drop_oct] = 0.0
            m_fun_new[drop_fun] = 0.0
        return m_oct_new, m_fun_new

    def forward(self, batch: Dict[str, torch.Tensor], modality_dropout: Tuple[float, float] = (0.0, 0.0)) -> Dict[str, torch.Tensor]:
        z_oct_raw = self.extract_oct_embedding(batch["oct"])
        z_fundus_raw = self.extract_fundus_embedding(batch["ir"], batch["faf"])

        z_oct = self.oct_proj(z_oct_raw)
        z_fundus = self.fundus_proj(z_fundus_raw)

        m_oct = batch["has_oct"].float()
        m_fundus = batch["has_fundus"].float()
        if self.training:
            m_oct, m_fundus = self.apply_modality_dropout(m_oct, m_fundus, *modality_dropout)

        gate_in = torch.cat([z_oct, z_fundus, m_oct.unsqueeze(1), m_fundus.unsqueeze(1)], dim=1)
        gate_logits = self.gate(gate_in)

        minus_inf = torch.full_like(gate_logits, -1e9)
        valid = torch.stack([m_oct > 0, m_fundus > 0], dim=1)
        gate_logits = torch.where(valid, gate_logits, minus_inf)
        alpha = torch.softmax(gate_logits, dim=1)

        fused = alpha[:, 0:1] * z_oct + alpha[:, 1:2] * z_fundus
        logits = self.head(fused)

        logits_oct = self.oct_model(batch["oct"])
        logits_fundus = self.fundus_model(batch["ir"], batch["faf"])
        return {
            "logits": logits,
            "alpha": alpha,
            "z_oct": z_oct,
            "z_fundus": z_fundus,
            "logits_oct": logits_oct,
            "logits_fundus": logits_fundus,
        }


# -----------------------------
# Train / Eval
# -----------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray], n_classes: int) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    base = metrics_from_cm(cm)
    out = {
        "acc": base["acc"],
        "bacc": base["bacc"],
        "macro_f1": base["macro_f1"],
        "per_class_recall": base["per_class_recall"],
        "cm": cm.tolist(),
    }
    if y_prob is not None and roc_auc_score is not None:
        try:
            out["auroc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            out["auroc_ovr"] = None
    return out


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


@torch.no_grad()
def evaluate_prob_fusion(model: ProbabilityFusionModel, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for batch in tqdm(loader, desc="Eval prob fusion"):
        batch = move_batch(batch, device)
        out = model(batch)
        probs = out["probs"]
        y_true.extend(batch["label"].cpu().numpy().tolist())
        y_pred.extend(probs.argmax(1).cpu().numpy().tolist())
        y_prob.append(probs.cpu().numpy())
    y_prob_np = np.concatenate(y_prob, axis=0) if y_prob else None
    return compute_metrics(np.array(y_true), np.array(y_pred), y_prob_np, model.num_classes)


def build_optimizer_for_late_fusion(model: MultimodalLateFusionModel, lr_head: float, lr_oct: float, lr_fun: float, wd: float):
    groups = [
        {
            "params": list(model.head.parameters())
            + list(model.gate.parameters())
            + list(model.oct_proj.parameters())
            + list(model.fundus_proj.parameters()),
            "lr": lr_head,
        },
        {"params": [p for p in model.oct_model.parameters() if p.requires_grad], "lr": lr_oct},
        {"params": [p for p in model.fundus_model.parameters() if p.requires_grad], "lr": lr_fun},
    ]
    return torch.optim.AdamW([g for g in groups if len(g["params"]) > 0], weight_decay=wd)


def _set_oct_requires_grad(model: OCTWrapper, requires_grad: bool) -> None:
    for p in model.parameters():
        p.requires_grad = requires_grad


def _unfreeze_last_oct_stages(model: OCTWrapper, n_last_stages: int) -> None:
    """
    Structure réelle dans oct_volume_search.py:
      model.model.encoder = Sequential(base.features, base.avgpool)
      base.features est l'index 0.
    On garde un accès robuste en validant la présence de features séquentielles.
    """
    if n_last_stages <= 0:
        return
    enc = getattr(model.model, "encoder", None)
    if not isinstance(enc, nn.Sequential) or len(enc) == 0:
        print("[warn] OCT unfreeze: encoder non trouvé, skip")
        return
    features = enc[0]
    if not isinstance(features, nn.Sequential):
        print("[warn] OCT unfreeze: features non séquentielles, skip")
        return
    n = min(n_last_stages, len(features))
    for stage in features[-n:]:
        for p in stage.parameters():
            p.requires_grad = True


def set_backbones_trainable(model: MultimodalLateFusionModel, freeze_backbones: bool, unfreeze_last_oct_stages: int, unfreeze_last_fundus_blocks: int):
    _set_oct_requires_grad(model.oct_model, not freeze_backbones)
    for p in model.fundus_model.parameters():
        p.requires_grad = not freeze_backbones

    if freeze_backbones:
        _set_oct_requires_grad(model.oct_model, False)
        for p in model.fundus_model.parameters():
            p.requires_grad = False

    _unfreeze_last_oct_stages(model.oct_model, unfreeze_last_oct_stages)

    if unfreeze_last_fundus_blocks > 0:
        unfreeze_last_blocks(model.fundus_model.model.ir.backbone, unfreeze_last_fundus_blocks)
        unfreeze_last_blocks(model.fundus_model.model.faf.backbone, unfreeze_last_fundus_blocks)
        for p in model.fundus_model.model.ir.proj.parameters():
            p.requires_grad = True
        for p in model.fundus_model.model.faf.proj.parameters():
            p.requires_grad = True


def train_late_fusion(args, model: MultimodalLateFusionModel, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, class_weights: torch.Tensor, out_dir: Path, class_names: List[str]):
    ce = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))
    set_backbones_trainable(model, args.freeze_backbones, args.unfreeze_last_oct_stages, args.unfreeze_last_fundus_blocks)
    opt = build_optimizer_for_late_fusion(model, args.lr_head, args.lr_oct_backbone, args.lr_fundus_backbone, args.weight_decay)

    best_bacc = -1.0
    history = []
    patience = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        all_y, all_p = [], []

        for batch in tqdm(train_loader, desc=f"Train epoch {epoch}"):
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
                out = model(batch, modality_dropout=(args.modality_dropout_oct, args.modality_dropout_fundus))
                loss = ce(out["logits"], batch["label"])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = batch["label"].size(0)
            total_loss += float(loss.item()) * bs
            total += bs
            all_y.append(batch["label"].detach().cpu().numpy())
            all_p.append(out["logits"].argmax(1).detach().cpu().numpy())

        tr_m = compute_metrics(np.concatenate(all_y), np.concatenate(all_p), None, args.num_classes)
        val_m = evaluate_late_fusion(model, val_loader, device, args)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total, 1),
            "train_bacc": tr_m["bacc"],
            "val_bacc": val_m["bacc"],
            "val_macro_f1": val_m["macro_f1"],
        }
        history.append(row)
        print(row)

        if val_m["bacc"] > best_bacc:
            best_bacc = val_m["bacc"]
            patience = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "val": val_m}, out_dir / "best_multimodal.pt")
            confusion_matrix_figure(np.array(val_m["cm"]), out_dir / "best_val_confusion_matrix.png", labels=class_names)
        else:
            patience += 1

        if patience >= args.early_stopping_patience:
            print(f"Early stopping @ epoch {epoch}")
            break

    save_json({"history": history, "best_bacc": best_bacc}, out_dir / "metrics_history.json")


@torch.no_grad()
def evaluate_late_fusion(model: MultimodalLateFusionModel, loader: DataLoader, device: torch.device, args, force_mask: Optional[str] = None):
    model.eval()
    y_true, y_pred, y_prob, pred_rows = [], [], [], []

    for batch in tqdm(loader, desc="Eval late fusion"):
        batch = move_batch(batch, device)
        if force_mask == "no_oct":
            batch["has_oct"] = torch.zeros_like(batch["has_oct"])
        if force_mask == "no_fundus":
            batch["has_fundus"] = torch.zeros_like(batch["has_fundus"])

        out = model(batch)
        probs = torch.softmax(out["logits"], dim=1)
        pred = probs.argmax(1)

        y_true.extend(batch["label"].cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
        y_prob.append(probs.cpu().numpy())

        for i, cid in enumerate(batch["case_id"]):
            pred_rows.append(
                {
                    "case_id": cid,
                    "y_true": int(batch["label"][i].item()),
                    "y_pred": int(pred[i].item()),
                    "has_oct": float(batch["has_oct"][i].item()),
                    "has_fundus": float(batch["has_fundus"][i].item()),
                    "alpha_oct": float(out["alpha"][i, 0].item()),
                    "alpha_fundus": float(out["alpha"][i, 1].item()),
                }
            )

    y_prob_np = np.concatenate(y_prob, axis=0) if y_prob else None
    m = compute_metrics(np.array(y_true), np.array(y_pred), y_prob_np, args.num_classes)
    return {**m, "pred_rows": pred_rows}


def run_modality_ablation(model: MultimodalLateFusionModel, loader: DataLoader, device: torch.device, args, out_dir: Path):
    full = evaluate_late_fusion(model, loader, device, args, force_mask=None)
    no_oct = evaluate_late_fusion(model, loader, device, args, force_mask="no_oct")
    no_fundus = evaluate_late_fusion(model, loader, device, args, force_mask="no_fundus")
    result = {
        "full": {"bacc": full["bacc"], "macro_f1": full["macro_f1"]},
        "no_oct": {"bacc": no_oct["bacc"], "macro_f1": no_oct["macro_f1"]},
        "no_fundus": {"bacc": no_fundus["bacc"], "macro_f1": no_fundus["macro_f1"]},
        "delta_bacc_no_oct": full["bacc"] - no_oct["bacc"],
        "delta_bacc_no_fundus": full["bacc"] - no_fundus["bacc"],
        "delta_macro_f1_no_oct": full["macro_f1"] - no_oct["macro_f1"],
        "delta_macro_f1_no_fundus": full["macro_f1"] - no_fundus["macro_f1"],
    }
    save_json(result, out_dir / "ablation.json")
    return result


# -----------------------------
# XAI (attribution honnête)
# -----------------------------
def normalize_map(x: np.ndarray) -> np.ndarray:
    x = x - x.min()
    return x / (x.max() + 1e-8)


def _fundus_to_rgb_uint8(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(MultimodalDataset.IMAGENET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(MultimodalDataset.IMAGENET_STD, device=x.device).view(3, 1, 1)
    img = (x * std + mean).clamp(0, 1)
    img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    return img


def _oct_slice_to_gray_uint8(x_slice: torch.Tensor) -> np.ndarray:
    img = x_slice.mean(0)
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return (img.detach().cpu().numpy() * 255).astype(np.uint8)


def save_overlay(base_uint8: np.ndarray, heat: np.ndarray, out_path: Path, cmap: str = "jet") -> None:
    plt.figure(figsize=(5, 5))
    if base_uint8.ndim == 2:
        plt.imshow(base_uint8, cmap="gray")
    else:
        plt.imshow(base_uint8)
    plt.imshow(heat, cmap=cmap, alpha=0.4)
    plt.axis("off")
    plt.tight_layout(pad=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _prepare_single_sample(sample: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            out[k] = v.unsqueeze(0).to(device)
        else:
            out[k] = v
    return out


def generate_saliency_multimodal(model: MultimodalLateFusionModel, sample: Dict[str, torch.Tensor], out_dir: Path, class_idx: Optional[int] = None, topk_oct: int = 3):
    """
    Méthode d'attribution: gradient input-level (saliency), pas un vrai Grad-CAM.
    Raison: backbones ConvNeXt+ViT hétérogènes; on fournit une attribution honnête et stable.
    """
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    batch = _prepare_single_sample(sample, device)

    batch["oct"] = batch["oct"].requires_grad_(True)
    batch["ir"] = batch["ir"].requires_grad_(True)
    batch["faf"] = batch["faf"].requires_grad_(True)

    out = model(batch)
    probs = torch.softmax(out["logits"], dim=1)
    pred = int(probs.argmax(1).item()) if class_idx is None else int(class_idx)
    score = out["logits"][0, pred]

    model.zero_grad(set_to_none=True)
    score.backward()

    meta = {
        "pred_class": pred,
        "pred_proba": float(probs[0, pred].item()),
        "alpha_oct": float(out["alpha"][0, 0].item()),
        "alpha_fundus": float(out["alpha"][0, 1].item()),
        "has_oct": float(batch["has_oct"][0].item()),
        "has_fundus": float(batch["has_fundus"][0].item()),
        "method": "input_gradient_saliency",
        "interpretation_warning": "Gating non causal. Attribution spatiale exploratoire; confirmer avec ablation et occlusion.",
    }

    if batch["has_fundus"][0] > 0:
        sal_ir = normalize_map(batch["ir"].grad.detach().abs().mean(1)[0].cpu().numpy())
        sal_faf = normalize_map(batch["faf"].grad.detach().abs().mean(1)[0].cpu().numpy())
        plt.imsave(out_dir / "fundus_ir_saliency.png", sal_ir, cmap="jet")
        plt.imsave(out_dir / "fundus_faf_saliency.png", sal_faf, cmap="jet")

        ir_rgb = _fundus_to_rgb_uint8(batch["ir"][0].detach())
        faf_rgb = _fundus_to_rgb_uint8(batch["faf"][0].detach())
        save_overlay(ir_rgb, sal_ir, out_dir / "fundus_ir_overlay.png")
        save_overlay(faf_rgb, sal_faf, out_dir / "fundus_faf_overlay.png")

        # Indicateur non causal IR vs FAF, utile pour debug XAI interne.
        ir_norm = float(np.mean(np.abs(batch["ir"].grad.detach().cpu().numpy())))
        faf_norm = float(np.mean(np.abs(batch["faf"].grad.detach().cpu().numpy())))
        s = ir_norm + faf_norm + 1e-8
        meta["fundus_grad_norm_ir"] = ir_norm
        meta["fundus_grad_norm_faf"] = faf_norm
        meta["fundus_relative_ir"] = ir_norm / s
        meta["fundus_relative_faf"] = faf_norm / s

    if batch["has_oct"][0] > 0:
        sal_oct = batch["oct"].grad.detach().abs().mean(2)[0].cpu().numpy()  # [S,H,W]
        sal_oct = normalize_map(sal_oct)
        slice_scores = sal_oct.mean(axis=(1, 2))
        topk = np.argsort(-slice_scores)[: max(1, topk_oct)]
        meta["oct_slice_scores"] = slice_scores.tolist()
        meta["oct_topk_slices"] = [int(i) for i in topk]

        for rank, si in enumerate(topk.tolist(), start=1):
            hm = sal_oct[si]
            plt.imsave(out_dir / f"oct_slice{si:02d}_saliency.png", hm, cmap="jet")
            base = _oct_slice_to_gray_uint8(batch["oct"][0, si].detach())
            save_overlay(base, hm, out_dir / f"oct_slice{si:02d}_overlay.png")
            if rank == 1:
                meta["oct_best_slice"] = int(si)

    save_json(meta, out_dir / "xai_saliency_meta.json")


def generate_occlusion_multimodal(model: MultimodalLateFusionModel, sample: Dict[str, torch.Tensor], out_dir: Path, patch: int = 32):
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    batch = _prepare_single_sample(sample, device)

    def score(inp):
        with torch.no_grad():
            o = model(inp)
            p = torch.softmax(o["logits"], dim=1)
            c = int(p.argmax(1).item())
            return float(p[0, c].item()), c

    base_score, cls = score(batch)
    meta = {
        "base_score": base_score,
        "pred_class": cls,
        "interpretation_warning": "Occlusion est exploratoire; confirmer avec ablation et cohérence clinique.",
    }

    if batch["has_fundus"][0] > 0:
        for mod in ["ir", "faf"]:
            img = batch[mod].clone()
            _, _, h, w = img.shape
            heat = np.zeros((h, w), dtype=np.float32)
            for y in range(0, h, patch):
                for x in range(0, w, patch):
                    occ = img.clone()
                    occ[:, :, y : y + patch, x : x + patch] = 0.0
                    b2 = dict(batch)
                    b2[mod] = occ
                    s, _ = score(b2)
                    heat[y : y + patch, x : x + patch] = base_score - s
            heat = normalize_map(heat)
            plt.imsave(out_dir / f"fundus_{mod}_occlusion.png", heat, cmap="jet")
            save_overlay(_fundus_to_rgb_uint8(batch[mod][0].detach()), heat, out_dir / f"fundus_{mod}_occlusion_overlay.png")

    if batch["has_oct"][0] > 0:
        vol = batch["oct"].clone()  # [1,S,3,H,W]
        n_slices = vol.shape[1]
        slice_imp = []
        for si in range(n_slices):
            occ = vol.clone()
            occ[:, si] = 0.0
            b2 = dict(batch)
            b2["oct"] = occ
            sc, _ = score(b2)
            slice_imp.append(base_score - sc)
        meta["oct_slice_occlusion_importance"] = slice_imp

    save_json(meta, out_dir / "xai_occlusion_meta.json")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Multimodal OCT+Fundus training/eval/XAI")
    p.add_argument("--mode", choices=["prob_fusion", "late_fusion", "xai"], required=True)
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--oct-ckpt", type=str, default="")
    p.add_argument("--fundus-ckpt", type=str, default="")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--out-dir", type=str, default="runs_multimodal")
    p.add_argument("--strict-load", action="store_true", help="Active strict=True for checkpoint loading")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=1998)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--fundus-size", type=int, default=256)
    p.add_argument("--label-col", type=str, default="auto", help="Column name for label (e.g., sgene or gene)")

    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-oct-backbone", type=float, default=1e-5)
    p.add_argument("--lr-fundus-backbone", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--freeze-backbones", action="store_true")
    p.add_argument("--unfreeze-last-oct-stages", type=int, default=1)
    p.add_argument("--unfreeze-last-fundus-blocks", type=int, default=4)
    p.add_argument("--modality-dropout-oct", type=float, default=0.15)
    p.add_argument("--modality-dropout-fundus", type=float, default=0.15)
    p.add_argument("--early-stopping-patience", type=int, default=6)
    p.add_argument("--amp", action="store_true")

    p.add_argument("--w-oct", type=float, default=0.5)
    p.add_argument("--w-fundus", type=float, default=0.5)

    p.add_argument("--case-id", type=str, default="")
    p.add_argument("--xai-method", choices=["saliency", "occlusion", "gradcam"], default="saliency")
    p.add_argument("--xai-topk-oct", type=int, default=3)
    p.add_argument("--xai-patch", type=int, default=32)
    return p.parse_args()


def build_loader(df: pd.DataFrame, classes: List[str], split: str, args, augment: bool, shuffle: bool, label_col: str) -> Tuple[MultimodalDataset, DataLoader]:
    ds = MultimodalDataset(df, classes, split=split, label_col=label_col, fundus_size=args.fundus_size, augment=augment)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=True)
    return ds, loader


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), out_dir / "config.json")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.manifest, dtype=str).fillna("")
    if args.label_col == "auto":
        if "sgene" in df.columns:
            label_col = "sgene"
        elif "gene" in df.columns:
            label_col = "gene"
        else:
            raise ValueError("No label column found. Expected 'sgene' or 'gene'.")
    else:
        label_col = args.label_col
        if label_col not in df.columns:
            raise ValueError(f"--label-col={label_col} not found in manifest columns: {list(df.columns)}")

    classes = sorted(df[label_col].unique().tolist())
    args.num_classes = args.num_classes or len(classes)

    oct_model = load_pretrained_oct_model(
        args.oct_ckpt,
        args.num_classes,
        device,
        pooling="meanmax",
        dropout=0.2,
        strict_load=args.strict_load,
    ) if args.oct_ckpt else None

    fundus_model = load_pretrained_fundus_model(
        args.fundus_ckpt,
        classes,
        device,
        strict_load=args.strict_load,
    ) if args.fundus_ckpt else None

    if args.mode == "prob_fusion":
        assert oct_model is not None and fundus_model is not None, "--oct-ckpt et --fundus-ckpt requis"
        _, loader = build_loader(df, classes, split=args.split, args=args, augment=False, shuffle=False, label_col=label_col)
        model = ProbabilityFusionModel(oct_model, fundus_model, num_classes=args.num_classes, w_oct=args.w_oct, w_fundus=args.w_fundus).to(device)
        metrics = evaluate_prob_fusion(model, loader, device)
        save_json(metrics, out_dir / f"metrics_prob_fusion_{args.split}.json")
        confusion_matrix_figure(np.array(metrics["cm"]), out_dir / f"confusion_prob_fusion_{args.split}.png", labels=classes)
        print(metrics)
        return

    if args.mode == "late_fusion":
        assert oct_model is not None and fundus_model is not None, "--oct-ckpt et --fundus-ckpt requis"
        model = MultimodalLateFusionModel(oct_model, fundus_model, num_classes=args.num_classes, fundus_size=args.fundus_size).to(device)

        _, train_loader = build_loader(df, classes, split="train", args=args, augment=True, shuffle=True, label_col=label_col)
        _, val_loader = build_loader(df, classes, split="val", args=args, augment=False, shuffle=False, label_col=label_col)

        train_df = df[df["split"] == "train"].copy()
        train_df = train_df.rename(columns={label_col: "gene"})
        class_weights = compute_soft_class_weights_from_df(train_df, classes, device)
        train_late_fusion(args, model, train_loader, val_loader, device, class_weights, out_dir, class_names=classes)

        best = torch.load(out_dir / "best_multimodal.pt", map_location=device)
        load_state_dict_report(model, best["model"], strict=False, model_name="MultimodalLateFusion")

        eval_res = evaluate_late_fusion(model, val_loader, device, args)
        save_json({k: v for k, v in eval_res.items() if k != "pred_rows"}, out_dir / "metrics_late_fusion_val.json")
        pd.DataFrame(eval_res["pred_rows"]).to_csv(out_dir / "predictions_val.csv", index=False)
        confusion_matrix_figure(np.array(eval_res["cm"]), out_dir / "confusion_late_fusion_val.png", labels=classes)

        ab = run_modality_ablation(model, val_loader, device, args, out_dir)
        print("Ablation:", ab)
        return

    if args.mode == "xai":
        assert args.checkpoint, "--checkpoint requis en mode xai"
        assert oct_model is not None and fundus_model is not None, "--oct-ckpt et --fundus-ckpt requis"
        model = MultimodalLateFusionModel(oct_model, fundus_model, num_classes=args.num_classes, fundus_size=args.fundus_size).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        load_state_dict_report(model, resolve_state_dict(ckpt), strict=False, model_name="MultimodalLateFusion")

        ds, _ = build_loader(df, classes, split=(args.split or "val"), args=args, augment=False, shuffle=False, label_col=label_col)
        sample = None
        for i in range(len(ds)):
            it = ds[i]
            if args.case_id and it["case_id"] == args.case_id:
                sample = it
                break
            if not args.case_id:
                sample = it
                break
        if sample is None:
            raise ValueError(f"case_id introuvable: {args.case_id}")

        xdir = out_dir / "xai" / str(sample["case_id"])
        # compat rétro: --xai-method gradcam redirigé honnêtement vers saliency
        if args.xai_method in {"saliency", "gradcam"}:
            generate_saliency_multimodal(model, sample, xdir, topk_oct=args.xai_topk_oct)
        else:
            generate_occlusion_multimodal(model, sample, xdir, patch=args.xai_patch)
        print(f"XAI saved to {xdir}")


if __name__ == "__main__":
    main()
