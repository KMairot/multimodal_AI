# ============================
# FUNDUS-ONLY TRAIN (IR+FAF) - 2 phases + 5 seeds
# Phase A: head-only (backbones gelés)
# Phase B: unfreeze partiel (N derniers blocks + norm) + LR discriminatif
# ============================

import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import timm
from PIL import Image
import torchvision.transforms as T
from torch.amp import GradScaler


# ----------------------------
# CONFIG
# ----------------------------
MANIFEST = r"C:\Users\kevin\Documents\GitHub\split_dataset\_tmp_build_ALL_GENES\export_manifest.csv"
GENES = ["EYS", "NR2E3", "PRPH2", "RHO", "RPGR", "USH2A"]

IR_SIZE = 256
FAF_SIZE = 256

BATCH_SIZE = 8          # 4 si OOM sur RTX 4050
NUM_WORKERS = 0         # mets 2 ou 4 si ça marche chez toi
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Run multiple seeds
SEEDS = [0, 1, 2, 3, 4]

# Phase A (head-only)
EPOCHS_A = 5
LR_HEAD_A = 1e-3

# Phase B (unfreeze partiel)
EPOCHS_B = 25
UNFREEZE_LAST_BLOCKS = 4
LR_BACKBONE_B = 5e-5
LR_HEAD_B = 5e-4

WEIGHT_DECAY = 0.05
DROP_PATH = 0.05
HEAD_DROPOUT = 0.1

USE_AMP = True  # tu peux mettre False pour debug

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


# ----------------------------
# SEED
# ----------------------------
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# SAFE image loader (PIL)
# ----------------------------
def load_pil_gray_as_rgb(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    # convert("RGB") = duplique proprement le canal grayscale en 3 canaux
    return Image.open(p).convert("RGB")


# ----------------------------
# Transforms
# ----------------------------
train_tf = T.Compose([
    T.RandomResizedCrop(IR_SIZE, scale=(0.85, 1.00), ratio=(0.95, 1.05)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=5),
    T.ColorJitter(brightness=0.10, contrast=0.15),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

val_tf = T.Compose([
    T.Resize((IR_SIZE, IR_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


# ----------------------------
# Dataset
# ----------------------------
class FundusDataset(Dataset):
    def __init__(self, df_subset: pd.DataFrame, genes, transform):
        self.df = df_subset.reset_index(drop=True)
        self.genes = list(genes)
        self.gene2idx = {g: i for i, g in enumerate(self.genes)}
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        y = self.gene2idx[r["gene"]]

        ir_img = load_pil_gray_as_rgb(r["ir_path"])
        faf_img = load_pil_gray_as_rgb(r["faf_path"])

        ir = self.transform(ir_img)
        faf = self.transform(faf_img)

        return {"ir": ir, "faf": faf, "y": torch.tensor(y, dtype=torch.long)}


# ----------------------------
# Model
# ----------------------------
class DinoBackbone(nn.Module):
    def __init__(self, model_name="vit_small_patch16_dinov3", out_dim=384, pretrained=True,
                 drop_path_rate=0.0):
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

    def forward(self, x):
        z = self.backbone(x)
        return self.proj(z)


class FundusOnlyNet(nn.Module):
    def __init__(self, genes, d=384, dinov3_name="vit_small_patch16_dinov3",
                 drop_path_rate=0.0, head_dropout=0.0):
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

    def forward(self, ir, faf):
        z1 = self.ir(ir)
        z2 = self.faf(faf)
        z = torch.cat([z1, z2], dim=1)
        return self.head(z)


# ----------------------------
# Freeze / Unfreeze helpers
# ----------------------------
def freeze_backbones(model: FundusOnlyNet):
    for p in model.ir.parameters():  p.requires_grad = False
    for p in model.faf.parameters(): p.requires_grad = False
    for p in model.head.parameters(): p.requires_grad = True


def unfreeze_last_blocks(vit_model: nn.Module, n_last_blocks=2):
    """
    vit_model = model.ir.backbone (ou model.faf.backbone)
    Freeze tout, puis unfreeze:
      - norm final
      - n derniers blocks
    """
    for p in vit_model.parameters():
        p.requires_grad = False

    if hasattr(vit_model, "norm"):
        for p in vit_model.norm.parameters():
            p.requires_grad = True

    if hasattr(vit_model, "blocks"):
        for blk in vit_model.blocks[-n_last_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True


def set_phaseB_trainable(model: FundusOnlyNet, n_last_blocks=2):
    for p in model.head.parameters():
        p.requires_grad = True

    unfreeze_last_blocks(model.ir.backbone,  n_last_blocks=n_last_blocks)
    unfreeze_last_blocks(model.faf.backbone, n_last_blocks=n_last_blocks)

    # proj doit être entraînable
    for p in model.ir.proj.parameters():  p.requires_grad = True
    for p in model.faf.proj.parameters(): p.requires_grad = True


# ----------------------------
# Metrics
# ----------------------------
def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def safe_row_normalize(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return cm / row_sums


@torch.no_grad()
def eval_metrics(model, loader):
    model.eval()
    all_y, all_p = [], []

    for batch in loader:
        ir = batch["ir"].to(DEVICE)
        faf = batch["faf"].to(DEVICE)
        y = batch["y"].to(DEVICE)

        logits = model(ir, faf)
        p = logits.argmax(1)

        all_y.append(y.cpu())
        all_p.append(p.cpu())

    model.train()

    y = torch.cat(all_y).numpy()
    p = torch.cat(all_p).numpy()

    acc = float((p == y).mean())
    n = len(GENES)
    cm = confusion_matrix_np(y, p, n_classes=n)

    recalls = []
    for c in range(n):
        denom = cm[c].sum()
        if denom == 0:
            continue
        recalls.append(cm[c, c] / denom)
    bacc = float(np.mean(recalls)) if len(recalls) else 0.0

    support = cm.sum(axis=1)
    recall_per_class = np.divide(np.diag(cm), np.maximum(support, 1))
    precision_per_class = np.divide(np.diag(cm), np.maximum(cm.sum(axis=0), 1))

    return acc, bacc, cm, recall_per_class, precision_per_class, support


def run_one_epoch(model, loader, opt, ce, scaler=None):
    model.train()
    total_loss = 0.0
    total = 0

    for batch in loader:
        ir = batch["ir"].to(DEVICE, non_blocking=True)
        faf = batch["faf"].to(DEVICE, non_blocking=True)
        y  = batch["y"].to(DEVICE, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        if USE_AMP and DEVICE == "cuda":
            with torch.autocast(device_type="cuda"):
                logits = model(ir, faf)
                loss = ce(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(ir, faf)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total += bs

    return total_loss / max(total, 1)


# ----------------------------
# Data prep (once)
# ----------------------------
df = pd.read_csv(MANIFEST, dtype=str).fillna("")

def filter_fundus(df_):
    return df_[(df_["has_ir"] == "1") & (df_["has_faf"] == "1")].reset_index(drop=True)

df_train = filter_fundus(df[df["split"] == "train"].copy())
df_val   = filter_fundus(df[df["split"] == "val"].copy())

print("Train fundus(IF):", len(df_train))
print("Val   fundus(IF):", len(df_val))
print("Train gene counts:\n", df_train["gene"].value_counts())
print("Val gene counts:\n", df_val["gene"].value_counts())

# class weights (doux)
counts = df_train["gene"].value_counts().reindex(GENES).fillna(0).values.astype(np.float32)
weights = counts.sum() / np.maximum(counts, 1.0)
weights = weights / weights.mean()
weights = np.sqrt(weights)
w = torch.tensor(weights, dtype=torch.float32, device=DEVICE)


# ----------------------------
# One run (one seed)
# ----------------------------
def train_one_seed(seed: int) -> float:
    seed_everything(seed)
    torch.backends.cudnn.benchmark = True

    train_ds = FundusDataset(df_train, GENES, transform=train_tf)
    val_ds   = FundusDataset(df_val,   GENES, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = FundusOnlyNet(
        GENES,
        d=384,
        dinov3_name="vit_small_patch16_dinov3",
        drop_path_rate=DROP_PATH,
        head_dropout=HEAD_DROPOUT,
    ).to(DEVICE)

    ce = nn.CrossEntropyLoss(weight=w)
    scaler = GradScaler("cuda") if (USE_AMP and DEVICE == "cuda") else None

    # ============================
    # PHASE A: head-only
    # ============================
    print("\n=== PHASE A: head-only ===")
    freeze_backbones(model)

    optA = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR_HEAD_A,
        weight_decay=WEIGHT_DECAY,
    )

    for ep in range(1, EPOCHS_A + 1):
        tr_loss = run_one_epoch(model, train_loader, optA, ce, scaler=scaler)
        val_acc, val_bacc, *_ = eval_metrics(model, val_loader)
        print(f"[A ep {ep:02d}/{EPOCHS_A}] tr_loss={tr_loss:.4f} | val_acc={val_acc:.3f} | val_bacc={val_bacc:.3f}")

    # ============================
    # PHASE B: unfreeze partial
    # ============================
    print("\n=== PHASE B: unfreeze partial ===")
    set_phaseB_trainable(model, n_last_blocks=UNFREEZE_LAST_BLOCKS)

    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("head."):
            head_params.append(p)
        else:
            backbone_params.append(p)

    optB = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE_B},
            {"params": head_params,     "lr": LR_HEAD_B},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    best = 0.0
    best_path = f"best_fundus_only_seed{seed}.pt"

    for ep in range(1, EPOCHS_B + 1):
        tr_loss = run_one_epoch(model, train_loader, optB, ce, scaler=scaler)
        val_acc, val_bacc, cm, rec, prec, sup = eval_metrics(model, val_loader)

        improved = val_bacc > best
        if improved:
            best = val_bacc
            torch.save(
                {"model": model.state_dict(), "epoch": ep, "val_bacc": val_bacc, "seed": seed},
                best_path
            )

        print(f"[B ep {ep:02d}/{EPOCHS_B}] tr_loss={tr_loss:.4f} | val_acc={val_acc:.3f} | val_bacc={val_bacc:.3f} | best_bacc={best:.3f}")

        if improved:
            labels = GENES
            df_cm = pd.DataFrame(
                cm,
                index=[f"true_{g}" for g in labels],
                columns=[f"pred_{g}" for g in labels],
            )
            df_cm_norm = pd.DataFrame(
                safe_row_normalize(cm),
                index=[f"true_{g}" for g in labels],
                columns=[f"pred_{g}" for g in labels],
            )

            print("\n=== Confusion matrix (counts) @ best ===")
            print(df_cm)

            print("\n=== Confusion matrix (row-normalized) @ best ===")
            print(df_cm_norm.round(3))

            print("\n=== Per-class metrics @ best ===")
            for i, g in enumerate(labels):
                print(f"{g:6s} | support={int(sup[i]):3d} | recall={rec[i]:.3f} | precision={prec[i]:.3f}")

    print(f"\nSeed {seed} DONE. Best bacc={best:.3f}. Saved: {best_path}")
    return best


# ----------------------------
# Main: run 5 seeds + summary
# ----------------------------
if __name__ == "__main__":
    results = []
    for seed in SEEDS:
        print(f"\n==================== SEED {seed} ====================")
        best_bacc = train_one_seed(seed)
        results.append(best_bacc)

    results = np.array(results, dtype=np.float32)
    print("\n==================== SUMMARY (5 seeds) ====================")
    print("best_bacc per seed:", [float(x) for x in results])
    print(f"mean={results.mean():.3f} | std={results.std():.3f}")