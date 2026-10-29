"""
Training Script for KisanMitra AI Disease Detection Models.

Supports MobileNetV3-Small (baseline) and EfficientNet-B0 with:
- Fixed deterministic random seed for strict reproducibility
- Class-weighted Cross-Entropy Loss to handle class imbalance
- Strong in-the-wild data augmentation
- AdamW optimizer with Cosine Annealing learning rate schedule
- Validation monitoring (Accuracy & Macro F1) and best checkpoint saving
- Sanity-check mode (50 images/class, 2 epochs) to verify code health before heavy runs
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    ALL_DISEASE_CLASSES,
    CLASS_TO_IDX,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    RANDOM_SEED,
)
from src.disease.dataset import (
    PlantDiseaseDataset,
    get_training_transforms,
    get_eval_transforms,
    compute_class_weights,
)
from src.disease.model import (
    get_disease_model,
    count_parameters,
    get_model_size_mb,
)


def set_seed(seed: int = RANDOM_SEED):
    """Sets deterministic random seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    """Resolves computational device (CUDA -> MPS -> CPU)."""
    if device_arg and device_arg.lower() != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets, _ in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate_val(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Runs evaluation on the validation split."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    total = len(all_targets)
    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = (
        float(np.mean(np.array(all_preds) == np.array(all_targets)))
        if total > 0
        else 0.0
    )
    val_f1 = (
        float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
        if total > 0
        else 0.0
    )

    return val_loss, val_acc, val_f1


def train_model(
    model_name: str = "mobilenet_v3_small",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    sanity_check: bool = False,
    device_name: str = "auto",
    seed: int = RANDOM_SEED,
    output_dir: Path = MODELS_DIR,
) -> Dict:
    """Full training pipeline with checkpointing and metric tracking."""
    set_seed(seed)
    device = resolve_device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"🚀 KISANMITRA AI — DISEASE MODEL TRAINING: {model_name.upper()}")
    print("=" * 80)
    print(f"Device:               {device}")
    print(f"Random Seed:          {seed}")
    print(f"Sanity Check Mode:    {sanity_check}")
    print(f"Planned Epochs:       {2 if sanity_check else epochs}")
    print(f"Batch Size:           {batch_size}")
    print(f"Learning Rate:        {lr}")
    print(f"Weight Decay:         {weight_decay}")

    # 1. Check manifests exist
    train_csv = DATA_PROCESSED_DIR / "disease_train.csv"
    val_csv = DATA_PROCESSED_DIR / "disease_val.csv"
    if not train_csv.exists() or not val_csv.exists():
        print(
            "\n❌ ERROR: Split manifests not found in data/processed/.\n"
            "👉 Please run: python3 src/disease/prepare_dataset.py"
        )
        sys.exit(1)

    # 2. Datasets & Loaders
    max_samples = 50 if sanity_check else None
    actual_epochs = 2 if sanity_check else epochs

    train_dataset = PlantDiseaseDataset(
        manifest_csv=train_csv,
        transform=get_training_transforms(),
        max_samples_per_class=max_samples,
    )
    val_dataset = PlantDiseaseDataset(
        manifest_csv=val_csv,
        transform=get_eval_transforms(),
        max_samples_per_class=max_samples,
    )

    print(f"\n📦 Loaded Dataset:")
    print(f"  • Train samples:      {len(train_dataset)}")
    print(f"  • Val samples:        {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 0 is safe for macOS and Colab CPU fallback
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Model Architecture
    num_classes = len(ALL_DISEASE_CLASSES)
    model = get_disease_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    print(f"\n🧠 Model Summary:")
    print(f"  • Total Parameters:   {total_params:,}")
    print(f"  • Trainable Params:   {trainable_params:,}")
    print(f"  • Model Size:         {size_mb:.2f} MB")

    # 4. Class-Weighted Loss & Optimizer
    class_weights = compute_class_weights(train_csv, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=actual_epochs,
        eta_min=1e-6,
    )

    # 5. Training Loop
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_val_f1 = 0.0
    best_checkpoint_path = output_dir / f"{model_name}_best.pt"
    start_train_time = time.time()

    print("\n" + "-" * 80)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<11} | {'Val Loss':<10} | {'Val Acc':<10} | {'Val F1':<8} | {'LR':<9}")
    print("-" * 80)

    for epoch in range(1, actual_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1 = evaluate_val(
            model, val_loader, criterion, device
        )
        curr_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"{epoch:>2}/{actual_epochs:<4} | "
            f"{train_loss:>10.4f}  | "
            f"{train_acc * 100:>9.2f}% | "
            f"{val_loss:>8.4f}  | "
            f"{val_acc * 100:>8.2f}% | "
            f"{val_f1:>6.4f} | "
            f"{curr_lr:>8.2e}"
        )

        # Save best model checkpoint based on Validation Macro F1
        if val_f1 > best_val_f1 or epoch == 1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "val_f1": val_f1,
                    "val_acc": val_acc,
                    "class_to_idx": CLASS_TO_IDX,
                    "config": {
                        "lr": lr,
                        "batch_size": batch_size,
                        "epochs": actual_epochs,
                        "weight_decay": weight_decay,
                        "seed": seed,
                        "sanity_check": sanity_check,
                    },
                },
                best_checkpoint_path,
            )

    elapsed_time = time.time() - start_train_time
    print("-" * 80)
    print(f"✓ Training finished in {elapsed_time / 60:.2f} minutes.")
    print(f"✓ Best Validation Macro F1: {best_val_f1:.4f}")
    print(f"✓ Best checkpoint saved to: {best_checkpoint_path.resolve()}")

    # Save history json
    history_file = REPORTS_DIR / f"{model_name}_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "best_val_f1": best_val_f1,
                "history": history,
                "config": {
                    "lr": lr,
                    "batch_size": batch_size,
                    "epochs": actual_epochs,
                    "weight_decay": weight_decay,
                    "seed": seed,
                    "sanity_check": sanity_check,
                    "total_params": total_params,
                    "size_mb": size_mb,
                },
            },
            f,
            indent=2,
        )
    print(f"✓ Training history logged to: {history_file.resolve()}\n")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train CNN model on PlantVillage disease dataset")
    parser.add_argument(
        "--model-name",
        type=str,
        default="mobilenet_v3_small",
        choices=["mobilenet_v3_small", "efficientnet_b0", "mobilenet_v3_large"],
        help="Architecture to train",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run quick sanity check: at most 50 images per class for 2 epochs",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    args = parser.parse_args()

    train_model(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        sanity_check=args.sanity_check,
        device_name=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
