"""
EcoSort AI - Model Training Pipeline
Trains Custom CNN, ResNet-18, and DenseNet-121 using identical dataset splits,
cross-entropy loss with class weights, AdamW optimizer with weight decay,
learning rate scheduling, early stopping, and checkpoint management.
"""

import os
import sys
import time
import json
import copy
import argparse

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any

# Optimize CPU thread allocation
if not torch.cuda.is_available() and hasattr(os, "cpu_count"):
    num_threads = min(8, os.cpu_count() or 4)
    torch.set_num_threads(num_threads)

from src.custom_cnn import build_custom_cnn
from src.resnet_model import build_resnet18
from src.densenet_model import build_densenet121
from src.dataset_loader import get_dataloaders, load_class_weights, CLASSES

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def get_model(model_name: str, num_classes: int = 6) -> nn.Module:
    """Instantiates the specified model architecture."""
    name = model_name.lower().strip()
    if name in ["custom_cnn", "cnn"]:
        return build_custom_cnn(num_classes=num_classes)
    elif name in ["resnet18", "resnet"]:
        return build_resnet18(num_classes=num_classes, pretrained=True, freeze_backbone=True)
    elif name in ["densenet121", "densenet"]:
        return build_densenet121(num_classes=num_classes, pretrained=True, freeze_backbone=True)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose: custom_cnn, resnet18, densenet121")


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Runs one training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
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

    epoch_loss = running_loss / max(1, total)
    epoch_acc = (correct / max(1, total)) * 100.0
    return {"loss": epoch_loss, "acc": epoch_acc}


def validate_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Runs one validation epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += images.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = (correct / max(1, total)) * 100.0
    return {"loss": epoch_loss, "acc": epoch_acc}


def train_model(
    model_name: str,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    patience: int = 2,
    max_train_samples: int = None,
    max_val_samples: int = None,
) -> Dict[str, Any]:
    """
    Complete training lifecycle with AdamW, ReduceLROnPlateau, Early Stopping,
    checkpoint saving, and history logging.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"Starting Training: {model_name.upper()} on {device}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Initial LR: {lr}")
    print(f"{'=' * 60}")

    # Load dataloaders
    train_loader, val_loader, _, classes = get_dataloaders(
        batch_size=batch_size,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    # Model, Loss, Optimizer, Scheduler
    model = get_model(model_name, num_classes=len(classes)).to(device)
    class_weights = load_class_weights(device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-6)

    history = {
        "model_name": model_name,
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "best_val_acc": 0.0,
        "training_time_seconds": 0.0,
    }

    best_model_state = None
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        scheduler.step(val_metrics["loss"])
        dur = time.time() - t0

        history["epochs"].append(epoch)
        history["train_loss"].append(round(train_metrics["loss"], 4))
        history["train_acc"].append(round(train_metrics["acc"], 2))
        history["val_loss"].append(round(val_metrics["loss"], 4))
        history["val_acc"].append(round(val_metrics["acc"], 2))
        history["lr"].append(current_lr)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({dur:.1f}s) - "
            f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['acc']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']:.2f}% | "
            f"LR: {current_lr:.6f}",
            flush=True,
        )

        # Early stopping and best model tracking
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  --> Best checkpoint saved (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.2f}%)", flush=True)
        else:
            patience_counter += 1
            print(f"  --> No improvement. EarlyStopping patience: {patience_counter}/{patience}", flush=True)
            if patience_counter >= patience:
                print(f"\n[Early Stopping Triggered at Epoch {epoch}]", flush=True)
                break

    total_time = time.time() - start_time
    history["training_time_seconds"] = round(total_time, 2)
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = round(best_val_loss, 4)
    history["best_val_acc"] = round(best_val_acc, 2)

    # Save checkpoint
    checkpoint_filename = f"{model_name.lower()}_best.pth"
    checkpoint_path = os.path.join(MODELS_DIR, checkpoint_filename)
    torch.save(
        {
            "model_name": model_name,
            "state_dict": best_model_state if best_model_state is not None else model.state_dict(),
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "classes": classes,
        },
        checkpoint_path,
    )
    print(f"Checkpoint successfully saved to: {checkpoint_path}")

    # Save history JSON
    history_filename = f"{model_name.lower()}_history.json"
    history_path = os.path.join(MODELS_DIR, history_filename)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to: {history_path}")

    return history


def train_all_models(
    epochs: int = 5,
    batch_size: int = 32,
    max_train_samples: int = None,
    max_val_samples: int = None,
):
    """Sequentially trains Custom CNN, ResNet-18, and DenseNet-121."""
    models_to_train = ["custom_cnn", "resnet18", "densenet121"]
    all_histories = {}
    for m in models_to_train:
        lr = 1e-3 if m == "custom_cnn" else 3e-4
        hist = train_model(
            model_name=m,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
        )
        all_histories[m] = hist
    print("\nAll 3 models trained successfully!")
    return all_histories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EcoSort AI Model Training")
    parser.add_argument("--model", type=str, default="all", choices=["custom_cnn", "resnet18", "densenet121", "all"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    args = parser.parse_args()

    if args.model == "all":
        train_all_models(
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
        )
    else:
        train_model(
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
        )
