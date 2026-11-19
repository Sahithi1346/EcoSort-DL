"""
EcoSort AI - Model Evaluation & Comparison Pipeline
Evaluates Custom CNN, ResNet-18, and DenseNet-121 on the test dataset.
Calculates Accuracy, Precision, Recall, F1-score, Confusion Matrix, and Inference Time.
Generates evaluation graphs and comparison metrics.
"""

import os
import sys
import time
import json

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import matplotlib.pyplot as plt

# Optimize CPU thread allocation
if not torch.cuda.is_available() and hasattr(os, "cpu_count"):
    num_threads = min(8, os.cpu_count() or 4)
    torch.set_num_threads(num_threads)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, Any, List

from src.custom_cnn import build_custom_cnn
from src.resnet_model import build_resnet18
from src.densenet_model import build_densenet121
from src.dataset_loader import get_dataloaders, CLASSES

MODELS_DIR = "models"
PLOTS_DIR = os.path.join(MODELS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_model_for_eval(model_name: str, checkpoint_path: str, device: torch.device):
    name = model_name.lower().strip()
    if name in ["custom_cnn", "cnn"]:
        model = build_custom_cnn(num_classes=len(CLASSES))
    elif name in ["resnet18", "resnet"]:
        model = build_resnet18(num_classes=len(CLASSES), pretrained=False)
    elif name in ["densenet121", "densenet"]:
        model = build_densenet121(num_classes=len(CLASSES), pretrained=False)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_single_model(
    model_name: str,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
    print(f"\nEvaluating: {model_name.upper()} on Test Set ({len(test_loader.dataset)} samples)...")

    model = load_model_for_eval(model_name, checkpoint_path, device)

    all_preds = []
    all_targets = []
    all_probs = []

    # Measure inference time
    t0 = time.time()
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    total_time = time.time() - t0
    total_samples = len(all_targets)
    avg_inference_time_ms = (total_time / max(1, total_samples)) * 1000.0

    # Compute classification metrics
    acc = accuracy_score(all_targets, all_preds) * 100.0
    prec_macro = precision_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    prec_weighted = precision_score(all_targets, all_preds, average="weighted", zero_division=0) * 100.0
    rec_macro = recall_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    rec_weighted = recall_score(all_targets, all_preds, average="weighted", zero_division=0) * 100.0
    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    f1_weighted = f1_score(all_targets, all_preds, average="weighted", zero_division=0) * 100.0

    cm = confusion_matrix(all_targets, all_preds)
    class_rep = classification_report(all_targets, all_preds, target_names=CLASSES, output_dict=True, zero_division=0)

    print(f"Results for {model_name}:")
    print(f"  Accuracy: {acc:.2f}%")
    print(f"  F1 (Macro): {f1_macro:.2f}% | F1 (Weighted): {f1_weighted:.2f}%")
    print(f"  Precision (Macro): {prec_macro:.2f}% | Recall (Macro): {rec_macro:.2f}%")
    print(f"  Avg Inference Latency: {avg_inference_time_ms:.2f} ms/sample")

    return {
        "model_name": model_name,
        "test_samples": total_samples,
        "accuracy": round(acc, 2),
        "precision_macro": round(prec_macro, 2),
        "precision_weighted": round(prec_weighted, 2),
        "recall_macro": round(rec_macro, 2),
        "recall_weighted": round(rec_weighted, 2),
        "f1_macro": round(f1_macro, 2),
        "f1_weighted": round(f1_weighted, 2),
        "inference_time_ms": round(avg_inference_time_ms, 2),
        "confusion_matrix": cm.tolist(),
        "classification_report": class_rep,
    }


def plot_confusion_matrices(eval_results: Dict[str, Any]):
    """Plots and saves confusion matrix heatmaps for each evaluated model."""
    for model_name, metrics in eval_results.items():
        cm = np.array(metrics["confusion_matrix"])
        # Normalized matrix
        cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)

        fig, ax = plt.subplots(figsize=(7, 6))
        cax = ax.matshow(cm_norm, cmap=plt.cm.Greens, vmin=0, vmax=1)
        fig.colorbar(cax)

        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels([c.capitalize() for c in CLASSES], rotation=45, ha="left")
        ax.set_yticklabels([c.capitalize() for c in CLASSES])

        # Annotate cells with count and percentage
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                count = cm[i, j]
                pct = cm_norm[i, j] * 100
                text_color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center", color=text_color, fontsize=9)

        ax.set_title(f"Confusion Matrix – {model_name.upper()}", pad=20, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontweight="bold")
        ax.set_ylabel("True Label", fontweight="bold")
        plt.tight_layout()

        save_path = os.path.join(PLOTS_DIR, f"confusion_matrix_{model_name}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved confusion matrix: {save_path}")


def plot_metrics_comparison(eval_results: Dict[str, Any]):
    """Generates comparison bar charts across Accuracy, Precision, Recall, and F1."""
    models = list(eval_results.keys())
    labels = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5.5))

    accs = [eval_results[m]["accuracy"] for m in models]
    precs = [eval_results[m]["precision_macro"] for m in models]
    recs = [eval_results[m]["recall_macro"] for m in models]
    f1s = [eval_results[m]["f1_macro"] for m in models]

    r1 = ax.bar(x - 1.5 * width, accs, width, label="Accuracy", color="#1b4d3e")
    r2 = ax.bar(x - 0.5 * width, precs, width, label="Precision (Macro)", color="#2d6a4f")
    r3 = ax.bar(x + 0.5 * width, recs, width, label="Recall (Macro)", color="#52b788")
    r4 = ax.bar(x + 1.5 * width, f1s, width, label="F1-Score (Macro)", color="#74c69d")

    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_title("EcoSort AI - Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models], fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")

    # Value labels
    for rects in [r1, r2, r3, r4]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "model_metrics_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved metrics comparison plot: {save_path}")


def plot_training_histories():
    """Plots training and validation loss & accuracy curves for all trained models."""
    models = ["custom_cnn", "resnet18", "densenet121"]
    histories = {}
    for m in models:
        hist_file = os.path.join(MODELS_DIR, f"{m}_history.json")
        if os.path.exists(hist_file):
            with open(hist_file, "r") as f:
                histories[m] = json.load(f)

    if not histories:
        return

    # Loss Curves
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"custom_cnn": "#e63946", "resnet18": "#1d4ed8", "densenet121": "#2a9d8f"}
    for m, hist in histories.items():
        c = colors.get(m, "#333333")
        epochs = hist["epochs"]
        ax.plot(epochs, hist["train_loss"], label=f"{m.upper()} (Train)", color=c, linestyle="--", linewidth=1.8)
        ax.plot(epochs, hist["val_loss"], label=f"{m.upper()} (Val)", color=c, linestyle="-", linewidth=2.2)

    ax.set_title("Training & Validation Loss per Epoch", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Cross-Entropy Loss", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    plt.tight_layout()
    loss_path = os.path.join(PLOTS_DIR, "training_loss_curves.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()

    # Accuracy Curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, hist in histories.items():
        c = colors.get(m, "#333333")
        epochs = hist["epochs"]
        ax.plot(epochs, hist["train_acc"], label=f"{m.upper()} (Train)", color=c, linestyle="--", linewidth=1.8)
        ax.plot(epochs, hist["val_acc"], label=f"{m.upper()} (Val)", color=c, linestyle="-", linewidth=2.2)

    ax.set_title("Training & Validation Accuracy per Epoch", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    plt.tight_layout()
    acc_path = os.path.join(PLOTS_DIR, "training_acc_curves.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print("Saved training/validation loss and accuracy curves.")


def evaluate_all_models(max_test_samples: int = None) -> Dict[str, Any]:
    """Runs evaluation across Custom CNN, ResNet-18, and DenseNet-121."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"EcoSort AI - Evaluation Pipeline Starting on {device}")
    print(f"{'=' * 60}")

    _, _, test_loader, _ = get_dataloaders(batch_size=32, max_test_samples=max_test_samples)

    models_to_eval = ["custom_cnn", "resnet18", "densenet121"]
    results = {}

    for m in models_to_eval:
        ckpt = os.path.join(MODELS_DIR, f"{m}_best.pth")
        if os.path.exists(ckpt):
            results[m] = evaluate_single_model(m, test_loader, device)
        else:
            print(f"Skipping {m}: Checkpoint {ckpt} not found.")

    if not results:
        print("No model checkpoints found to evaluate!")
        return {}

    # Identify best performing model based on test accuracy / F1
    best_model_name = max(results.keys(), key=lambda k: results[k]["accuracy"])
    best_f1_model = max(results.keys(), key=lambda k: results[k]["f1_macro"])

    overall_results = {
        "best_accuracy_model": best_model_name,
        "best_f1_model": best_f1_model,
        "classes": CLASSES,
        "models": results,
    }

    # Save to JSON
    json_path = os.path.join(MODELS_DIR, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(overall_results, f, indent=4)
    print(f"\nSaved evaluation metrics to: {json_path}")

    # Generate plots
    plot_confusion_matrices(results)
    plot_metrics_comparison(results)
    plot_training_histories()

    print("\n" + "=" * 60)
    print("ECOSORT AI - MODEL COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Model':<16} | {'Accuracy':<10} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10} | {'Latency':<10}")
    print("-" * 75)
    for m, met in results.items():
        print(f"{m.upper():<16} | {met['accuracy']:>8.2f}% | {met['f1_macro']:>8.2f}% | {met['precision_macro']:>8.2f}% | {met['recall_macro']:>8.2f}% | {met['inference_time_ms']:>6.1f} ms")
    print("=" * 60)
    print(f"BEST-PERFORMING MODEL: {best_model_name.upper()} (Accuracy: {results[best_model_name]['accuracy']}%, F1: {results[best_model_name]['f1_macro']}%)")
    print("=" * 60)

    return overall_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate EcoSort AI models")
    parser.add_argument("--max_test_samples", type=int, default=None)
    args = parser.parse_args()
    evaluate_all_models(max_test_samples=args.max_test_samples)
