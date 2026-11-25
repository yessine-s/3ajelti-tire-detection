"""
Evaluation metrics and test set evaluation for Tyre Defect Detection.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import config


def compute_metrics(y_true, y_pred, y_probs=None):
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_probs (np.ndarray): Prediction probabilities for each class

    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}

    # Overall metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-class metrics
    precision_per_class = precision_score(
        y_true, y_pred, average=None, zero_division=0, labels=range(config.NUM_CLASSES)
    )
    recall_per_class = recall_score(
        y_true, y_pred, average=None, zero_division=0, labels=range(config.NUM_CLASSES)
    )
    f1_per_class = f1_score(
        y_true, y_pred, average=None, zero_division=0, labels=range(config.NUM_CLASSES)
    )

    metrics["per_class"] = {}
    for idx, name in enumerate(config.CLASS_NAMES):
        metrics["per_class"][name] = {
            "precision": float(precision_per_class[idx]),
            "recall": float(recall_per_class[idx]),
            "f1": float(f1_per_class[idx])
        }

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    return metrics


def evaluate_model(model, dataloader, criterion, device=None, return_predictions=False):
    """
    Evaluate model on a dataloader.

    Args:
        model (torch.nn.Module): Model to evaluate
        dataloader (DataLoader): Dataloader with batches
        criterion (torch.nn.Module): Loss function
        device (torch.device): Device to use
        return_predictions (bool): Whether to return predictions

    Returns:
        dict or tuple: Metrics dict, or (metrics, predictions, labels) if return_predictions=True
    """
    if device is None:
        device = config.DEVICE

    use_amp = bool(config.USE_AMP and torch.cuda.is_available() and device.type == "cuda")

    model.to(device)
    criterion = criterion.to(device)
    model.eval()

    all_losses = []
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass (AMP when enabled)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            # Store results
            all_losses.append(loss.item())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())

    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    avg_loss = np.mean(all_losses)

    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = float(avg_loss)

    if return_predictions:
        return metrics, all_preds, all_labels, all_probs
    else:
        return metrics


def plot_confusion_matrix(y_true, y_pred, save_path=None, title="Confusion Matrix"):
    """
    Plot confusion matrix.

    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        save_path (str or Path): Path to save figure
        title (str): Title for plot

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        cbar_kws={"label": "Count"},
        ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    return fig


def evaluate_on_test_set(model, test_loader, criterion, device=None):
    """
    Comprehensive evaluation on test set.
    Saves results to JSON and confusion matrix to PNG.

    Args:
        model (torch.nn.Module): Model to evaluate
        test_loader (DataLoader): Test dataloader
        criterion (torch.nn.Module): Loss function
        device (torch.device): Device to use

    Returns:
        dict: Test results
    """
    if device is None:
        device = config.DEVICE

    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)

    # Evaluate
    metrics, preds, labels, probs = evaluate_model(
        model, test_loader, criterion, device=device, return_predictions=True
    )

    # Print classification report
    print("\nClassification Report:")
    print("-" * 60)
    print(classification_report(
        labels, preds,
        target_names=config.CLASS_NAMES,
        digits=4
    ))

    # Print summary
    print("\nTest Set Summary:")
    print("-" * 60)
    print(f"Overall Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Weighted F1 Score:     {metrics['f1_weighted']:.4f}")
    print(f"Macro F1 Score:        {metrics['f1_macro']:.4f}")
    print(f"Average Loss:          {metrics['loss']:.4f}")

    print("\nPer-Class Metrics:")
    for name, scores in metrics["per_class"].items():
        print(f"\n  {name}:")
        print(f"    Precision: {scores['precision']:.4f}")
        print(f"    Recall:    {scores['recall']:.4f}")
        print(f"    F1 Score:  {scores['f1']:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(
        labels, preds,
        save_path=config.CONFUSION_MATRIX_FILE,
        title="Test Set Confusion Matrix"
    )

    # Save results to JSON
    results = {
        "overall_accuracy": float(metrics["accuracy"]),
        "weighted_f1": float(metrics["f1_weighted"]),
        "macro_f1": float(metrics["f1_macro"]),
        "average_loss": float(metrics["loss"]),
        "per_class_metrics": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"]
    }

    with open(config.TEST_RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nTest results saved to {config.TEST_RESULTS_FILE}")
    print("="*60 + "\n")

    return results
