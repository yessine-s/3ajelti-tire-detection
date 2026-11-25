"""
Training logic for 3-phase fine-tuning of EfficientNet-B2.
Includes early stopping, checkpoint saving, logging, and GPU monitoring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import config
import model as model_module
import evaluate
from dataset import compute_class_weights


def get_gpu_memory_info(device=None):
    """
    Get GPU memory usage statistics.

    Args:
        device (torch.device): Device to check

    Returns:
        dict: Memory statistics or None if CPU
    """
    if device is None:
        device = config.DEVICE

    if not torch.cuda.is_available() or device.type != "cuda":
        return None

    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    allocated = torch.cuda.memory_allocated(0) / 1e9

    return {
        "total_gb": total,
        "reserved_gb": reserved,
        "allocated_gb": allocated,
        "available_gb": total - allocated,
        "percent_used": (allocated / total) * 100
    }


class EarlyStopper:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=5, metric="loss", mode="min"):
        """
        Args:
            patience (int): Number of epochs with no improvement to stop
            metric (str): Metric to monitor ("loss", "accuracy", "f1_weighted")
            mode (str): "min" for loss, "max" for accuracy/F1
        """
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.best_epoch = None

    def __call__(self, value, epoch):
        """
        Check if training should stop.

        Args:
            value (float): Current metric value
            epoch (int): Current epoch number

        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
        else:
            # Check if improved
            if self.mode == "min":
                improved = value < self.best_value
            else:  # max
                improved = value > self.best_value

            if improved:
                self.best_value = value
                self.best_epoch = epoch
                self.counter = 0
            else:
                self.counter += 1

        should_stop = self.counter >= self.patience
        return should_stop


def train_epoch(model, train_loader, criterion, optimizer, device=None, show_progress=False,
                scaler=None, use_amp=False):
    """
    Train for one epoch.

    Args:
        model (torch.nn.Module): Model to train
        train_loader (DataLoader): Training dataloader
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to use
        show_progress (bool): Whether to show progress bar

    Returns:
        dict: Epoch metrics (loss, accuracy, f1_weighted)
    """
    if device is None:
        device = config.DEVICE

    model.to(device)
    model.train()

    all_losses = []
    all_preds = []
    all_labels = []

    # Use tqdm for progress
    # Show progress bar if requested or always on GPU (for monitoring)
    loader = tqdm(train_loader, desc="Training", disable=(not show_progress and not torch.cuda.is_available()))

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Forward + backward pass (AMP when enabled)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Store results
        all_losses.append(loss.item())
        all_preds.extend(logits.argmax(dim=1).cpu().detach().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    from sklearn.metrics import accuracy_score, f1_score

    epoch_metrics = {
        "loss": np.mean(all_losses),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    }

    return epoch_metrics


def train_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
                early_stopper, phase_name, num_epochs, checkpoint_path, device=None):
    """
    Train for a phase (Phase 1, 2, or 3).

    Args:
        model (torch.nn.Module): Model to train
        train_loader (DataLoader): Training dataloader
        val_loader (DataLoader): Validation dataloader
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler (torch.optim.lr_scheduler._LRScheduler): LR scheduler
        early_stopper (EarlyStopper): Early stopping object
        phase_name (str): Name of phase (e.g., "Phase 1")
        num_epochs (int): Number of epochs to train
        checkpoint_path (Path): Path to save best checkpoint
        device (torch.device): Device to use

    Returns:
        list: List of epoch logs (dicts)
    """
    if device is None:
        device = config.DEVICE

    use_amp = bool(config.USE_AMP and torch.cuda.is_available() and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"\n{'='*70}")
    print(f"{phase_name.upper()}")
    print(f"{'='*70}")
    model_module.print_model_summary(model, phase=phase_name)

    epoch_logs = []
    best_val_metric = None

    for epoch in range(num_epochs):
        # Train epoch
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            use_amp=use_amp,
        )

        # Validate
        val_metrics = evaluate.evaluate_model(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Log epoch
        log_dict = {
            "phase": phase_name,
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1_weighted": train_metrics["f1_weighted"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_weighted": val_metrics["f1_weighted"]
        }
        epoch_logs.append(log_dict)

        # Print epoch summary with clear formatting
        print(f"\n{phase_name} Epoch {epoch + 1}/{num_epochs}")
        print(f"  {'='*70}")
        print(f"  Learning Rate: {current_lr:.2e}")
        print(f"  {'='*70}")
        print(f"  TRAIN -> Loss: {train_metrics['loss']:.4f} | Accuracy: {train_metrics['accuracy']:.4f} | F1-Score: {train_metrics['f1_weighted']:.4f}")
        print(f"  VAL   -> Loss: {val_metrics['loss']:.4f} | Accuracy: {val_metrics['accuracy']:.4f} | F1-Score: {val_metrics['f1_weighted']:.4f}")
        print(f"  {'='*70}")

        # Show GPU memory usage if available
        mem_info = get_gpu_memory_info(device)
        if mem_info:
            print(f"  GPU Memory: {mem_info['allocated_gb']:.2f}/{mem_info['total_gb']:.2f} GB ({mem_info['percent_used']:.1f}%)")

        # Check early stopping metric
        early_stop_metric = val_metrics["f1_weighted"]

        # Check if best validation metric
        if best_val_metric is None or early_stop_metric > best_val_metric:
            best_val_metric = early_stop_metric
            # Save checkpoint
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  [Best checkpoint saved] (Val F1: {early_stop_metric:.4f})")

        # Check early stopping
        should_stop = early_stopper(early_stop_metric, epoch + 1)
        if should_stop:
            print(f"\n  [Early stopping triggered after {epoch + 1} epochs]")
            break

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()

    # Load best checkpoint
    print(f"\nLoading best checkpoint from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    return epoch_logs


def setup_optimizer(model, lr, optimizer_type="adam"):
    """
    Setup optimizer.

    Args:
        model (torch.nn.Module): Model to optimize
        lr (float): Learning rate
        optimizer_type (str): "adam" or "sgd"

    Returns:
        torch.optim.Optimizer: Optimizer
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_type.lower() == "adam":
        return optim.Adam(trainable_params, lr=lr)
    elif optimizer_type.lower() == "sgd":
        return optim.SGD(trainable_params, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def setup_scheduler(optimizer, num_epochs):
    """
    Setup LR scheduler (Cosine Annealing).

    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        num_epochs (int): Number of epochs

    Returns:
        torch.optim.lr_scheduler._LRScheduler: Scheduler
    """
    return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)


def train_full_pipeline(train_loader, val_loader, test_loader, test_dataset,
                       lr_phase1=None, lr_phase2=None, lr_phase3=None,
                       batch_size=None, rotation_degrees=None, dropout_rate=None,
                       device=None):
    """
    Complete 3-phase training pipeline.

    Args:
        train_loader (DataLoader): Training dataloader
        val_loader (DataLoader): Validation dataloader
        test_loader (DataLoader): Test dataloader
        test_dataset (Dataset): Test dataset
        lr_phase1, lr_phase2, lr_phase3 (float): Learning rates for phases
        batch_size (int): Batch size (for logging)
        rotation_degrees (int): Rotation degrees (for logging)
        dropout_rate (float): Dropout rate
        device (torch.device): Device to use

    Returns:
        tuple: (model, all_epoch_logs, criterion)
    """
    if device is None:
        device = config.DEVICE

    if lr_phase1 is None:
        lr_phase1 = config.PHASE_1_LR
    if lr_phase2 is None:
        lr_phase2 = config.PHASE_2_LR
    if lr_phase3 is None:
        lr_phase3 = config.PHASE_3_LR

    # Create model
    model = model_module.create_model(dropout_rate=dropout_rate)
    model.to(device)

    # Compute class weights
    class_weights = compute_class_weights(train_loader)

    # Loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = criterion.to(device)

    # Track all epoch logs
    all_epoch_logs = []

    # ========================================================================
    # PHASE 1: Freeze backbone, train head only
    # ========================================================================
    model_module.freeze_backbone(model)

    optimizer_phase1 = setup_optimizer(model, lr_phase1)
    scheduler_phase1 = setup_scheduler(optimizer_phase1, config.PHASE_1_EPOCHS)
    early_stopper_phase1 = EarlyStopper(patience=config.EARLY_STOPPING_PATIENCE, metric="f1", mode="max")

    logs_phase1 = train_phase(
        model, train_loader, val_loader, criterion, optimizer_phase1, scheduler_phase1,
        early_stopper_phase1, "Phase 1: Freeze Backbone", config.PHASE_1_EPOCHS,
        config.CHECKPOINT_PHASE_1, device
    )
    all_epoch_logs.extend(logs_phase1)

    # ========================================================================
    # PHASE 2: Unfreeze last 3 blocks, train them + head
    # ========================================================================
    model_module.unfreeze_last_n_blocks(model, n_blocks=3)

    optimizer_phase2 = setup_optimizer(model, lr_phase2)
    scheduler_phase2 = setup_scheduler(optimizer_phase2, config.PHASE_2_EPOCHS)
    early_stopper_phase2 = EarlyStopper(patience=config.EARLY_STOPPING_PATIENCE, metric="f1", mode="max")

    logs_phase2 = train_phase(
        model, train_loader, val_loader, criterion, optimizer_phase2, scheduler_phase2,
        early_stopper_phase2, "Phase 2: Unfreeze Last 3 Blocks", config.PHASE_2_EPOCHS,
        config.CHECKPOINT_PHASE_2, device
    )
    all_epoch_logs.extend(logs_phase2)

    # ========================================================================
    # PHASE 3: Unfreeze all layers, fine-tune entire model
    # ========================================================================
    model_module.unfreeze_all(model)

    optimizer_phase3 = setup_optimizer(model, lr_phase3)
    scheduler_phase3 = setup_scheduler(optimizer_phase3, config.PHASE_3_EPOCHS)
    early_stopper_phase3 = EarlyStopper(patience=config.EARLY_STOPPING_PATIENCE, metric="f1", mode="max")

    logs_phase3 = train_phase(
        model, train_loader, val_loader, criterion, optimizer_phase3, scheduler_phase3,
        early_stopper_phase3, "Phase 3: Unfreeze All Layers", config.PHASE_3_EPOCHS,
        config.CHECKPOINT_PHASE_3, device
    )
    all_epoch_logs.extend(logs_phase3)

    # Load best Phase 3 checkpoint
    print(f"\nLoading best Phase 3 checkpoint...")
    model.load_state_dict(torch.load(config.CHECKPOINT_PHASE_3, map_location=device))

    # Save training log to CSV
    logs_df = pd.DataFrame(all_epoch_logs)
    logs_df.to_csv(config.TRAINING_LOG_FILE, index=False)
    print(f"\nTraining log saved to {config.TRAINING_LOG_FILE}")

    return model, all_epoch_logs, criterion
