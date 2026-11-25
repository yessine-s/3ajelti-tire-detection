"""
Hyperparameter tuning using Optuna.
Tunes learning rates, dropout, batch size, and rotation degrees.
Uses Phase 1 training for quick evaluation.
"""

import optuna
from optuna.trial import Trial
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import f1_score
import config
import model as model_module
import train as train_module
import dataset as dataset_module
import evaluate


def create_optuna_study():
    """
    Create Optuna study for hyperparameter tuning.

    Returns:
        optuna.study.Study: Study object
    """
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.SEED)
    )
    return study


def objective(trial: Trial, train_loader, val_loader):
    """
    Objective function for Optuna optimization.
    Trains only Phase 1 for quick evaluation.

    Args:
        trial (optuna.trial.Trial): Optuna trial object
        train_loader (DataLoader): Training dataloader
        val_loader (DataLoader): Validation dataloader

    Returns:
        float: Validation weighted F1 score
    """
    # Suggest hyperparameters
    lr_phase1 = trial.suggest_float("lr_phase1", 1e-4, 1e-2, log=True)
    lr_phase2 = trial.suggest_float("lr_phase2", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    rotation_degrees = trial.suggest_int("rotation_degrees", 10, 25)

    print(f"\n{'='*70}")
    print(f"Trial {trial.number + 1}")
    print(f"{'='*70}")
    print(f"Suggested hyperparameters:")
    print(f"  LR Phase 1: {lr_phase1:.2e}")
    print(f"  LR Phase 2: {lr_phase2:.2e}")
    print(f"  Dropout: {dropout:.4f}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Rotation Degrees: {rotation_degrees} degrees")
    print(f"{'='*70}")

    # If batch size is different, recreate dataloaders
    if batch_size != config.BATCH_SIZE:
        print(f"Recreating dataloaders with batch size {batch_size}...")
        loaders = dataset_module.get_dataloaders(
            batch_size=batch_size,
            rotation_degrees=rotation_degrees
        )
        train_loader = loaders["train"]
        val_loader = loaders["val"]

    # Create model with custom dropout
    model = model_module.create_model(dropout_rate=dropout)
    model.to(config.DEVICE)

    # Compute class weights
    class_weights = dataset_module.compute_class_weights(train_loader)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = criterion.to(config.DEVICE)

    # Freeze backbone
    model_module.freeze_backbone(model)

    # Setup optimizer and scheduler
    optimizer = train_module.setup_optimizer(model, lr_phase1)
    scheduler = train_module.setup_scheduler(optimizer, config.OPTUNA_TRIAL_EPOCHS)

    # Train Phase 1 for quick evaluation
    print(f"\nTraining Phase 1 for {config.OPTUNA_TRIAL_EPOCHS} epochs...")

    best_f1 = 0.0

    for epoch in range(config.OPTUNA_TRIAL_EPOCHS):
        # Train epoch
        train_metrics = train_module.train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, show_progress=False
        )

        # Validate
        val_metrics = evaluate.evaluate_model(model, val_loader, criterion, config.DEVICE)

        # Step scheduler
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        val_f1 = val_metrics["f1_weighted"]

        print(f"  Epoch {epoch + 1}/{config.OPTUNA_TRIAL_EPOCHS}: "
              f"Train Loss={train_metrics['loss']:.4f}, "
              f"Val F1={val_f1:.4f}")

        # Track best F1
        if val_f1 > best_f1:
            best_f1 = val_f1

        # Report intermediate value for pruning
        trial.report(val_f1, epoch)

        # Check for pruning
        if trial.should_prune():
            print(f"  Trial pruned.")
            raise optuna.TrialPruned()

    print(f"\nTrial {trial.number + 1} completed. Best Val F1: {best_f1:.4f}")

    return best_f1


def run_hyperparameter_tuning(train_loader, val_loader, n_trials=None):
    """
    Run hyperparameter tuning with Optuna.

    Args:
        train_loader (DataLoader): Training dataloader
        val_loader (DataLoader): Validation dataloader
        n_trials (int or None): Number of trials to run

    Returns:
        dict: Best hyperparameters found
    """
    if n_trials is None:
        n_trials = config.OPTUNA_TRIALS

    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING WITH OPTUNA")
    print("="*70)
    print(f"Running {n_trials} trials for Phase 1 only (quick evaluation)")
    print(f"Objective: Maximize validation weighted F1 score")
    print("="*70 + "\n")

    # Create study
    study = create_optuna_study()

    # Optimize
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader),
        n_trials=n_trials,
        show_progress_bar=False
    )

    # Get best trial
    best_trial = study.best_trial
    best_params = best_trial.params

    print("\n" + "="*70)
    print("OPTUNA TUNING COMPLETED")
    print("="*70)
    print(f"\nBest Trial: {best_trial.number + 1}")
    print(f"Best Validation F1: {best_trial.value:.4f}")
    print(f"\nBest Hyperparameters:")
    for param_name, param_value in best_params.items():
        if "lr" in param_name:
            print(f"  {param_name}: {param_value:.2e}")
        else:
            print(f"  {param_name}: {param_value}")
    print("="*70 + "\n")

    # Save best parameters
    save_best_params(best_params)

    return best_params


def save_best_params(best_params):
    """
    Save best hyperparameters to JSON file.

    Args:
        best_params (dict): Best parameters found by Optuna
    """
    json_params = {k: v for k, v in best_params.items()}

    with open(config.BEST_OPTUNA_PARAMS_FILE, "w") as f:
        json.dump(json_params, f, indent=4)

    print(f"Best parameters saved to {config.BEST_OPTUNA_PARAMS_FILE}")


def load_best_params(params_file=None):
    """
    Load best hyperparameters from JSON file.

    Args:
        params_file (str or Path or None): Path to params file

    Returns:
        dict: Hyperparameters
    """
    if params_file is None:
        params_file = config.BEST_OPTUNA_PARAMS_FILE

    params_file = Path(params_file)

    if not params_file.exists():
        print(f"Warning: Params file not found at {params_file}")
        return None

    with open(params_file, "r") as f:
        params = json.load(f)

    print(f"Loaded best parameters from {params_file}")
    return params
