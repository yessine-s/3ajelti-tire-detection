"""
Main orchestration script for the complete Tyre Defect Detection pipeline.
Handles data loading, hyperparameter tuning, training, and evaluation.
"""

import torch
import numpy as np
import random
import config
import dataset as dataset_module
import train as train_module
import evaluate
import inference


def set_seed(seed=None):
    """Set random seed for reproducibility."""
    if seed is None:
        seed = config.SEED

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Don't force deterministic on GPU - it's slower and not necessary for training
    # Config.py already sets benchmark=True for speed
    # Only set deterministic if not using GPU
    if not torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # If GPU is available, let config.py control the settings (benchmark=True for speed)

    print(f"Random seed set to {seed}")


def main(use_tuning=True, n_optuna_trials=None):
    """
    Main pipeline execution.

    Args:
        use_tuning (bool): Whether to run Optuna hyperparameter tuning
        n_optuna_trials (int or None): Number of Optuna trials (uses config default if None)
    """
    print("\n" + "="*70)
    print("TYRE DEFECT DETECTION - COMPLETE PIPELINE")
    print("="*70)
    print(f"Device: {config.DEVICE}")

    # Warn if using CPU
    if not torch.cuda.is_available():
        print("\n[INFO] Running on CPU - Training will be slower but will work fine!")
        print("[INFO] Recommended: Use --no-tuning flag for faster training")
        print("[INFO] Config has been optimized for CPU with reduced batch sizes and epochs")

    print("\n" + "="*70 + "\n")

    # Set seed for reproducibility
    set_seed(config.SEED)

    # ========================================================================
    # STEP 1: DATA LOADING
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)

    # Load dataloaders with default settings
    loaders = dataset_module.get_dataloaders()
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    print(f"\nDataset sizes:")
    print(f"  Train: {len(loaders['train_dataset'])} images")
    print(f"  Val: {len(val_loader.dataset)} images")
    print(f"  Test: {len(test_loader.dataset)} images")

    # ========================================================================
    # STEP 2: HYPERPARAMETER TUNING (Optional)
    # ========================================================================
    best_params = None

    if use_tuning:
        import hyperparameter_tuning as hpo_module
        print("\n" + "="*70)
        print("STEP 2: HYPERPARAMETER TUNING WITH OPTUNA")
        print("="*70)

        best_params = hpo_module.run_hyperparameter_tuning(
            train_loader, val_loader, n_trials=n_optuna_trials
        )

        # Extract parameters for training
        lr_phase1 = best_params.get("lr_phase1", config.PHASE_1_LR)
        lr_phase2 = best_params.get("lr_phase2", config.PHASE_2_LR)
        batch_size = best_params.get("batch_size", config.BATCH_SIZE)
        rotation_degrees = best_params.get("rotation_degrees", config.TRAIN_AUG_ROTATION)
        dropout_rate = best_params.get("dropout", config.DROPOUT_RATE)

        # Recreate dataloaders with tuned batch size and rotation
        print(f"\nRecreating dataloaders with tuned hyperparameters...")
        print(f"  Batch Size: {batch_size}")
        print(f"  Rotation Degrees: {rotation_degrees} degrees")
        print(f"  Dropout: {dropout_rate:.4f}")

        loaders = dataset_module.get_dataloaders(
            batch_size=batch_size,
            rotation_degrees=rotation_degrees
        )
        train_loader = loaders["train"]
        val_loader = loaders["val"]
        test_loader = loaders["test"]
    else:
        print("\n" + "="*70)
        print("STEP 2: HYPERPARAMETER TUNING - SKIPPED")
        print("="*70)
        print("Using default hyperparameters from config.py")

        lr_phase1 = config.PHASE_1_LR
        lr_phase2 = config.PHASE_2_LR
        batch_size = config.BATCH_SIZE
        rotation_degrees = config.TRAIN_AUG_ROTATION
        dropout_rate = config.DROPOUT_RATE

    # ========================================================================
    # STEP 3: 3-PHASE FINE-TUNING
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: 3-PHASE FINE-TUNING")
    print("="*70)

    model, epoch_logs, criterion = train_module.train_full_pipeline(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_dataset=loaders.get("test_dataset"),
        lr_phase1=lr_phase1,
        lr_phase2=lr_phase2,
        lr_phase3=config.PHASE_3_LR,
        batch_size=batch_size,
        rotation_degrees=rotation_degrees,
        dropout_rate=dropout_rate,
        device=config.DEVICE
    )

    # ========================================================================
    # STEP 4: TEST SET EVALUATION
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: TEST SET EVALUATION")
    print("="*70)

    test_results = evaluate.evaluate_on_test_set(
        model, test_loader, criterion, device=config.DEVICE
    )

    # ========================================================================
    # STEP 5: EXAMPLE INFERENCE
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: EXAMPLE INFERENCE")
    print("="*70)

    # Get first test image
    test_images = list(config.TEST_DIR.glob("*.jpg"))[:3]

    if test_images:
        print(f"\nMaking predictions on {len(test_images)} sample test images:")

        for img_path in test_images:
            class_name, confidence, class_probs = inference.predict(
                img_path, model, device=config.DEVICE, return_probs=True
            )

            print(f"\n  {img_path.name}:")
            print(f"    Predicted: {class_name} (confidence: {confidence:.4f})")
            for class_name_i, prob in class_probs.items():
                print(f"      {class_name_i}: {prob:.4f}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  - Training log: {config.TRAINING_LOG_FILE}")
    print(f"  - Test results: {config.TEST_RESULTS_FILE}")
    print(f"  - Confusion matrix: {config.CONFUSION_MATRIX_FILE}")
    print(f"  - Phase 1 checkpoint: {config.CHECKPOINT_PHASE_1}")
    print(f"  - Phase 2 checkpoint: {config.CHECKPOINT_PHASE_2}")
    print(f"  - Phase 3 checkpoint: {config.CHECKPOINT_PHASE_3}")

    if use_tuning:
        print(f"  - Best Optuna params: {config.BEST_OPTUNA_PARAMS_FILE}")

    print(f"\nTest Results Summary:")
    print(f"  Overall Accuracy: {test_results['overall_accuracy']:.4f}")
    print(f"  Weighted F1 Score: {test_results['weighted_f1']:.4f}")
    print(f"  Macro F1 Score: {test_results['macro_f1']:.4f}")

    print("\n" + "="*70 + "\n")

    return model, test_results


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    use_tuning = True
    n_optuna_trials = None

    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "--no-tuning":
            use_tuning = False
        elif sys.argv[1].isdigit():
            n_optuna_trials = int(sys.argv[1])

    # Run pipeline
    model, test_results = main(
        use_tuning=use_tuning,
        n_optuna_trials=n_optuna_trials
    )
