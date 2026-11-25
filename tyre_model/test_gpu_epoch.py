#!/usr/bin/env python3
"""
Quick GPU training test - runs 1 epoch of Phase 1 to verify GPU is working
"""

import sys
import time
sys.path.insert(0, '.')

import torch
import config
from dataset import get_dataloaders
from model import create_model
from train import train_epoch
import torch.nn as nn

def main():
    print("\n" + "="*70)
    print("GPU TRAINING TEST - 1 EPOCH")
    print("="*70)

    # Check GPU
    print(f"\nDevice: {config.DEVICE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"AMP Enabled: {config.USE_AMP}")

    # Load data
    print("\nLoading data...")
    dataloaders = get_dataloaders()
    train_loader = dataloaders['train']

    # Build model
    print("Building model...")
    model = create_model(num_classes=config.NUM_CLASSES)

    # Setup training
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Warm up GPU
    print("\nWarming up GPU...")
    dummy_input = torch.randn(config.BATCH_SIZE, 3, config.TARGET_IMG_SIZE, config.TARGET_IMG_SIZE)
    dummy_input = dummy_input.to(config.DEVICE)
    model.to(config.DEVICE)
    with torch.no_grad():
        _ = model(dummy_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Train 1 epoch
    print("\nTraining 1 epoch...")
    start = time.time()

    metrics = train_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device=config.DEVICE,
        show_progress=True,
        scaler=None,
        use_amp=config.USE_AMP
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.time() - start

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nTime: {elapsed:.2f} seconds")
    print(f"Images per second: {len(dataloaders['train_dataset']) / elapsed:.1f}")
    print(f"\nMetrics:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-weighted: {metrics['f1_weighted']:.4f}")

    # GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU Memory: {allocated:.2f}/{total:.2f} GB")

    # Speed benchmark
    print("\n" + "="*70)
    if elapsed < 60:
        print("SUCCESS: GPU is working!")
        print(f"Expected ~3-5 min per epoch (full training)")
        print(f"Total training time estimate: 30-40 min with GPU")
    else:
        print("WARNING: Training is slow, GPU may not be being used")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
