"""
Model architecture for Tyre Defect Detection.
Uses EfficientNet-B2 pretrained on ImageNet with custom classifier head.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
import config


def create_model(dropout_rate=None, num_classes=None):
    """
    Create EfficientNet-B2 model with custom classifier head.

    Args:
        dropout_rate (float or None): Dropout rate before classifier
        num_classes (int or None): Number of output classes

    Returns:
        torch.nn.Module: Model with custom head
    """
    if dropout_rate is None:
        dropout_rate = config.DROPOUT_RATE

    if num_classes is None:
        num_classes = config.NUM_CLASSES

    # Load pretrained EfficientNet-B2
    weights = EfficientNet_B2_Weights.IMAGENET1K_V1
    model = efficientnet_b2(weights=weights)

    # Get number of features in the classifier
    num_features = model.classifier[1].in_features

    # Replace classifier with custom head
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(num_features, num_classes)
    )

    return model


def freeze_backbone(model):
    """
    Freeze all backbone parameters (everything except classifier).

    Args:
        model (torch.nn.Module): Model to freeze
    """
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_all(model):
    """
    Unfreeze all model parameters.

    Args:
        model (torch.nn.Module): Model to unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True


def unfreeze_last_n_blocks(model, n_blocks=3):
    """
    Unfreeze last n blocks of EfficientNet-B2 + classifier.

    EfficientNet-B2 has 'features' (which contains _blocks) and 'classifier'.

    Args:
        model (torch.nn.Module): Model to partially unfreeze
        n_blocks (int): Number of blocks to unfreeze from the end
    """
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Support both torchvision EfficientNet (features is Sequential)
    # and timm-like EfficientNet (features._blocks).
    blocks = []
    if hasattr(model, "features") and hasattr(model.features, "_blocks"):
        blocks = list(model.features._blocks)
    elif hasattr(model, "features"):
        blocks = list(model.features.children())

    if not blocks:
        raise RuntimeError(
            "Unable to find backbone blocks for partial unfreezing. "
            "Expected model.features._blocks or model.features children."
        )

    # Unfreeze last n blocks (clamped to available count)
    n_blocks = max(1, min(n_blocks, len(blocks)))
    for block in blocks[-n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    # Always unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True


def get_trainable_params(model):
    """
    Get number of trainable parameters.

    Args:
        model (torch.nn.Module): Model

    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_total_params(model):
    """
    Get total number of parameters.

    Args:
        model (torch.nn.Module): Model

    Returns:
        int: Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def print_model_summary(model, phase=""):
    """
    Print model parameter summary.

    Args:
        model (torch.nn.Module): Model
        phase (str): Phase name for logging
    """
    total = get_total_params(model)
    trainable = get_trainable_params(model)
    frozen = total - trainable

    print(f"\n{'='*60}")
    if phase:
        print(f"Model Summary - {phase}")
    else:
        print(f"Model Summary")
    print(f"{'='*60}")
    print(f"Total Parameters:     {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"Frozen Parameters:    {frozen:,}")
    print(f"{'='*60}\n")
