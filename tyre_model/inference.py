"""
Inference module for Tyre Defect Detection.
Provides predict() function for single image predictions with detailed tire information.
"""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
import config
import model as model_module
from dataset import get_transforms
from tire_info import get_tire_info


def predict(image_path, model, device=None, return_probs=False):
    """
    Predict defect class for a single image with tire information.

    Args:
        image_path (str or Path): Path to image file
        model (torch.nn.Module): Trained model
        device (torch.device): Device to use
        return_probs (bool): Whether to return probability for all classes

    Returns:
        tuple: (predicted_class_name, confidence_score) if return_probs=False
               (predicted_class_name, confidence_score, class_probs, tire_info) if return_probs=True
    """
    if device is None:
        device = config.DEVICE

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Set model to eval mode
    model.to(device)
    model.eval()

    # Load and preprocess image
    try:
        image = Image.open(image_path).convert("L")  # Grayscale
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {e}")

    # Convert grayscale to RGB by duplicating channel
    image_arr = np.array(image)
    image_rgb = np.stack([image_arr, image_arr, image_arr], axis=-1)
    image = Image.fromarray(image_rgb.astype(np.uint8))

    # Get validation transform (no augmentation)
    transform = get_transforms("val")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Forward pass
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)

    # Get prediction
    predicted_idx = probs.argmax(dim=1).item()
    confidence = probs[0, predicted_idx].item()
    predicted_class = config.CLASS_NAMES[predicted_idx]

    # Get tire information
    tire_info = get_tire_info(predicted_class)

    if return_probs:
        class_probs = {
            config.CLASS_NAMES[i]: float(probs[0, i].item())
            for i in range(config.NUM_CLASSES)
        }
        return predicted_class, confidence, class_probs, tire_info
    else:
        return predicted_class, confidence, tire_info


def predict_batch(image_paths, model, device=None, return_probs=False):
    """
    Predict defect classes for multiple images with tire information.

    Args:
        image_paths (list): List of paths to image files
        model (torch.nn.Module): Trained model
        device (torch.device): Device to use
        return_probs (bool): Whether to return probabilities for all classes

    Returns:
        list: List of predictions, each element contains tire information
    """
    predictions = []
    for image_path in image_paths:
        try:
            pred = predict(image_path, model, device, return_probs)
            predictions.append(pred)
        except Exception as e:
            print(f"Error predicting {image_path}: {e}")
            predictions.append(None)

    return predictions


def load_checkpoint(checkpoint_path, device=None):
    """
    Load a trained model checkpoint.

    Args:
        checkpoint_path (str or Path): Path to checkpoint file
        device (torch.device): Device to use

    Returns:
        torch.nn.Module: Loaded model
    """
    if device is None:
        device = config.DEVICE

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Create model
    model = model_module.create_model()

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    return model
