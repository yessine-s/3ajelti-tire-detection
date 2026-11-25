"""
Simple photo testing script with tire information
Usage: python test_photo.py YOUR_PHOTO_PATH
Example: python test_photo.py C:/Users/USER/Desktop/my_tire.jpg
"""

import sys
from pathlib import Path
import inference
import config

if len(sys.argv) < 2:
    print("Usage: python test_photo.py YOUR_PHOTO_PATH")
    print("Example: python test_photo.py C:/Users/USER/Desktop/my_tire.jpg")
    sys.exit(1)

photo_path = sys.argv[1]

# Load model
print(f"Loading model...")
model = inference.load_checkpoint(config.CHECKPOINT_PHASE_3, device=config.DEVICE)

# Predict
print(f"Predicting on: {photo_path}\n")
predicted_class, confidence, class_probs, tire_info = inference.predict(
    photo_path,
    model,
    device=config.DEVICE,
    return_probs=True
)

# Display results
print("="*70)
print(f"  {tire_info['icon']} TIRE DETECTION RESULT - 3ajeltii {tire_info['icon']}")
print("="*70)
print(f"\nPhoto: {Path(photo_path).name}")
print(f"\nDetected Problem: {predicted_class}")
print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

print(f"\n{'─'*70}")
print(f"Model signal: {tire_info['model_signal']}")
print(f"{'─'*70}")

print(f"\nDescription:")
print(f"{tire_info['description']}")

print(f"\n⚡ Recommendation:")
print(f"{tire_info['advice']}")

print(f"\nAll Class Probabilities:")
for class_name, prob in class_probs.items():
    print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")

print("\n" + "="*70 + "\n")
