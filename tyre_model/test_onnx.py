import onnxruntime as ort
import numpy as np
from PIL import Image
import config

# ── Load ONNX model ────────────────────────────────────────────────────
session = ort.InferenceSession('outputs/tire_model.onnx')
print("✅ ONNX model loaded")

# ── Preprocess image (same as training) ───────────────────────────────
def preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = (img - mean) / std

    # HWC → CHW → NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

# ── Run prediction ─────────────────────────────────────────────────────
def predict(image_path):
    img     = preprocess(image_path)
    outputs = session.run(None, {'input': img})
    logits  = outputs[0][0]

    # Softmax to get probabilities
    exp     = np.exp(logits - np.max(logits))
    probs   = exp / exp.sum()

    pred_idx   = np.argmax(probs)
    pred_class = config.CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100

    print(f"\n🛞 Prediction: {pred_class}")
    print(f"📊 Confidence: {confidence:.1f}%")
    print(f"\nAll probabilities:")
    for cls, prob in zip(config.CLASS_NAMES, probs):
        bar = '█' * int(prob * 20)
        print(f"  {cls:20s}: {prob*100:5.1f}% {bar}")

    return pred_class, confidence

# ── Test with your image ───────────────────────────────────────────────
import sys
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    # Default: use first image found in test data
    import os
    for cls in config.CLASS_NAMES:
        test_dir = f"data/test/images"
        if os.path.exists(test_dir):
            images = [f for f in os.listdir(test_dir)
                     if f.lower().endswith(('.jpg','.jpeg','.png'))]
            if images:
                image_path = os.path.join(test_dir, images[0])
                print(f"Using test image: {image_path}")
                break

predict(image_path)