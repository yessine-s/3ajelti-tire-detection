import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
import config

# ── Recreate the exact same model architecture ─────────────────────────
def create_model():
    weights = EfficientNet_B2_Weights.IMAGENET1K_V1
    model = efficientnet_b2(weights=None)  # No pretrained weights needed
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=config.DROPOUT_RATE, inplace=True),
        nn.Linear(num_features, config.NUM_CLASSES)
    )
    return model

# ── Load your best checkpoint ──────────────────────────────────────────
print("Loading checkpoint...")
model = create_model()
checkpoint = torch.load(
    'outputs/best_checkpoint_phase3.pt',
    map_location='cpu'
)

# Handle different checkpoint formats
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
elif 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()
print("✅ Model loaded successfully")

# ── Export to ONNX ─────────────────────────────────────────────────────
dummy_input = torch.randn(1, 3, 224, 224)  # Batch=1, RGB, 224x224

print("Converting to ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    'outputs/tire_model.onnx',
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input':  {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
print("✅ ONNX model saved to outputs/tire_model.onnx")

# ── Verify the export ──────────────────────────────────────────────────
import onnx
onnx_model = onnx.load('outputs/tire_model.onnx')
onnx.checker.check_model(onnx_model)
print("✅ ONNX model is valid!")

# ── Check model size ───────────────────────────────────────────────────
import os
size = os.path.getsize('outputs/tire_model.onnx') / 1024 / 1024
print(f"📦 Model size: {size:.2f} MB")

# ── Test inference ─────────────────────────────────────────────────────
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('outputs/tire_model.onnx')
test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {'input': test_input})
probs = outputs[0][0]

print("\n🧪 Test inference result:")
for i, (cls, prob) in enumerate(zip(config.CLASS_NAMES, probs)):
    print(f"  {cls}: {prob:.4f}")

print("\n✅ Export complete! Ready for mobile integration.")