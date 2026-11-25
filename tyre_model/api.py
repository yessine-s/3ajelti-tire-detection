from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Allow requests from Expo app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load ONNX model ────────────────────────────────────────────────────
CLASS_NAMES    = ['Bulge', 'Cracks', 'Non-defective']
IMAGENET_MEAN  = np.array([0.485, 0.456, 0.406])
IMAGENET_STD   = np.array([0.229, 0.224, 0.225])

session = ort.InferenceSession('outputs/tire_model.onnx')
print("✅ Model loaded!")

# ── Preprocess ─────────────────────────────────────────────────────────
def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)           # HWC → CHW
    img = np.expand_dims(img, axis=0)      # → NCHW
    return img.astype(np.float32)

# ── Softmax ────────────────────────────────────────────────────────────
def softmax(logits):
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()

# ── Predict endpoint ───────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img      = preprocess(contents)

    outputs  = session.run(None, {'input': img})
    logits   = outputs[0][0]
    probs    = softmax(logits)

    pred_idx    = int(np.argmax(probs))
    pred_class  = CLASS_NAMES[pred_idx]
    confidence  = float(probs[pred_idx] * 100)

    return {
        "label":      pred_class,
        "confidence": round(confidence, 1),
        "isCracked":  pred_class != 'Non-defective',
        "allScores": [
            {"class": cls, "prob": round(float(probs[i]) * 100, 1)}
            for i, cls in enumerate(CLASS_NAMES)
        ]
    }

# ── Health check ───────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "✅ Tire Detection API is running!"}