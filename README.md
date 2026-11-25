# 3ajeltii — Tire Defect Detection

3ajeltii is an Expo mobile app and Python inference project for classifying tire photos as **Bulge**, **Cracks**, or **Non-defective**. The mobile app captures or selects a photo and sends it to a local FastAPI service, which returns a predicted class and confidence scores.

> **Safety:** This is an experimental image-classification project, not a tire inspection or roadworthiness assessment. A “Non-defective” result does not confirm that a tire is safe, and the model cannot determine safe remaining mileage. If you notice cuts, cracks, bulges, or other damage, stop using the tire and have it assessed by a qualified tire professional. See [NHTSA tire safety guidance](https://www.nhtsa.gov/vehicle-safety/tires).

## Project layout

- `TireAI/` — Expo SDK 58 / React Native mobile client.
- `TireAI/assets/tire_model.onnx` — ONNX model asset used by the project.
- `tyre_model/` — Python model training, evaluation, export, and FastAPI inference code.
- `tyre_model/data/` — expected local training and evaluation data; intentionally not included in this repository.
- `tyre_model/outputs/` — local training outputs and the ONNX file used by the API; generated locally and ignored by Git.

## Requirements

- Node.js **22.13 or newer** for Expo SDK 58.
- npm.
- Python 3.10 or newer for the API and training scripts.
- A phone and development computer on the same network when testing against the local API.

## Clone the repository

```powershell
git clone https://github.com/yessine-s/3ajelti-tire-detection.git
cd 3ajelti-tire-detection
```
## Run the mobile app

```powershell
cd TireAI
npm ci
Copy-Item .env.example .env.local
```

Edit `TireAI/.env.local` and set `EXPO_PUBLIC_API_URL` to the LAN address of the computer running the API, for example `http://192.168.1.20:8000`. Use your own address; a phone cannot reach the computer through `localhost`.

Then start Expo:

```powershell
npx expo start
```

Open the project in a compatible Expo Go app or a development build. If you change `.env.local`, restart Expo so the public environment variable is bundled again.

## Run the local inference API

From the repository root, create and activate a virtual environment, then install the API dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r tyre_model/requirements-api.txt
```

The API expects its model at `tyre_model/outputs/tire_model.onnx`. The mobile model asset is included in the repository, so copy it into the API output directory:

```powershell
New-Item -ItemType Directory -Force tyre_model/outputs | Out-Null
Copy-Item TireAI/assets/tire_model.onnx tyre_model/outputs/tire_model.onnx
Set-Location tyre_model
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

The API provides `GET /` for a health check and `POST /predict` for an image upload. The current CORS configuration is permissive for local development; restrict allowed origins and add appropriate access controls before deploying it to a public server.

## Train or evaluate the Python model

Training data and generated checkpoints are not committed. Place a properly licensed dataset under `tyre_model/data/` using the train, valid, and test folders and label CSV files expected by `tyre_model/config.py`. Then install the training requirements and run the pipeline:

```powershell
Set-Location tyre_model
python -m pip install -r requirements.txt
python main.py --no-tuning
```

Run `python main.py` to include Optuna hyperparameter tuning. Training creates files under `tyre_model/outputs/`; these files are intentionally ignored by Git. The inference API uses the ONNX model path described above.

## Configuration and privacy

- Set `EXPO_PUBLIC_API_URL` in `TireAI/.env.local`; `.env.local` is ignored by Git. Expo public variables are embedded in the client bundle, so never put passwords, API tokens, or other secrets in an `EXPO_PUBLIC_` variable.
- Training images, virtual environments, checkpoints, and local keys are excluded from the repository. Provide your own data and model artifacts where needed.
- No license file is included yet. Add a license before inviting reuse if you intend to grant permission to redistribute or modify this project.