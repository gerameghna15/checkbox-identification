from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, PaliGemmaModel
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
import json
import io

# -------------------------
# Device
# -------------------------
DEVICE = "cpu"

# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent

# This points to: project_root/checkbox_model/finetuned/
MODEL_DIR = (BASE_DIR / ".." / "checkbox_model" / "finetuned").resolve()

# -------------------------
# Load model config
# -------------------------

with open(MODEL_DIR / "model_config.json", "r") as f:
    cfg = json.load(f)

# -------------------------
# Load processor & backbone
# -------------------------
processor = AutoProcessor.from_pretrained(cfg["backbone"])
backbone = PaliGemmaModel.from_pretrained(cfg["backbone"]).to(DEVICE)
backbone.eval()

# Freeze backbone (important)
for p in backbone.parameters():
    p.requires_grad = False

# -------------------------
# Rebuild classifier head (must match training)
# -------------------------
classifier = nn.Sequential(
    nn.Linear(cfg["classifier_architecture"]["input_dim"], 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2)
).to(DEVICE)

# Load trained weights
classifier.load_state_dict(
    torch.load(MODEL_DIR / "classifier_head.pt", map_location=DEVICE)
)
classifier.eval()

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Checkbox Classification API")

# -------------------------
# Feature extraction (MATCH TRAINING EXACTLY)
# -------------------------
@torch.no_grad()
def extract_features(image: Image.Image) -> torch.Tensor:
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    outputs = backbone(**inputs)

    # SAME AS TRAINING: mean pool + squeeze
    feat = outputs.last_hidden_state.mean(dim=1).squeeze()

    # Ensure shape = [1, input_dim]
    if feat.dim() == 1:
        feat = feat.unsqueeze(0)

    return feat

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
async def predict_checkbox(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        features = extract_features(image)
        logits = classifier(features)
        pred = logits.argmax(dim=1).item()

        label = "checked" if pred == 1 else "unchecked"

        return JSONResponse(content={
            "prediction": label,
            "pred_id": pred
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# -------------------------
# Health check endpoint 
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
