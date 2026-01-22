import random
import json
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from collections import Counter
from transformers import AutoProcessor, PaliGemmaModel

# -------------------------
# Reproducibility
# -------------------------
random.seed(42)
torch.manual_seed(42)

# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_ROOT / "data" / "processed" / "cropped_checkboxes_binary_small" / "test"
MODEL_DIR = PROJECT_ROOT / "checkbox_model" / "finetuned"

MODEL_CONFIG_PATH = MODEL_DIR / "model_config.json"
HEAD_PATH = MODEL_DIR / "classifier_head.pt"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

DEVICE = "cpu"


# -------------------------
# Model builder (must match training)
# -------------------------
def build_classifier(input_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )


# -------------------------
# Load config
# -------------------------
if not MODEL_CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing model_config.json at: {MODEL_CONFIG_PATH.resolve()}")

if not HEAD_PATH.exists():
    raise FileNotFoundError(f"Missing classifier_head.pt at: {HEAD_PATH.resolve()}")

with open(MODEL_CONFIG_PATH, "r") as f:
    cfg = json.load(f)

BACKBONE_ID = cfg["backbone"]
INPUT_DIM = int(cfg["classifier_architecture"]["input_dim"])


# -------------------------
# Load processor + backbone
# -------------------------
print("🔄 Loading backbone:", BACKBONE_ID)
processor = AutoProcessor.from_pretrained(BACKBONE_ID)
backbone = PaliGemmaModel.from_pretrained(BACKBONE_ID).to(DEVICE)
backbone.eval()

for p in backbone.parameters():
    p.requires_grad = False


# -------------------------
# Load classifier head
# -------------------------
print("🔄 Loading classifier head:", HEAD_PATH)
classifier = build_classifier(INPUT_DIM).to(DEVICE)
classifier.load_state_dict(torch.load(HEAD_PATH, map_location=DEVICE))
classifier.eval()


# -------------------------
# Collect test images
# -------------------------
def get_all_test_images():
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test folder not found: {TEST_DIR.resolve()}")

    all_images = []
    for cls in ["checked", "unchecked"]:
        cls_dir = TEST_DIR / cls
        if not cls_dir.exists():
            continue

        for p in cls_dir.rglob("*"):
            if p.suffix.lower() in VALID_EXTS:
                all_images.append((p, cls))

    if not all_images:
        raise RuntimeError(f"No images found in: {TEST_DIR.resolve()}")

    return all_images


@torch.no_grad()
def extract_features(image: Image.Image) -> torch.Tensor:
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    outputs = backbone(**inputs)

    feat = outputs.last_hidden_state.mean(dim=1).squeeze()

    # Ensure shape = [1, input_dim]
    if feat.dim() == 1:
        feat = feat.unsqueeze(0)

    return feat


@torch.no_grad()
def predict_label(img_path: Path) -> str:
    image = Image.open(img_path).convert("RGB")
    features = extract_features(image)

    logits = classifier(features)
    pred_id = logits.argmax(dim=1).item()

    return "checked" if pred_id == 1 else "unchecked"


def main():
    N = 20  # balanced demo size
    all_images = get_all_test_images()

    checked_imgs = [(p, y) for (p, y) in all_images if y == "checked"]
    unchecked_imgs = [(p, y) for (p, y) in all_images if y == "unchecked"]

    if len(checked_imgs) == 0 or len(unchecked_imgs) == 0:
        raise RuntimeError("Test folder must contain both 'checked' and 'unchecked' images.")

    half = N // 2
    half = min(half, len(checked_imgs), len(unchecked_imgs))

    sampled = random.sample(checked_imgs, half) + random.sample(unchecked_imgs, half)
    random.shuffle(sampled)
    N = len(sampled)

    correct = 0
    wrong = 0
    errors = []
    pred_counter = Counter()

    print(f"\n🚀 Running LOCAL inference on {N} balanced random test images (seed=42)...\n")

    for i, (img_path, true_label) in enumerate(sampled, start=1):
        pred_label = predict_label(img_path)
        pred_counter[pred_label] += 1

        if pred_label == true_label:
            correct += 1
            status = "✅ CORRECT"
        else:
            wrong += 1
            status = "❌ WRONG"
            errors.append((img_path.name, true_label, pred_label))

        print(f"[{i:02d}] {status} | True: {true_label:9s} | Pred: {pred_label:9s} | File: {img_path.name}")

    accuracy = correct / N if N > 0 else 0.0

    print("\n" + "=" * 50)
    print("📊 Local Mini Evaluation Summary (Balanced Sample)")
    print("=" * 50)
    print(f"Total Samples: {N}")
    print(f"Correct:       {correct}")
    print(f"Wrong:         {wrong}")
    print(f"Mini Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\n📌 Prediction Distribution:")
    for k, v in pred_counter.items():
        print(f"  {k}: {v}")

    if errors:
        print("\n❗ Wrong Predictions (showing up to 10):")
        for fname, true_label, pred_label in errors[:10]:
            print(f"  File: {fname} | True: {true_label} | Pred: {pred_label}")
    else:
        print("\n🎉 No errors found in this local mini test run!")


if __name__ == "__main__":
    main()
