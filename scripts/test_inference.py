import random
import requests
from pathlib import Path
from collections import Counter

API_URL = "http://127.0.0.1:8000/predict"

# Always resolve relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "cropped_checkboxes_binary_small" / "test"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Reproducibility
random.seed(42)


def get_all_test_images():
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test folder not found: {TEST_DIR.resolve()}")

    class_folders = ["checked", "unchecked"]
    all_images = []

    for cls in class_folders:
        cls_dir = TEST_DIR / cls
        if not cls_dir.exists():
            continue

        for p in cls_dir.rglob("*"):
            if p.suffix.lower() in VALID_EXTS:
                all_images.append((p, cls))

    if not all_images:
        raise RuntimeError(f"No images found inside: {TEST_DIR.resolve()}")

    return all_images


def predict_image(img_path: Path):
    with open(img_path, "rb") as f:
        files = {"file": (img_path.name, f, "image/jpeg")}
        response = requests.post(API_URL, files=files, timeout=180)

    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")

    pred_json = response.json()
    return pred_json.get("prediction", "N/A")


def main():
    N = 20  # total samples for demo (balanced)
    all_images = get_all_test_images()

    checked_imgs = [(p, y) for (p, y) in all_images if y == "checked"]
    unchecked_imgs = [(p, y) for (p, y) in all_images if y == "unchecked"]

    if len(checked_imgs) == 0 or len(unchecked_imgs) == 0:
        raise RuntimeError("Test folder must contain both 'checked' and 'unchecked' images.")

    # Balanced sampling: half checked + half unchecked
    half = N // 2
    half = min(half, len(checked_imgs), len(unchecked_imgs))

    sampled = random.sample(checked_imgs, half) + random.sample(unchecked_imgs, half)
    random.shuffle(sampled)
    N = len(sampled)

    correct = 0
    wrong = 0
    errors = []
    pred_counter = Counter()

    print(f"\n🚀 Running inference on test images...\n")

    for i, (img_path, true_label) in enumerate(sampled, start=1):
        try:
            pred_label = predict_image(img_path)
            pred_counter[pred_label] += 1

            if pred_label == true_label:
                correct += 1
                status = "✅ CORRECT"
            else:
                wrong += 1
                status = "❌ WRONG"
                errors.append((img_path.name, true_label, pred_label))

            print(f"[{i:02d}] {status} | True: {true_label:9s} | Pred: {pred_label:9s} | File: {img_path.name}")

        except Exception as e:
            wrong += 1
            errors.append((img_path.name, true_label, "ERROR"))
            print(f"[{i:02d}] ⚠️ ERROR | True: {true_label:9s} | File: {img_path.name} | {str(e)}")

    accuracy = correct / N if N > 0 else 0.0

    print("\n" + "=" * 50)
    print("📊 Mini Evaluation Summary (Balanced Sample)")
    print("=" * 50)
    print(f"Total Samples: {N}")
    print(f"Correct:       {correct}")
    print(f"Wrong:         {wrong}")
    print(f"Mini Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\n📌 Prediction Distribution:")
    for k, v in pred_counter.items():
        print(f"  {k}: {v}")

    if errors:
        print("\n❗ Wrong Predictions / Errors (showing up to 10):")
        for fname, true_label, pred_label in errors[:10]:
            print(f"  File: {fname} | True: {true_label} | Pred: {pred_label}")
    else:
        print("\n🎉 No errors found in this mini test run!")


if __name__ == "__main__":
    main()
