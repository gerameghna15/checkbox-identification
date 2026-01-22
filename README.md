# Checkbox Identification using PaliGemma Vision Embeddings

This project improves checkbox state classification (**checked vs unchecked**) using **PaliGemma-3B vision embeddings** and a lightweight trained classifier head.

---

## 1. Objective
Adapt a multi-modal model to improve checkbox identification accuracy on a checkbox dataset by training a lightweight classifier head on top of frozen vision embeddings.

---

## 2. Dataset

### 2.1 Source
Dataset link (Roboflow):  
https://universe.roboflow.com/checkbox-detection-ztyeq/checkbox-kwtcz-qkbid/dataset/7

### 2.2 Preprocessing Summary
1. Started from the original dataset (COCO format).
2. Cropped checkbox regions from the original images.
3. Created a clean **binary split**:
   - `checked`
   - `unchecked`
4. Created two processed versions:
   - **Cropped (Full)** dataset
   - **Cropped (Small)** dataset *(mini version for faster training/testing)*

### 2.3 Dataset Summary

| Dataset Stage        | Train | Validation | Test |
|----------------------|-------|------------|------|
| Original (COCO)      | 611   | 151        | 81   |
| Cropped (Full)       | 4403  | 978        | 714  |
| Cropped (Small)      | 1000  | 300        | 400  |

### 2.4 Processed Dataset Folder Structure
The model training/evaluation uses the following folder structure:

```text
data/processed/cropped_checkboxes_binary_small/
├── train/
│   ├── checked/
│   └── unchecked/
├── valid/
│   ├── checked/
│   └── unchecked/
└── test/
    ├── checked/
    └── unchecked/

## Test Dataset

Only the test dataset (~12 MB) is included in this repository to allow immediate inference and reproducibility.
The full processed datasets are hosted on Hugging Face.

3. Approach
3.1 Baseline (No Training)
Backbone: google/paligemma-3b-mix-224

Vision embeddings: frozen

Classifier head: random Linear layer (no training)

Purpose: establish a lower-bound baseline

3.2 Fine-tuned Model (Classifier Head Training)
This notebook trains a lightweight classifier head on top of frozen PaliGemma vision embeddings for checkbox state classification.

Backbone: google/paligemma-3b-mix-224

Vision embeddings: frozen

Trainable classifier head: MLP

Linear → ReLU → Dropout → Linear

Training:

Epochs: 15

Optimizer: Adam

Loss: CrossEntropyLoss

4. Results
4.1 Baseline Performance
Accuracy: 0.50

Macro F1-score: 0.33

4.2 Fine-tuned Performance
Accuracy: 0.82

Macro F1-score: 0.81

All results, confusion matrices, and classification report visuals are stored under:

results/
├── baseline/
├── finetuned/
└── comparison/
5. Running the FastAPI App (Bonus)
5.1 Install Dependencies
pip install -r app/requirements.txt
5.2 Start the API Server
From the project root:

uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
5.3 Health Check
Open in browser:

http://127.0.0.1:8000/health
Expected output:

{"status":"ok"}
5.4 Example Prediction Request (cURL)
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@sample.jpg"
Example response:

{
  "prediction": "checked",
  "pred_id": 1
}
📌 Note: The @sample.jpg is required in curl to attach the image file correctly.

6. Model Files
Fine-tuned model artifacts are stored in:

checkbox_model/finetuned/
├── classifier_head.pt
├── model_config.json
├── model_info.json
└── training_config.json
7. Demo / Inference Scripts
7.1 API-based Demo (Random Test Images)
Runs inference through the FastAPI endpoint using random test images:

python3 scripts/test_inference.py
7.2 Local Demo (No API Required)
Runs inference locally without starting the FastAPI server:

python3 scripts/local_test_inference.py
8. Execution Note (Colab + Compute Constraints)
This experiment was executed on Google Colab using GPU resources. Some paths in the notebooks reference Google Drive/Colab environments. The saved results are preserved for baseline vs fine-tuned comparison and reproducibility under limited compute constraints.

9. Notes
The PaliGemma backbone is loaded from HuggingFace at runtime.

Only the classifier head is trained and saved.

The mini dataset version was created to speed up training and evaluation.

10. Full Project Directory Structure
checkbox_identification/
├── app/
│   ├── app.py
│   └── requirements.txt
├── checkbox_model/
│   ├── baseline/
│   │   ├── baseline_config.json
│   │   └── model_info.json
│   └── finetuned/
│       ├── classifier_head.pt
│       ├── model_config.json
│       ├── model_info.json
│       └── training_config.json
├── data/
│   ├── processed/
│   │   ├── cropped_checkboxes_binary/
│   │   │   ├── train/
│   │   │   ├── valid/
│   │   │   └── test/
│   │   └── cropped_checkboxes_binary_small/
│   │       ├── train/
│   │       ├── valid/
│   │       └── test/
│   └── raw/
│       └── Checkbox.v7i.coco/
├── notebooks/
│   ├── cropping.ipynb
│   ├── data_exploration.ipynb
│   ├── 03_baseline_paligemma_embeddings_ipynb_.ipynb
│   └── 04_finetuned_paligemma_classifier.ipynb
├── results/
│   ├── baseline/
│   ├── finetuned/
│   └── comparison/
└── scripts/
    ├── local_test_inference.py
    └── test_inference.py

---

