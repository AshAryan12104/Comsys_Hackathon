# 🎯 FaceCom: Robust Face and Gender Recognition in Challenging Environments

This repository contains our solution for the FaceCom Hackathon, where we address two real-world computer vision tasks:

- **Task A: Gender Classification**  
- **Task B: Face Verification / Matching under visual distortions**

---

## 📁 Dataset Structure (After Extraction)

data/
│
├── Task_A/
│   ├── train/
│   │   ├── male/
│   │   └── female/
│   └── val/
│       ├── male/
│       └── female/
│
├── Task_B/
│   ├── train/
│   │   ├── 001_frontal/
│   │   │   ├── 001_frontal.jpg
│   │   │   └── distortion/
│   │   │       ├── 001_frontal_distorted1.jpg
│   │   │       └── ...
│   │   └── ...
│   └── val/
│       └── same format as train/

---

## 🧠 Tasks & Approach

### 🔹 Task A – Gender Classification
- Binary classification: `male` or `female`
- CNN-based backbone (ResNet18)
- Trained using BCEWithLogitsLoss
- Evaluated on accuracy, precision, recall, F1

### 🔹 Task B – Face Matching / Verification
- Face embedding model trained with contrastive loss
- Compares distorted face to identity folders
- Uses cosine similarity for matching
- Labels: `1` if matched correctly, else `0`

---

## 🏗️ Project Structure

FaceCom-Robust-Face-and-Gender-Recognition/
│
├── README.md
├── requirements.txt
├── config.yaml
├── main.py
├── train.py
├── evaluate.py
├── matcher.py
├── train_matcher.py
│
├── models/
│   ├── multitask_model.py
│   ├── backbone.py
│   └── embedding_model.py
│
├── utils/
│   ├── data_loader.py
│   ├── transforms.py
│   ├── metrics.py
│   └── helpers.py
│
├── outputs/
│   ├── checkpoints/
│   │   ├── best_model.pt
│   │   └── matcher_model.pt
│   └── results/
│       └── matching_results.csv
│
├── data/
│   └── Task_A/, Task_B/
│
├── summary/
│   └── FaceCom_Technical_Summary.pdf
│
└── test_embedding_similarity.py

---

## ⚙️ Setup Instructions


## 🚀 How to Run
### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/FaceCom-Robust-Face-and-Gender-Recognition.git
cd FaceCom-Robust-Face-and-Gender-Recognition
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset
Extract and place the dataset inside the data/ folder.
Maintain the provided structure for Task_A and Task_B.

🚀 How to Run
🔹 Task A: Gender Classification
    Train:
```bash
python main.py --mode train --config config.yaml
```

    Evaluate Model
```bash
python main.py --mode evaluate --config config.yaml
```

🔹 Task B: Face Matching / Verification
    Train Embedding Model:
```bash
python train_matcher.py
```
    Generate Matching Predictions:
```bash
python matcher.py
```
    (Optional) Test Specific Similarities:
```bash
python test_embedding_similarity.py
```

---

## 📊 Evaluation Metrics
### Task A - Gender Classification:
- Accuracy
- Precision
- Recall
- F1-Score

### Task B - Face Recognition:
- Cosine Similarity
- Binary Labels:
- 1: Match
- 0: No Match
- Output saved in: outputs/results/matching_results.csv

---

## 📌 Summary Highlights
- Multitask learning enables better feature generalization.
- Strong robustness under blur, glare, fog, and low light.
- Trained & evaluated using PyTorch with GPU support.
- Modular code and easy-to-use config file.
- Uses pretrained ResNet for transfer learning.

---

## 📄 License
This repository is released for academic and hackathon use only.

---

## 🤝 Team
- **Name:** Md Aryan Rehman
- **Role:** Developer / Researcher
- **Contact:** [aryanrehman12104@gmail.com ]
