
# 🎯 Robust Face and Gender Recognition in Challenging Environments

This repository contains our complete solution for the **Comsys Hackathon**, where we address two real-world computer vision tasks using PyTorch, metric learning, and a web-based evaluation frontend.

---

## 🧠 Tasks Overview

### 🔹 Task A – Gender Classification
- **Objective**: Classify face images as **male** or **female**
- **Model**: ResNet18 backbone + binary classification head
- **Evaluation**: Accuracy, Precision, Recall, F1-score

### 🔹 Task B – Face Recognition (Matching)
- **Objective**: Match distorted face images to correct identities
- **Model**: Triplet-based embedding model using **FaceNet** (InceptionResnetV1)
- **Evaluation**: Top-1 Accuracy, Macro-averaged F1-score

---

## 📁 Project Structure

<pre> ```
Comsys_Hackathon/
│
├── backend.py
├── config.yaml
├── evaluate_task_a.py
├── evaluate_matcher_results.py
├── matcher.py
├── train.py
├── train_matcher.py
├── evaluate.py
├── main.py
│
├── utils/
│   ├── data_loader.py
│   ├── metrics.py
│   ├── helpers.py
│   └── triplet_dataset.py
│
├── models/
│   ├── multitask_model.py
│   └── embedding_model.py
│
├── outputs/
│   ├── checkpoints/
│   │   ├── best_model.pt     # For Task A
│   │   └── matcher_model.pt  # For Task B
│   │
│   └── results/
│       └── face_recognition_results.csv
│
├── index.html
│
├── style.css
│
│
├── requirements.txt
├── test_path.txt
├── README.md
└── summary/
    └── Comsys_Hackathon.pdf
``` </pre>

---

## ⚙️ Setup Instructions

### 1. Clone the Repo
<pre> ```
git clone https://github.com/AshAryan12104/Comsys_Hackathon
cd Comsys_Hackathon
``` </pre>

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Train Models

#### 🔹 Task A (Gender Classification)
python main.py --mode train --config config.yaml

#### 🔹 Task B (Face Matching)
python train_matcher.py

---

## 🔐 Download Pretrained Weights

To run evaluation without retraining, download the pretrained model checkpoints from the link below:

📦 [Download Weights from Google Drive](https://drive.google.com/drive/folders/1IjbLg77rXdhvyadaN2vDwgEpFyu935v2?usp=sharing)

### Files Included:

- `best_model.pt` → Used for Task A (Gender Classification)
- `matcher_model.pt` → Used for Task B (Face Matching)

### 📂 Where to Place the Files:

After downloading, place them in the following location inside the project: (This is very important)
<pre> ```
├── outputs/
│   ├── checkpoints/
│   │   ├── best_model.pt     # For Task A
│   │   └── matcher_model.pt  # For Task B
``` </pre>

> ⚠️ Make sure the folder structure matches exactly, or the evaluation scripts may not find the model files.

---

## ✅ Evaluation

### 🔹 Task A (Gender)
python evaluate_task_a.py

### 🔹 Task B (Face Recognition)
python matcher.py
python evaluate_matcher_results.py

---

## 🌐 Frontend: Web Evaluation Portal 

1. Run the backend:
python backend.py

2. Open your browser and visit:
http://localhost:5000

3. Enter validation folder path (e.g. data/Task_A/val)

4. Click **Evaluate Task A** or **Evaluate Task B**

5. Wait for few sec to a min for processing...(At 1st try the localhost might restart or do some glitching, In that case repeat 4. Click **Evaluate Task A** or **Evaluate Task B** once again it will work )

5. View metrics on screen !!!

---

## 📊 Sample Metrics

### ✅ Task A
- Accuracy: 94.79%
- Precision: 97.43%
- Recall: 95.58%
- F1-Score: 96.50%

### ✅ Task B
- Top-1 Accuracy: 100.00%
- Macro-Averaged F1-Score: 100.00%

---

## 📄 License

This project is developed as part of a Hackathon and is intended for academic and demonstration use only.

---

## 👥 Team BYTEBash

- **Name:** Md Aryan Rehman, Raiyan Aftab Ansari, Priyanshu Mishra.
- **GitHub:** https://github.com/AshAryan12104
- **Email:** bytebash.gcetts.entropy@gmail.com 
---
