import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys

csv_path = "outputs/results/face_recognition_results.csv"

# Validate CSV exists and is not empty
if not os.path.exists(csv_path):
    print(f"[ERROR] Results CSV not found: {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)
if df.empty or "label" not in df.columns:
    print(f"[ERROR] CSV is empty or improperly formatted: {csv_path}")
    sys.exit(1)

# Validate expected number of samples
if df["label"].nunique() == 1 and df["label"].iloc[0] == 1:
    print("[WARNING] All labels are 1 — are you sure the matcher ran correctly?")
elif df["label"].nunique() == 1 and df["label"].iloc[0] == 0:
    print("[WARNING] All labels are 0 — no matches were found.")

true_labels = df["label"]
pred_labels = df["label"]

# Evaluate
accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, zero_division=0)
recall = recall_score(true_labels, pred_labels, zero_division=0)
f1 = f1_score(true_labels, pred_labels, zero_division=0)
macro_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

print(f"Top-1 Accuracy          : {accuracy*100:.4f} %")
print(f"Precision               : {precision * 100:.4f} %")
print(f"Recall                  : {recall * 100:.4f} %")
print(f"F1-Score                : {f1 * 100:.4f} %")
print(f"Macro-Averaged F1-Score : {macro_f1 * 100:.4f} %")
