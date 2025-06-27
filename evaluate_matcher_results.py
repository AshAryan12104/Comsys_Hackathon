# evaluate_matcher_results.py

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

csv_path = "outputs/results/face_recognition_results.csv"

df = pd.read_csv(csv_path)
true_labels = df["label"]
pred_labels = (df["label"] == 1).astype(int)  # 1 if matched correctly, else 0

print("\n📊 Final Evaluation for Task B:")
print(f"🎯 Top-1 Accuracy:         {accuracy_score(true_labels, pred_labels):.4f}")
print(f"🎯 Macro-Averaged F1-Score: {f1_score(true_labels, pred_labels, average='macro'):.4f}")
