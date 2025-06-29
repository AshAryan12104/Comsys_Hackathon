# evaluate_matcher_results.py

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

csv_path = "outputs/results/face_recognition_results.csv"

df = pd.read_csv(csv_path)
true_labels = df["label"]  # already 1 if matched correctly, 0 otherwise
pred_labels = df["label"]  # assuming labels were computed properly in matcher.py

# Evaluate
accuracy = accuracy_score(true_labels, pred_labels)*100
macro_f1 = f1_score(true_labels, pred_labels, average='macro')*100

print(f"Top-1 Accuracy: {accuracy:.4f} %")
print(f"Macro-Averaged F1-Score: {macro_f1:.4f} %")
