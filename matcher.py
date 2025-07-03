
# matcher.py

import os
import sys
import torch
import csv
import argparse
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from models.embedding_model import EmbeddingModel

parser = argparse.ArgumentParser()
parser.add_argument('--test_dir', type=str, default="data/Task_B/val", help="Path to test directory")
args = parser.parse_args()

# === CONFIG ===
identity_dir = args.test_dir
test_root = args.test_dir
output_csv = "outputs/results/face_recognition_results.csv"
model_path = "outputs/checkpoints/matcher_model.pt"
threshold = 0.65  # Adjust based on performance

progress_file = "outputs/results/matcher_progress.txt"
def write_progress(pct):
    with open(progress_file, "w") as f:
        f.write(str(pct))

# === DEVICE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# === LOAD MODEL ===
model = EmbeddingModel(backbone="facenet").to(device)
model.eval()

# === Validate Test Directory ===
if not os.path.exists(test_root):
    print(f"[ERROR] Test path '{test_root}' does not exist.")
    sys.exit(1)

valid = False
for identity in os.listdir(test_root):
    identity_path = os.path.join(test_root, identity)
    if not os.path.isdir(identity_path):
        continue

    ref_imgs = [f for f in os.listdir(identity_path) if f.endswith('.jpg')]
    distortion_path = os.path.join(identity_path, 'distortion')
    if not ref_imgs:
        print(f"[WARN] Identity '{identity}' has no reference image.")
        continue
    if not os.path.isdir(distortion_path):
        print(f"[WARN] No 'distortion' folder in '{identity_path}'")
        continue

    distorted_imgs = [f for f in os.listdir(distortion_path) if f.endswith('.jpg')]
    if not distorted_imgs:
        print(f"[WARN] No distorted images in '{distortion_path}'")
        continue

    valid = True
    break

if not valid:
    print("[ERROR] No valid identities with reference and distorted images found.")
    sys.exit(1)


# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

# === UTILS ===
def get_embedding(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(image).squeeze().cpu().numpy()
    return emb

# === BUILD IDENTITY EMBEDDINGS ===
reference_embeddings = {}
for identity in os.listdir(identity_dir):
    identity_path = os.path.join(identity_dir, identity)
    if not os.path.isdir(identity_path): continue

    images = [img for img in os.listdir(identity_path) if img.endswith(".jpg")]
    if not images: continue

    ref_path = os.path.join(identity_path, images[0])  # use the first reference image
    embedding = get_embedding(ref_path)
    reference_embeddings[identity] = embedding

# === MATCHING ===
print("[INFO] Matching distorted images...")

os.makedirs(os.path.dirname(output_csv), exist_ok=True)
with open(output_csv, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image", "predicted_identity", "match_score", "label"])

    # === Pre-calculate total number of distorted images ===
    total_imgs = sum(
    len(os.listdir(os.path.join(test_root, folder, "distortion")))
    for folder in os.listdir(test_root)
    if os.path.isdir(os.path.join(test_root, folder, "distortion"))
    )

    processed = 0

    for identity in tqdm(os.listdir(test_root)):
        identity_folder = os.path.join(test_root, identity)
        distortion_folder = os.path.join(identity_folder, "distortion")
        if not os.path.isdir(distortion_folder): continue

        for distorted_img in os.listdir(distortion_folder):
            if not distorted_img.endswith(".jpg"): continue

            test_path = os.path.join(distortion_folder, distorted_img)
            test_emb = get_embedding(test_path)

            best_match = None
            best_score = -1

            for ref_id, ref_emb in reference_embeddings.items():
                score = cosine_similarity(test_emb.reshape(1, -1), ref_emb.reshape(1, -1))[0][0]
                if score > best_score:
                    best_score = score
                    best_match = ref_id

            label = 1 if identity == best_match and best_score >= threshold else 0
            writer.writerow([distorted_img, best_match, round(best_score, 4), label])
            processed += 1
            percent = min(int((processed / total_imgs) * 100),100)
            with open("outputs/results/matcher_progress.txt", "w") as f:
                f.write(str(percent))


print(f"[DONE] Results saved to {output_csv}")
write_progress(100)

