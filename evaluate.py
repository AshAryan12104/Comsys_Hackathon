# evaluate.py

import torch
from utils.data_loader import get_loaders
from utils.helpers import set_seed, compute_metrics
from models.multitask_model import MultiTaskModel

def evaluate(config):
    set_seed(config['project']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Evaluating on device: {device}")

    # Load validation data
    _, val_loader = get_loaders(config)

    # Load model
    model = MultiTaskModel(
        backbone=config['model']['backbone'],
        pretrained=config['model']['pretrained'],
        num_classes_identity=config['model']['num_classes_identity']
    ).to(device)

    model_path = "outputs/checkpoints/best_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Evaluate
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, gender in val_loader:
            images = images.to(device)
            gender = gender.to(device)

            gender_logits, _ = model(images)
            preds = (torch.sigmoid(gender_logits.view(-1)) >= 0.5).int()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(gender.cpu().numpy())

    # Metrics
    metrics = compute_metrics(all_labels, all_preds)

    print("\n--- Gender Classification Metrics ---")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")
