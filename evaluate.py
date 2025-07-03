# evaluate.py

import torch
import warnings
from utils.data_loader import get_loaders
from utils.helpers import set_seed, compute_metrics
from models.multitask_model import MultiTaskModel
import os

def evaluate(config):
    warnings.filterwarnings("ignore")  #  Suppress all warnings

    set_seed(config['project']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check override from test_path.txt
    val_path = os.getenv("OVERRIDE_VAL_PATH", None)
    if val_path:
        config["dataset"]["root"] = val_path

    _, val_loader = get_loaders(config)

    model = MultiTaskModel(
        backbone=config['model']['backbone'],
        pretrained=config['model']['pretrained'],
        num_classes_identity=config['model']['num_classes_identity']
    ).to(device)

    model_path = "outputs/checkpoints/best_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, gender in val_loader:
            images = images.to(device)
            gender = gender.to(device)

            gender_logits, _ = model(images)
            preds = (torch.sigmoid(gender_logits.view(-1)) >= 0.5).int()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(gender.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds)

    # ✅ Print only what frontend needs
    print(f"Accuracy: {metrics['accuracy']*100:.4f} %")
    print(f"Precision: {metrics['precision']*100:.4f} %")
    print(f"Recall: {metrics['recall']*100:.4f} %")
    print(f"F1-Score: {metrics['f1']*100:.4f} %")
