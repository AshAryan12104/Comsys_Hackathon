# train.py

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.data_loader import get_loaders
from utils.helpers import set_seed, compute_metrics
from models.multitask_model import MultiTaskModel

def train(config):
    set_seed(config['project']['seed'])

    train_loader, val_loader = get_loaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")

    model = MultiTaskModel(
        backbone=config['model']['backbone'],
        pretrained=config['model']['pretrained'],
        num_classes_identity=config['model']['num_classes_identity']
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])

    best_f1 = 0.0
    num_epochs = config['train']['epochs']

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            images = images.to(device)
            labels = labels.float().to(device)  # labels: 0 (female), 1 (male)

            logits, _ = model(images)  # ignore identity logits
            logits = logits.view(-1)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f" Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

        # ---------------- Evaluation ----------------
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs, _ = model(images)
                preds = (torch.sigmoid(outputs.view(-1)) >= 0.5).int()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = compute_metrics(all_labels, all_preds)
        print(f" Val F1: {metrics['f1']:.4f} | Acc: {metrics['accuracy']:.4f}")

        # Save best model
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            os.makedirs("outputs/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "outputs/checkpoints/best_model.pt")
            print(f" Saved best model (F1: {best_f1:.4f})")

    print(" Training complete.")
