# train_matcher.py

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.embedding_model import EmbeddingModel
from utils.triplet_dataset import TripletDataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training on device: {device}")

# CONFIG
train_dir = "data/Task_B/train"
checkpoint_path = "outputs/checkpoints/matcher_model.pt"

# Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)

# Transforms
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

# Training loop
def train():
    dataset = TripletDataset(train_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = EmbeddingModel(backbone="facenet").to(device)
    loss_fn = TripletLoss(margin=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    epochs = 10

    for epoch in range(epochs):
        total_loss = 0.0
        for anchor, positive, negative in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = loss_fn(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f" Model saved to {checkpoint_path}")

if __name__ == "__main__":
    train()
