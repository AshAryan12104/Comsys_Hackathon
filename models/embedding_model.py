# models/embedding_model.py

import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
from torchvision import models

class EmbeddingModel(nn.Module):
    def __init__(self, backbone="resnet18"):
        super().__init__()
        
        if backbone == "resnet18":
            base = models.resnet18(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1])  # remove FC layer
            self.embedding = nn.Linear(base.fc.in_features, 128)

        elif backbone == "resnet50":
            base = models.resnet50(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
            self.embedding = nn.Linear(base.fc.in_features, 128)

        elif backbone == "facenet":
            self.feature_extractor = InceptionResnetV1(pretrained='vggface2').eval()
            self.embedding = nn.Identity()  # Already outputs 512-dim embedding

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        x = self.feature_extractor(x)
        if isinstance(self.embedding, nn.Identity):
            return x.view(x.size(0), -1)
        else:
            x = x.view(x.size(0), -1)
            return self.embedding(x)
