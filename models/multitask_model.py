# models/multitask_model.py

import torch
import torch.nn as nn
from torchvision import models

class MultiTaskModel(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, num_classes_identity=100):
        super(MultiTaskModel, self).__init__()

        # Load pretrained ResNet
        base_model = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  # Remove FC layer

        # Classifier heads
        self.gender_head = nn.Linear(base_model.fc.in_features, 1)  # Binary
        self.identity_head = nn.Linear(base_model.fc.in_features, num_classes_identity)  # Multi-class

    def forward(self, x):
        features = self.backbone(x).squeeze()
        gender_logits = self.gender_head(features)
        identity_logits = self.identity_head(features)
        return gender_logits, identity_logits
