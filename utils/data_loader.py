from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import os

def get_loaders(config):
    transform = transforms.Compose([
        transforms.Resize((config['dataset']['image_size'], config['dataset']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_root = os.path.join(config['dataset']['root'], "train")
    val_root = config['dataset'].get("val_path", os.path.join(config['dataset']['root'], "val"))

    train_dataset = ImageFolder(root=train_root, transform=transform)
    val_dataset = ImageFolder(root=val_root, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False)

    return train_loader, val_loader
