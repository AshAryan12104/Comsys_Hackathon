from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

def get_loaders(config):
    transform = transforms.Compose([
        transforms.Resize((config['dataset']['image_size'], config['dataset']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root="data/Task_A/train", transform=transform)
    val_dataset = ImageFolder(root="data/Task_A/val", transform=transform)


    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False)

    return train_loader, val_loader
