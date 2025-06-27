# utils/triplet_dataset.py

import os
import random
from PIL import Image
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.identities = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.image_paths = {identity: self._get_images(identity) for identity in self.identities}

    def _get_images(self, identity):
        identity_path = os.path.join(self.root_dir, identity)
        images = [os.path.join(identity_path, f) for f in os.listdir(identity_path) if f.endswith(".jpg")]
        return images

    def __len__(self):
        return len(self.identities) * 10  # sampling multiplier

    def __getitem__(self, idx):
        anchor_id = random.choice(self.identities)
        positive_imgs = self.image_paths[anchor_id]
        if len(positive_imgs) < 2:
            return self.__getitem__(idx)  # skip identities with < 2 images

        anchor_img = random.choice(positive_imgs)
        positive_img = random.choice([img for img in positive_imgs if img != anchor_img])

        negative_id = random.choice([i for i in self.identities if i != anchor_id])
        negative_img = random.choice(self.image_paths[negative_id])

        anchor = Image.open(anchor_img).convert("RGB")
        positive = Image.open(positive_img).convert("RGB")
        negative = Image.open(negative_img).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative
