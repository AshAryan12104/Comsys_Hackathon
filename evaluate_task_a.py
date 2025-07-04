# evaluate_task_a.py
import os
import sys
import argparse
from evaluate import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_dir', type=str, default='data/Task_A/val', help="Path to validation data")
    args = parser.parse_args()

# Validate path
if not os.path.exists(args.val_dir):
    print(f"[ERROR] Provided path '{args.val_dir}' does not exist.")
    sys.exit(1)

# Validate structure: expecting 2 folders (male/female)
subdirs = [d for d in os.listdir(args.val_dir) if os.path.isdir(os.path.join(args.val_dir, d))]
if len(subdirs) != 2:
    print(f"[ERROR] Validation folder must contain exactly 2 class folders (e.g., male/female). Found: {subdirs}")
    sys.exit(1)

# Check at least one image per folder
for sub in subdirs:
    sub_path = os.path.join(args.val_dir, sub)
    imgs = [f for f in os.listdir(sub_path) if f.endswith(('.jpg', '.png'))]
    if not imgs:
        print(f"[ERROR] Folder '{sub}' has no valid images.")
        sys.exit(1)


# Pass the path as part of config
evaluate({
    'project': {
        'seed': 42
    },
    'dataset': {
        'root': 'data/Task_A',        # Root for training (if needed)
        'val_path': args.val_dir,     # 👈 This is used for validation
        'image_size': 224,
        'num_workers': 2
    },
    'model': {
        'backbone': 'resnet18',
        'pretrained': True,
        'num_classes_identity': 2,
        'multitask': False
    },
    'train': {
        'epochs': 10,
        'batch_size': 32,
        'lr': 0.0005,
        'weight_decay': 0.0001
    }
})
