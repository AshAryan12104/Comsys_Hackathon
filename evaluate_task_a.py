# evaluate_task_a.py
from evaluate import evaluate
import os

if __name__ == "__main__":
    test_path_file = "test_path.txt"

    if os.path.exists(test_path_file):
        with open(test_path_file, "r") as f:
            path = f.read().strip()
            print("[DEBUG] Running evaluation with path:", path)
            os.environ["OVERRIDE_VAL_PATH"] = path  # Pass val path via env
    else:
        print("[DEBUG] test_path.txt not found. Using default val path.")

    evaluate({
        'project': {
            'seed': 42
        },
        'dataset': {
            'root': 'data/Task_A',
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
