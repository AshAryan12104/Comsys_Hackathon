import yaml
import argparse
from train import train
from evaluate import evaluate

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True)
    parser.add_argument('--task', choices=['gender'], default='gender', help='Which task to run')
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.task == 'gender':
        if args.mode == 'train':
            train(config)
        elif args.mode == 'evaluate':
            evaluate(config)
