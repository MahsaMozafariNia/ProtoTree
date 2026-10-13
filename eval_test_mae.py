"""
Evaluate a saved ProtoTree checkpoint on the face_dataset test set and print hard/soft MAE.

Rebuilds the test loader from the run's own saved args (metadata/args.pickle), so the image
transforms and batch size match training, reads the splits from --csv_dir (default
./protopairnet_csv_dataset; the age bins come from its train_set.csv, as in training), then
reuses prototree.test.eval_mae -- the same function main_tree.py uses for val/test MAE.

Usage:
    python eval_test_mae.py --run_dir ./runs/prototree_face/run_N [--checkpoint pruned_and_projected]

FACE_DATA_ROOT env var (or --data_root) overrides the image folder stored in args, as in training.
"""
import argparse
import os

import torch

from util.args import load_args
from util.data import get_dataloaders
from prototree.test import eval_mae


def main():
    parser = argparse.ArgumentParser(description='Test-set hard/soft MAE for a trained ProtoTree')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Run directory created by main_tree.py (contains metadata/ and checkpoints/)')
    parser.add_argument('--checkpoint', type=str, default='pruned_and_projected',
                        help='Checkpoint subfolder under <run_dir>/checkpoints, e.g. pruned_and_projected '
                             '(final saved model), pruned, best_valid_model, last_epoch')
    parser.add_argument('--csv_dir', type=str, default='./protopairnet_csv_dataset',
                        help='Folder with train_set.csv/valid_set.csv/test_set.csv')
    parser.add_argument('--data_root', type=str, default=None,
                        help='UTKFace image folder (default: FACE_DATA_ROOT env var, else the run args)')
    parser.add_argument('--disable_cuda', action='store_true', help='Run on CPU')
    cli = parser.parse_args()

    args = load_args(os.path.join(cli.run_dir, 'metadata'))
    if args.dataset != 'face_dataset':
        raise ValueError(f"MAE is only defined for face_dataset, this run used '{args.dataset}'")
    args.disable_cuda = cli.disable_cuda or args.disable_cuda

    # get_data reads FACE_CSV_DIR / FACE_DATA_ROOT before the args values, so set them here
    os.environ['FACE_CSV_DIR'] = cli.csv_dir
    if cli.data_root is not None:
        os.environ['FACE_DATA_ROOT'] = cli.data_root

    device = torch.device('cuda') if not args.disable_cuda and torch.cuda.is_available() else torch.device('cpu')

    _, _, _, testloader, classes, _ = get_dataloaders(args)

    model_path = os.path.join(cli.run_dir, 'checkpoints', cli.checkpoint, 'model.pth')
    tree = torch.load(model_path, map_location=device)

    info = eval_mae(tree, testloader, device)
    print(f"Checkpoint: {model_path}")
    print(f"Test set: {cli.csv_dir}/test_set.csv, {len(testloader.dataset)} samples, {len(classes)} classes")
    print(f"Test hard MAE: {info['hard_mae']:.4f} years")
    print(f"Test soft MAE: {info['soft_mae']:.4f} years")


if __name__ == '__main__':
    main()
