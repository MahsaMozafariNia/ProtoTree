"""
Fine-tune a pruned-and-projected ProtoTree so its routing matches the projected prototypes again.

Projection replaces every prototype with its nearest training patch, which shifts all
min-distances, but the per-prototype margins (the routing thresholds) stay tuned to the old
prototypes. With a low temperature almost every image then takes the same path. This script:

  1. Recalibrates the margins (skip with --no_recalibrate): for each prototype, the margin is
     set so that the same fraction of training images routes right as with the pruned model
     right before projection (checkpoints/pruned).
  2. Fine-tunes only the margins (AdamW, no lr decay) and the leaves (derivative-free update, as
     in prototree/train.py). Backbone, add-on layers and prototypes stay frozen (and in eval
     mode), so the prototypes remain real training patches. The temperature is annealed from
     --temp_start down to the model's own training temperature, so gradient reaches the margins.
  3. Keeps the model with the best validation accuracy (only at the final temperature; the
     recalibrated model is a candidate too) and reports its test accuracy and MAE.

Output goes to <run_dir>/finetune/run_N (log.txt, log_finetune.csv, checkpoints/).

Usage:
    python finetune_projected.py --run_dir ./runs/prototree_face/run_N [--epochs 20] [--lr_margin 1e-3]

FACE_CSV_DIR / FACE_DATA_ROOT env vars (or --csv_dir / --data_root) override the run args, as in training.
"""
import argparse
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from util.args import load_args, save_args
from util.data import get_dataloaders
from util.log import Log, get_run_log_dir
from prototree.test import eval, eval_mae
from prototree.train import leaf_targets


def effective_margin(tree) -> torch.Tensor:
    # Same transformation as ProtoTree.forward: the margin actually used for routing
    delta_min = 1e-4
    return F.relu(tree.prototype_margin.detach() - delta_min) + delta_min


@torch.no_grad()
def collect_min_distances(tree, loader, device) -> torch.Tensor:
    tree.eval()
    min_distances = []
    for xs, _ in loader:
        _, info = tree.forward(xs.to(device))
        min_distances.append(info['min_distances'].cpu())
    return torch.cat(min_distances, dim=0)  # (num_images, num_prototypes)


@torch.no_grad()
def recalibrate_margins(tree, pruned_tree, loader, device, log: Log):
    """
    An image routes right at a prototype's node when its min-distance is below the margin.
    For each prototype, set the new margin to the quantile of the new (projected) distances that
    sends the same fraction of images right as the pruned model did before projection.
    """
    d_old = collect_min_distances(pruned_tree, loader, device)
    d_new = collect_min_distances(tree, loader, device)
    m_old = effective_margin(pruned_tree).cpu()
    m_proj = effective_margin(tree).cpu()

    frac_right_old = (d_old < m_old.view(1, -1)).float().mean(dim=0)
    frac_right_proj = (d_new < m_proj.view(1, -1)).float().mean(dim=0)
    m_new = torch.stack([torch.quantile(d_new[:, j], frac_right_old[j].item())
                         for j in range(d_new.shape[1])])
    tree.prototype_margin.data.copy_(m_new.to(tree.prototype_margin.device))

    msg = (f"Recalibration over {d_new.shape[0]} training images:\n"
           f"  min-distance mean before projection={d_old.mean():.4f}, after projection={d_new.mean():.4f}\n"
           f"  margin mean before={m_proj.mean():.4f}, after recalibration={m_new.mean():.4f}\n"
           f"  fraction routed right (mean over prototypes): pruned={frac_right_old.mean():.4f}, "
           f"projected before recalibration={frac_right_proj.mean():.4f}")
    print(msg, flush=True)
    log.log_message(msg)


def finetune_epoch(tree, train_loader, optimizer, device, train_leaves: bool) -> dict:
    # eval() for the whole tree: the frozen backbone's BatchNorm statistics must not change
    tree.eval()
    nr_batches = float(len(train_loader))
    class_targets = leaf_targets(tree, train_loader, device)
    with torch.no_grad():
        _old_dist_params = {leaf: leaf._dist_params.detach().clone() for leaf in tree.leaves}

    total_loss, total_acc = 0., 0.
    for xs, ys in train_loader:
        xs, ys = xs.to(device), ys.to(device)
        optimizer.zero_grad()
        ys_pred, info = tree.forward(xs)
        if tree._log_probabilities:
            loss = F.nll_loss(ys_pred, ys)
        else:
            loss = F.nll_loss(torch.log(torch.clamp(ys_pred, min=1e-6)), ys)
        loss.backward()
        optimizer.step()

        if train_leaves:
            # Derivative-free leaf update, same as prototree/train.py:train_epoch
            with torch.no_grad():
                target = class_targets[ys]
                ys_pred_safe = torch.clamp(ys_pred, min=1e-6)
                for leaf in tree.leaves:
                    if tree._log_probabilities:
                        update = torch.exp(torch.logsumexp(info['pa_tensor'][leaf.index] + leaf.distribution() + torch.log(target) - ys_pred, dim=0))
                    else:
                        update = torch.sum((info['pa_tensor'][leaf.index] * leaf.distribution() * target)/ys_pred_safe, dim=0)
                    leaf._dist_params -= (_old_dist_params[leaf]/nr_batches)
                    F.relu_(leaf._dist_params)
                    leaf._dist_params += update

        total_loss += loss.item()
        total_acc += torch.eq(torch.argmax(ys_pred, dim=1), ys).float().mean().item()

    return {'loss': total_loss / nr_batches, 'train_accuracy': total_acc / nr_batches}


def temperature_at(epoch: int, temp_start: float, temp_end: float, anneal_epochs: int) -> float:
    # Geometric anneal from temp_start (epoch 1) to temp_end (epoch anneal_epochs), then constant
    if epoch >= anneal_epochs:
        return temp_end
    t = (epoch - 1) / max(1, anneal_epochs - 1)
    return temp_start * (temp_end / temp_start) ** t


def evaluate(tree, loader, name, device, log: Log) -> dict:
    acc = eval(tree, loader, name, device, log)['test_accuracy']
    mae = eval_mae(tree, loader, device, log)
    return {'acc': acc, 'hard_mae': mae['hard_mae'], 'soft_mae': mae['soft_mae']}


def fmt(r: dict) -> str:
    return f"acc={r['acc']:.4f}, hard_mae={r['hard_mae']:.2f}yr, soft_mae={r['soft_mae']:.2f}yr"


def main():
    parser = argparse.ArgumentParser(description='Recalibrate and fine-tune the margins and leaves of a projected ProtoTree')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Run directory created by main_tree.py (contains metadata/ and checkpoints/)')
    parser.add_argument('--checkpoint', type=str, default='pruned_and_projected',
                        help='Checkpoint subfolder to fine-tune')
    parser.add_argument('--reference_checkpoint', type=str, default='pruned',
                        help='Checkpoint before projection, used for margin recalibration')
    parser.add_argument('--no_recalibrate', action='store_true', help='Skip margin recalibration')
    parser.add_argument('--epochs', type=int, default=20, help='Fine-tuning epochs (0 = recalibrate only)')
    parser.add_argument('--anneal_epochs', type=int, default=10,
                        help='Epochs over which the temperature goes from --temp_start to the final temperature')
    parser.add_argument('--temp_start', type=float, default=0.05, help='Temperature at the first fine-tuning epoch')
    parser.add_argument('--temp_end', type=float, default=None,
                        help='Final temperature (default: the temperature stored in the checkpoint)')
    parser.add_argument('--lr_margin', type=float, default=1e-3, help='Learning rate for the margins (no decay)')
    parser.add_argument('--freeze_leaves', action='store_true', help='Fine-tune only the margins, not the leaves')
    parser.add_argument('--csv_dir', type=str, default=None,
                        help='Folder with train_set.csv/valid_set.csv/test_set.csv (default: FACE_CSV_DIR env var, else the run args)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='UTKFace image folder (default: FACE_DATA_ROOT env var, else the run args)')
    parser.add_argument('--disable_cuda', action='store_true', help='Run on CPU')
    cli = parser.parse_args()

    args = load_args(os.path.join(cli.run_dir, 'metadata'))
    if args.dataset != 'face_dataset':
        raise ValueError(f"This script expects face_dataset, this run used '{args.dataset}'")
    args.disable_cuda = cli.disable_cuda or args.disable_cuda

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # get_data reads FACE_CSV_DIR / FACE_DATA_ROOT before the args values, so set them here
    if cli.csv_dir is not None:
        os.environ['FACE_CSV_DIR'] = cli.csv_dir
    if cli.data_root is not None:
        os.environ['FACE_DATA_ROOT'] = cli.data_root
    csv_src = 'FACE_CSV_DIR' if 'FACE_CSV_DIR' in os.environ else 'run args'
    root_src = 'FACE_DATA_ROOT' if 'FACE_DATA_ROOT' in os.environ else 'run args'
    csv_dir = os.environ.get('FACE_CSV_DIR', args.face_csv_dir)
    data_root = os.environ.get('FACE_DATA_ROOT', args.face_data_root)

    log = Log(get_run_log_dir(os.path.join(cli.run_dir, 'finetune')))
    save_args(cli, log.metadata_dir)
    log.create_log('log_finetune', 'epoch', 'temp', 'train_acc', 'train_loss', 'val_acc', 'val_hard_mae', 'val_soft_mae')
    header = (f"Log dir:   {log.log_dir}\n"
              f"CSV dir:   {csv_dir}   (from {csv_src})\n"
              f"Data root: {data_root}   (from {root_src})")
    print(header, flush=True)
    log.log_message(header)

    device = torch.device('cuda') if not args.disable_cuda and torch.cuda.is_available() else torch.device('cpu')
    trainloader, projectloader, validloader, testloader, classes, _ = get_dataloaders(args)

    model_path = os.path.join(cli.run_dir, 'checkpoints', cli.checkpoint, 'model.pth')
    tree = torch.load(model_path, map_location=device).to(device)
    temp_end = cli.temp_end if cli.temp_end is not None else tree.prototype_temp
    tree.prototype_temp = temp_end
    print(f"Loaded {model_path} ({tree.num_branches} branches, {tree.num_leaves} leaves, temp={temp_end})", flush=True)

    # Baseline: the projected model as saved
    projected_test = evaluate(tree, testloader, 'projected', device, log)
    print(f"Projected (as saved) test: {fmt(projected_test)}", flush=True)

    if not cli.no_recalibrate:
        ref_path = os.path.join(cli.run_dir, 'checkpoints', cli.reference_checkpoint, 'model.pth')
        pruned_tree = torch.load(ref_path, map_location=device).to(device)
        pruned_tree.prototype_temp = temp_end
        if pruned_tree.num_prototypes != tree.num_prototypes:
            raise ValueError(f"{ref_path} has {pruned_tree.num_prototypes} prototypes, {model_path} has {tree.num_prototypes}")
        print(f"Recalibrating margins against {ref_path}", flush=True)
        recalibrate_margins(tree, pruned_tree, projectloader, device, log)
        del pruned_tree
        torch.cuda.empty_cache()
        tree.save(f'{log.checkpoint_dir}/recalibrated')

    # Epoch 0 = starting point (recalibrated, or as projected), a candidate for the best model
    val = evaluate(tree, validloader, 0, device, log)
    log.log_values('log_finetune', 0, temp_end, 'n.a.', 'n.a.', val['acc'], val['hard_mae'], val['soft_mae'])
    print(f"Epoch 0 ({'recalibrated' if not cli.no_recalibrate else 'projected'}): val {fmt(val)}", flush=True)
    best_val_acc, best_epoch, best_tree = val['acc'], 0, deepcopy(tree)

    # Only the margins get gradients; leaves are updated derivative-free
    for param in tree.parameters():
        param.requires_grad = False
    tree.prototype_margin.requires_grad = True
    optimizer = torch.optim.AdamW([tree.prototype_margin], lr=cli.lr_margin, weight_decay=0.)

    for epoch in range(1, cli.epochs + 1):
        tree.prototype_temp = temperature_at(epoch, cli.temp_start, temp_end, cli.anneal_epochs)
        train_info = finetune_epoch(tree, trainloader, optimizer, device, not cli.freeze_leaves)
        val = evaluate(tree, validloader, epoch, device, log)
        log.log_values('log_finetune', epoch, tree.prototype_temp, train_info['train_accuracy'], train_info['loss'],
                       val['acc'], val['hard_mae'], val['soft_mae'])
        m = effective_margin(tree)
        print(f"Epoch {epoch}: temp={tree.prototype_temp:.5f}, train_acc={train_info['train_accuracy']:.4f}, "
              f"val {fmt(val)}, margin mean={m.mean():.4f} min={m.min():.4f} max={m.max():.4f}", flush=True)

        # Select only at the final temperature, so the saved model routes like the trained one
        if tree.prototype_temp == temp_end and val['acc'] > best_val_acc:
            best_val_acc, best_epoch, best_tree = val['acc'], epoch, deepcopy(tree)

    best_tree.save(f'{log.checkpoint_dir}/finetuned_best')
    best_test = evaluate(best_tree, testloader, 'finetuned_best', device, log)
    summary = (f"\nBest model: epoch {best_epoch} (val acc={best_val_acc:.4f}), "
               f"saved to {log.checkpoint_dir}/finetuned_best\n"
               f"Projected (as saved) test: {fmt(projected_test)}\n"
               f"Fine-tuned best      test: {fmt(best_test)}")
    print(summary, flush=True)
    log.log_message(summary)


if __name__ == '__main__':
    main()
