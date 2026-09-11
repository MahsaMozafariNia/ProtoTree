import torch
import os
import argparse
from prototree.prototree import ProtoTree
from util.log import Log

def _save(tree: ProtoTree, optimizer, scheduler, directory: str):
    tree.eval()
    tree.save(directory)
    tree.save_state(directory)
    torch.save(optimizer.state_dict(), f'{directory}/optimizer_state.pth')
    torch.save(scheduler.state_dict(), f'{directory}/scheduler_state.pth')

def save_last_epoch(tree: ProtoTree, optimizer, scheduler, log: Log):
    # Overwritten every epoch, so a killed/crashed run can still resume from its most recent epoch
    _save(tree, optimizer, scheduler, f'{log.checkpoint_dir}/last_epoch')

def save_half_epoch(tree: ProtoTree, optimizer, scheduler, log: Log):
    _save(tree, optimizer, scheduler, f'{log.checkpoint_dir}/half_epoch')

def save_best_valid_tree(tree: ProtoTree, optimizer, scheduler, best_valid_acc: float, valid_acc: float, log: Log):
    tree.eval()
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        _save(tree, optimizer, scheduler, f'{log.checkpoint_dir}/best_valid_model')
    return best_valid_acc

def save_tree_description(tree: ProtoTree, optimizer, scheduler, description: str, log: Log):
    _save(tree, optimizer, scheduler, f'{log.checkpoint_dir}/{description}')
