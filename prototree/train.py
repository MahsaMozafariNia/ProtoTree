import argparse
from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader

from prototree.prototree import ProtoTree

from util.log import Log

def _report_nan(name: str, tensor: torch.Tensor, epoch: int, batch: int) -> bool:
    """Prints and returns True the first time `tensor` contains a NaN/Inf, otherwise returns False."""
    nan = torch.isnan(tensor).any().item()
    inf = torch.isinf(tensor).any().item()
    if nan or inf:
        finite = tensor[torch.isfinite(tensor)]
        rng = (finite.min().item(), finite.max().item()) if finite.numel() else ('n/a', 'n/a')
        print(f"[NaN CHECK] epoch={epoch} batch={batch} tensor={name} nan={nan} inf={inf} "
              f"finite_range={rng} shape={tuple(tensor.shape)}", flush=True)
    return nan or inf

def train_epoch(tree: ProtoTree,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                epoch: int,
                disable_derivative_free_leaf_optim: bool,
                device,
                log: Log = None,
                log_prefix: str = 'log_train_epochs',
                progress_prefix: str = 'Train Epoch'
                ) -> dict:
    
    tree = tree.to(device)
    # Make sure the model is in eval mode
    tree.eval()
    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.
    # Create a log if required
    log_loss = f'{log_prefix}_losses'

    nr_batches = float(len(train_loader))
    with torch.no_grad():
        _old_dist_params = dict()
        for leaf in tree.leaves:
            _old_dist_params[leaf] = leaf._dist_params.detach().clone()
        # Optimize class distributions in leafs
        eye = torch.eye(tree._num_classes).to(device)

    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs, ys) in enumerate(train_loader):
        # Make sure the model is in train mode
        tree.train()
        # Reset the gradients
        optimizer.zero_grad()

        xs, ys = xs.to(device), ys.to(device)

        # Perform a forward pass through the network
        ys_pred, info = tree.forward(xs)

        # --- NaN diagnostics: forward pass output, before this batch touches any parameter ---
        _report_nan('ys_pred (post-forward)', ys_pred, epoch, i)
        _report_nan('prototype_margin (pre-update)', tree.prototype_margin, epoch, i)
        _report_nan('prototype_vectors (pre-update)', tree.prototype_layer.prototype_vectors, epoch, i)

        # Learn prototypes and network with gradient descent.
        # If disable_derivative_free_leaf_optim, leaves are optimized with gradient descent as well.
        # Compute the loss
        if tree._log_probabilities:
            loss = F.nll_loss(ys_pred, ys)
        else:
            loss = F.nll_loss(torch.log(ys_pred), ys)

        if _report_nan('loss', loss.detach().view(1), epoch, i):
            print(f"[NaN CHECK] epoch={epoch} batch={i} ys min/max: "
                  f"{ys_pred.min().item():.3e}/{ys_pred.max().item():.3e}", flush=True)

        # Compute the gradient
        loss.backward()

        # --- NaN diagnostics: gradients, before clipping/stepping ---
        grad_norm = torch.nn.utils.clip_grad_norm_(tree.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm) or grad_norm.item() > 50:
            print(f"[NaN CHECK] epoch={epoch} batch={i} grad_norm(pre-clip)={grad_norm.item():.3e}", flush=True)
        if tree.prototype_margin.grad is not None:
            _report_nan('prototype_margin.grad', tree.prototype_margin.grad, epoch, i)
        if tree.prototype_layer.prototype_vectors.grad is not None:
            _report_nan('prototype_vectors.grad', tree.prototype_layer.prototype_vectors.grad, epoch, i)

        # Update model parameters
        optimizer.step()
        # Keep prototypes in the same [0, 1] range as the add-on layer's Sigmoid output
        with torch.no_grad():
            tree.prototype_layer.prototype_vectors.data.clamp_(0.0, 1.0)

        # --- NaN diagnostics: right after this batch's update -- if either fires here, this
        # batch is the one that broke it, and the grad/loss prints above say why ---
        _report_nan('prototype_margin (post-update)', tree.prototype_margin, epoch, i)
        _report_nan('prototype_vectors (post-update)', tree.prototype_layer.prototype_vectors, epoch, i)

        if not disable_derivative_free_leaf_optim:
            #Update leaves with derivate-free algorithm
            #Make sure the tree is in eval mode
            tree.eval()
            with torch.no_grad():
                target = eye[ys] #shape (batchsize, num_classes)
                # Floor ys_pred away from 0 before dividing by it below -- without this, a
                # near-zero predicted probability blows the update up to inf/NaN, which then
                # corrupts leaf._dist_params and, via the next forward/backward pass, prototype_vectors.
                ys_pred_safe = torch.clamp(ys_pred, min=1e-6)
                for leaf in tree.leaves:
                    if tree._log_probabilities:
                        # log version
                        update = torch.exp(torch.logsumexp(info['pa_tensor'][leaf.index] + leaf.distribution() + torch.log(target) - ys_pred, dim=0))
                    else:
                        update = torch.sum((info['pa_tensor'][leaf.index] * leaf.distribution() * target)/ys_pred_safe, dim=0)
                    leaf._dist_params -= (_old_dist_params[leaf]/nr_batches)
                    F.relu_(leaf._dist_params) #dist_params values can get slightly negative because of floating point issues. therefore, set to zero.
                    leaf._dist_params += update
                    _report_nan(f'leaf[{leaf.index}]._dist_params', leaf._dist_params, epoch, i)

        # Count the number of correct classifications
        ys_pred_max = torch.argmax(ys_pred, dim=1)
        
        correct = torch.sum(torch.eq(ys_pred_max, ys))
        acc = correct.item() / float(len(xs))

        # Compute metrics over this batch
        total_loss+=loss.item()
        total_acc+=acc

        if log is not None:
            log.log_values(log_loss, epoch, i + 1, loss.item(), acc)

    train_info['loss'] = total_loss/float(i+1)
    train_info['train_accuracy'] = total_acc/float(i+1)
    return train_info


def train_epoch_kontschieder(tree: ProtoTree,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                epoch: int,
                disable_derivative_free_leaf_optim: bool,
                device,
                log: Log = None,
                log_prefix: str = 'log_train_epochs',
                progress_prefix: str = 'Train Epoch'
                ) -> dict:

    tree = tree.to(device)

    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.

    # Create a log if required
    log_loss = f'{log_prefix}_losses'
    if log is not None and epoch==1:
        log.create_log(log_loss, 'epoch', 'batch', 'loss', 'batch_train_acc')
    
    # Reset the gradients
    optimizer.zero_grad()

    if disable_derivative_free_leaf_optim:
        print("WARNING: kontschieder arguments will be ignored when training leaves with gradient descent")
    else:
        if tree._kontschieder_normalization:
            # Iterate over the dataset multiple times to learn leaves following Kontschieder's approach
            for _ in range(10):
                # Train leaves with derivative-free algorithm using normalization factor
                train_leaves_epoch(tree, train_loader, epoch, device)
        else:
            # Train leaves with Kontschieder's derivative-free algorithm, but using softmax
            train_leaves_epoch(tree, train_loader, epoch, device)
    # Train prototypes and network.
    # If disable_derivative_free_leaf_optim, leafs are optimized with gradient descent as well.
    # Make sure the model is in train mode
    tree.train()
    for i, (xs, ys) in enumerate(train_loader):
        xs, ys = xs.to(device), ys.to(device)

        # Reset the gradients
        optimizer.zero_grad()
        # Perform a forward pass through the network
        ys_pred, _ = tree.forward(xs)
        # Compute the loss
        if tree._log_probabilities:
            loss = F.nll_loss(ys_pred, ys)
        else:
            loss = F.nll_loss(torch.log(ys_pred), ys)
        # Compute the gradient
        loss.backward()
        # Clip gradients so a single bad batch can't push a parameter to inf/NaN
        torch.nn.utils.clip_grad_norm_(tree.parameters(), max_norm=1.0)
        # Update model parameters
        optimizer.step()
        # Keep prototypes in the same [0, 1] range as the add-on layer's Sigmoid output
        with torch.no_grad():
            tree.prototype_layer.prototype_vectors.data.clamp_(0.0, 1.0)

        # Count the number of correct classifications
        ys_pred = torch.argmax(ys_pred, dim=1)
        
        correct = torch.sum(torch.eq(ys_pred, ys))
        acc = correct.item() / float(len(xs))

        # Compute metrics over this batch
        total_loss+=loss.item()
        total_acc+=acc

        if log is not None:
            log.log_values(log_loss, epoch, i + 1, loss.item(), acc)

    train_info['loss'] = total_loss/float(i+1)
    train_info['train_accuracy'] = total_acc/float(i+1)
    return train_info

# Updates leaves with derivative-free algorithm
def train_leaves_epoch(tree: ProtoTree,
                        train_loader: DataLoader,
                        epoch: int,
                        device,
                        progress_prefix: str = 'Train Leafs Epoch'
                        ) -> dict:

    #Make sure the tree is in eval mode for updating leafs
    tree.eval()

    with torch.no_grad():
        _old_dist_params = dict()
        for leaf in tree.leaves:
            _old_dist_params[leaf] = leaf._dist_params.detach().clone()
        # Optimize class distributions in leafs
        eye = torch.eye(tree._num_classes).to(device)

        # Iterate through the data set
        update_sum = dict()

        # Create empty tensor for each leaf that will be filled with new values
        for leaf in tree.leaves:
            update_sum[leaf] = torch.zeros_like(leaf._dist_params)

        for i, (xs, ys) in enumerate(train_loader):
            xs, ys = xs.to(device), ys.to(device)
            #Train leafs without gradient descent
            out, info = tree.forward(xs)
            target = eye[ys] #shape (batchsize, num_classes)
            # See the same guard in train_epoch: floor out away from 0 before dividing by it.
            out_safe = torch.clamp(out, min=1e-6)
            for leaf in tree.leaves:
                if tree._log_probabilities:
                    # log version
                    update = torch.exp(torch.logsumexp(info['pa_tensor'][leaf.index] + leaf.distribution() + torch.log(target) - out, dim=0))
                else:
                    update = torch.sum((info['pa_tensor'][leaf.index] * leaf.distribution() * target)/out_safe, dim=0)
                update_sum[leaf] += update

        for leaf in tree.leaves:
            leaf._dist_params -= leaf._dist_params #set current dist params to zero
            leaf._dist_params += update_sum[leaf] #give dist params new value