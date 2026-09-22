from prototree.prototree import ProtoTree
from util.log import Log, get_run_log_dir

from util.args import get_args, save_args, get_optimizer
from util.data import get_dataloaders
from util.init import init_tree
from util.net import get_network, freeze, update_temperature
from util.visualize import gen_vis
from util.analyse import *
from util.save import *
from prototree.train import train_epoch, train_epoch_kontschieder
from prototree.test import eval, eval_fidelity, eval_mae
from prototree.prune import prune
from prototree.project import project, project_with_class_constraints
from prototree.upsample import upsample

import os
import random
import numpy as np
import torch
from shutil import copy
from copy import deepcopy

def run_tree(args=None):
    args = args or get_args()
    # Seed everything before any randomness (prototype init, augmentation, batch shuffling,
    # dropout) happens, so runs with the same arguments are reproducible.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Create a logger, in a fresh run_N subdirectory of args.log_dir so repeated runs don't overwrite each other
    args.log_dir = get_run_log_dir(args.log_dir)
    log = Log(args.log_dir)
    print("Log dir: ", args.log_dir, flush=True)
    # Create a csv log for storing the validation accuracy/MAE, mean train accuracy and mean loss for each epoch
    # ("val" here is the test set, doubling as the held-out signal used to pick the best checkpoint -- see below)
    log.create_log('log_epoch_overview', 'epoch', 'val_acc', 'mean_train_acc', 'mean_train_crossentropy_loss_during_epoch', 'val_hard_mae', 'val_soft_mae')
    # Log the run arguments
    save_args(args, log.metadata_dir)
    print("Args:", flush=True)
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}", flush=True)
    if not args.disable_cuda and torch.cuda.is_available():
        # device = torch.device('cuda')
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')
        
    # Log which device was actually used
    log.log_message('Device used: '+str(device))

    # Create a log for logging the loss values
    log_prefix = 'log_train_epochs'
    log_loss = log_prefix+'_losses'
    log.create_log(log_loss, 'epoch', 'batch', 'loss', 'batch_train_acc')

    # Obtain the dataset and dataloaders
    trainloader, projectloader, testloader, classes, num_channels = get_dataloaders(args)
    print('data is loaded', flush=True)
    # Create a convolutional network based on arguments and add 1x1 conv layer
    features_net, add_on_layers = get_network(num_channels, args)
    print('network is loaded', flush=True)
    # Create a ProtoTree
    tree = ProtoTree(num_classes=len(classes),
                    feature_net = features_net,
                    args = args,
                    add_on_layers = add_on_layers)
    tree = tree.to(device=device)
    # Determine which optimizer should be used to update the tree parameters
    optimizer, params_to_freeze, params_to_train = get_optimizer(tree, args)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=args.gamma)
    tree, epoch = init_tree(tree, optimizer, scheduler, device, args)
    
    tree.save(f'{log.checkpoint_dir}/tree_init')
    log.log_message("Max depth %s, so %s internal nodes and %s leaves"%(args.depth, tree.num_branches, tree.num_leaves))
    analyse_output_shape(tree, trainloader, log, device)

    leaf_labels = dict()
    best_train_acc = 0.
    # No separate held-out validation split exists in this pipeline, so the test set doubles
    # as the validation signal used to pick the "best" checkpoint during training.
    best_valid_acc = 0.
    half_epoch = args.epochs // 2


    print("Delta Requires grad:", tree.prototype_margin.requires_grad)
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p is tree.prototype_margin:
                print("Found delta in optimizer")


    with torch.no_grad():
        if args.delta_initialized_value == 0:
            xs0, _, _, _ = next(iter(trainloader))
            _, info = tree.forward(xs0.to(device))
            mean_d = info['min_distances'].mean(dim=0)  # (P,)
            tree.prototype_margin.copy_(mean_d.detach())
            del info
        else:
            tree.prototype_margin.copy_(torch.full_like(tree.prototype_margin, args.delta_initialized_value))
        
    torch.cuda.empty_cache()

    if epoch < args.epochs + 1:
        '''
            TRAIN AND EVALUATE TREE
        '''
        for epoch in range(epoch, args.epochs + 1):
            log.log_message("\nEpoch %s"%str(epoch))
            # Freeze (part of) network for some epochs if indicated in args
            freeze(tree, epoch, params_to_freeze, params_to_train, args, log)
            update_temperature(tree, epoch, args, log)
            log_learning_rates(optimizer, args, log)

            # Train tree
            if tree._kontschieder_train:
                train_info = train_epoch_kontschieder(tree, trainloader, optimizer, epoch, args.disable_derivative_free_leaf_optim, device, log, log_prefix)
            else:
                train_info = train_epoch(tree, trainloader, optimizer, epoch, args.disable_derivative_free_leaf_optim, device, log, log_prefix)
            best_train_acc = max(best_train_acc, train_info['train_accuracy'])
            save_last_epoch(tree, optimizer, scheduler, log)
            if epoch == half_epoch:
                save_half_epoch(tree, optimizer, scheduler, log)
            leaf_labels = analyse_leafs(tree, epoch, len(classes), leaf_labels, args.pruning_threshold_leaves, log)

            # Evaluate tree every 5 epochs (and always on the last one), regardless of --epochs
            if epoch % 5 == 0 or epoch == args.epochs:
                eval_info = eval(tree, testloader, epoch, device, log)
                original_test_acc = eval_info['test_accuracy']
                best_valid_acc = save_best_valid_tree(tree, optimizer, scheduler, best_valid_acc, eval_info['test_accuracy'], log)
                mae_str = ""
                val_hard_mae, val_soft_mae = "n.a.", "n.a."
                if args.dataset == 'face_dataset':
                    mae_info = eval_mae(tree, testloader, device, log)
                    val_hard_mae, val_soft_mae = mae_info['hard_mae'], mae_info['soft_mae']
                    mae_str = f", val_hard_mae={val_hard_mae:.2f}yr, val_soft_mae={val_soft_mae:.2f}yr"
                log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'], train_info['train_accuracy'], train_info['loss'], val_hard_mae, val_soft_mae)
                print(f"Epoch {epoch}: train_acc={train_info['train_accuracy']:.4f}, val_acc={eval_info['test_accuracy']:.4f}{mae_str}", flush=True)
            else:
                log.log_values('log_epoch_overview', epoch, "n.a.", train_info['train_accuracy'], train_info['loss'], "n.a.", "n.a.")
                print(f"Epoch {epoch}: train_acc={train_info['train_accuracy']:.4f}", flush=True)

            scheduler.step()

    else: #tree was loaded and not trained, so evaluate only
        '''
            EVALUATE TREE
        '''
        eval_info = eval(tree, testloader, epoch, device, log)
        original_test_acc = eval_info['test_accuracy']
        best_valid_acc = save_best_valid_tree(tree, optimizer, scheduler, best_valid_acc, eval_info['test_accuracy'], log)
        val_hard_mae, val_soft_mae = "n.a.", "n.a."
        if args.dataset == 'face_dataset':
            mae_info = eval_mae(tree, testloader, device, log)
            val_hard_mae, val_soft_mae = mae_info['hard_mae'], mae_info['soft_mae']
        log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'], "n.a.", "n.a.", val_hard_mae, val_soft_mae)

    '''
        EVALUATE AND ANALYSE TRAINED TREE
    '''
    log.log_message("Training Finished. Best training accuracy was %s, best valid accuracy was %s\n"%(str(best_train_acc), str(best_valid_acc)))
    trained_tree = deepcopy(tree)
    leaf_labels = analyse_leafs(tree, epoch+1, len(classes), leaf_labels, args.pruning_threshold_leaves, log)
    analyse_leaf_distributions(tree, log)

    '''
        LOAD BEST VALID MODEL BEFORE PRUNING
    '''
    # Prune/project the checkpoint that scored best on the validation signal during training,
    # rather than whatever the last epoch happened to land on.
    best_valid_model_path = f'{log.checkpoint_dir}/best_valid_model/model.pth'
    if os.path.isfile(best_valid_model_path):
        log.log_message(f"Loading best valid model (val_acc={best_valid_acc}) from {best_valid_model_path} for pruning")
        tree = torch.load(best_valid_model_path, map_location=device)
        tree = tree.to(device=device)
    else:
        log.log_message("No best_valid_model checkpoint found; pruning the last-epoch tree instead")

    '''
        PRUNE
    '''
    pruned = prune(tree, args.pruning_threshold_leaves, log)
    name = "pruned"
    save_tree_description(tree, optimizer, scheduler, name, log)
    pruned_tree = deepcopy(tree)
    # Analyse and evaluate pruned tree
    leaf_labels = analyse_leafs(tree, epoch+2, len(classes), leaf_labels, args.pruning_threshold_leaves, log)
    analyse_leaf_distributions(tree, log)
    eval_info = eval(tree, testloader, name, device, log)
    pruned_test_acc = eval_info['test_accuracy']

    pruned_tree = tree

    '''
        PROJECT
    '''
    project_info, tree = project_with_class_constraints(deepcopy(pruned_tree), projectloader, device, args, log)
    name = "pruned_and_projected"
    save_tree_description(tree, optimizer, scheduler, name, log)
    pruned_projected_tree = deepcopy(tree)
    # Analyse and evaluate pruned tree with projected prototypes
    average_distance_nearest_image(project_info, tree, log)
    leaf_labels = analyse_leafs(tree, epoch+3, len(classes), leaf_labels, args.pruning_threshold_leaves, log)
    analyse_leaf_distributions(tree, log)
    eval_info = eval(tree, testloader, name, device, log)
    pruned_projected_test_acc = eval_info['test_accuracy']
    eval_info_samplemax = eval(tree, testloader, name, device, log, 'sample_max')
    get_avg_path_length(tree, eval_info_samplemax, log)
    eval_info_greedy = eval(tree, testloader, name, device, log, 'greedy')
    get_avg_path_length(tree, eval_info_greedy, log)
    fidelity_info = eval_fidelity(tree, testloader, device, log)

    # Upsample prototype for visualization
    project_info = upsample(tree, project_info, projectloader, name, args, log)
    # visualize tree
    gen_vis(tree, name, args, classes)

    
    return trained_tree.to('cpu'), pruned_tree.to('cpu'), pruned_projected_tree.to('cpu'), original_test_acc, pruned_test_acc, pruned_projected_test_acc, project_info, eval_info_samplemax, eval_info_greedy, fidelity_info


if __name__ == '__main__':
    args = get_args()
    run_tree(args)