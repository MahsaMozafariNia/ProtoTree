import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from prototree.prototree import ProtoTree
from util.log import Log
from features.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet50_features_inat, resnet101_features, resnet152_features
from features.densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from features.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,vgg19_features, vgg19_bn_features

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet50_inat': resnet50_features_inat,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}

"""
    Create network with pretrained features and 1x1 convolutional layer

"""
FACE_PRETRAINED_PATH = './pretrained_models/resnet_baseline_face_dataset_size224_lr0.0001_ep50_powerful_augument_False.pth'

def _load_face_pretrained(features: nn.Module):
    """
    Loads resnet50 weights fine-tuned on face_dataset (saved by ga-reg's train_baseline.py) into
    a ProtoTree ResNet_features backbone. The checkpoint wraps the resnet50 as
    BaselineRegressionModel.model (keys prefixed "model.") and carries its own regression head
    (keys prefixed "model.fc.") which ProtoTree's feature extractor doesn't have, so both the
    prefix and the head are stripped before loading.
    """
    checkpoint = torch.load(FACE_PRETRAINED_PATH, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    state_dict = {k[len('model.'):]: v for k, v in state_dict.items()
                  if k.startswith('model.') and not k.startswith('model.fc.')}
    features.load_state_dict(state_dict, strict=False)

def get_network(num_in_channels: int, args: argparse.Namespace):
    # Define a conv net for estimating the probabilities at each decision node
    if args.pretrained_face:
        features = base_architecture_to_features[args.net](pretrained=False)
        _load_face_pretrained(features)
    else:
        features = base_architecture_to_features[args.net](pretrained=not args.disable_pretrained)
    features_name = str(features).upper()
    if features_name.startswith('VGG') or features_name.startswith('RES'):
        first_add_on_layer_in_channels = \
            [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
    elif features_name.startswith('DENSE'):
        first_add_on_layer_in_channels = \
            [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
    else:
        raise Exception('other base base_architecture NOT implemented')
    
    add_on_layers = nn.Sequential(
                    nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=args.num_features, kernel_size=1, bias=False),
                    nn.Dropout2d(p=args.dropout),
                    nn.Sigmoid()
                    )
    return features, add_on_layers

def freeze(tree: ProtoTree, epoch: int, params_to_freeze: list, params_to_train: list, args: argparse.Namespace, log: Log):
    if args.freeze_epochs>0:
        if epoch == 1:
            log.log_message("\nNetwork frozen")
            for parameter in params_to_freeze:
                parameter.requires_grad = False
        elif epoch == args.freeze_epochs + 1:
            log.log_message("\nNetwork unfrozen")
            for parameter in params_to_freeze:
                parameter.requires_grad = True

def _get_annealed_value(start_val: float, end_val: float, epoch: int, args: argparse.Namespace) -> float:
    """Geometric interpolation from start_val to end_val between anneal_start_frac and anneal_end_frac epochs."""
    a_start = 0 if getattr(args, 'temp_drop', False) else max(1, int(round(args.anneal_start_frac * args.epochs)))
    a_end   = max(a_start + 1, int(round(args.anneal_end_frac * args.epochs)))
    if epoch <= a_start:
        return start_val
    elif epoch >= a_end:
        return end_val
    else:
        frac = (epoch - a_start) / float(a_end - a_start)
        return start_val * (end_val / start_val) ** frac


def update_temperature(tree: ProtoTree, epoch: int, args: argparse.Namespace, log: Log):
    """
    Geometric temperature schedule matching the Gaussian Regression Tree reference.
    Holds prototype_temp until a_start, decays smoothly to target_temp by a_end, then holds.
    Falls back to a single step-drop at temp_start_epoch if anneal_start_frac is not set.
    """
    if hasattr(args, 'anneal_start_frac') and args.anneal_start_frac >= 0:
        desired_temp = _get_annealed_value(args.prototype_temp, args.target_temp, epoch, args)
        if abs(desired_temp - tree.prototype_temp) > 1e-12:
            log.log_message(f"\nprototype_temp {tree.prototype_temp:.5f} -> {desired_temp:.5f}")
        tree.prototype_temp = desired_temp
    elif args.temp_start_epoch >= 0 and epoch == args.temp_start_epoch:
        tree.prototype_temp = args.target_temp
        log.log_message("\nprototype_temp dropped from %s to target_temp=%s"%(str(args.prototype_temp), str(args.target_temp)))


def get_label_smoothing(epoch: int, args: argparse.Namespace) -> float:
    """Anneals label_smoothing from args.label_smoothing down to args.label_smoothing_end.
    Uses the same anneal_start_frac/anneal_end_frac schedule as temperature.
    Falls back to static args.label_smoothing if label_smoothing_end is not set or anneal_start_frac < 0.
    """
    start = getattr(args, 'label_smoothing', 0.0)
    end   = getattr(args, 'label_smoothing_end', -1.0)
    if end < 0 or not (hasattr(args, 'anneal_start_frac') and args.anneal_start_frac >= 0):
        return start
    return _get_annealed_value(start, end, epoch, args)

