
import numpy as np
import argparse
import os
import torch
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Normalize, Compose, Lambda
import pandas as pd
from PIL import Image


def get_data(args: argparse.Namespace):
    """
    Load the proper dataset based on the parsed arguments
    :param args: The arguments in which is specified which dataset should be used
    :return: a 5-tuple consisting of:
                - The train data set
                - The project data set (usually train data set without augmentation)
                - The test data set
                - a tuple containing all possible class labels
                - a tuple containing the shape (depth, width, height) of the input images
    """
    if args.dataset =='CUB-200-2011':
        return get_birds(True, './data/CUB_200_2011/dataset/train_corners', './data/CUB_200_2011/dataset/train_crop', './data/CUB_200_2011/dataset/test_full')
    if args.dataset == 'CARS':
        return get_cars(True, './data/cars/dataset/train', './data/cars/dataset/train', './data/cars/dataset/test')
    if args.dataset == 'face_dataset':
        csv_dir = os.environ.get('FACE_CSV_DIR', args.face_csv_dir)
        data_root = os.environ.get('FACE_DATA_ROOT', args.face_data_root)
        return get_faces(args, data_root,
                         f'{csv_dir}/train_set.csv',
                         f'{csv_dir}/train_set.csv',
                         f'{csv_dir}/test_set.csv')
    raise Exception(f'Could not load data set "{args.dataset}"!')

def get_dataloaders(args: argparse.Namespace):
    """
    Get data loaders
    """
    # Obtain the dataset
    trainset, projectset, testset, classes, shape  = get_data(args)
    c, w, h = shape
    # Determine if GPU should be used
    cuda = not args.disable_cuda and torch.cuda.is_available()
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              pin_memory=cuda
                                              )
    projectloader = torch.utils.data.DataLoader(projectset,
                                            #    batch_size=args.batch_size,
                                              batch_size=int(args.batch_size/4), #make batch size smaller to prevent out of memory errors during projection
                                              shuffle=False,
                                              pin_memory=cuda
                                              )
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=cuda
                                             )
    print("Num classes (k) = ", len(classes), flush=True)
    return trainloader, projectloader, testloader, classes, c


def get_birds(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size = 224): 
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])
    if augment:
        transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.RandomOrder([
            transforms.RandomPerspective(distortion_scale=0.2, p = 0.5),
            transforms.ColorJitter((0.6,1.4), (0.6,1.4), (0.6,1.4), (-0.02,0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, shear=(-2,2),translate=[0.05,0.05]),
            ]),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transform_no_augment

    trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform_no_augment)
    testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
    classes = trainset.classes
    for i in range(len(classes)):
        classes[i]=classes[i].split('.')[1]
    return trainset, projectset, testset, classes, shape


def get_cars(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size = 224): 
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])

    if augment:
        transform = transforms.Compose([
            transforms.Resize(size=(img_size+32, img_size+32)), #resize to 256x256
            transforms.RandomOrder([
            transforms.RandomPerspective(distortion_scale=0.5, p = 0.5),
            transforms.ColorJitter((0.6,1.4), (0.6,1.4), (0.6,1.4), (-0.4,0.4)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15,shear=(-2,2)),
            ]),
            transforms.RandomCrop(size=(img_size, img_size)), #crop to 224x224
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transform_no_augment

    trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform_no_augment)
    testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
    classes = trainset.classes

    return trainset, projectset, testset, classes, shape


def _percentile_bin_edges(ages: np.ndarray, num_bins: int) -> np.ndarray:
    """
    Compute `num_bins` age bins from percentiles of `ages`, so that (as closely as the data
    allows) each bin holds an equal share of the samples. The outer edges are extended to
    +/- inf so that ages outside the range seen here (e.g. in a held-out split) still fall
    into the nearest bin instead of being dropped.
    """
    percentiles = np.linspace(0, 100, num_bins + 1)
    edges = np.unique(np.percentile(ages, percentiles))
    if len(edges) - 1 < num_bins:
        raise ValueError(
            f"Could only form {len(edges) - 1} distinct percentile bins for the requested "
            f"{num_bins} (= 2^depth) bins; the age values have too many ties. Reduce --depth."
        )
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


class UTKFaceBinned(torch.utils.data.Dataset):
    """
    UTKFace-style face dataset (image_name, age, ethnicity, gender columns) that turns the
    continuous age into a classification label by assigning it to one of `len(bin_edges) - 1`
    percentile-based age bins, so ProtoTree's classification pipeline can be used unchanged.
    """

    def __init__(self, root_dir: str, csv_file: str, bin_edges: np.ndarray, transform):
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_file)
        self.bin_edges = bin_edges
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(os.path.join(self.root_dir, row['image_name'])).convert('RGB')
        image = self.transform(image)
        # Internal edges only (drop the -inf/+inf outer edges) so digitize returns 0..num_bins-1
        label = int(np.digitize([float(row['age'])], self.bin_edges[1:-1], right=False)[0])
        return image, label


def get_faces(args, data_root: str, csv_file_train: str, csv_file_project: str, csv_file_test: str, img_size=224):
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.RandomOrder([
        transforms.RandomPerspective(distortion_scale=0.2, p = 0.5),
        transforms.ColorJitter((0.6,1.4), (0.6,1.4), (0.6,1.4), (-0.02,0.02)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, shear=(-2,2),translate=[0.05,0.05]),
        ]),
        transforms.ToTensor(),
        normalize,
    ])

    # Number of age bins matches the number of leaves in the (complete) tree of this depth,
    # so every leaf is naturally responsible for one percentile-based age bracket.
    num_bins = 2 ** args.depth
    train_ages = pd.read_csv(csv_file_train)['age'].astype(float).values
    bin_edges = _percentile_bin_edges(train_ages, num_bins)

    trainset = UTKFaceBinned(data_root, csv_file_train, bin_edges, transform)
    projectset = UTKFaceBinned(data_root, csv_file_project, bin_edges, transform_no_augment)
    testset = UTKFaceBinned(data_root, csv_file_test, bin_edges, transform_no_augment)

    classes = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if not np.isfinite(lo):
            classes.append(f'<{hi:.0f}')
        elif not np.isfinite(hi):
            classes.append(f'>={lo:.0f}')
        else:
            classes.append(f'{lo:.0f}-{hi:.0f}')

    return trainset, projectset, testset, classes, shape

