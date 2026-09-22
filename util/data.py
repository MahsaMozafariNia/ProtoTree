
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
import random
from PIL import Image


def get_data(args: argparse.Namespace):
    """
    Load the proper dataset based on the parsed arguments
    :param args: The arguments in which is specified which dataset should be used
    :return: a 6-tuple consisting of:
                - The train data set
                - The project data set (usually train data set without augmentation)
                - The validation data set (held out, used for checkpoint selection during training)
                - The test data set (held out, only touched for the final reported numbers)
                - a tuple containing all possible class labels
                - a tuple containing the shape (depth, width, height) of the input images
    """
    if args.dataset =='CUB-200-2011':
        trainset, projectset, testset, classes, shape = get_birds(True, './data/CUB_200_2011/dataset/train_corners', './data/CUB_200_2011/dataset/train_crop', './data/CUB_200_2011/dataset/test_full')
        # No separate validation split exists for this dataset -- the test set doubles as validation,
        # matching this function's historical behaviour before face_dataset got its own valid_set.csv.
        return trainset, projectset, testset, testset, classes, shape
    if args.dataset == 'CARS':
        trainset, projectset, testset, classes, shape = get_cars(True, './data/cars/dataset/train', './data/cars/dataset/train', './data/cars/dataset/test')
        return trainset, projectset, testset, testset, classes, shape
    if args.dataset == 'face_dataset':
        csv_dir = os.environ.get('FACE_CSV_DIR', args.face_csv_dir)
        data_root = os.environ.get('FACE_DATA_ROOT', args.face_data_root)
        return get_faces(args, data_root,
                         f'{csv_dir}/train_set.csv',
                         f'{csv_dir}/train_set.csv',
                         f'{csv_dir}/valid_set.csv',
                         f'{csv_dir}/test_set.csv')
    raise Exception(f'Could not load data set "{args.dataset}"!')

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def get_dataloaders(args: argparse.Namespace):
    """
    Get data loaders
    """
    # Obtain the dataset
    trainset, projectset, validset, testset, classes, shape  = get_data(args)
    c, w, h = shape
    # Determine if GPU should be used
    cuda = not args.disable_cuda and torch.cuda.is_available()

    # Deterministic worker seeding so re-running with the same --seed reproduces the same
    # augmentation/shuffling, including inside DataLoader worker subprocesses.
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader_kwargs = dict(
        pin_memory=cuda,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              **loader_kwargs
                                              )
    projectloader = torch.utils.data.DataLoader(projectset,
                                            #    batch_size=args.batch_size,
                                              batch_size=int(args.batch_size/4), #make batch size smaller to prevent out of memory errors during projection
                                              shuffle=False,
                                              **loader_kwargs
                                              )
    validloader = torch.utils.data.DataLoader(validset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             **loader_kwargs
                                             )
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             **loader_kwargs
                                             )
    print("Num classes (k) = ", len(classes), flush=True)
    return trainloader, projectloader, validloader, testloader, classes, c


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


class UTKFaceBinned(torch.utils.data.Dataset):
    """
    UTKFace-style face dataset (image_name, age, ethnicity, gender columns) that turns the
    continuous age into a classification label by assigning it to one of `len(bin_edges) - 1`
    equal-width age bins, so ProtoTree's classification pipeline can be used unchanged.
    """

    def __init__(self, root_dir: str, csv_file: str, bin_edges: np.ndarray, transform):
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_file)
        self.bin_edges = bin_edges
        self.transform = transform
        # ImageFolder-style (path, label) list, in row order (matches the unshuffled
        # projectloader), for code that expects torchvision.datasets.ImageFolder's .imgs
        # (e.g. prototree/upsample.py indexing project_loader.dataset.imgs[i]).
        self.imgs = [(os.path.join(root_dir, row.image_name), self._label(row.age))
                     for row in self.data.itertuples()]

    def _label(self, age: float) -> int:
        # Internal edges only (drop the -inf/+inf outer edges) so digitize returns 0..num_bins-1
        return int(np.digitize([float(age)], self.bin_edges[1:-1], right=False)[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(os.path.join(self.root_dir, row['image_name'])).convert('RGB')
        image = self.transform(image)
        return image, self._label(row['age'])


def get_faces(args, data_root: str, csv_file_train: str, csv_file_project: str, csv_file_valid: str, csv_file_test: str, img_size=224):
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

    # --double_leaves halves the number of classes (bins) relative to the number of leaves
    # (2^depth), so every bin is reachable through two independent leaves/routing paths.
    num_bins = 2 ** (args.depth - 1) if args.double_leaves else 2 ** args.depth
    train_ages = pd.read_csv(csv_file_train)['age'].astype(float).values
    if args.bin_strategy == 'quantile':
        # Percentile edges of the training ages, so each bin holds roughly num_train/num_bins
        # samples (fixes the severe class imbalance equal-width bins produce, e.g. UTKFace's
        # 25-30 bin outnumbering its 70-75 bin ~14 to 1) -- at the cost of uneven bin widths.
        percentiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.unique(np.percentile(train_ages, percentiles))
        if len(bin_edges) - 1 < num_bins:
            # Integer ages create duplicate percentile edges at fine enough resolution. The
            # model doesn't require num_classes == num_leaves, so proceed with fewer, wider
            # bins rather than erroring out.
            bins_formula = "2^(depth-1)" if args.double_leaves else "2^depth"
            print(f"face_dataset: requested {num_bins} quantile bins ({bins_formula}) but the training "
                  f"ages only support {len(bin_edges) - 1} distinct ones (too many ties); "
                  f"using {len(bin_edges) - 1} classes instead.", flush=True)
    else:
        # One equal-width age bin per leaf: max_age / num_leaves, rounded to whole years.
        max_age = train_ages.max()
        bin_width = round(max_age / num_bins)
        bin_edges = np.array([i * bin_width for i in range(num_bins + 1)], dtype=float)
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf

    trainset = UTKFaceBinned(data_root, csv_file_train, bin_edges, transform)
    projectset = UTKFaceBinned(data_root, csv_file_project, bin_edges, transform_no_augment)
    validset = UTKFaceBinned(data_root, csv_file_valid, bin_edges, transform_no_augment)
    testset = UTKFaceBinned(data_root, csv_file_test, bin_edges, transform_no_augment)

    classes = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if not np.isfinite(lo):
            classes.append(f'<{hi:.0f}')
        elif not np.isfinite(hi):
            classes.append(f'>={lo:.0f}')
        else:
            classes.append(f'{lo:.0f}-{hi:.0f}')

    return trainset, projectset, validset, testset, classes, shape

