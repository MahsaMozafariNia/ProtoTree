from util.args import load_args
from util.data import get_dataloaders
from util.visualize_prediction import gen_pred_vis
import argparse
import torch
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from copy import deepcopy
import os

def get_local_expl_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser('Explain a prediction')
    parser.add_argument('--run_dir',
                        type=str,
                        required=True,
                        help='Run directory created by main_tree.py (contains metadata/, checkpoints/ and the upsampled prototypes)')
    parser.add_argument('--checkpoint',
                        type=str,
                        default='pruned_and_projected',
                        help='Checkpoint folder of the model to explain. Either a subfolder name of <run_dir>/checkpoints '
                             '(e.g. pruned_and_projected) or a full path to a checkpoint folder (e.g. <run_dir>/finetune/run_1/checkpoints/finetuned_best)')
    parser.add_argument('--sample_dir',
                        type=str,
                        default=None,
                        help='Directory to image to be explained, or to a folder containing multiple test images')
    parser.add_argument('--num_test_samples',
                        type=int,
                        default=0,
                        help='face_dataset only: explain the first N images of <csv_dir>/test_set.csv instead of --sample_dir')
    parser.add_argument('--csv_dir',
                        type=str,
                        default=None,
                        help='face_dataset: folder with train_set.csv/valid_set.csv/test_set.csv (default: FACE_CSV_DIR env var, else the run args)')
    parser.add_argument('--data_root',
                        type=str,
                        default=None,
                        help='face_dataset: image folder (default: FACE_DATA_ROOT env var, else the run args)')
    parser.add_argument('--results_dir',
                        type=str,
                        default='local_explanations',
                        help='Directory (inside the run directory) where local explanations will be saved')
    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Flag that disables GPU usage if set')
    parser.add_argument('--image_size',
                        type=int,
                        default=224,
                        help='Resize images to this size')
    parser.add_argument('--upsample_threshold',
                        type=float,
                        default=0.98,
                        help='Threshold (between 0 and 1) for visualizing the nearest patch of an image after upsampling. The higher this threshold, the larger the patches.')
    cli = parser.parse_args()
    if cli.sample_dir is None and cli.num_test_samples <= 0:
        parser.error('give either --sample_dir or --num_test_samples')

    # Start from the arguments the tree was trained with, so get_dataloaders builds the same data/bins
    args = load_args(os.path.join(cli.run_dir, 'metadata'))
    args.log_dir = cli.run_dir  # upsampling_results/ and pruned_and_projected/node_vis of the run live here
    args.results_dir = cli.results_dir
    args.image_size = cli.image_size
    args.upsample_threshold = cli.upsample_threshold
    args.disable_cuda = cli.disable_cuda or args.disable_cuda
    args.checkpoint = cli.checkpoint if os.path.isdir(cli.checkpoint) else os.path.join(cli.run_dir, 'checkpoints', cli.checkpoint)
    args.sample_dir = cli.sample_dir
    args.num_test_samples = cli.num_test_samples

    # get_data reads FACE_CSV_DIR / FACE_DATA_ROOT before the args values, so set them here
    if cli.csv_dir is not None:
        os.environ['FACE_CSV_DIR'] = cli.csv_dir
    if cli.data_root is not None:
        os.environ['FACE_DATA_ROOT'] = cli.data_root
    return args

def explain_local(args, tree, classes, device):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    test_transform = transforms.Compose([
                        transforms.Resize(size=(args.image_size, args.image_size)),
                        transforms.ToTensor(),
                        normalize
                    ])

    sample = test_transform(Image.open(args.sample_dir).convert('RGB')).unsqueeze(0).to(device)

    gen_pred_vis(tree, sample, args.sample_dir, args.results_dir, args, classes)

    with torch.no_grad():
        pred, _ = tree.forward(sample, sampling_strategy='greedy')
    return classes[torch.argmax(pred, dim=1)[0].item()]



if __name__ == '__main__':
    args = get_local_expl_args()
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')
    # Log which device was actually used
    print('Device used: ',str(device))

    # Load trained ProtoTree
    tree = torch.load(os.path.join(args.checkpoint, 'model.pth'), map_location=device).to(device=device)
    tree.eval()
    print('Model explained: ', args.checkpoint)
    # Obtain the class names (for face_dataset: the age bins)
    _, _, testloader, classes, _, _ = get_dataloaders(args)

    if args.num_test_samples > 0:
        testset = testloader.dataset
        csv_file = os.path.join(os.environ.get('FACE_CSV_DIR', args.face_csv_dir), 'test_set.csv')
        ages = pd.read_csv(csv_file)['age'].values
        for i in range(min(args.num_test_samples, len(testset.imgs))):
            img_path, label = testset.imgs[i]
            args_1 = deepcopy(args)
            args_1.sample_dir = img_path.replace('\\', '/')
            print("Image to explain: ", args_1.sample_dir)
            pred_bin = explain_local(args_1, tree, classes, device)
            print("True age: %s (bin %s), predicted bin: %s" % (ages[i], classes[label], pred_bin), flush=True)
    elif os.path.isdir(args.sample_dir):
        class_name = args.sample_dir.rstrip('/').split('/')[-1]
        if not os.path.exists(os.path.join(os.path.join(args.log_dir, args.results_dir),class_name)):
            os.makedirs(os.path.join(os.path.join(args.log_dir, args.results_dir),class_name))
        for filename in os.listdir(args.sample_dir):
            print(filename)
            if filename.endswith(".jpg") or filename.endswith(".png"):
                args_1 = deepcopy(args)
                args_1.sample_dir = args.sample_dir.rstrip('/')+"/"+filename
                args_1.results_dir = os.path.join(args.results_dir, class_name)
                print("Predicted: ", explain_local(args_1, tree, classes, device))
    else:
        print("Image to explain: ", args.sample_dir)
        print("Predicted: ", explain_local(args, tree, classes, device))
