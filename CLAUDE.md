# ProtoTree — Head Pose Estimation

## What this is
ProtoTree-based interpretable decision tree for head pose (yaw) estimation. Trained on 300W-LP, evaluated on AFLW2000.

## Dataset location
Dataset loading lives in **Gaussian-Regression_Tree/util/data.py** — that's the only reason we touch that sibling directory.

## Architecture
- Backbone: ResNet50 (ImageNet pretrained)
- Add-on: 1x1 Conv2d(2048→256) → Dropout2d(p=args.dropout) → Sigmoid
- Tree: depth=7, 128 leaves, prototype_temp annealed 0.2→0.01

## Key training script
`run_headpose.sh` → `main_tree.py`

## Current augmentations (data.py lines ~375-384)
Training: RandomPerspective(0.2, p=0.5) + ColorJitter + RandomAffine(shear, translate) — NO HorizontalFlip (correct for head pose, flip changes yaw sign)

## Known overfitting issue
Train acc ~0.80 vs val acc ~0.30. Full dataset (451347) didn't close the gap.
- `--dropout` defaults to 0.0 — **not being used** despite the layer existing
- `--weight_decay` currently 0.001

## Experiment log (run IDs on the cluster)
- 451332: label_smooth=0.5, 0.5 subset → best: val=0.302, test_soft_mae=4.52°
- 451333: label_smooth=1.0 → val=0.268 (worse)
- 451334: label_smooth anneal 1.5→0.5 → val=0.303
- 451336: label_smooth end=0.1 → val=0.307
- 451337: label_smooth end=0.01 → val=0.300, overfit worst (train=0.813)
- 451347: same as 451332 but full dataset → val=0.28 at epoch 30, not helping
