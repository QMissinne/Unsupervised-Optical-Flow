import argparse
import torch
from tqdm import tqdm

import albumentations as albu
from albumentations.pytorch import ToTensorV2

from dataset import getDataloaders
from models import FlowNetS, LightFlowNet, PWC_Net

from utils_eval import evaluate


# --------------------------------------------------
# Meters
# --------------------------------------------------

class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(1, self.count)


# --------------------------------------------------
# Evaluation loop
# --------------------------------------------------

@torch.no_grad()
def evaluate_split(model, dataloader, device, split_name):
    model.eval()

    epe_meter = AverageMeter()
    aae_meter = AverageMeter()

    for imgs, flow_gt in tqdm(dataloader, desc=f"Evaluating {split_name}", leave=False):
        imgs = imgs.to(device)
        flow_gt = flow_gt.to(device)

        # Forward (FlowNet returns tuple in eval mode)
        flow_pred = model(imgs)[0]

        # Your existing metric function
        epe, aae = evaluate(flow_pred, flow_gt)

        epe_meter.update(epe.item(), imgs.size(0))
        aae_meter.update(aae.item(), imgs.size(0))

    return epe_meter.avg, aae_meter.avg


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Dataset root')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best_weight.pt')
    parser.add_argument('--model', type=str, default='flownet',
                        choices=['flownet', 'lightflownet', 'pwc'],
                        help='Model type')
    parser.add_argument('--batch-size', type=int, default=8)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    if args.model == "lightflownet":
        model = LightFlowNet()
    elif args.model == "pwc":
        model = PWC_Net()
    else:
        model = FlowNetS()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # --------------------------------------------------
    # Dataloaders (NO augmentation!)
    # --------------------------------------------------
    frames_transforms = albu.Compose([
        albu.Normalize((0., 0., 0.), (1., 1., 1.)),
        ToTensorV2()
    ])

    train_loader, val_loader, test_loader = getDataloaders(
        batch_size=args.batch_size,
        root=args.root,
        frames_transforms=frames_transforms,
        frames_aug_transforms=None,   # IMPORTANT: no augmentation
        co_aug_transforms=None
    )


    # --------------------------------------------------
    # Evaluate all splits
    # --------------------------------------------------
    results = {}

    for split_name, loader in [
        ("TRAIN", train_loader),
        ("VAL",   val_loader),
        ("TEST",  test_loader),
    ]:
        epe, aae = evaluate_split(model, loader, device, split_name)
        results[split_name] = (epe, aae)

    # --------------------------------------------------
    # Print summary (clearly marked)
    # --------------------------------------------------
    print("\n================== FLOWNET EVALUATION ==================")
    print(f"Checkpoint: {args.checkpoint}")
    print("--------------------------------------------------------")
    for split in ["TRAIN", "VAL", "TEST"]:
        epe, aae = results[split]
        print(f"{split:>5} | EPE: {epe:8.4f} | AAE: {aae:8.4f}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
