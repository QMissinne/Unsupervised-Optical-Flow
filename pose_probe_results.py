import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from models import FlowPoseProbe
from dataset import getPoseDataloaders


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def collect_predictions(dataloader, model, device):
    all_rot_pred = []
    all_rot_gt = []
    all_trans_pred = []
    all_trans_gt = []

    with torch.no_grad():
        for frames, (gt_axis, gt_trans) in dataloader:
            frames = frames.to(device)
            gt_axis = gt_axis.to(device)
            gt_trans = gt_trans.to(device)

            pred_axis, pred_trans = model(frames)

            rp, tp = tensor_to_np(pred_axis, pred_trans)
            rg, tg = tensor_to_np(gt_axis, gt_trans)

            all_rot_pred.append(rp)
            all_rot_gt.append(rg)
            all_trans_pred.append(tp)
            all_trans_gt.append(tg)

    return (np.concatenate(all_rot_pred),
            np.concatenate(all_rot_gt),
            np.concatenate(all_trans_pred),
            np.concatenate(all_trans_gt))

def tensor_to_np(axis, trans):
    """Bx1x1x3 → Nx3 numpy"""
    axis = axis.reshape(-1, 3).cpu().numpy()
    trans = trans.reshape(-1, 3).cpu().numpy()
    return axis, trans


def safe_corr(a, b):
    """Pearson correlation that handles constant arrays."""
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan
    return pearsonr(a, b)[0]


# ------------------------------------------------------------
# New high-quality visualization
# ------------------------------------------------------------

def linear_regression_line(x, y):
    """Returns slope m, intercept b for y = m x + b."""
    if len(x) < 2:
        return 0, 0
    return np.polyfit(x, y, 1)


def make_visualisation(rot_pred_test, rot_gt_test,
                       trans_pred_test, trans_gt_test,
                       rot_pred_train=None, rot_gt_train=None,
                       trans_pred_train=None, trans_gt_train=None,
                       save_path="pose_probe_results.png"):

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Pose Probe – Train (yellow) vs Test (blue)", fontsize=18)

    components = ["X", "Y", "Z"]

    # ------------------------------------------------------------------
    # Helper: single scatter panel (train + test + regression lines)
    # ------------------------------------------------------------------
    def scatter_panel(ax, gt_train, pred_train, gt_test, pred_test, title):
        # TRAIN scatter -------------------------------------------------
        if gt_train is not None:
            ax.scatter(gt_train, pred_train,
                       s=4, alpha=0.45, color="gold", label="Train")

            m, b = linear_regression_line(gt_train, pred_train)
            xs = np.linspace(gt_train.min(), gt_train.max(), 200)
            ax.plot(xs, m * xs + b, color="gold", lw=2,
                    label="Train regression")

        # TEST scatter --------------------------------------------------
        ax.scatter(gt_test, pred_test,
                   s=4, alpha=0.45, color="dodgerblue", label="Test")

        m, b = linear_regression_line(gt_test, pred_test)
        xs = np.linspace(gt_test.min(), gt_test.max(), 200)
        ax.plot(xs, m * xs + b, color="dodgerblue", lw=2,
                label="Test regression")

        # Diagonal perfect-prediction line -----------------------------
        min_v = min(gt_test.min(), pred_test.min())
        max_v = max(gt_test.max(), pred_test.max())
        ax.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1)

        ax.set_title(title)
        ax.set_xlabel("GT")
        ax.set_ylabel("Pred")
        ax.legend()

    # ------------------------------------------------------------------
    # 1) Rotation scatter plots (3 components)
    # ------------------------------------------------------------------
    for i in range(3):
        scatter_panel(
            axs[0, i],
            rot_gt_train[:, i] if rot_gt_train is not None else None,
            rot_pred_train[:, i] if rot_pred_train is not None else None,
            rot_gt_test[:, i],
            rot_pred_test[:, i],
            f"Rotation Component: {components[i]}"
        )

    # ------------------------------------------------------------------
    # 2) Translation scatter plots (3 components)
    # ------------------------------------------------------------------
    for i in range(3):
        scatter_panel(
            axs[1, i],
            trans_gt_train[:, i] if trans_gt_train is not None else None,
            trans_pred_train[:, i] if trans_pred_train is not None else None,
            trans_gt_test[:, i],
            trans_pred_test[:, i],
            f"Translation Component: {components[i]}"
        )

    # ------------------------------------------------------------------
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=250)
    plt.close()
    print(f"\n✅ Saved combined train/test visualisation → {save_path}\n")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main(args):

    train_loader, _, test_loader = getPoseDataloaders(
        batch_size=args.batch_size,
        root=args.data_root,
        split_root=args.split_root
    )

    model = FlowPoseProbe().to(args.device)
    ckpt = torch.load(args.model_checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- Train predictions ----
    rot_pred_train, rot_gt_train, trans_pred_train, trans_gt_train = \
        collect_predictions(train_loader, model, args.device)

    # ---- Test predictions ----
    rot_pred_test, rot_gt_test, trans_pred_test, trans_gt_test = \
        collect_predictions(test_loader, model, args.device)

    save_path = os.path.join("visualizations", args.output)

    make_visualisation(
        rot_pred_test, rot_gt_test,
        trans_pred_test, trans_gt_test,
        rot_pred_train, rot_gt_train,
        trans_pred_train, trans_gt_train,
        save_path
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_root", default="./splits/tartan")
    parser.add_argument("--output", default="pose_probe_results.png")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args)
