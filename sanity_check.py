import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
from scipy.stats import pearsonr

from models import FlowPoseProbe
from dataset import TartanPoseDataset

import albumentations as albu
from albumentations.pytorch import ToTensorV2


# ============================================================
#   RELATIVE POSE VERIFICATION UTILITIES
# ============================================================

def compute_rel_poses(abs_poses):
    """
    abs[i] = world_T_cam[i]
    rel[i] = cam_{i-1} -> cam_i    (camera-to-camera transform)
    rel[0] = identity
    """
    rel = [np.eye(4, dtype=np.float32)]
    for i in range(1, len(abs_poses)):
        T_prev = abs_poses[i - 1]
        T_curr = abs_poses[i]
        rel_pose = np.linalg.inv(T_prev) @ T_curr   # T_{prev→curr}
        rel.append(rel_pose.astype(np.float32))
    return np.stack(rel)


def verify_rel_pose_consistency(abs_poses, rel_poses):
    """
    Checks: abs[i-1] * rel[i] ≈ abs[i]
    Errors should be near zero.
    """
    trans_errs = []
    rot_errs = []

    for i in range(1, len(abs_poses)):
        T_prev = abs_poses[i - 1]
        T_curr = abs_poses[i]

        T_pred = T_prev @ rel_poses[i]

        # pose error
        T_err = np.linalg.inv(T_pred) @ T_curr
        terr = np.linalg.norm(T_err[:3, 3])

        R_err = T_err[:3, :3]
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2.0, -1, 1))

        trans_errs.append(terr)
        rot_errs.append(angle)

    print("\n=== Relative Pose Consistency Check ===")
    print(f"Mean translation error : {np.mean(trans_errs):.6e} m")
    print(f"Max  translation error : {np.max(trans_errs):.6e} m")
    print(f"Mean rotation error    : {np.mean(rot_errs):.6e} rad")
    print(f"Max  rotation error    : {np.max(rot_errs):.6e} rad")
    print("If these are tiny (1e-6), rel poses are consistent.\n")


def reconstruct_from_rel_cam(abs_poses, rel_poses):
    """
    Reconstruct the global trajectory using:

        T_rec[i] = T_rec[i-1] @ rel[i]

    consistent with:
        rel[i] = inv(abs[i-1]) @ abs[i]
    """
    T = abs_poses[0].copy()
    traj = [T[:3, 3].copy()]

    for i in range(1, len(rel_poses)):
        T = T @ rel_poses[i]
        traj.append(T[:3, 3].copy())

    return np.stack(traj)


def plot_global_vs_relative(abs_poses, rel_poses, out_path, seq_name="trajectory"):
    abs_traj = abs_poses[:, :3, 3]
    rel_traj = reconstruct_from_rel_cam(abs_poses, rel_poses)

    plt.figure(figsize=(7, 7))
    plt.plot(abs_traj[:, 0], abs_traj[:, 2], 'g-', label="Global GT", linewidth=2)
    plt.plot(rel_traj[:, 0], rel_traj[:, 2], 'r--', label="Reconstructed (relative)", linewidth=2)

    plt.title(f"Global vs Reconstructed Trajectory\n{seq_name}")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(out_path)
    plt.close()

    print(f"[Trajectory] Saved comparison plot → {out_path}")

    # drift statistics
    L = min(len(abs_traj), len(rel_traj))
    drift = np.linalg.norm(abs_traj[:L] - rel_traj[:L], axis=1)

    print("\n=== Trajectory Drift Statistics ===")
    print(f"Mean drift:  {drift.mean():.4f} m")
    print(f"Max drift:   {drift.max():.4f} m")
    print(f"Final drift: {drift[-1]:.4f} m\n")


# ============================================================
# 1. ---------- GT POSE SANITY CHECKS -------------------------
# ============================================================

def load_abs_poses(pose_file):
    abs_poses = []
    with open(pose_file, "r") as f:
        for line in f:
            vals = list(map(float, line.split()))
            tx, ty, tz = vals[:3]
            qx, qy, qz, qw = vals[3:]

            T = np.eye(4)
            T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            abs_poses.append(T)
    return np.stack(abs_poses)


def plot_gt_trajectory(rel_poses):
    T = np.eye(4)
    traj = [T[:3, 3].copy()]

    for dT in rel_poses:
        T = T @ dT
        traj.append(T[:3, 3].copy())

    traj = np.stack(traj)

    plt.figure(figsize=(6, 6))
    plt.plot(traj[:, 0], traj[:, 2], 'g-')
    plt.axis("equal")
    plt.title("GT Relative Trajectory (X–Z)")
    plt.grid()
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig("visualizations/gt_relative_traj.png")
    plt.close()


# ============================================================
# 2. ------------- BASELINES ---------------------------------
# ============================================================

class ZeroPose(torch.nn.Module):
    def forward(self, x):
        B = x.size(0)
        return (torch.zeros(B, 1, 1, 3, device=x.device),
                torch.zeros(B, 1, 1, 3, device=x.device))


class RandomProbe(torch.nn.Module):
    """MinimalPoseDecoder with random embedding."""
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1024, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 6)
        )

    def forward(self, x):
        B = x.size(0)
        feat = torch.randn(B, 1024, device=x.device)
        out = self.mlp(feat)
        axis = out[:, :3].view(B, 1, 1, 3)
        trans = out[:, 3:].view(B, 1, 1, 3)
        return axis, trans


# ============================================================
# 3. ---------- Evaluation Utilities --------------------------
# ============================================================

def evaluate_model(model, loader, device):
    model.eval()
    all_pred_axis, all_pred_trans = [], []
    all_gt_axis, all_gt_trans = [], []

    with torch.no_grad():
        for frames, (gt_axis, gt_trans) in loader:
            frames = frames.to(device)
            gt_axis = gt_axis.to(device)
            gt_trans = gt_trans.to(device)

            pred_axis, pred_trans = model(frames)

            all_pred_axis.append(pred_axis.cpu().numpy())
            all_gt_axis.append(gt_axis.cpu().numpy())
            all_pred_trans.append(pred_trans.cpu().numpy())
            all_gt_trans.append(gt_trans.cpu().numpy())

    pred_axis = np.concatenate(all_pred_axis).reshape(-1, 3)
    pred_trans = np.concatenate(all_pred_trans).reshape(-1, 3)
    gt_axis = np.concatenate(all_gt_axis).reshape(-1, 3)
    gt_trans = np.concatenate(all_gt_trans).reshape(-1, 3)

    rot_err = np.linalg.norm(pred_axis - gt_axis, axis=1)
    trans_err = np.linalg.norm(pred_trans - gt_trans, axis=1)

    corr_rot = pearsonr(gt_axis[:, 0], pred_axis[:, 0])[0]
    corr_trans = pearsonr(gt_trans[:, 0], pred_trans[:, 0])[0]

    return rot_err.mean(), trans_err.mean(), corr_rot, corr_trans


# ============================================================
# 4. ---------- MAIN SCRIPT ----------------------------------
# ============================================================

def main(args):
    # Load dataset
    transforms = albu.Compose([
        albu.Normalize((0., 0., 0.), (1., 1., 1.)),
        ToTensorV2()
    ])

    val_set = TartanPoseDataset(
        args.data_root,
        os.path.join(args.split_root, "val_files.txt"),
        frames_transforms=transforms
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ======== GET ONE SEQUENCE FROM VAL SET =========
    first_img = val_set.imgPairs[0][0]
    example_seq = os.path.dirname(os.path.dirname(first_img))
    pose_file = os.path.join(example_seq, "pose_lcam_front.txt")

    print(f"[SanityCheck] Sequence: {example_seq}")
    print(f"[SanityCheck] Loading ABS poses: {pose_file}")

    abs_poses = load_abs_poses(pose_file)
    rel_poses = compute_rel_poses(abs_poses)

    # ---- NEW: VERIFY RELATIVE POSES ----
    verify_rel_pose_consistency(abs_poses, rel_poses)

    # ---- Plot trajectories ----
    plot_gt_trajectory(rel_poses)
    plot_global_vs_relative(abs_poses, rel_poses,
                            f"visualizations/global_vs_relative.png",
                            seq_name=os.path.basename(example_seq))

    # ======== BASELINES & PROBE ========
    device = torch.device(args.device)

    print("\n=== ZeroPose ===")
    z_rot, z_trans, z_cr, z_ct = evaluate_model(ZeroPose().to(device), val_loader, device)

    print("\n=== Random Probe ===")
    r_rot, r_trans, r_cr, r_ct = evaluate_model(RandomProbe().to(device), val_loader, device)

    print("\n=== FlowPoseProbe ===")
    probe = FlowPoseProbe().to(device)
    ckpt = torch.load(args.flow_probe_ckpt, map_location=device)
    probe.load_state_dict(ckpt["model_state_dict"])
    p_rot, p_trans, p_cr, p_ct = evaluate_model(probe, val_loader, device)

    print("\n==================== RESULTS ====================")
    print("Model           | RotErr | TransErr | CorrRot | CorrTrans")
    print("---------------------------------------------------------")
    print(f"ZeroPose        | {z_rot:.4f} | {z_trans:.4f} | {z_cr:.3f} | {z_ct:.3f}")
    print(f"RandomProbe     | {r_rot:.4f} | {r_trans:.4f} | {r_cr:.3f} | {r_ct:.3f}")
    print(f"FlowPoseProbe   | {p_rot:.4f} | {p_trans:.4f} | {p_cr:.3f} | {p_ct:.3f}")
    print("=========================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_root", default="./splits/tartan")
    parser.add_argument("--flow_probe_ckpt", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args)
