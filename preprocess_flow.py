# preprocess_flow.py
# -----------------------------------------------------------
# Computes 320×320 optical flow ground truth for TartanAirV2.
# Saves .npy files shaped [320, 320, 2].
# Standalone version: includes depth decoder locally.
# -----------------------------------------------------------

import os
import glob
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# -----------------------------------------------------------
# INCLUDED FROM evaluation/depth_eval.py
# Minimal function required for flow computation
# -----------------------------------------------------------

def decode_tartan_depth(path):
    """
    Reads TartanAir v2 packed depth PNG (lossless RGBA).
    Reinterprets bytes as float32 depth.
    Returns depth (meters) as float32 array H×W.
    """
    depth_rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # H x W x 4 (uint8)
    if depth_rgba is None:
        raise FileNotFoundError(f"[depth_eval] Could not read: {path}")

    # reinterpret bytes → float32
    depth = depth_rgba.view("<f4")     # little-endian float32
    depth = np.squeeze(depth, axis=-1) # (H,W)
    return depth.astype(np.float32)

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------

dataset_root = "/data/quentin/tartanair/"
camera = "lcam_front"

# Camera intrinsics for 320×320 images
fx = fy = 320.0
cx = cy = 320.0

K_320 = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)

# -----------------------------------------------------------
# Load absolute poses
# -----------------------------------------------------------

def load_abs_poses(pose_file):
    """
    Loads absolute poses from TartanAir V2 pose file:
    Each line: tx ty tz qx qy qz qw
    Returns list of 4×4 SE3 matrices.
    """
    poses = []
    data = np.loadtxt(pose_file).reshape(-1, 7)

    for tx, ty, tz, qx, qy, qz, qw in data:
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = [tx, ty, tz]
        poses.append(T)

    return poses

# -----------------------------------------------------------
# Compute flow from depth + SE(3)
# -----------------------------------------------------------

def compute_flow_from_depth(depth, K, T):
    """
    depth: (H,W)
    K: intrinsics 3×3
    T: 4×4 transform T_t→t+1
    Output: (H,W,2) optical flow
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))

    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    pts = np.stack([X, Y, Z, np.ones_like(Z)], axis=-1)  # (H,W,4)

    pts_next = pts @ T.T
    Xn, Yn, Zn = pts_next[..., 0], pts_next[..., 1], pts_next[..., 2]

    un = fx * (Xn / Zn) + cx
    vn = fy * (Yn / Zn) + cy

    flow = np.stack([un - u, vn - v], axis=-1).astype(np.float32)
    return flow

# -----------------------------------------------------------
# Flow generation per sequence
# -----------------------------------------------------------

def generate_flow_for_sequence(seq):
    """
    seq example: 'House/Easy/P000'
    """
    print(f"==> Computing GT flow for: {seq}")

    seq_dir = os.path.join(dataset_root, seq)
    pose_file = os.path.join(seq_dir, f"pose_{camera}.txt")
    depth_dir = os.path.join(seq_dir, f"depth_{camera}")
    out_dir = os.path.join(seq_dir, f"flow_{camera}_gt")

    if not os.path.isfile(pose_file):
        print(f"[ERROR] Pose file missing: {pose_file}")
        return

    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    if len(depth_files) == 0:
        print(f"[ERROR] No depth files in: {depth_dir}")
        return

    abs_poses = load_abs_poses(pose_file)
    os.makedirs(out_dir, exist_ok=True)

    for i in range(len(depth_files) - 1):
        depth_path = depth_files[i]

        # Decode + resize depth to exactly 320×320
        depth = decode_tartan_depth(depth_path)
        depth = cv2.resize(depth, (320, 320), cv2.INTER_NEAREST)

        # SE(3) motion T_t→t+1
        T = np.linalg.inv(abs_poses[i]) @ abs_poses[i + 1]

        flow = compute_flow_from_depth(depth, K_320, T)

        out_name = os.path.basename(depth_path).replace(".png", "_flow.npy")
        np.save(os.path.join(out_dir, out_name), flow)

    print(f"✔ Saved {len(depth_files) - 1} flow maps → {out_dir}")

# -----------------------------------------------------------
# Script entry
# -----------------------------------------------------------

if __name__ == "__main__":
    sequences = [
        "House/Easy/P000",
        "House/Easy/P001",
        "House/Easy/P002",
        "House/Easy/P003",
        "House/Easy/P004",
        "House/Easy/P005",
        "House/Easy/P006",
        "House/Easy/P007",
        "House/Hard/P000",
        "House/Hard/P001",
        "House/Hard/P002",
        "House/Hard/P003",
        "House/Hard/P004",
        "House/Hard/P005",
        "House/Hard/P006",
        "House/Hard/P007",
    ]

    for seq in sequences:
        generate_flow_for_sequence(seq)

    print("\nAll flow GT generated successfully.")
