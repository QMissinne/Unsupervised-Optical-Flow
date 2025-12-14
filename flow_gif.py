import argparse
import os
import cv2
import torch
import numpy as np
import imageio
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn.functional as F

from dataset import computeImg   # Make sure computeImg is available
from models import FlowNetS, Unsupervised, LightFlowNet, PWC_Net

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# ---------------------------------------------------------
# Load GT flow for TartanAir naming scheme
# ---------------------------------------------------------
def load_gt_flow(gt_dir, frame_index):
    """
    Loads TartanAir GT flow stored as:
        <frame_index>_lcam_front_depth_flow.npy

    frame_index : int, e.g. 1 → "000001"
    """
    frame_str = str(frame_index).zfill(6)
    fname = f"{frame_str}_lcam_front_depth_flow.npy"
    path = os.path.join(gt_dir, fname)

    if not os.path.exists(path):
        raise FileNotFoundError(f"GT flow not found: {path}")

    flow = np.load(path).astype(np.float32)  # shape (H, W, 2)
    return flow


# ---------------------------------------------------------
# Build 3-panel visualization
# ---------------------------------------------------------
def make_triplet(input_img, pred_flow_img, gt_flow_img):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    h = max(input_img.shape[0], pred_flow_img.shape[0], gt_flow_img.shape[0])

    def pad(im):
        pad_h = h - im.shape[0]
        return np.pad(im, ((0, pad_h), (0, 0), (0, 0)), mode="constant")

    return np.concatenate([pad(input_img), pad(pred_flow_img), pad(gt_flow_img)], axis=1)

# - --------------------------------------------------------
# Find model by UUID
# ---------------------------------------------------------

def find_model_by_uuid(model_root, uuid):
    """
    Finds a model directory whose name ends with the given UUID,
    then returns the path to best_weight.pt inside it.
    """
    matches = [
        d for d in os.listdir(model_root)
        if os.path.isdir(os.path.join(model_root, d)) and d.endswith(uuid)
    ]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No model directory with UUID '{uuid}' found in {model_root}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple model directories match UUID '{uuid}': {matches}"
        )

    weight_path = os.path.join(model_root, matches[0], "best_weight.pt")

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Missing best_weight.pt in {matches[0]}")

    return weight_path, matches[0]


# =========================================================
# MAIN SCRIPT
# =========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True,
                        help="Folder containing RGB frames (000001_lcam_front.png)")
    parser.add_argument('--gt_path', required=True,
                        help="Folder containing GT flow .npy files")
    parser.add_argument("--unsup", action="store_true")
    parser.add_argument('--model', default='flownet', type=str)
    parser.add_argument('--fps', default=10, type=int)
    parser.add_argument('--gif', action="store_true")
    parser.add_argument('--model_uuid', default='', type=str, help="Specify model UUID")

    args = parser.parse_args()
    run_id = args.model_uuid if args.model_uuid is not None else "default"
    file_names = sorted(os.listdir(args.path))

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------
    if args.unsup:
        mymodel = Unsupervised(conv_predictor=args.model)
        model_path = os.path.join("Unsupervised", type(mymodel.predictor).__name__)
    else:
        if "light" in args.model.lower():
            mymodel = LightFlowNet()
            model_path = "LightFlowNet"
        elif "pwc" in args.model.lower():
            mymodel = PWC_Net()
            model_path = "PWC_Net"
        else:
            mymodel = FlowNetS()
            model_path = "FlowNetS"

    model_dir = os.path.join("model_weight", model_path)

    if args.model_uuid is None:
        weight_path = os.path.join(model_dir, "best_weight.pt")
    else:
        weight_path, run_dir = find_model_by_uuid(model_dir, args.model_uuid)
        print(f"[INFO] Using run: {run_dir}")
        print(f"[INFO] Loading weights: {weight_path}")

    weights = torch.load(weight_path, map_location=device)

    result_dir = os.path.join("result", run_dir)
    os.makedirs(result_dir, exist_ok=True)

    mymodel.load_state_dict(weights["model_state_dict"])
    if args.unsup:
        mymodel = mymodel.predictor
    mymodel = mymodel.to(device).eval()


    frames_transforms = albu.Compose([
        albu.Normalize((0., 0., 0.), (1., 1., 1.)),
        ToTensorV2()
    ])

    tmp_dir = os.path.join(result_dir, "gif_frames")
    os.makedirs(tmp_dir, exist_ok=True)
    output_frames = []

    # ---------------------------------------------------------
    # MAIN LOOP — 3-panel visualization
    # ---------------------------------------------------------
    for i in tqdm(range(len(file_names) - 1), desc="Computing flow"):

        # Load RGB frames
        f1_path = os.path.join(args.path, file_names[i])
        f2_path = os.path.join(args.path, file_names[i + 1])

        frame1_raw = cv2.imread(f1_path)
        frame2_raw = cv2.imread(f2_path)
        h, w = frame1_raw.shape[:2]

        # Extract frame index from filename: "000123_lcam_front.png"
        frame_idx = int(file_names[i].split("_")[0])

        # Load GT flow as numpy (H, W, 2)
        gt_flow = load_gt_flow(args.gt_path, frame_idx)

        # Preprocess input for model
        t1 = frames_transforms(image=frame1_raw)['image']
        t2 = frames_transforms(image=frame2_raw)['image']
        frames = torch.cat((t1, t2), dim=0).unsqueeze(0).to(device)

        # Predict flow
        with torch.no_grad():
            flow_pred = mymodel(frames)[0]
            flow_pred_rs = F.interpolate(flow_pred, (h, w), mode='bilinear',
                                          align_corners=False)[0].cpu().numpy()

        # Visualize flows using computeImg()
        pred_flow_img = computeImg(flow_pred_rs, verbose=False)
        gt_flow_img   = computeImg(gt_flow, verbose=False)

        # Create final triple image
        merged = make_triplet(frame1_raw, pred_flow_img, gt_flow_img)

        # Save frame for GIF/video
        out_path = os.path.join(tmp_dir, f"frame_{i:05d}_{run_id}.png")
        cv2.imwrite(out_path, cv2.cvtColor(merged, cv2.COLOR_RGB2BGR))
        output_frames.append(merged)

    # ---------------------------------------------------------
    # Save final GIF or MP4
    # ---------------------------------------------------------
    gif_path = os.path.join(result_dir, f"trajectory_{run_id}.gif")
    mp4_path = os.path.join(result_dir, f"trajectory_{run_id}.mp4")

    if args.gif:
        imageio.mimsave(gif_path, output_frames, fps=args.fps)
        print("Saved GIF:", gif_path)
    else:
        writer = imageio.get_writer(mp4_path, fps=args.fps)
        for f in output_frames:
            writer.append_data(f)
        writer.close()
        print("Saved video:", mp4_path)

    print("DONE.")
