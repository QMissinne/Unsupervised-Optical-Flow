import os
from dataset import TartanFlowDataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2

tf = albu.Compose([albu.Normalize(), ToTensorV2()])

ds = TartanFlowDataset(
    "/data/quentin/tartanair",
    "splits/tartan/train_files.txt",
    frames_transforms=tf
)

print(len(ds))
print(ds[0][0].shape, ds[0][1].shape)

root = "/data/quentin/tartanair/"
split = "/home/quentin/Unsupervised-Optical-Flow/splits/tartan/train_files.txt"

with open(split) as f:
    lines = f.read().splitlines()

print("Total split entries:", len(lines))

env, diff, traj, frame = lines[0].split()
print("Example entry:", env, diff, traj, frame)

# Construct expected folders
traj_dir = os.path.join(root, env, diff, traj)
img_dir = os.path.join(traj_dir, "resized_image_lcam_front")
flow_dir = os.path.join(traj_dir, "flow_lcam_front_gt")

print("Trajectory folder exists:", os.path.isdir(traj_dir))
print("Image folder exists:", os.path.isdir(img_dir))
print("Flow folder exists:", os.path.isdir(flow_dir))

# Test image
img0 = os.path.join(img_dir, f"{frame}_lcam_front.png")
img1 = os.path.join(img_dir, f"{str(int(frame)+1).zfill(6)}_lcam_front.png")

print("Frame 0 exists? ", os.path.isfile(img0))
print("Frame 1 exists? ", os.path.isfile(img1))

# Test flow
flow_path = os.path.join(flow_dir, f"{frame}_lcam_front_depth_flow.npy")
print("Flow exists? ", os.path.isfile(flow_path))