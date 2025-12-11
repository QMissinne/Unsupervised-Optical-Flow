from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch
import numpy as np
from utils import *
import cv2
import os
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from scipy.spatial.transform import Rotation as R

class TartanPoseDataset(Dataset):
    """
    Loads frame pairs + relative pose (axis-angle + translation)
    for training the FlowPoseProbe.
    
    Requires:
        <root>/<env>/<difficulty>/<traj>/resized_image_lcam_front/
        <root>/<env>/<difficulty>/<traj>/relative_poses.npy
    """

    def __init__(self, root, split_file, frames_transforms=None):
        self.root = root
        self.frames_transforms = frames_transforms

        self.imgPairs = []
        self.rel_poses = []   # list of SE(3) matrices

        self._load_split(split_file)

    # ----------------------------------------------------------
    # Read split file exactly like Flow dataset
    # ----------------------------------------------------------
    def _load_split(self, split_file):
        with open(split_file, "r") as f:
            entries = f.read().splitlines()

        for line in entries:
            env, difficulty, traj, frame_str = line.split()

            traj_dir = os.path.join(self.root, env, difficulty, traj)

            img_dir = os.path.join(traj_dir, "resized_image_lcam_front")
            pose_path = os.path.join(traj_dir, "relative_poses.npy")

            if not os.path.exists(pose_path):
                print(f"[WARNING] Missing pose file: {pose_path}")
                continue

            # load all poses ONCE
            if not hasattr(self, "_pose_cache"):
                self._pose_cache = {}
            if pose_path not in self._pose_cache:
                self._pose_cache[pose_path] = np.load(pose_path)  # shape [N,4,4]
            rel_pose_arr = self._pose_cache[pose_path]

            frame_idx = int(frame_str)

            # image paths
            img1 = os.path.join(img_dir, f"{frame_str}_lcam_front.png")
            img2 = os.path.join(img_dir, f"{str(frame_idx + 1).zfill(6)}_lcam_front.png")

            if not (os.path.exists(img1) and os.path.exists(img2)):
                continue

            if frame_idx >= len(rel_pose_arr) - 1:
                continue

            self.imgPairs.append([img1, img2])
            self.rel_poses.append(rel_pose_arr[frame_idx + 1])  # T_{t->t+1}

        print(f"[TartanPoseDataset] Loaded {len(self.imgPairs)} samples from {split_file}")

    # ----------------------------------------------------------
    # Convert SE3 to axisangle + translation
    # ----------------------------------------------------------
    @staticmethod
    def se3_to_pose(T):
        Rmat = T[:3, :3]
        tvec = T[:3, 3]
        axisangle = R.from_matrix(Rmat).as_rotvec().astype(np.float32)
        return axisangle, tvec.astype(np.float32)

    # ----------------------------------------------------------
    # __getitem__
    # ----------------------------------------------------------
    def __getitem__(self, idx):
        # load frames
        frame1 = cv2.cvtColor(cv2.imread(self.imgPairs[idx][0]), cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(cv2.imread(self.imgPairs[idx][1]), cv2.COLOR_BGR2RGB)

        if self.frames_transforms:
            frame1 = self.frames_transforms(image=frame1)["image"]
            frame2 = self.frames_transforms(image=frame2)["image"]

        frames = torch.cat((frame1, frame2), dim=0)  # shape [6,H,W]

        # load pose
        T_rel = self.rel_poses[idx]
        axisangle, tvec = self.se3_to_pose(T_rel)

        # convert to tensors
        axisangle = torch.from_numpy(axisangle).view(1, 1, 1, 3)
        tvec = torch.from_numpy(tvec).view(1, 1, 1, 3)

        return frames, (axisangle, tvec)

    def __len__(self):
        return len(self.imgPairs)


class OpticalFlowDataset(Dataset):
    def __init__(self, root, frames_transforms=None, frames_aug_transforms=None,co_aug_transforms=None):
        self.root = root
        self.frames_transforms = frames_transforms
        self.frames_aug_transforms = frames_aug_transforms
        self.co_aug_transforms = co_aug_transforms
        self.target3 = {'image0': 'image', 'image1': 'image', 'image2': 'image'}
        self.target2 = {'image0': 'image', 'image1': 'image'}
        self.imgPairs = []
        self.flows = []
        self.func = readflo

    def __getitem__(self, idx):

        flow = self.func(self.flows[idx])
        frame1 = cv2.cvtColor(cv2.imread(self.imgPairs[idx][0]), cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(cv2.imread(self.imgPairs[idx][1]), cv2.COLOR_BGR2RGB)

        if self.co_aug_transforms is not None:
            transformed = albu.Compose(self.co_aug_transforms, p=1, additional_targets=self.target3)(image=frame1,
                                                                                                     image0=frame2,
                                                                                                     image1=flow)
            frame1 = transformed['image']
            frame2 = transformed['image0']
            flow = transformed['image1']

        if self.frames_aug_transforms is not None:
            transformed = albu.Compose(self.frames_aug_transforms, p=1, additional_targets=self.target2)(image=frame1,
                                                                                                         image0=frame2)
            frame1 = transformed['image']
            frame2 = transformed['image0']

        if self.frames_transforms is not None:

            frame1 = self.frames_transforms(image=frame1)['image']
            frame2 = self.frames_transforms(image=frame2)['image']

        flow = torch.from_numpy(flow.transpose(2, 0, 1).copy())
        frames = torch.cat((frame1, frame2), dim=0)
        return frames, flow

    def __len__(self):
        return len(self.imgPairs)


class SintelDataset(OpticalFlowDataset):
    def __init__(self, root, frames_transforms=None, frames_aug_transforms=None, co_aug_transforms=None):
        super().__init__(root, frames_transforms, frames_aug_transforms, co_aug_transforms)
        scenes = sorted(os.listdir(os.path.join(self.root, "final")))
        genres = ['albedo', 'clean', 'final']
        for scene in scenes:
            frame_names = [path.split('.')[0] for path in sorted(os.listdir(os.path.join(self.root, "final", scene)))]
            for i in range(len(frame_names) - 1):
                self.flows += [os.path.join(self.root, 'flow', scene, frame_names[i] + ".flo")] * len(genres)

                self.imgPairs += [[os.path.join(self.root, genre, scene, frame_names[i] + '.png'),
                                   os.path.join(self.root, genre, scene, frame_names[i + 1] + '.png')] for genre in
                                  genres]

        assert (len(self.imgPairs) == len(self.flows))


class FlyingChairs(OpticalFlowDataset):
    def __init__(self, root, frames_transforms=None, frames_aug_transforms=None, co_aug_transforms=None):
        super().__init__(root, frames_transforms, frames_aug_transforms, co_aug_transforms)
        for file_name in sorted(os.listdir(self.root)):
            if "flow" in file_name:
                self.flows.append(os.path.join(self.root, file_name))
            elif "img1" in file_name:
                self.imgPairs.append([os.path.join(self.root, file_name)])
            else:
                self.imgPairs[-1].append(os.path.join(self.root, file_name))

        assert (len(self.imgPairs) == len(self.flows))


class Chairs3D(OpticalFlowDataset):
    def __init__(self, root, frames_transforms=None, frames_aug_transforms=None, co_aug_transforms=None):
        super().__init__(root, frames_transforms, frames_aug_transforms, co_aug_transforms)
        self.func = readPFM
        for flow, im1, im2 in zip(sorted(os.listdir(os.path.join(self.root, "flow"))),
                                  sorted(os.listdir(os.path.join(self.root, "t0"))),
                                  sorted(os.listdir(os.path.join(self.root, "t1")))):
            self.flows.append(os.path.join(self.root, "flow", flow))
            self.imgPairs.append([os.path.join(self.root,"t0", im1)])
            self.imgPairs[-1].append(os.path.join(self.root,"t1", im2))

        assert (len(self.imgPairs) == len(self.flows))


class OceanData(OpticalFlowDataset):
    def __init__(self, root, frames_transforms=None, frames_aug_transforms=None, co_aug_transforms=None):
        super().__init__(root, frames_transforms, frames_aug_transforms, co_aug_transforms)
        file_names = sorted(os.listdir(self.root))

        for i in range(0, len(file_names) - 1):
            self.imgPairs.append([os.path.join(self.root, file_names[i]), os.path.join(self.root, file_names[i + 1])])

    def __getitem__(self, idx):

        frame1 = cv2.cvtColor(cv2.imread(self.imgPairs[idx][0]), cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(cv2.imread(self.imgPairs[idx][1]), cv2.COLOR_BGR2RGB)

        if self.co_aug_transforms is not None:
            transformed = albu.Compose(self.co_aug_transforms, p=1, additional_targets=self.target2)(image=frame1,
                                                                                                     image0=frame2)
            frame1 = transformed['image']
            frame2 = transformed['image0']

        if self.frames_aug_transforms is not None:
            transformed = albu.Compose(self.frames_aug_transforms, p=1, additional_targets=self.target2)(image=frame1,
                                                                                                         image0=frame2)
            frame1 = transformed['image']
            frame2 = transformed['image0']

        if self.frames_transforms is not None:

            frame1 = self.frames_transforms(image=frame1)['image']
            frame2 = self.frames_transforms(image=frame2)['image']

        frames = torch.cat((frame1, frame2), dim=0)
        return frames

class TartanFlowDataset(OpticalFlowDataset):
    """
    TartanAir optical-flow dataset for your preprocessed structure:

        resized_image_lcam_front/<frame>_lcam_front.png
        flow_lcam_front_gt/<frame>_lcam_front_depth_flow.npy

    Uses split files:                    # print("Pred flows shape:", pred_flows.shape)

        splits/tartan/train_files.txt
        splits/tartan/val_files.txt

    Returns:
        frames : [6, H, W]
        flow   : [2, H, W]
    """

    def __init__(self, root, split_file,
                 frames_transforms=None,
                 frames_aug_transforms=None,
                 co_aug_transforms=None):

        super().__init__(root, frames_transforms, frames_aug_transforms, co_aug_transforms)

        self.root = root
        self.split_file = split_file
        self.func = np.load                      # load flow .npy
        self.flow_dirname = "flow_lcam_front_gt" # <-- your actual folder

        self._read_split_file()


    # -------------------------------------------------------------------------
    # Read split file and build imgPairs[] and flows[]
    # -------------------------------------------------------------------------
    def _read_split_file(self):
        assert os.path.isfile(self.split_file), f"Split file not found: {self.split_file}"

        with open(self.split_file, "r") as f:
            entries = f.read().splitlines()

        for line in entries:
            env, difficulty, traj, frame_str = line.split()

            traj_dir = os.path.join(self.root, env, difficulty, traj)

            img_dir  = os.path.join(traj_dir, "resized_image_lcam_front")
            flow_dir = os.path.join(traj_dir, self.flow_dirname)

            # Build image paths
            img0 = os.path.join(img_dir, f"{frame_str}_lcam_front.png")
            img1 = os.path.join(img_dir, f"{str(int(frame_str)+1).zfill(6)}_lcam_front.png")

            if not (os.path.exists(img0) and os.path.exists(img1)):
                continue

            # Correct flow filename based on your dataset
            flow_path = os.path.join(flow_dir, f"{frame_str}_lcam_front_depth_flow.npy")
            if not os.path.exists(flow_path):
                continue

            self.imgPairs.append([img0, img1])
            self.flows.append(flow_path)

        print(f"[TartanFlowDataset] Loaded {len(self.imgPairs)} frame pairs from {self.split_file}")


    # -------------------------------------------------------------------------
    #   __getitem__
    # -------------------------------------------------------------------------
    def __getitem__(self, idx):
        # Load flow as numpy (H, W, 2)
        flow = self.func(self.flows[idx]).astype(np.float32)

        # print("DEBUG flow (before returns): min/max/mean/std =",
        # float(flow.min()), float(flow.max()),
        # float(flow.mean()), float(flow.std()))

        # Load frames (H, W, 3)
        frame1 = cv2.cvtColor(cv2.imread(self.imgPairs[idx][0]), cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(cv2.imread(self.imgPairs[idx][1]), cv2.COLOR_BGR2RGB)

        # print("====================================================================")
        # print("Original frame1 shape:", frame1.shape)
        # print("pixel value range:", frame1.min(), frame1.max())
        # # print pixel value statistics
        # print("Mean pixel value:", frame1.mean())
        # print("Std pixel value:", frame1.std())
        # print("full image pixel count:", frame1.size)
        
        # ---------------------------------------------------------------------
        # Resize flow if image size != flow size
        # ---------------------------------------------------------------------
        H, W = flow.shape[:2]
        target_h, target_w = frame1.shape[:2]

        if (H != target_h) or (W != target_w):
            scale_y = target_h / H
            scale_x = target_w / W

            # Resize the flow field
            flow_resized = cv2.resize(flow, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # Scale flow vectors
            flow_resized[..., 0] *= scale_x   # horizontal displacement
            flow_resized[..., 1] *= scale_y   # vertical displacement

            flow = flow_resized
        # ---------------------------------------------------------------------

        # --------------------------- Co-augment ------------------------------
        if self.co_aug_transforms is not None:
            transformed = albu.Compose(
                self.co_aug_transforms, p=1,
                additional_targets={'image0': 'image', 'image1': 'image'}
            )(image=frame1, image0=frame2, image1=flow)

            frame1 = transformed['image']
            frame2 = transformed['image0']
            flow   = transformed['image1']

        # --------------------------- Frame augment ---------------------------
        if self.frames_aug_transforms is not None:
            transformed = albu.Compose(
                self.frames_aug_transforms, p=1,
                additional_targets={'image0': 'image'}
            )(image=frame1, image0=frame2)

            frame1 = transformed['image']
            frame2 = transformed['image0']

        # --------------------------- Final transforms ------------------------
        if self.frames_transforms is not None:
            frame1 = self.frames_transforms(image=frame1)['image']
            frame2 = self.frames_transforms(image=frame2)['image']

        # Convert flow to torch [2, H, W]
        flow = torch.from_numpy(flow.transpose(2, 0, 1).copy())

        # Concatenate images: [3+3, H, W]
        frames = torch.cat((frame1, frame2), dim=0)

        return frames, flow

def getPoseDataloaders(batch_size, root, split_root="./splits/tartan", augment=False):
    train_split = os.path.join(split_root, "train_files.txt")
    val_split   = os.path.join(split_root, "val_files.txt")
    test_split  = os.path.join(split_root, "test_files.txt")

    # normal transforms (you may use your Albumentations setup)
    frames_transforms = albu.Compose([
        albu.Normalize((0.,0.,0.), (1.,1.,1.)),
        ToTensorV2()
    ])

    train_set = TartanPoseDataset(root, train_split, frames_transforms)
    val_set   = TartanPoseDataset(root, val_split, frames_transforms)
    test_set  = TartanPoseDataset(root, test_split, frames_transforms)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


def getDataloaders(batch_size, root='../sintel/training', frames_transforms=None, frames_aug_transforms=None,
                   co_aug_transforms=None):

    test_size = 50
    if "sintel" in root:
        train_dataset = SintelDataset(root, frames_transforms, frames_aug_transforms, co_aug_transforms)
        val_dataset = SintelDataset(root, frames_transforms)
        val_size = 133
    elif "ocean" in root:
        train_dataset = OceanData(root, frames_transforms, frames_aug_transforms, co_aug_transforms)
        val_dataset = OceanData(root, frames_transforms)
        val_size = 100
    elif "Flying" in root:
        print("Using Flying Chairs dataset")
        train_dataset = FlyingChairs(root, frames_transforms, frames_aug_transforms, co_aug_transforms)
        val_dataset = FlyingChairs(root, frames_transforms)
        val_size = 640
        test_size = 200
    elif "tartan" in root.lower():
        print("Using TartanAir dataset")

        split_dir = os.path.join(os.path.dirname(__file__), "splits", "tartan")

        train_split = os.path.join(split_dir, "train_files.txt")
        val_split   = os.path.join(split_dir, "val_files.txt")
        test_split  = os.path.join(split_dir, "test_files.txt")

        train_dataset = TartanFlowDataset(
            root, train_split,
            frames_transforms, frames_aug_transforms, co_aug_transforms
        )

        val_dataset = TartanFlowDataset(
            root, val_split,
            frames_transforms
        )

        test_dataset = TartanFlowDataset(
            root, test_split,
            frames_transforms
        )

        # ---- NO RANDOM SPLIT HERE ----
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            pin_memory=torch.cuda.is_available(), num_workers=4
        )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            pin_memory=torch.cuda.is_available(), num_workers=4
        )

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            pin_memory=torch.cuda.is_available(), num_workers=4
        )

        return train_loader, val_loader, test_loader

        
    else:
        print("Using Chairs3D dataset")
        train_dataset = Chairs3D(os.path.join(root, "train"), frames_transforms, frames_aug_transforms, co_aug_transforms)
        val_dataset = Chairs3D(os.path.join(root, "train"), frames_transforms)
        test_dataset = Chairs3D(os.path.join(root, "test"), frames_transforms)
        val_size = 640
        test_size = 0

    torch.manual_seed(1)
    indices = torch.randperm(len(train_dataset)).tolist()
    train_idx, valid_idx, test_idx = indices[val_size + test_size:], indices[:val_size], indices[
                                                                                         val_size:test_size + val_size]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)
    if len(test_idx) > 0:
        test_sampler = SubsetRandomSampler(test_idx)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                                pin_memory=torch.cuda.is_available(), num_workers=4)
    else:
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                pin_memory=torch.cuda.is_available(), num_workers=4)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              pin_memory=torch.cuda.is_available(), num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                            pin_memory=torch.cuda.is_available(), num_workers=4)

    return train_loader, val_loader, test_loader


