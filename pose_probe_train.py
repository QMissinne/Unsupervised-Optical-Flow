import argparse
import time
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models import FlowNetS
from models import FlowNetSEncoder, FlowPoseProbe  # new pose module
from dataset import getPoseDataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------
# AverageMeter – same as existing sup/unsup scripts
# -------------------------------------------------------------------
class AverageMeter(object):
    def __init__(self, keep_all=False):
        self.reset()
        self.data = [] if keep_all else None

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        if self.data is not None:
            self.data.append(value)
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


# -------------------------------------------------------------------
# Simple pose loss: L1 on rotation + translation
# -------------------------------------------------------------------
def pose_loss(pred, target):

    pred_axisangle, pred_trans = pred # shape missmatch! pred is (B,1,1,3), target is (B,1,1,1,3)
    tgt_axisangle, tgt_trans = target # shape missmatch! pred is (B,1,1,3), target is (B,1,1,1,3)

    # squeeze target to match pred shape
    # (B,1,1,1,3) -> (B,1,1,3)
    tgt_axisangle = tgt_axisangle.squeeze(3)
    tgt_trans = tgt_trans.squeeze(3)    


    loss_rot = nn.functional.l1_loss(pred_axisangle, tgt_axisangle)
    loss_trans = nn.functional.l1_loss(pred_trans, tgt_trans)

    rmse_rot = torch.sqrt(torch.mean((pred_axisangle - tgt_axisangle) ** 2))
    rmse_trans = torch.sqrt(torch.mean((pred_trans - tgt_trans) ** 2))

    total_loss = loss_rot + loss_trans
    return (
        total_loss,
        loss_rot.item(),
        loss_trans.item(),
        rmse_rot.item(),
        rmse_trans.item()
    )


# -------------------------------------------------------------------
# Training loop adapted from your style
# -------------------------------------------------------------------
def epoch(model, dataloader, optimizer=None):
    model.train() if optimizer is not None else model.eval()

    avg_loss = AverageMeter()
    avg_rot_loss = AverageMeter()
    avg_trans_loss = AverageMeter()
    avg_batch_time = AverageMeter()
    avg_rot_rmse = AverageMeter()
    avg_trans_rmse = AverageMeter()

    tic = time.time()

    for i, batch in enumerate(dataloader):

        imgs, pose_target = batch
        imgs = imgs.to(device)
        pose_target = [t.to(device) for t in pose_target]

        with torch.set_grad_enabled(optimizer is not None):
            pred = model(imgs)
            loss, loss_rot, loss_trans, rmse_rot, rmse_trans = pose_loss(pred, pose_target)

        # optimization
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # timing + logs
        batch_time = time.time() - tic
        tic = time.time()

        avg_loss.update(loss.item())
        avg_rot_loss.update(loss_rot)
        avg_trans_loss.update(loss_trans)
        avg_batch_time.update(batch_time)
        avg_rot_rmse.update(rmse_rot)
        avg_trans_rmse.update(rmse_trans)

        if i % 50 == 0:
            print("[{} Batch {}/{}] "
                  "Time {:.3f}s ({:.3f}s) "
                  "Loss {:.4f} ({:.4f}) "
                  "Rot {:.4f} ({:.4f}) "
                  "Trans {:.4f} ({:.4f})".format(
                    "TRAIN" if optimizer is not None else "VAL",
                    i, len(dataloader),
                    avg_batch_time.val, avg_batch_time.avg,
                    avg_loss.val, avg_loss.avg,
                    avg_rot_loss.val, avg_rot_loss.avg,
                    avg_trans_loss.val, avg_trans_loss.avg))

    print(
        "\n==> Epoch summary:"
        " Loss {:.4f},"
        " Rot_L1 {:.4f}, Trans_L1 {:.4f},"
        " Rot_RMSE {:.4f}, Trans_RMSE {:.4f}\n".format(
            avg_loss.avg,
            avg_rot_loss.avg, avg_trans_loss.avg,
            avg_rot_rmse.avg, avg_trans_rmse.avg,
        )
    )


    return (
        avg_loss.avg,
        avg_rot_loss.avg,
        avg_trans_loss.avg,
        avg_rot_rmse.avg,
        avg_trans_rmse.avg
    )

# -------------------------------------------------------------------
# Main script
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to pretrained FlowNetS (supervised or unsupervised).")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Dataset root path for pose data.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    # -------------------------------------------------------------
    # 1. Load pretrained FlowNetS
    # -------------------------------------------------------------
    pretrained = FlowNetS()
    print("Loading pretrained weights from:", args.model_checkpoint)
    state = torch.load(args.model_checkpoint, map_location=device)
    pretrained.load_state_dict(state["model_state_dict"], strict=False)

    # -------------------------------------------------------------
    # 2. Build the pose probe (frozen encoder + minimal MLP)
    # -------------------------------------------------------------
    print("\n>>> Using SUPERVISED FlowPoseProbe")
    pose_model = FlowPoseProbe(pretrained_flownets=pretrained).to(device)

    # Confirm encoder is frozen
    print("\nTrainable parameters:")
    for name, p in pose_model.named_parameters():
        print(f"{name}: requires_grad={p.requires_grad}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, pose_model.parameters()),
        lr=args.lr
    )

    # -------------------------------------------------------------
    # 3. Data
    # -------------------------------------------------------------
    print(">>> Loading Supervised Pose Dataset (TartanPoseDataset)")
    train_loader, val_loader, test_loader = getPoseDataloaders(
        batch_size=args.batch_size,
        root=args.data_root
    )

    # -------------------------------------------------------------
    # 4. Logging + checkpoint paths
    # -------------------------------------------------------------
    exp_name = "PoseProbe_" + os.path.basename(args.model_checkpoint).replace(".pt", "")
    logdir = os.path.join("runs", exp_name)
    ckptdir = os.path.join("Checkpoints", exp_name)
    os.makedirs(ckptdir, exist_ok=True)
    tb = SummaryWriter(logdir)

    best_val_loss = 1e9

    # -------------------------------------------------------------
    # 5. Training loop
    # -------------------------------------------------------------
    for epoch_idx in range(args.epochs):
        print("\n=========== EPOCH {} ===========".format(epoch_idx + 1))

        train_loss, train_rot, train_trans, train_rmse_rot, train_rmse_trans = \
            epoch(pose_model, train_loader, optimizer)

        val_loss, val_rot, val_trans, val_rmse_rot, val_rmse_trans = \
            epoch(pose_model, val_loader)
        # TensorBoard
        tb.add_scalars("loss/total", {"train": train_loss, "val": val_loss}, epoch_idx)
        tb.add_scalars("loss/rotation", {"train": train_rot, "val": val_rot}, epoch_idx)
        tb.add_scalars("loss/translation", {"train": train_trans, "val": val_trans}, epoch_idx)
        tb.add_scalars("rmse/rotation", {"train": train_rmse_rot, "val": val_rmse_rot}, epoch_idx)
        tb.add_scalars("rmse/translation", {"train": train_rmse_trans, "val": val_rmse_trans}, epoch_idx)


        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model_state_dict": pose_model.state_dict(),
                 "best_loss": best_val_loss},
                os.path.join(ckptdir, "best_pose_probe.pt")
            )
            print("Saved new best model!")

    tb.close()

    # -------------------------------------------------------------
    # 6. Final evaluation (Train + Test)
    # -------------------------------------------------------------

    print("\n=== FINAL EVALUATION ===")

    # Evaluate on TRAIN set (no gradient)
    train_loss, train_rot, train_trans, train_rmse_rot, train_rmse_trans = epoch(pose_model, train_loader)

    # Evaluate on TEST set
    test_loss, test_rot, test_trans, test_rmse_rot, test_rmse_trans = epoch(pose_model, test_loader)

    print("\n== FINAL TRAIN RESULTS ==")
    print(f"Total:          {train_loss:.4f}")
    print(f"Rot:            {train_rot:.4f}")
    print(f"RMSE Rot:       {train_rmse_rot:.4f}")
    print(f"RMSE Trans:     {train_rmse_trans:.4f}")
    print(f"Trans:          {train_trans:.4f}")


    print("\n== FINAL TEST RESULTS ==")
    print(f"Total:          {test_loss:.4f}")
    print(f"Rot:            {test_rot:.4f}")
    print(f"RMSE Rot:       {test_rmse_rot:.4f}")
    print(f"RMSE Trans:     {test_rmse_trans:.4f}")
    print(f"Trans:          {test_trans:.4f}")