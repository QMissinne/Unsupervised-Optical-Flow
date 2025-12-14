import numpy as np
import sys
import os
import cv2
import torch.nn.functional as F
import torch
import re
import matplotlib.pyplot as plt

TAG_FLOAT = 202021.25
# testing vim, and git push

# ---------------------------------------------------------
# DEBUG EPE
# ---------------------------------------------------------

DEBUG_EPE = False
DEBUG_EPE_MAXPRINT = 5
_epe_call_count = 0


def readflo(file):
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file, 'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    data = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
    flow = np.resize(data, (int(h[0]), int(w[0]), 2))
    f.close()

    return flow


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data[:, :, :2]


def makeColorwheel():

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])  # r g b

    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY, 1) / RY)
    col += RY

    # YG
    colorwheel[col:YG + col, 0] = 255 - np.floor(255 * np.arange(0, YG, 1) / YG)
    colorwheel[col:YG + col, 1] = 255
    col += YG

    # GC
    colorwheel[col:GC + col, 1] = 255
    colorwheel[col:GC + col, 2] = np.floor(255 * np.arange(0, GC, 1) / GC)
    col += GC

    # CB
    colorwheel[col:CB + col, 1] = 255 - np.floor(255 * np.arange(0, CB, 1) / CB)
    colorwheel[col:CB + col, 2] = 255
    col += CB

    # BM
    colorwheel[col:BM + col, 2] = 255
    colorwheel[col:BM + col, 0] = np.floor(255 * np.arange(0, BM, 1) / BM)
    col += BM

    # MR
    colorwheel[col:MR + col, 2] = 255 - np.floor(255 * np.arange(0, MR, 1) / MR)
    colorwheel[col:MR + col, 0] = 255
    return colorwheel


def computeColor(u, v):
    colorwheel = makeColorwheel()
    nan_u = np.isnan(u)
    nan_v = np.isnan(v)
    nan_u = np.where(nan_u)
    nan_v = np.where(nan_v)

    u[nan_u] = 0
    u[nan_v] = 0
    v[nan_u] = 0
    v[nan_v] = 0

    ncols = colorwheel.shape[0]
    radius = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)  # -1~1 maped to 1~ncols
    k0 = fk.astype(np.uint8)  # 1, 2, ..., ncols
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    img = np.empty([k1.shape[0], k1.shape[1], 3])
    ncolors = colorwheel.shape[1]
    for i in range(ncolors):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255
        col1 = tmp[k1] / 255
        col = (1 - f) * col0 + f * col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx] * (1 - col[idx])  # increase saturation with radius
        col[~idx] *= 0.75  # out of range
        img[:, :, 2 - i] = np.floor(255 * col).astype(np.uint8)

    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img.astype(np.uint8)


def computeImg(flow, verbose=False, savePath=None):
    eps = sys.float_info.epsilon
    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWN_FLOW = 1e10
    if flow.shape[0] == 2:
        u = flow[0, :, :]
        v = flow[1, :, :]
    else:
        u = flow[:, :, 0]
        v = flow[:, :, 1]

    maxu = -999
    maxv = -999

    minu = 999
    minv = 999

    maxrad = -1
    # fix unknown flow
    greater_u = np.where(u > UNKNOWN_FLOW_THRESH)
    greater_v = np.where(v > UNKNOWN_FLOW_THRESH)
    u[greater_u] = 0
    u[greater_v] = 0
    v[greater_u] = 0
    v[greater_v] = 0

    maxu = max([maxu, np.amax(u)])
    minu = min([minu, np.amin(u)])

    maxv = max([maxv, np.amax(v)])
    minv = min([minv, np.amin(v)])
    rad = np.sqrt(np.multiply(u, u) + np.multiply(v, v))
    maxrad = max([maxrad, np.amax(rad)])

    u = u / (maxrad + eps)
    v = v / (maxrad + eps)
    img = computeColor(u, v)
    if savePath is not None:
        cv2.imwrite(savePath, img)
    if verbose:
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def computerArrows(flow, step=16, verbose=False, savePath=None, img=None):
    h, w = flow.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    if img is None:
        vis = np.ones((h, w)).astype('uint8')*255
    else:
        vis = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x2, y2), (x1, y1) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

    if savePath is not None:
        cv2.imwrite(savePath, vis)
    if verbose:
        cv2.imshow('arrowsViz', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return vis


def disp_function(pred_flo, true_flo):
    height, width = true_flo.shape[1:]
    pred_flo = F.interpolate(pred_flo, (height, width), mode='bilinear', align_corners=False)
    pred_flo = computeImg(pred_flo[0].cpu().numpy())
    if true_flo.shape[0] == 2:
        true_flo = computeImg(true_flo.cpu().numpy())
        image1, image2 = np.expand_dims(pred_flo, axis=0), np.expand_dims(true_flo, axis=0)
        return np.concatenate((image1, image2), axis=0)
    else:
        true_flo = true_flo[:3]
        true_flo = true_flo.transpose(0, 2)
        true_flo = true_flo.transpose(0, 1)
        image1, image2 = np.expand_dims(pred_flo, axis=0), np.expand_dims(true_flo.cpu().numpy(), axis=0)
        return np.concatenate((image1, image2), axis=0)


def EPE(flow_pred, flow_true, real=False):
    global _epe_call_count
    _epe_call_count += 1

    if DEBUG_EPE and _epe_call_count <= DEBUG_EPE_MAXPRINT:
        print("\n[EPE DEBUG] call", _epe_call_count, "real=", real)
        print("  pred shape:", tuple(flow_pred.shape), "dtype:", flow_pred.dtype, "device:", flow_pred.device)
        print("  true shape:", tuple(flow_true.shape), "dtype:", flow_true.dtype, "device:", flow_true.device)
        pred_stats = (flow_pred.min().item(), flow_pred.max().item(), flow_pred.mean().item(), flow_pred.std().item())
        true_stats = (flow_true.min().item(), flow_true.max().item(), flow_true.mean().item(), flow_true.std().item())
        print("  pred stats min/max/mean/std:", pred_stats)
        print("  true stats min/max/mean/std:", true_stats)

    if real:
        batch_size, _, h, w = flow_true.shape
        if DEBUG_EPE and _epe_call_count <= DEBUG_EPE_MAXPRINT:
            print(f"    [real=True] upsamping prediction from {flow_pred.shape[2:]} to {(h, w)}")
        flow_pred = F.interpolate(flow_pred, (h, w), mode='bilinear', align_corners=False)

        if DEBUG_EPE and _epe_call_count <= DEBUG_EPE_MAXPRINT:
            pred_stats2 = (flow_pred.min().item(), flow_pred.max().item(), flow_pred.mean().item(), flow_pred.std().item())
            print("    after upsample, pred stats min/max/mean/std:", pred_stats2)

    else:
        batch_size, _, h, w = flow_pred.shape
        if DEBUG_EPE and _epe_call_count <= DEBUG_EPE_MAXPRINT:
            print(f"    [real=False] upsamping ground truth from {flow_true.shape[2:]} to {(h, w)}")
        flow_true = F.interpolate(flow_true, (h, w), mode='area')
        if DEBUG_EPE and _epe_call_count <= DEBUG_EPE_MAXPRINT:
            true_stats2 = (flow_true.min().item(), flow_true.max().item(), flow_true.mean().item(), flow_true.std().item())
            print("    after upsample, true stats min/max/mean/std:", true_stats2)
    
    epe_map = torch.norm(flow_pred - flow_true, 2, 1) # B x H x W
    epe = epe_map.mean()

    # print("EPE:", epe.item())

    return epe

def EPE_pixel(flow_pred, flow_true):
    """
    Pixel-consistent EPE:
    - upsamples pred to GT resolution
    - scales vector magnitudes by resolution ratio
    """
    _, _, H, W = flow_true.shape
    _, _, h, w = flow_pred.shape

    flow_pred_up = F.interpolate(flow_pred, (H, W), mode='bilinear', align_corners=False)

    scale_x = W / w
    scale_y = H / h

    flow_pred_up = flow_pred_up.clone()
    flow_pred_up[:, 0] *= scale_x
    flow_pred_up[:, 1] *= scale_y

    return torch.norm(flow_pred_up - flow_true, 2, 1).mean()


def EPE_all(flows_pred, flow_true, weights=(0.005, 0.01, 0.02, 0.08, 0.32)):

    if len(flows_pred) < 5:
        weights = [0.005]*len(flows_pred)
    loss = 0

    for i in range(len(weights)):
        loss += weights[i] * EPE(flows_pred[i], flow_true, real=False)

    return loss


def AAE(flow_pred, flow_true):
    batch_size, _, h, w = flow_true.shape
    flow_pred = F.interpolate(flow_pred, (h, w), mode='bilinear', align_corners=False)
    numerator = torch.sum(torch.mul(flow_pred, flow_pred), dim=1) + 1
    denominator = torch.sqrt(torch.sum(flow_pred ** 2, dim=1) + 1) * torch.sqrt(torch.sum(flow_true ** 2, dim=1) + 1)
    result = torch.clamp(torch.div(numerator, denominator), min=-1.0, max=1.0)

    return torch.acos(result).mean()


def evaluate(flow_pred, flow_true):

    epe = EPE(flow_pred, flow_true, real=True)
    aae = AAE(flow_pred, flow_true)
    return epe, aae


def charbonnier(x, alpha=0.25, epsilon=1.e-3):
    return torch.pow(torch.pow(x, 2) + epsilon**2, alpha)


def smoothness_loss(flow):
    b, c, h, w = flow.size()
    v_translated = torch.cat((flow[:, :, 1:, :], torch.zeros(b, c, 1, w, device=flow.device)), dim=-2)
    h_translated = torch.cat((flow[:, :, :, 1:], torch.zeros(b, c, h, 1, device=flow.device)), dim=-1)
    s_loss = charbonnier(flow - v_translated) + charbonnier(flow - h_translated)
    s_loss = torch.sum(s_loss, dim=1) / 2

    return torch.sum(s_loss)/b


def photometric_loss(wraped, frame1):
    h, w = wraped.shape[2:]
    frame1 = F.interpolate(frame1, (h, w), mode='bilinear', align_corners=False)
    p_loss = charbonnier(wraped - frame1)
    p_loss = torch.sum(p_loss, dim=1)/3
    return torch.sum(p_loss)/frame1.size(0)


def unsup_loss(pred_flows, wraped_imgs, frame1, weights=(0.005, 0.01, 0.02, 0.08, 0.32)):
    if len(pred_flows) < 5:
        weights = [0.005]*len(pred_flows)
    photometric = 0
    smooth = 0
    for i in range(len(weights)):
        photometric += weights[i] * photometric_loss(wraped_imgs[i], frame1)
        smooth += 0.45 * (weights[i] * smoothness_loss(pred_flows[i]))

    loss = photometric + smooth
    return loss, photometric, smooth

def _debug_test_scaling_effect():
    import torch
    import torch.nn.functional as F

    print("\n============================")
    print("TEST A: scaling effect demo")
    print("============================")

    # Create a low-res flow field with a known constant displacement:
    # say "1 pixel at 80x80 grid"
    pred = torch.zeros(1, 2, 80, 80)
    pred[:, 0] = 1.0  # +1 in x everywhere

    # Construct the "physically equivalent" full-res flow at 320x320:
    # if 80->320 is x4, then equivalent displacement is +4 px at 320
    gt = torch.zeros(1, 2, 320, 320)
    gt[:, 0] = 4.0

    # Original EPE(real=True): upsamples pred but does NOT scale magnitude.
    # It will compare ~1 vs 4 => EPE ~3
    epe_orig = EPE(pred, gt, real=True).item()
    print("Original EPE(real=True) on 1@80 vs 4@320 ->", epe_orig, "(should be ~3.0 if no scaling)")

def debug_overlay_flow(
    frames,
    flow,
    step=10,
    max_arrows=2000,
    out_dir="visualizations/debug",
    fname="debug_flow.png",
    title=None
):
    """
    frames: Tensor [6, H, W] or [B, 6, H, W]
    flow:   Tensor [2, H, W] or [B, 2, H, W]
    """

    os.makedirs(out_dir, exist_ok=True)

    # ---- handle batch ----
    if frames.dim() == 4:
        frames = frames[0]
    if flow.dim() == 4:
        flow = flow[0]

    frames = frames.detach().cpu()
    flow   = flow.detach().cpu()

    # ---- split frames ----
    img1 = frames[:3]
    img2 = frames[3:6]

    def norm_img(x):
        x = x - x.min()
        x = x / (x.max() + 1e-6)
        return x

    img1 = norm_img(img1)
    img2 = norm_img(img2)

    img1_np = img1.permute(1, 2, 0).numpy()
    img2_np = img2.permute(1, 2, 0).numpy()

    H, W = img1.shape[1:]

    # ---- overlay image ----
    overlay = torch.zeros(3, H, W)
    overlay[0] = img1.mean(0)  # red channel
    overlay[2] = img2.mean(0)  # blue channel
    overlay_np = overlay.permute(1, 2, 0).numpy()

    # ---- flow arrows ----
    fx = flow[0].numpy()
    fy = flow[1].numpy()

    ys, xs = np.mgrid[0:H:step, 0:W:step]
    xs = xs.flatten()
    ys = ys.flatten()

    fx = fx[ys, xs]
    fy = fy[ys, xs]

    if len(xs) > max_arrows:
        idx = np.random.choice(len(xs), max_arrows, replace=False)
        xs, ys, fx, fy = xs[idx], ys[idx], fx[idx], fy[idx]

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # frame 1
    axes[0].imshow(img1_np)
    axes[0].set_title("Frame t")
    axes[0].axis("off")

    # frame 2
    axes[1].imshow(img2_np)
    axes[1].set_title("Frame t+1")
    axes[1].axis("off")

    # overlay + flow
    axes[2].imshow(overlay_np)
    axes[2].quiver(
        xs, ys, fx, fy,
        color="white",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.002
    )
    axes[2].set_title("Overlay + GT Flow")
    axes[2].axis("off")

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    save_path = os.path.join(out_dir, fname)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"[DEBUG] Saved flow visualization to: {save_path}")

def debug_flow_correspondence(
    frames,
    flow,
    out_dir="visualizations/debug",
    fname="correspondence_check.png",
    patch_size=5
):
    """
    frames: Tensor [6,H,W] or [B,6,H,W]
    flow:   Tensor [2,H,W] or [B,2,H,W]
    """

    os.makedirs(out_dir, exist_ok=True)

    # -----------------------
    # Handle batch
    # -----------------------
    if frames.dim() == 4:
        frames = frames[0]
    if flow.dim() == 4:
        flow = flow[0]

    frames = frames.detach().cpu()
    flow   = flow.detach().cpu()

    img1 = frames[:3]
    img2 = frames[3:6]

    # Normalize images for visualization
    def norm(x):
        x = x - x.min()
        x = x / (x.max() + 1e-6)
        return x

    img1 = norm(img1).permute(1, 2, 0).numpy()
    img2 = norm(img2).permute(1, 2, 0).numpy()

    H, W, _ = img1.shape

    # -----------------------
    # Pick 9 locations
    # -----------------------
    ys = [H // 6, H // 2, 5 * H // 6]
    xs = [W // 6, W // 2, 5 * W // 6]

    points = [(y, x) for y in ys for x in xs]

    half = patch_size // 2

    img1_marked = img1.copy()
    img2_marked = img2.copy()

    # -----------------------
    # Paint patches
    # -----------------------
    for (y, x) in points:
        # Clamp
        y0 = max(0, y - half)
        y1 = min(H, y + half + 1)
        x0 = max(0, x - half)
        x1 = min(W, x + half + 1)

        # Paint in frame t
        img1_marked[y0:y1, x0:x1] = [1.0, 1.0, 0.0]  # yellow

        # Flow displacement
        dx = flow[0, y, x].item()
        dy = flow[1, y, x].item()

        y2 = int(round(y + dy))
        x2 = int(round(x + dx))

        # Clamp destination
        y2 = max(0, min(H - 1, y2))
        x2 = max(0, min(W - 1, x2))

        y0 = max(0, y2 - half)
        y1 = min(H, y2 + half + 1)
        x0 = max(0, x2 - half)
        x1 = min(W, x2 + half + 1)

        # Paint in frame t+1
        img2_marked[y0:y1, x0:x1] = [1.0, 1.0, 0.0]  # yellow

        # draw line for clarity
        rr = np.linspace(y, y2, 20).astype(int)
        cc = np.linspace(x, x2, 20).astype(int)
        rr = np.clip(rr, 0, H-1)
        cc = np.clip(cc, 0, W-1)
        img2_marked[rr, cc] = [1.0, 0.0, 0.0]  # red line


    # -----------------------
    # Stack & save
    # -----------------------
    vis = np.concatenate([img1_marked, img2_marked], axis=1)

    plt.figure(figsize=(10, 5))
    plt.imshow(vis)
    plt.title("GT Flow Correspondence Check (Frame t → t+1)")
    plt.axis("off")
    plt.tight_layout()

    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"[DEBUG] Saved correspondence check to: {path}")

