import torch
import torch.nn.functional as F
import numpy as np

@torch.no_grad()
def resize_flow(flow, H, W, scale_vectors: bool):
    """
    flow: [B,2,h,w] -> [B,2,H,W]
    If scale_vectors=True, scales u/v by resolution ratio (physically correct if flow is in pixels at that grid).
    If scale_vectors=False, preserves magnitudes (matches your current training convention).
    """
    B, C, h, w = flow.shape
    out = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=False)
    if scale_vectors:
        out = out.clone()
        out[:, 0] *= (W / w)
        out[:, 1] *= (H / h)
    return out

@torch.no_grad()
def EPE(flow_pred, flow_true, *, scale_vectors_when_upsampling: bool = False, mask_invalid: bool = True):
    """
    Returns mean EPE over valid pixels.
    - Default scale_vectors_when_upsampling=False to match your training/eval convention.
    """
    B, _, H, W = flow_true.shape
    flow_pred = resize_flow(flow_pred, H, W, scale_vectors_when_upsampling)

    epe_map = torch.sqrt(((flow_pred - flow_true) ** 2).sum(dim=1))  # [B,H,W]

    if mask_invalid:
        valid = torch.isfinite(flow_true).all(dim=1)
        return epe_map[valid].mean()
    else:
        return epe_map.mean()

@torch.no_grad()
def AAE(flow_pred, flow_true, *, scale_vectors_when_upsampling: bool = False, degrees: bool = True, mask_invalid: bool = True):
    """
    Middlebury-style AAE with +1 trick:
      acos( (p·t + 1) / (||p'|| ||t'||) )
    """
    B, _, H, W = flow_true.shape
    flow_pred = resize_flow(flow_pred, H, W, scale_vectors_when_upsampling)

    # Correct dot product: pred · true
    dot = (flow_pred * flow_true).sum(dim=1) + 1.0
    norm_p = torch.sqrt((flow_pred ** 2).sum(dim=1) + 1.0)
    norm_t = torch.sqrt((flow_true ** 2).sum(dim=1) + 1.0)

    cos = dot / (norm_p * norm_t + 1e-8)
    cos = torch.clamp(cos, -1.0, 1.0)
    ang = torch.acos(cos)  # radians

    if degrees:
        ang = ang * (180.0 / np.pi)

    if mask_invalid:
        valid = torch.isfinite(flow_true).all(dim=1)
        return ang[valid].mean()
    else:
        return ang.mean()

@torch.no_grad()
def evaluate(flow_pred, flow_true, *, scale_vectors_when_upsampling: bool = False, degrees: bool = True):
    epe = EPE(flow_pred, flow_true, scale_vectors_when_upsampling=scale_vectors_when_upsampling)
    aae = AAE(flow_pred, flow_true, scale_vectors_when_upsampling=scale_vectors_when_upsampling, degrees=degrees)
    return epe, aae
