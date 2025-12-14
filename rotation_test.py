import torch
import math

def axis_angle_to_quaternion(w):
    theta = torch.norm(w, dim=-1, keepdim=True)
    axis = w / (theta + 1e-8)
    half = 0.5 * theta
    q_xyz = axis * torch.sin(half)
    q_w   = torch.cos(half)
    q = torch.cat([q_xyz, q_w], dim=-1)
    return q / (q.norm(dim=-1, keepdim=True) + 1e-8)

def quat_geodesic(q1, q2):
    q1 = q1 / (q1.norm(dim=-1, keepdim=True) + 1e-8)
    q2 = q2 / (q2.norm(dim=-1, keepdim=True) + 1e-8)
    dot = torch.sum(q1*q2, dim=-1).abs().clamp(-1+1e-7, 1-1e-7)
    return 2.0 * torch.acos(dot)

# Pick some axis-angle
w = torch.tensor([[0.2, -0.1, 0.05]], dtype=torch.float32)  # radians
axis = w / (w.norm(dim=-1, keepdim=True) + 1e-8)

# Make an equivalent representation (add 2π along same axis)
w_equiv = w + (2*math.pi) * axis

# Your current loss (parameter-space)
l1_param = torch.abs(w - w_equiv).mean()

# Correct rotation distance
q1 = axis_angle_to_quaternion(w)
q2 = axis_angle_to_quaternion(w_equiv)
geo = quat_geodesic(q1, q2)

print("L1(axis-angle params) =", float(l1_param))
print("Geodesic rotation angle =", float(geo), "rad")