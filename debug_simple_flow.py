import torch
from models import FlowNetS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = FlowNetS().to(device)
net.train()

x = torch.randn(1, 6, 320, 320, device=device)  # random input
flows = net(x)  # should be a tuple of 5 flows

print("Number of flows:", len(flows))
for lvl, f in enumerate(flows):
    print(f"Level {lvl} shape:", f.shape)
    print("  min :", f.min().item())
    print("  max :", f.max().item())
    print("  mean:", f.mean().item())
    print("  std :", f.std().item())
    print("  unique sample:", torch.unique(f[0,0,:5,:5]))
