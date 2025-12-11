import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_channels, out_channels, kernel_size=3, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=False),
        nn.ReLU(inplace=True))


def predict_flow(in_channels):
    return nn.Conv2d(in_channels, 2, 5, stride=1, padding=2, bias=False)


def upconv(in_channels, out_channels):
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                         nn.ReLU(inplace=True))


def concatenate(tensor1, tensor2, tensor3):
    _, _, h1, w1 = tensor1.shape
    _, _, h2, w2 = tensor2.shape
    _, _, h3, w3 = tensor3.shape
    h, w = min(h1, h2, h3), min(w1, w2, w3)
    return torch.cat((tensor1[:, :, :h, :w], tensor2[:, :, :h, :w], tensor3[:, :, :h, :w]), 1)


class FlowNetS(nn.Module):
    def __init__(self):
        super(FlowNetS, self).__init__()

        self.conv1 = conv(6, 64, kernel_size=7)
        self.conv2 = conv(64, 128, kernel_size=5)
        self.conv3 = conv(128, 256, kernel_size=5)
        self.conv3_1 = conv(256, 256, stride=1)
        self.conv4 = conv(256, 512)
        self.conv4_1 = conv(512, 512, stride=1)
        self.conv5 = conv(512, 512)
        self.conv5_1 = conv(512, 512, stride=1)
        self.conv6 = conv(512, 1024)

        self.predict_flow6 = predict_flow(1024)  # conv6 output
        self.predict_flow5 = predict_flow(1026)  # upconv5 + 2 + conv5_1
        self.predict_flow4 = predict_flow(770)  # upconv4 + 2 + conv4_1
        self.predict_flow3 = predict_flow(386)  # upconv3 + 2 + conv3_1
        self.predict_flow2 = predict_flow(194)  # upconv2 + 2 + conv2

        self.upconv5 = upconv(1024, 512)
        self.upconv4 = upconv(1026, 256)
        self.upconv3 = upconv(770, 128)
        self.upconv2 = upconv(386, 64)

        self.upconvflow6 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upconvflow5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upconvflow4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upconvflow3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

    def forward(self, x):

        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)

        flow6 = self.predict_flow6(out_conv6)
        up_flow6 = self.upconvflow6(flow6)
        out_upconv5 = self.upconv5(out_conv6)
        concat5 = concatenate(out_upconv5, out_conv5, up_flow6)

        flow5 = self.predict_flow5(concat5)
        up_flow5 = self.upconvflow5(flow5)
        out_upconv4 = self.upconv4(concat5)
        concat4 = concatenate(out_upconv4, out_conv4, up_flow5)

        flow4 = self.predict_flow4(concat4)
        up_flow4 = self.upconvflow4(flow4)
        out_upconv3 = self.upconv3(concat4)
        concat3 = concatenate(out_upconv3, out_conv3, up_flow4)

        flow3 = self.predict_flow3(concat3)
        up_flow3 = self.upconvflow3(flow3)
        out_upconv2 = self.upconv2(concat3)
        concat2 = concatenate(out_upconv2, out_conv2, up_flow3)

        finalflow = self.predict_flow2(concat2)

        if self.training:
            return finalflow, flow3, flow4, flow5, flow6
        else:
            return finalflow,


class LightFlowNet(nn.Module):
    def __init__(self):
        super(LightFlowNet, self).__init__()
        pass

    def forward(self, x):
        pass


class PWC_Net(nn.Module):
    def __init__(self):
        super(PWC_Net, self).__init__()
        pass

    def forward(self, x):
        pass


def generate_grid(B, H, W, device):
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    grid = torch.transpose(grid, 1, 2)
    grid = torch.transpose(grid, 2, 3)
    grid = grid.to(device)
    return grid


class Unsupervised(nn.Module):
    def __init__(self, conv_predictor="flownet"):
        super(Unsupervised, self).__init__()

        if "light" in conv_predictor:
            self.predictor = LightFlowNet()
        elif "pwc" in conv_predictor:
            self.predictor = PWC_Net()
        else:
            self.predictor = FlowNetS()

    def stn(self, flow, frame):
        b, _, h, w = flow.shape
        frame = F.interpolate(frame, size=(h, w), mode='bilinear', align_corners=False)
        flow = torch.transpose(flow, 1, 2)
        flow = torch.transpose(flow, 2, 3)

        grid = flow + generate_grid(b, h, w, flow.device)

        factor = torch.FloatTensor([[[[2 / w, 2 / h]]]]).to(flow.device)
        grid = grid * factor - 1
        warped_frame = F.grid_sample(frame, grid)

        return warped_frame

    # def stn(self, flow, frame):
    #     """
    #     Implements Eq (3)-(4) exactly as in the FlowNet / Unsupervised Flow paper.
    #     flow:  Bx2xhxw   (pixel displacement)
    #     frame: Bx3xH_fullxW_full
    #     """

    #     B, C, H_full, W_full = frame.shape

    #     # 1. Upsample
    #     flow_up = F.interpolate(flow, size=(H_full, W_full), mode='bilinear', align_corners=True)

    #     # 2. Scale displacement correctly
    #     scale_x = W_full / flow.shape[3]
    #     scale_y = H_full / flow.shape[2]
    #     flow_up[:,0] *= scale_x
    #     flow_up[:,1] *= scale_y

    #     # 2. Base grid: pixel-center coordinates
    #     # meshgrid gives integer pixel indices i,j
    #     y, x = torch.meshgrid(
    #         torch.arange(H_full, device=flow.device),
    #         torch.arange(W_full, device=flow.device),
    #         indexing='ij'
    #     )

    #     # convert to float and add 0.5 (pixel center)
    #     x = x.float() + 0.5  
    #     y = y.float() + 0.5

    #     # 3. Apply flow: (x2, y2) = (x1 + u, y1 + v)
    #     x2 = x.unsqueeze(0) + flow_up[:, 0]   # BxHxW
    #     y2 = y.unsqueeze(0) + flow_up[:, 1]   # BxHxW

    #     # 4. Normalize to [-1, 1] for grid_sample
    #     x_norm = (x2 / W_full) * 2 - 1
    #     y_norm = (y2 / H_full) * 2 - 1

    #     # 5. Create grid: BxHxWx2
    #     grid = torch.stack((x_norm, y_norm), dim=-1)

    #     # 6. Bilinear sampling (Eq. 4)
    #     warped = F.grid_sample(
    #         frame,
    #         grid,
    #         mode='bilinear',
    #         padding_mode='zeros',   # matches paper (not border)
    #         align_corners=False     # matches bilinear kernel in Eq. 4
    #     )

    #     return warped




    def forward(self, x):
        
        # print("Unsupervised.training:", self.training)
        # print("Predictor.training:", self.predictor.training)
        flow_predictions = self.predictor(x)
        frame2 = x[:, 3:, :, :]
        warped_images = [self.stn(flow, frame2) for flow in flow_predictions]

        # print("\n=== DEBUG: checking flow outputs inside model ===")
        # for lvl, f in enumerate(flow_predictions):
        #     print(f"Level {lvl}: mean={f.mean().item():.6f}, std={f.std().item():.6f}, "
        #         f"min={f.min().item():.6f}, max={f.max().item():.6f}")
        # print("===================================================\n")

        return flow_predictions, warped_images

# ------------------------------------------------------------------
#   MINIMAL POSE DECODER (Global Pooling + MLP)
# ------------------------------------------------------------------

class MinimalPoseDecoder(nn.Module):
    """
    A minimal probe head to test what pose information lives in the encoder.
    Input:  B x C x H x W   (encoder feature map)
    Output: axisangle (B,1,1,3), translation (B,1,1,3)
    """

    def __init__(self, in_channels=1024, hidden_dim=256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_rot = nn.Linear(hidden_dim, 3)
        self.fc_trans = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.pool(x).view(B, C)

        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))

        axisangle = self.fc_rot(h).view(B, 1, 1, 3)
        translation = self.fc_trans(h).view(B, 1, 1, 3)

        return axisangle, translation


# ------------------------------------------------------------------
#   ENCODER-ONLY WRAPPER FOR FlowNetS  (NO CHANGES TO FlowNetS CLASS)
# ------------------------------------------------------------------

class FlowNetSEncoder(nn.Module):
    """
    Wraps your existing FlowNetS to expose the encoder only.
    Does NOT modify FlowNetS itself.
    """

    def __init__(self, flownet_s: FlowNetS):
        super().__init__()

        # copy the encoder layers from the provided FlowNetS instance
        self.conv1 = flownet_s.conv1
        self.conv2 = flownet_s.conv2
        self.conv3 = flownet_s.conv3
        self.conv3_1 = flownet_s.conv3_1
        self.conv4 = flownet_s.conv4
        self.conv4_1 = flownet_s.conv4_1
        self.conv5 = flownet_s.conv5
        self.conv5_1 = flownet_s.conv5_1
        self.conv6 = flownet_s.conv6

    def forward(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6  # B,1024,H/64,W/64


# ------------------------------------------------------------------
#   POSE PROBE MODEL (Frozen encoder + trainable MLP)
# ------------------------------------------------------------------

class FlowPoseProbe(nn.Module):
    """
    Uses your existing FlowNetS encoder (frozen) + minimal MLP pose head.
    """

    def __init__(self, pretrained_flownets=None, hidden_dim=256):
        super().__init__()

        # If user passes a pretrained FlowNetS, use it.
        # Otherwise create a fresh one.
        self.encoder_backbone = pretrained_flownets if pretrained_flownets else FlowNetS()

        # Wrap only the encoder part
        self.encoder = FlowNetSEncoder(self.encoder_backbone)

        # Freeze FlowNetS weights
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Add pose decoder
        self.pose_decoder = MinimalPoseDecoder(in_channels=1024, hidden_dim=hidden_dim)

    def forward(self, x):
        features = self.encoder(x)  # B,1024,H/64,W/64
        axisangle, translation = self.pose_decoder(features)
        return axisangle, translation