import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

class ConvBlock(nn.Module):
    """Double convolution block: (Conv2d -> BatchNorm2d -> ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class ResNetBasicBlock(nn.Module):
    """Basic residual block with two 3x3 convolutions and skip connection."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)

class ResNetEncoder(nn.Module):
    """Customizable ResNet-style encoder for multispectral and SAR imagery input."""
    def __init__(self, in_channels: int, base_channels: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = nn.Sequential(
            ResNetBasicBlock(base_channels, base_channels, stride=1),
            ResNetBasicBlock(base_channels, base_channels, stride=1)
        )
        self.layer2 = nn.Sequential(
            ResNetBasicBlock(base_channels, base_channels * 2, stride=2),
            ResNetBasicBlock(base_channels * 2, base_channels * 2, stride=1)
        )
        self.layer3 = nn.Sequential(
            ResNetBasicBlock(base_channels * 2, base_channels * 4, stride=2),
            ResNetBasicBlock(base_channels * 4, base_channels * 4, stride=1)
        )
        self.layer4 = nn.Sequential(
            ResNetBasicBlock(base_channels * 4, base_channels * 8, stride=2),
            ResNetBasicBlock(base_channels * 8, base_channels * 8, stride=1)
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        x = self.stem(x)         # Down 2x (base_channels)
        features.append(x)
        x = self.layer1(x)       # (base_channels)
        features.append(x)
        x = self.layer2(x)       # Down 4x (base_channels * 2)
        features.append(x)
        x = self.layer3(x)       # Down 8x (base_channels * 4)
        features.append(x)
        x = self.layer4(x)       # Down 16x (base_channels * 8)
        features.append(x)
        return features

class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale context."""
    def __init__(self, in_channels: int, out_channels: int, rates: List[int] = [1, 6, 12, 18]):
        super().__init__()
        modules = []
        # 1x1 Conv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        for rate in rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
            
        # Global Pooling branch
        self.global_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d((len(rates) + 2) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        res = [conv(x) for conv in self.convs]
        
        # Upsample global pooling features
        gp = self.global_pooling(x)
        gp_up = F.interpolate(gp, size=size, mode='bilinear', align_corners=False)
        res.append(gp_up)
        
        out = torch.cat(res, dim=1)
        return self.project(out)

class GridAttentionGate(nn.Module):
    """Attention Gate for skip connections to focus on salient regions."""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Downsample x spatial size to g if needed, but in standard UNet we upsample g to x
        g_proj = self.W_g(g)
        
        # Spatial dimensions alignment
        if g_proj.shape[-2:] != x.shape[-2:]:
            g_proj = F.interpolate(g_proj, size=x.shape[-2:], mode='bilinear', align_corners=False)
            
        x_proj = self.W_x(x)
        psi_input = self.relu(g_proj + x_proj)
        attention_weights = self.psi(psi_input)
        return x * attention_weights

class MultiEncoderUNet(nn.Module):
    """
    NASA-grade Multi-Encoder Attention UNet with Deep Supervision.
    Supports fusion of multimodal input (e.g. SAR, Multispectral, DEM).
    """
    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 64):
        super().__init__()
        # Encoder for the main input
        self.encoder = ResNetEncoder(in_channels, base_channels)
        
        # ASPP module at the bottleneck (deepest layer)
        self.aspp = ASPPModule(base_channels * 8, base_channels * 8)
        
        # Decoders & Attention Gates
        # Decoder 4: Up from base_channels * 8 (512) to base_channels * 4 (256)
        self.att4 = GridAttentionGate(F_g=base_channels * 8, F_l=base_channels * 4, F_int=base_channels * 4)
        self.upconv4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(base_channels * 8, base_channels * 4)
        
        # Decoder 3: Up to base_channels * 2 (128)
        self.att3 = GridAttentionGate(F_g=base_channels * 4, F_l=base_channels * 2, F_int=base_channels * 2)
        self.upconv3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(base_channels * 4, base_channels * 2)
        
        # Decoder 2: Up to base_channels (64)
        self.att2 = GridAttentionGate(F_g=base_channels * 2, F_l=base_channels, F_int=base_channels)
        self.upconv2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(base_channels * 2, base_channels)
        
        # Decoder 1: Stem skip matching
        self.att1 = GridAttentionGate(F_g=base_channels, F_l=base_channels, F_int=base_channels // 2)
        self.upconv1 = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        self.decoder1 = ConvBlock(base_channels * 2, base_channels)
        
        # Deep supervision projection layers
        self.ds_out4 = nn.Conv2d(base_channels * 4, num_classes, kernel_size=1)
        self.ds_out3 = nn.Conv2d(base_channels * 2, num_classes, kernel_size=1)
        self.ds_out2 = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        self.final_out = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        
        # Encoder features
        # e0 stem (down 2x), e1 (stem scale), e2 (down 4x), e3 (down 8x), e4 (down 16x)
        e0, e1, e2, e3, e4 = self.encoder(x)
        
        # Bottleneck ASPP
        bottleneck = self.aspp(e4)
        
        # Decoder 4
        g4 = self.upconv4(bottleneck)
        x4_skip = self.att4(g=bottleneck, x=e3)
        d4 = self.decoder4(torch.cat([g4, x4_skip], dim=1))
        
        # Decoder 3
        g3 = self.upconv3(d4)
        x3_skip = self.att3(g=d4, x=e2)
        d3 = self.decoder3(torch.cat([g3, x3_skip], dim=1))
        
        # Decoder 2
        g2 = self.upconv2(d3)
        x2_skip = self.att2(g=d3, x=e1)
        d2 = self.decoder2(torch.cat([g2, x2_skip], dim=1))
        
        # Decoder 1
        g1 = self.upconv1(d2)
        x1_skip = self.att1(g=d2, x=e0)
        d1 = self.decoder1(torch.cat([g1, x1_skip], dim=1))
        
        # Output convolutions
        out = self.final_out(d1)
        # Final prediction needs to be interpolated to full input size
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        # If in training mode, output deep supervision features
        if self.training:
            out4 = F.interpolate(self.ds_out4(d4), size=input_size, mode='bilinear', align_corners=False)
            out3 = F.interpolate(self.ds_out3(d3), size=input_size, mode='bilinear', align_corners=False)
            out2 = F.interpolate(self.ds_out2(d2), size=input_size, mode='bilinear', align_corners=False)
            return {
                "out": out,
                "ds4": out4,
                "ds3": out3,
                "ds2": out2
            }
        return {"out": out}

class FocalLoss(nn.Module):
    """Focal Loss to address class imbalance in segmentation."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class DiceLoss(nn.Module):
    """Dice Loss for segmentation overlapping evaluation."""
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.sigmoid(inputs)
        
        # Flatten label/prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class HybridDisasterLoss(nn.Module):
    """Hybrid loss combining Dice, Binary Cross-Entropy (BCE), and Focal Loss."""
    def __init__(self, w_dice: float = 1.0, w_bce: float = 0.5, w_focal: float = 1.0):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.w_dice = w_dice
        self.w_bce = w_bce
        self.w_focal = w_focal

    def forward(self, preds: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        # Multi-class target tensor has shape (B, C, H, W)
        main_loss = self._compute_single_loss(preds["out"], targets)
        
        if "ds4" in preds:
            # Add Deep Supervision losses with decreasing weights
            loss4 = self._compute_single_loss(preds["ds4"], targets)
            loss3 = self._compute_single_loss(preds["ds3"], targets)
            loss2 = self._compute_single_loss(preds["ds2"], targets)
            return main_loss + 0.4 * loss4 + 0.2 * loss3 + 0.1 * loss2
            
        return main_loss

    def _compute_single_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Combine target multi-class loss
        bce = F.binary_cross_entropy_with_logits(pred, target)
        dice = self.dice(pred, target)
        focal = self.focal(pred, target)
        return self.w_bce * bce + self.w_dice * dice + self.w_focal * focal

def compute_segmentation_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Computes IoU and F1 Score for prediction and target tensors."""
    preds_bin = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds_bin * targets).sum(dim=(0, 2, 3))
    union = (preds_bin + targets).sum(dim=(0, 2, 3)) - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    precision = (intersection + 1e-6) / (preds_bin.sum(dim=(0, 2, 3)) + 1e-6)
    recall = (intersection + 1e-6) / (targets.sum(dim=(0, 2, 3)) + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return {
        "mean_iou": iou.mean().item(),
        "mean_f1": f1.mean().item()
    }
