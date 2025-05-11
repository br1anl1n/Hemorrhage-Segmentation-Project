"""
Implements U-Net architecture: Creates a specialized neural network designed for 
medical image segmentation. U-Net gets its name from its U-shaped structure.

Defines the model components:
    Encoder path (going down): Captures context and features by progressively 
    reducing image size while increasing depth
    Decoder path (going up): Rebuilds the spatial information while using features 
    from the encoder
    Skip connections: Direct pathways that connect matching levels of the encoder 
    and decoder to preserve fine details

Handles flexible inputs: The model can accept brain scans with different numbers of channels (from various window settings).
Sets up the segmentation output: Produces a single-channel mask highlighting where the hemorrhage is located.
Implements specialized loss functions: 
    Dice loss: Measures overlap between predicted and actual hemorrhage regions
    Combined loss: Merges binary cross-entropy with Dice loss for better training

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

def dice_loss(pred, target, smooth=1.0, weight=1.0):
    """
    Compute Dice loss between predicted and target segmentation masks.
    Adds weighting to focus more on foreground pixels.
    
    Args:
        pred: Model output tensor of shape (B, C, H, W)
        target: Target mask tensor of shape (B, C, H, W)
        smooth: Small constant to avoid division by zero
        weight: Weight to apply to foreground pixels (higher values increase focus on hemorrhage pixels)
    
    Returns:
        Dice loss (1 - Dice coefficient)
    """
    # Apply sigmoid to get probabilities
    pred = torch.sigmoid(pred)
    
    # Flatten the predictions and targets
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Apply weighting to focus more on foreground pixels
    weights = torch.ones_like(target_flat)
    weights[target_flat > 0] = weight
    
    # Weighted intersection and union
    intersection = (pred_flat * target_flat * weights).sum()
    pred_sum = (pred_flat * weights).sum()
    target_sum = (target_flat * weights).sum()
    
    dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    return 1 - dice

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Compute Focal Loss to address class imbalance.
    Focuses more on hard, misclassified examples.
    
    Args:
        pred: Model output tensor
        target: Target mask tensor
        alpha: Weighting factor
        gamma: Focusing parameter (higher values increase focus on hard, misclassified examples)
    
    Returns:
        Focal loss
    """
    # Apply sigmoid to get probabilities
    pred = torch.sigmoid(pred)
    
    # Flatten the predictions and targets
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Calculate BCE
    bce = F.binary_cross_entropy(pred_flat, target_flat, reduction='none')
    
    # Apply the focal term
    pt = torch.exp(-bce)  # pt is the probability of the correct class
    focal_weight = alpha * (1 - pt) ** gamma
    
    # Return weighted loss
    return (focal_weight * bce).mean()

def combined_loss(pred, target, alpha=0.5, dice_weight=5.0, focal_gamma=2.0):
    """
    Combination of BCE, Dice loss, and Focal Loss.
    
    Args:
        pred: Model output tensor of shape (B, C, H, W)
        target: Target mask tensor of shape (B, C, H, W)
        alpha: Weight for BCE vs. Dice loss
        dice_weight: Weight for foreground pixels in Dice loss
        focal_gamma: Focusing parameter for focal loss
    
    Returns:
        Weighted combination of losses
    """
    focal = focal_loss(pred, target, gamma=focal_gamma)
    dice = dice_loss(pred, target, weight=dice_weight)
    
    return alpha * focal + (1 - alpha) * dice