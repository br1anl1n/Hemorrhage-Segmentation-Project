# imports and libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import sys
import datetime
import cv2 
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Add mixed precision training imports
from torch.cuda.amp import GradScaler, autocast

# Import the rest of your modules
sys.path.append(str(Path(__file__).parent.parent))

# Import your modules
from data.data_loader import HemorrhageDataset


class Resize:
    """Transform to resize images and masks."""
    def __init__(self, output_size):
        self.output_size = output_size
        
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        # Get current dimensions
        c, h, w = image.shape
        
        if h == self.output_size and w == self.output_size:
            return sample
        
        # Resize image - handle different channel numbers
        resized_image = torch.zeros((c, self.output_size, self.output_size), dtype=image.dtype)
        for i in range(c):
            channel = image[i].numpy()
            resized_channel = cv2.resize(channel, (self.output_size, self.output_size), 
                                         interpolation=cv2.INTER_LINEAR)
            resized_image[i] = torch.from_numpy(resized_channel)
        
        # Resize mask - preserve mask being in [0, 1]
        mask_np = mask.numpy()
        resized_mask = np.zeros((mask_np.shape[0], self.output_size, self.output_size), dtype=np.float32)
        for i in range(mask_np.shape[0]):
            channel = mask_np[i]
            resized_channel = cv2.resize(channel, (self.output_size, self.output_size), 
                                         interpolation=cv2.INTER_NEAREST)  # Nearest for masks to preserve binary nature
            resized_mask[i] = resized_channel
        
        # Update the sample with resized image and mask
        sample['image'] = resized_image
        sample['mask'] = torch.from_numpy(resized_mask)
        
        return sample


class ImprovedRandomAugmentation:
    """Enhanced augmentations with more sophisticated transforms."""
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        # Convert to numpy arrays for easier manipulation
        image_np = image.numpy()  # CHW format
        mask_np = mask.numpy()    # CHW format
        
        # Standard geometric transforms with higher probability
        if np.random.random() > 0.3:  # Increased from 0.5
            image_np = np.flip(image_np, axis=2).copy()  # Flip width dimension
            mask_np = np.flip(mask_np, axis=2).copy()
        
        if np.random.random() > 0.3:  # Increased from 0.5
            image_np = np.flip(image_np, axis=1).copy()  # Flip height dimension
            mask_np = np.flip(mask_np, axis=1).copy()
        
        if np.random.random() > 0.3:  # Increased from 0.5
            k = np.random.randint(1, 4)  # 1, 2, or 3 (90, 180, 270 degrees)
            image_np = np.rot90(image_np, k=k, axes=(1, 2)).copy()  # Rotate in H-W plane
            mask_np = np.rot90(mask_np, k=k, axes=(1, 2)).copy()
        
        # Random cropping and padding
        if np.random.random() > 0.6:
            h, w = image_np.shape[1], image_np.shape[2]
            
            # Random crop size between 80-95% of original
            crop_factor = np.random.uniform(0.8, 0.95)
            crop_h = int(h * crop_factor)
            crop_w = int(w * crop_factor)
            
            # Random crop position
            top = np.random.randint(0, h - crop_h + 1)
            left = np.random.randint(0, w - crop_w + 1)
            
            # Crop both image and mask
            image_crop = image_np[:, top:top+crop_h, left:left+crop_w].copy()
            mask_crop = mask_np[:, top:top+crop_h, left:left+crop_w].copy()
            
            # Resize back to original size
            resized_image = np.zeros_like(image_np)
            resized_mask = np.zeros_like(mask_np)
            
            for i in range(image_np.shape[0]):
                resized_image[i] = cv2.resize(image_crop[i], (w, h), 
                                            interpolation=cv2.INTER_LINEAR)
            
            for i in range(mask_np.shape[0]):
                resized_mask[i] = cv2.resize(mask_crop[i], (w, h), 
                                           interpolation=cv2.INTER_NEAREST)
            
            image_np = resized_image
            mask_np = resized_mask
        
        # More aggressive brightness/contrast variations
        if np.random.random() > 0.4:  # Increased from 0.5
            # Apply to each channel independently
            for i in range(image_np.shape[0]):
                # Random brightness adjustment
                factor = np.random.uniform(0.7, 1.3)  # More extreme range
                image_np[i] = image_np[i] * factor
                
                # Random contrast adjustment
                factor = np.random.uniform(0.7, 1.3)  # More extreme range
                mean = np.mean(image_np[i])
                image_np[i] = (image_np[i] - mean) * factor + mean
                
                # Clip to valid range
                image_np[i] = np.clip(image_np[i], 0, 1)
        
        # Gaussian noise (especially helpful for medical imaging)
        if np.random.random() > 0.7:
            for i in range(image_np.shape[0]):
                noise = np.random.normal(0, 0.015, image_np[i].shape)  
                image_np[i] = image_np[i] + noise
                image_np[i] = np.clip(image_np[i], 0, 1)
        
        # Elastic deformation
        if np.random.random() > 0.6:  # More frequent elastic deformation
            h, w = image_np.shape[1], image_np.shape[2]
            dx = np.random.rand(h, w) * 2 - 1
            dy = np.random.rand(h, w) * 2 - 1
            
            # Smooth the displacement fields
            dx = cv2.GaussianBlur(dx, (0, 0), 7)
            dy = cv2.GaussianBlur(dy, (0, 0), 7)
            
            # Scale the displacement fields
            strength = np.random.uniform(3, 8)  # Increased upper bound
            dx *= strength
            dy *= strength
            
            # Create mesh grid
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            
            # Add displacement
            map_x = (x + dx).astype(np.float32)
            map_y = (y + dy).astype(np.float32)
            
            # Apply displacement to each channel
            for i in range(image_np.shape[0]):
                image_np[i] = cv2.remap(image_np[i], map_x, map_y, cv2.INTER_LINEAR)
            
            # Only apply to mask if it's binary (0 or 1 values only)
            if np.array_equal(np.unique(mask_np), np.array([0., 1.])):
                for i in range(mask_np.shape[0]):
                    mask_np[i] = cv2.remap(mask_np[i], map_x, map_y, cv2.INTER_NEAREST)
        
        # Convert back to tensors
        sample['image'] = torch.from_numpy(image_np)
        sample['mask'] = torch.from_numpy(mask_np)
        
        return sample


class FastHemorrhageDataset(HemorrhageDataset):
    """Optimized dataset for faster training."""
    
    def __init__(self, base_dir, labels_dir, window_types=None, transform=None, img_size=288, augment=True):
        """Initialize with additional parameters for faster loading."""
        super().__init__(base_dir, labels_dir, window_types, transform)
        
        # Add resizing transform
        self.img_size = img_size
        self.resize_transform = Resize(img_size)
        
        # Add augmentation
        self.augment = augment
        self.augmentation = ImprovedRandomAugmentation()  # Use our improved augmentation
        
        # Pre-load a small subset of images and masks for even faster training
        self.cached_samples = {}
    
    def __getitem__(self, idx):
        """Get sample with additional optimizations."""
        # Get sample from parent class
        if idx in self.cached_samples:
            return self.cached_samples[idx]
        
        sample = super().__getitem__(idx)
        
        # Apply resize transform
        sample = self.resize_transform(sample)
        
        # Apply augmentation (only for training samples)
        if self.augment:
            sample = self.augmentation(sample)
        
        # Cache for future access
        if len(self.cached_samples) < 100:  # Limit cache size to avoid memory issues
            self.cached_samples[idx] = sample
        
        return sample


class WeightedHemorrhageDataset(FastHemorrhageDataset):
    """Dataset with weighted sampling based on hemorrhage size."""
    
    def __init__(self, base_dir, labels_dir, window_types=None, transform=None, 
                 img_size=288, augment=True):
        super().__init__(base_dir, labels_dir, window_types, transform, img_size, augment)
        
        # Calculate weights for each sample based on hemorrhage size
        self.weights = []
        self.hemorrhage_sizes = []
        
        # Process all samples to get hemorrhage sizes
        print("Calculating sample weights...")
        for i in tqdm(range(len(self))):
            # Get sample
            sample = super().__getitem__(i)
            mask = sample['mask']
            
            # Calculate hemorrhage size as percentage of image
            hemorrhage_size = mask.sum() / (mask.shape[1] * mask.shape[2])
            self.hemorrhage_sizes.append(hemorrhage_size.item())
        
        # Convert to numpy array
        self.hemorrhage_sizes = np.array(self.hemorrhage_sizes)
        
        # Calculate median size
        median_size = np.median(self.hemorrhage_sizes)
        
        # Assign weights: higher weight to small hemorrhages and large hemorrhages
        # Medium sized ones get lower weight
        for size in self.hemorrhage_sizes:
            if size < 0.01:  # Very small hemorrhages
                weight = 3.0
            elif size < median_size / 2:  # Small hemorrhages
                weight = 2.0
            elif size > median_size * 2:  # Large hemorrhages
                weight = 1.5
            else:  # Medium hemorrhages
                weight = 1.0
            self.weights.append(weight)
        
        self.weights = np.array(self.weights)
        
    def get_weighted_sampler(self):
        """Create a weighted sampler for this dataset."""
        from torch.utils.data import WeightedRandomSampler
        
        # Normalize weights to sum to 1
        weights = self.weights / self.weights.sum()
        
        # Create sampler
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(self),
            replacement=True
        )
        
        return sampler


# Define enhanced U-Net components
class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Residual connection with 1x1 conv if needed
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class AttentionGate(nn.Module):
    """Attention Gate for U-Net with size handling"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
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
        
    def forward(self, g, x):
        # Get input dimensions
        g_h, g_w = g.size(2), g.size(3)
        x_h, x_w = x.size(2), x.size(3)
        
        # Process gating signal
        g1 = self.W_g(g)
        
        # Process skip connection
        x1 = self.W_x(x)
        
        # Resize g1 to match x1's dimensions if different
        if g_h != x_h or g_w != x_w:
            g1 = F.interpolate(g1, size=(x_h, x_w), mode='bilinear', align_corners=True)
        
        # Apply attention mechanism
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class EnhancedUNet(nn.Module):
    """U-Net with residual blocks and deep supervision."""
    def __init__(self, n_channels=3, n_classes=1, filters_base=64):
        super(EnhancedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder
        self.inc = ResidualBlock(n_channels, filters_base)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(filters_base, filters_base*2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(filters_base*2, filters_base*4)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(filters_base*4, filters_base*8)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(filters_base*8, filters_base*16)
        )
        
        # Attention gates
        self.att1 = AttentionGate(filters_base*16, filters_base*8, filters_base*8)
        self.att2 = AttentionGate(filters_base*8, filters_base*4, filters_base*4)
        self.att3 = AttentionGate(filters_base*4, filters_base*2, filters_base*2)
        self.att4 = AttentionGate(filters_base*2, filters_base, filters_base)
        
        # Decoder
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(filters_base*16, filters_base*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters_base*8),
            nn.ReLU(inplace=True)
        )
        self.up_conv1 = ResidualBlock(filters_base*16, filters_base*8)
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(filters_base*8, filters_base*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters_base*4),
            nn.ReLU(inplace=True)
        )
        self.up_conv2 = ResidualBlock(filters_base*8, filters_base*4)
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(filters_base*4, filters_base*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters_base*2),
            nn.ReLU(inplace=True)
        )
        self.up_conv3 = ResidualBlock(filters_base*4, filters_base*2)
        
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(filters_base*2, filters_base, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters_base),
            nn.ReLU(inplace=True)
        )
        self.up_conv4 = ResidualBlock(filters_base*2, filters_base)
        
        # Deep supervision outputs
        self.out1 = nn.Conv2d(filters_base*8, n_classes, kernel_size=1)
        self.out2 = nn.Conv2d(filters_base*4, n_classes, kernel_size=1)
        self.out3 = nn.Conv2d(filters_base*2, n_classes, kernel_size=1)
        self.final = nn.Conv2d(filters_base, n_classes, kernel_size=1)
        
        # Spatial dropout for regularization
        self.dropout2d = nn.Dropout2d(p=0.2)
        
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Apply spatial dropout for regularization
        x5 = self.dropout2d(x5)
        
        # Decoder with attention and skip connections
        x4_att = self.att1(x5, x4)
        x = self.up1(x5)
        x = torch.cat([x4_att, x], dim=1)
        x = self.up_conv1(x)
        ds1 = self.out1(x)  # Deep supervision output 1
        ds1 = F.interpolate(ds1, scale_factor=8, mode='bilinear', align_corners=True)
        
        x3_att = self.att2(x, x3)
        x = self.up2(x)
        x = torch.cat([x3_att, x], dim=1)
        x = self.up_conv2(x)
        ds2 = self.out2(x)  # Deep supervision output 2
        ds2 = F.interpolate(ds2, scale_factor=4, mode='bilinear', align_corners=True)
        
        x2_att = self.att3(x, x2)
        x = self.up3(x)
        x = torch.cat([x2_att, x], dim=1)
        x = self.up_conv3(x)
        ds3 = self.out3(x)  # Deep supervision output 3
        ds3 = F.interpolate(ds3, scale_factor=2, mode='bilinear', align_corners=True)
        
        x1_att = self.att4(x, x1)
        x = self.up4(x)
        x = torch.cat([x1_att, x], dim=1)
        x = self.up_conv4(x)
        
        # Main output
        out = self.final(x)
        
        # During training, return all outputs; during inference, return only main output
        if self.training:
            return out, ds1, ds2, ds3
        else:
            return out


# Improved loss functions
def boundary_weighted_dice_loss(pred, target, smooth=1.0, boundary_weight=10.0):
    """
    Enhanced Dice loss that gives more weight to boundary regions.
    This helps with accurate segmentation of borders.
    """
    # Apply sigmoid to get probabilities
    pred = torch.sigmoid(pred)
    
    # Get boundaries by applying dilation and erosion
    kernel_size = 3
    
    # Move target to CPU for boundary calculation
    # Convert to numpy for morphological operations
    target_np = target.detach().cpu().numpy()
    boundaries = np.zeros_like(target_np)
    
    # Process each batch element and channel
    for b in range(target_np.shape[0]):
        for c in range(target_np.shape[1]):
            # Create boundary mask
            dilated = cv2.dilate(target_np[b, c], 
                                np.ones((kernel_size, kernel_size), np.uint8))
            eroded = cv2.erode(target_np[b, c], 
                              np.ones((kernel_size, kernel_size), np.uint8))
            boundary = dilated - eroded
            boundaries[b, c] = boundary
    
    # Back to tensor and move to the appropriate device
    boundaries = torch.from_numpy(boundaries).to(target.device)
    
    # Create weight map with higher weight at boundaries
    weights = torch.ones_like(target)
    weights = weights + boundaries * (boundary_weight - 1)
    
    # Flatten
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    weights_flat = weights.reshape(-1)
    
    # Weighted Dice
    intersection = (pred_flat * target_flat * weights_flat).sum()
    pred_sum = (pred_flat * weights_flat).sum()
    target_sum = (target_flat * weights_flat).sum()
    
    dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    return 1 - dice


def focal_loss(pred, target, alpha=0.25, gamma=2.5):
    """
    Compute Focal Loss to address class imbalance.
    Focuses more on hard, misclassified examples.
    
    Args:
        pred: Model output tensor (logits)
        target: Target mask tensor
        alpha: Weighting factor
        gamma: Focusing parameter (higher values increase focus on hard, misclassified examples)
    
    Returns:
        Focal loss
    """
    # Calculate BCE with logits (combines sigmoid and BCE)
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    
    # Apply the focal term
    # For predictions from sigmoid, use:
    pred_sigmoid = torch.sigmoid(pred)
    pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
    
    # Apply the focal weight
    focal_weight = alpha * (1 - pt) ** gamma
    
    # Return weighted loss
    return (focal_weight * bce).mean()


class DeepSupervisionLoss(nn.Module):
    """Loss function with deep supervision."""
    def __init__(self, alpha=0.3, focal_gamma=2.5, boundary_weight=8.0):
        super(DeepSupervisionLoss, self).__init__()
        self.alpha = alpha
        self.focal_gamma = focal_gamma
        self.boundary_weight = boundary_weight
        
    def forward(self, preds, target):
        # Main prediction
        if isinstance(preds, tuple):
            # Deep supervision mode
            main_pred, ds1, ds2, ds3 = preds
            
            # Main loss
            focal = focal_loss(main_pred, target, gamma=self.focal_gamma)
            dice = boundary_weighted_dice_loss(main_pred, target, boundary_weight=self.boundary_weight)
            main_loss = self.alpha * focal + (1 - self.alpha) * dice
            
            # Deep supervision losses
            ds1_focal = focal_loss(ds1, target, gamma=self.focal_gamma)
            ds1_dice = boundary_weighted_dice_loss(ds1, target, boundary_weight=self.boundary_weight)
            ds1_loss = self.alpha * ds1_focal + (1 - self.alpha) * ds1_dice
            
            ds2_focal = focal_loss(ds2, target, gamma=self.focal_gamma)
            ds2_dice = boundary_weighted_dice_loss(ds2, target, boundary_weight=self.boundary_weight)
            ds2_loss = self.alpha * ds2_focal + (1 - self.alpha) * ds2_dice
            
            ds3_focal = focal_loss(ds3, target, gamma=self.focal_gamma)
            ds3_dice = boundary_weighted_dice_loss(ds3, target, boundary_weight=self.boundary_weight)
            ds3_loss = self.alpha * ds3_focal + (1 - self.alpha) * ds3_dice
            
            # Combine losses with gradually decreasing weights
            total_loss = main_loss + 0.4 * ds1_loss + 0.2 * ds2_loss + 0.1 * ds3_loss
            return total_loss
        else:
            # Normal mode - single prediction
            focal = focal_loss(preds, target, gamma=self.focal_gamma)
            dice = boundary_weighted_dice_loss(preds, target, boundary_weight=self.boundary_weight)
            return self.alpha * focal + (1 - self.alpha) * dice


def apply_post_processing(pred, threshold=0.5, min_size=50):
    """
    Apply post-processing to predictions to improve final mask quality.
    
    Args:
        pred: Model prediction (B, C, H, W)
        threshold: Binary threshold value
        min_size: Minimum component size to keep
        
    Returns:
        Processed prediction
    """
    # Make a copy to avoid modifying the original
    processed = pred.clone().detach().cpu().numpy()
    
    # Create binary mask
    binary = (processed > threshold).astype(np.uint8)
    
    # Process each batch and channel
    for b in range(binary.shape[0]):
        for c in range(binary.shape[1]):
            mask = binary[b, c]
            
            # Remove small connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            # Skip background (label 0)
            for label in range(1, num_labels):
                size = stats[label, cv2.CC_STAT_AREA]
                if size < min_size:
                    mask[labels == label] = 0
            
            # Fill holes in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > min_size:
                    cv2.drawContours(mask, [contour], 0, 1, -1)  # Fill the contour
            
            # Smooth boundaries with morphological operations
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Update the binary mask
            binary[b, c] = mask
    
    # Convert back to tensor and return
    return torch.from_numpy(binary).float()


def visualize_predictions(model, val_loader, device, epoch):
    """Visualize model predictions for a few validation samples."""
    model.eval()
    vis_dir = Path('results/visualizations/predictions')
    vis_dir.mkdir(exist_ok=True, parents=True)
    
    with torch.no_grad():
        # Get a batch
        batch = next(iter(val_loader))
        inputs = batch['image'].to(device)
        masks = batch['mask'].to(device)
        ids = batch['id']
        
        # Generate predictions
        with autocast():
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                main_pred = outputs[0]  # Get main prediction from deep supervision
            else:
                main_pred = outputs
                
        preds = torch.sigmoid(main_pred)
        preds_binary = (preds > 0.5).float()
        
        # Apply post-processing to improve predictions
        processed_preds = apply_post_processing(preds_binary)
        
        # Visualize first few samples
        num_samples = min(3, inputs.size(0))
        
        for i in range(num_samples):
            # Get data
            img = inputs[i].cpu().numpy()
            mask = masks[i, 0].cpu().numpy()  # Remove channel dimension
            raw_pred = preds_binary[i, 0].cpu().numpy()  # Raw prediction
            proc_pred = processed_preds[i, 0].cpu().numpy()  # Processed prediction
            img_id = ids[i]
            
            # Handle multi-channel grayscale images - use first channel for visualization
            if img.shape[0] > 1:
                # For visualization, just use the first channel (e.g., brain_window)
                img_for_display = img[0]
            else:
                img_for_display = img[0]  # Still take first dimension even if just one channel
            
            # Create figure
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # Display image
            axes[0].imshow(img_for_display, cmap='gray')  # Use grayscale colormap
            axes[0].set_title(f"Image (ID: {img_id})")
            axes[0].axis('off')
            
            # Display ground truth mask
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title("Ground Truth")
            axes[1].axis('off')
            
            # Display raw prediction
            axes[2].imshow(raw_pred, cmap='gray')
            axes[2].set_title("Raw Prediction")
            axes[2].axis('off')
            
            # Display processed prediction
            axes[3].imshow(proc_pred, cmap='gray')
            axes[3].set_title("Processed Prediction")
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.savefig(vis_dir / f"epoch_{epoch}_sample_{i}.png")
            plt.close()
    
    print(f"Saved prediction visualizations for epoch {epoch}")


def plot_training_history(history):
    """Plot training and validation loss and Dice coefficient."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Dice coefficient
    ax2.plot(epochs, history['train_dice'], 'b-', label='Training Dice')
    ax2.plot(epochs, history['val_dice'], 'r-', label='Validation Dice')
    ax2.set_title('Training and Validation Dice Coefficient')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Dice Coefficient')
    ax2.legend()
    ax2.grid(True)
    
    # Save the figure
    plt.tight_layout()
    fig.savefig('results/visualizations/training_history.png')
    plt.close(fig)


def predict_with_tta(model, inputs, device):
    """
    Apply test-time augmentation for more robust predictions.
    
    Args:
        model: Trained model
        inputs: Input batch of images
        device: Device to run inference on
        
    Returns:
        Processed prediction after test-time augmentation
    """
    model.eval()
    
    # Generate augmented versions
    versions = [
        inputs,  # Original
        torch.flip(inputs, [2]),  # Horizontal flip
        torch.flip(inputs, [3]),  # Vertical flip
        torch.rot90(inputs, 1, [2, 3])  # 90 degree rotation
    ]
    
    # Get predictions for each version
    with torch.no_grad():
        preds = []
        for v in versions:
            with autocast():
                pred = model(v.to(device))
                if isinstance(pred, tuple):
                    pred = pred[0]  # Get main prediction
                pred = torch.sigmoid(pred)
                preds.append(pred)
        
        # Restore original orientation for flipped/rotated versions
        preds[1] = torch.flip(preds[1], [2])  # Un-flip horizontal
        preds[2] = torch.flip(preds[2], [3])  # Un-flip vertical 
        preds[3] = torch.rot90(preds[3], -1, [2, 3])  # Un-rotate
        
        # Average predictions
        avg_pred = torch.mean(torch.stack(preds), dim=0)
        
        # Apply post-processing
        processed_pred = apply_post_processing(avg_pred, threshold=0.5, min_size=50)
        
        return processed_pred


def train_with_mixed_precision(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, device, num_epochs=40):
    """Training loop with mixed precision for faster training."""
    model = model.to(device)
    
    # Initialize metrics tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': []
    }
    
    best_val_loss = float('inf')
    best_val_dice = 0.0
    patience = 5  # For early stopping
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        batch_count = 0
        
        train_iter = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch in train_iter:
            inputs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, masks)
                
                # Calculate Dice coefficient
                if isinstance(outputs, tuple):
                    main_pred = outputs[0]  # Get main prediction
                else:
                    main_pred = outputs
                
                pred = torch.sigmoid(main_pred)
                pred_binary = (pred > 0.5).float()
                dice = (2. * (pred_binary * masks).sum()) / (pred_binary.sum() + masks.sum() + 1e-8)
            
            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update metrics
            train_loss += loss.item() * inputs.size(0)
            train_dice += dice.item() * inputs.size(0)
            batch_count += 1
            
            # Update progress bar
            train_iter.set_postfix(loss=loss.item(), dice=dice.item())
        
        # Calculate epoch statistics
        sample_count = batch_count * train_loader.batch_size
        epoch_train_loss = train_loss / sample_count
        epoch_train_dice = train_dice / sample_count
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        batch_count = 0
        
        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
            for batch in val_iter:
                inputs = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # Forward pass
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)
                    
                    # Calculate Dice coefficient
                    if isinstance(outputs, tuple):
                        main_pred = outputs[0]  # Get main prediction
                    else:
                        main_pred = outputs
                    
                    pred = torch.sigmoid(main_pred)
                    pred_binary = (pred > 0.5).float()
                    dice = (2. * (pred_binary * masks).sum()) / (pred_binary.sum() + masks.sum() + 1e-8)
                
                val_loss += loss.item() * inputs.size(0)
                val_dice += dice.item() * inputs.size(0)
                batch_count += 1
                
                val_iter.set_postfix(loss=loss.item(), dice=dice.item())
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, OneCycleLR):
                # OneCycleLR should be called after each batch, but we'll update it here
                # to make sure it's updated at least once per epoch
                pass
            else:
                scheduler.step()
        
        # Calculate epoch statistics
        sample_count = batch_count * val_loader.batch_size
        epoch_val_loss = val_loss / sample_count
        epoch_val_dice = val_dice / sample_count
        
        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_dice'].append(epoch_train_dice)
        history['val_dice'].append(epoch_val_dice)
        
        # Print epoch results
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Dice: {epoch_train_dice:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Dice: {epoch_val_dice:.4f}')
        
        # Create model directory if it doesn't exist
        model_dir = Path('results/models')
        model_dir.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the best model based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_dir / f'enhanced_unet_best_loss_{timestamp}.pth')
            print(f'Saved new best model with validation loss: {best_val_loss:.4f}')
            patience_counter = 0  # Reset patience counter
        
        # Save the best model based on Dice score
        if epoch_val_dice > best_val_dice:
            best_val_dice = epoch_val_dice
            torch.save(model.state_dict(), model_dir / f'enhanced_unet_best_dice_{timestamp}.pth')
            print(f'Saved new best model with validation Dice: {best_val_dice:.4f}')
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1
        
        # Check for early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {patience} epochs without improvement")
            break
        
        # Save models at regular intervals (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), model_dir / f'enhanced_unet_epoch_{epoch+1}.pth')
            print(f'Saved checkpoint at epoch {epoch+1}')
        
        # Visualize a few predictions
        if (epoch + 1) % 5 == 0 or epoch == 0:  # First epoch and every 5 epochs
            visualize_predictions(model, val_loader, device, epoch+1)
    
    # Training complete
    time_elapsed = time.time() - start_time
    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print(f'Best validation Dice score: {best_val_dice:.4f}')
    
    # Plot training history
    plot_training_history(history)
    
    return model, history


def main():
    """Main function for training hemorrhage segmentation model."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create directories
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/visualizations', exist_ok=True)
    
    # Define paths
    base_dir = Path("/content/hemorrhage_project/hemorrhage-project/renders/intraventricular").expanduser()
    labels_dir = Path("/content/hemorrhage_project/hemorrhage-project/labels").expanduser()
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parameters for improved training
    img_size = 288  # Increased from 256 for more detail
    batch_size = 8  # Increased since GPU memory allows
    window_types = ['brain_window', 'max_contrast_window', 'subdural_window']  # Use multiple window types for better context
    
    print("Creating datasets...")
    try:
        # Use WeightedHemorrhageDataset for better balance
        dataset = WeightedHemorrhageDataset(
            base_dir, 
            labels_dir, 
            window_types=window_types,
            img_size=img_size,
            augment=True  # Enable augmentation for training
        )
        
        # Use weighted sampler if available
        try:
            train_sampler = dataset.get_weighted_sampler()
            use_weighted_sampler = True
            print("Using weighted sampling for improved class balance")
        except:
            use_weighted_sampler = False
            print("Weighted sampling not available, using random sampling")
    
    except Exception as e:
        print(f"Error creating weighted dataset: {e}")
        print("Falling back to standard dataset")
        
        # Use FastHemorrhageDataset as fallback
        dataset = FastHemorrhageDataset(
            base_dir, 
            labels_dir, 
            window_types=window_types,
            img_size=img_size,
            augment=True
        )
        use_weighted_sampler = False
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create train-val split
    train_idx, val_idx = train_test_split(
        range(len(dataset)), 
        test_size=0.2, 
        random_state=42
    )
    
    # Create data loaders
    if use_weighted_sampler:
        train_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            num_workers=4,  # Adjust based on your system
            pin_memory=True  # Faster data transfer to GPU
        )
    else:
        train_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=SubsetRandomSampler(train_idx),
            num_workers=4, 
            pin_memory=True
        )
    
    # Create a separate validation dataset without augmentation
    val_dataset = FastHemorrhageDataset(
        base_dir, 
        labels_dir, 
        window_types=window_types,
        img_size=img_size,
        augment=False  # No augmentation for validation
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_idx),
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    
    # Create the enhanced model
    print("Initializing model...")
    n_channels = len(window_types)
    model = EnhancedUNet(n_channels=n_channels, n_classes=1, filters_base=64)
    print(f"Model created: EnhancedUNet with residual blocks and deep supervision")
    print(f"Input channels: {n_channels} (one per window type)")
    
    # Create loss function
    criterion = DeepSupervisionLoss(alpha=0.3, focal_gamma=2.5, boundary_weight=8.0)
    print(f"Using DeepSupervisionLoss with boundary weighting")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.0005,  # Lower initial learning rate for stability
        weight_decay=0.0001  # L2 regularization to prevent overfitting
    )
    print(f"Using AdamW optimizer with weight decay")
    
    # Create learning rate scheduler - OneCycleLR for faster convergence
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.001,  # Peak learning rate
        epochs=40,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # Spend 10% of training time in warmup
        div_factor=10.0,  # Initial LR = max_lr / div_factor
        final_div_factor=100.0  # Final LR = max_lr / final_div_factor
    )
    print(f"Using OneCycleLR scheduler with warmup")
    
    # Create gradient scaler for mixed precision training
    scaler = GradScaler()
    print(f"Using mixed precision training for speed and memory efficiency")
    
    # Train the model
    print("Starting training...")
    model, history = train_with_mixed_precision(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        num_epochs=40  # Increased for better learning
    )
    
    # Final evaluation with test-time augmentation
    print("Performing final evaluation with test-time augmentation...")
    model.eval()
    test_dice = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Final Evaluation"):
            inputs = batch['image']
            masks = batch['mask'].to(device)
            
            # Get prediction with TTA
            tta_preds = predict_with_tta(model, inputs, device)
            # Move tta_preds to the same device as masks (FIX)
            tta_preds = tta_preds.to(device)
            
            # Calculate Dice
            dice = (2. * (tta_preds * masks).sum()) / (tta_preds.sum() + masks.sum() + 1e-8)
            test_dice += dice.item() * inputs.size(0)
            batch_count += 1
    
    final_dice = test_dice / (batch_count * val_loader.batch_size)
    print(f"Final Dice Score with TTA: {final_dice:.4f}")
    
    print("Training and evaluation completed!")
    return model, history, final_dice


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
