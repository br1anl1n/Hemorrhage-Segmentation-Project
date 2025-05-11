import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import torch
import sys
import random

# Add the project directory to the path to import custom modules
project_dir = Path(__file__).parent.parent
sys.path.append(str(project_dir))

# Import your modules
from src.data.data_loader import HemorrhageDataset

def verify_masks(num_samples=20):
    """
    Verify that masks are being properly generated from ROI data.
    Checks for issues like empty masks, low mask intensity, etc.
    
    Args:
        num_samples: Number of samples to check
    """
    # Define paths
    base_dir = Path("/content/hemorrhage_project/hemorrhage-project/renders/intraventricular").expanduser()
    labels_dir = Path("/content/hemorrhage_project/hemorrhage-project/labels").expanduser()
    output_dir = Path("results/visualizations/mask_verification")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Try different window types
    for window_type in ['brain_window', 'max_contrast_window', 'subdural_window', 'brain_bone_window']:
        try:
            print(f"\nVerifying masks with {window_type}...")
            
            # Create dataset
            dataset = HemorrhageDataset(base_dir, labels_dir, window_types=[window_type])
            
            if len(dataset) == 0:
                print(f"No samples found for {window_type}")
                continue
            
            # Get statistics on masks
            mask_stats = []
            empty_mask_count = 0
            
            # Take a random sample of indices to check
            indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
            
            for i in indices:
                sample = dataset[i]
                image = sample['image'].numpy()
                mask = sample['mask'].numpy()
                
                # Check mask properties
                mask_sum = mask.sum()
                mask_mean = mask.mean()
                mask_max = mask.max()
                
                if mask_sum == 0:
                    empty_mask_count += 1
                
                mask_stats.append({
                    'id': sample['id'],
                    'sum': mask_sum,
                    'mean': mask_mean,
                    'max': mask_max,
                    'source': sample['source']
                })
                
                # Visualize the image and mask
                # Convert from CHW to HWC for visualization
                if image.shape[0] == 3:
                    vis_image = image.transpose((1, 2, 0))
                else:
                    # If more channels, just use first 3
                    vis_image = image[:3].transpose((1, 2, 0))
                
                # Remove channel dimension from mask
                vis_mask = mask[0]
                
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                axes[0].imshow(vis_image)
                axes[0].set_title(f"Image (ID: {sample['id']})")
                axes[0].axis('off')
                
                # Mask
                axes[1].imshow(vis_mask, cmap='gray')
                axes[1].set_title(f"Mask (Sum: {mask_sum:.1f})")
                axes[1].axis('off')
                
                # Overlay
                # Make a copy of the image for overlay
                overlay = vis_image.copy()
                if vis_image.max() > 1.0:
                    overlay = overlay / 255.0
                
                # Create alpha channel from mask
                alpha = vis_mask * 0.7
                # Create color mask (red)
                red_mask = np.zeros_like(overlay)
                red_mask[:, :, 0] = 1.0  # Red channel
                
                # Apply mask with alpha blending
                for c in range(3):
                    overlay[:, :, c] = overlay[:, :, c] * (1 - alpha) + red_mask[:, :, c] * alpha
                
                axes[2].imshow(overlay)
                axes[2].set_title("Overlay")
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(output_dir / f"{window_type}_{sample['id']}.png")
                plt.close()
            
            # Report statistics
            print(f"Analyzed {len(mask_stats)} samples from {window_type}")
            print(f"Empty masks: {empty_mask_count} ({empty_mask_count/len(mask_stats)*100:.1f}%)")
            
            # Calculate average stats
            avg_sum = sum(stat['sum'] for stat in mask_stats) / len(mask_stats)
            avg_mean = sum(stat['mean'] for stat in mask_stats) / len(mask_stats)
            
            print(f"Average mask sum: {avg_sum:.2f}")
            print(f"Average mask mean: {avg_mean:.4f}")
            
            # Count by source
            source_counts = {}
            for stat in mask_stats:
                source = stat['source']
                if source not in source_counts:
                    source_counts[source] = 0
                source_counts[source] += 1
            
            print("\nSamples by source:")
            for source, count in source_counts.items():
                print(f"  {source}: {count}")
            
        except Exception as e:
            print(f"Error processing {window_type}: {e}")

if __name__ == "__main__":
    verify_masks()