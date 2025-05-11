import matplotlib.pyplot as plt
from pathlib import Path
from data.data_loader import HemorrhageDataset

def check_grayscale_conversion():
    """Visualize samples before and after grayscale conversion."""
    # Set paths
    base_dir = Path("/content/hemorrhage_project/hemorrhage-project/renders/intraventricular").expanduser()
    labels_dir = Path("/content/hemorrhage_project/hemorrhage-project/labels").expanduser()
    
    # Create dataset
    dataset = HemorrhageDataset(base_dir, labels_dir, window_types=['brain_window', 'max_contrast_window'])
    
    # Get a sample
    sample = dataset[0]
    img = sample['image'].numpy()
    mask = sample['mask'].numpy()
    
    # Check the shape to verify grayscale conversion
    print(f"Image shape: {img.shape}")  # Should be (num_windows, H, W)
    print(f"Mask shape: {mask.shape}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Show first window (brain_window)
    axes[0].imshow(img[0], cmap='gray')
    axes[0].set_title('Brain Window (Grayscale)')
    
    # Show second window (max_contrast_window)
    axes[1].imshow(img[1], cmap='gray')
    axes[1].set_title('Max Contrast Window (Grayscale)')
    
    # Show mask
    axes[2].imshow(mask[0], cmap='gray')
    axes[2].set_title('Hemorrhage Mask')
    
    plt.tight_layout()
    plt.savefig('results/visualizations/grayscale_verification.png')
    plt.close()
    
    print("Grayscale verification image saved to results/visualizations/grayscale_verification.png")

if __name__ == "__main__":
    check_grayscale_conversion()