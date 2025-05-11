import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import pandas as pd
from pathlib import Path
import ast

def roi_to_mask(roi_data, width=512, height=512):
    """
    Convert ROI polygon data to a binary mask.
    
    Args:
        roi_data: List of dictionaries with x,y coordinates
        width, height: Dimensions of the output mask
        
    Returns:
        Binary mask as numpy array
    """
    # Create an empty mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Convert normalized coordinates to pixel coordinates
    points = []
    for point in roi_data:
        x = int(point['x'] * width)
        y = int(point['y'] * height)
        points.append([x, y])
    
    # Convert points to the format expected by cv2.fillPoly
    points = np.array(points, dtype=np.int32)
    points = points.reshape((-1, 1, 2))
    
    # Fill the polygon
    cv2.fillPoly(mask, [points], 255)
    
    return mask

def visualize_masks_from_csv():
    """Visualize masks created from ROI data in CSV files."""
    # Get label data
    labels_dir = Path("/content/hemorrhage_project/hemorrhage-project/labels").expanduser()
    
    # Find the csv files that have ROI data
    csv_files = list(labels_dir.glob("**/*.csv"))
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'ROI' in df.columns:
                print(f"Processing {csv_file.name}...")
                
                # Get a few samples
                samples = df.sample(min(3, len(df))) if len(df) > 0 else df
                
                for i, (_, row) in enumerate(samples.iterrows()):
                    try:
                        # Extract ROI data
                        roi_string = row['ROI']
                        
                        # Parse the ROI string into a Python object
                        # Different files might have different formats
                        try:
                            # Try direct JSON parsing
                            roi_data = json.loads(roi_string)
                        except:
                            # Try evaluating as a Python literal
                            roi_data = ast.literal_eval(roi_string)
                        
                        # If ROI data is a list of lists (multiple regions), process each
                        if isinstance(roi_data, list) and all(isinstance(item, list) for item in roi_data):
                            print(f"  Sample {i+1} has {len(roi_data)} regions")
                            
                            masks = []
                            for region in roi_data:
                                mask = roi_to_mask(region)
                                masks.append(mask)
                            
                            # Combine masks
                            combined_mask = np.zeros_like(masks[0])
                            for mask in masks:
                                combined_mask = np.maximum(combined_mask, mask)
                            
                            mask_to_visualize = combined_mask
                        else:
                            # Single region
                            mask_to_visualize = roi_to_mask(roi_data)
                        
                        # Display
                        plt.figure(figsize=(10, 5))
                        plt.subplot(1, 2, 1)
                        plt.imshow(mask_to_visualize, cmap='gray')
                        plt.title(f"Mask from {row.get('Origin', 'Unknown')}")
                        
                        # Try to load corresponding image if possible (placeholder for now)
                        plt.subplot(1, 2, 2)
                        plt.text(0.5, 0.5, "Image would be loaded here\nif mapping is established", 
                                ha='center', va='center')
                        plt.title("Corresponding Image (Placeholder)")
                        
                        plt.tight_layout()
                        plt.savefig(f"results/visualizations/mask_sample_{csv_file.stem}_{i}.png")
                        plt.close()
                    
                    except Exception as e:
                        print(f"  Error processing row: {e}")
                
                print(f"  Processed {len(samples)} samples")
        
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

if __name__ == "__main__":
    # Create visualizations directory if it doesn't exist
    Path("results/visualizations").mkdir(parents=True, exist_ok=True)
    
    visualize_masks_from_csv()