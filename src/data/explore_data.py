"""
Investigates label files: It opens and examines the CSV files containing hemorrhage
labels to understand their structure, columns, and contents.
Analyzes distribution of hemorrhage types: It tries to find information about 
different types of hemorrhages in the dataset and creates visual charts of their distribution.
Maps out image directories: It explores the folder structure where CT scan images
are stored and identifies different viewing modes ("window settings").
Provides sample visualizations: It randomly selects and displays sample images 
from each window setting so researchers can see what they're working with.
Analyzes image properties: It calculates statistics about the images such as:

    Image dimensions (height and width)
    Average brightness (mean intensity)
    Contrast levels (standard deviation of intensity)

Creates helpful visualizations: It generates charts and histograms showing the
 distribution of these image properties.

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

def load_hemorrhage_labels():
    """Load and explore the hemorrhage labels directory."""
    labels_dir = Path("/content/hemorrhage_project/hemorrhage-project/labels").expanduser()
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found at {labels_dir}")
    
    # List all CSV files in the labels directory and its subdirectories
    csv_files = list(labels_dir.glob("**/*.csv"))
    print(f"Found {len(csv_files)} CSV files in the labels directory:")
    for csv_file in csv_files:
        print(f"  - {csv_file.relative_to(labels_dir)}")
    
    # Try to read one of the CSV files to understand its structure
    if csv_files:
        sample_csv = csv_files[0]
        try:
            df = pd.read_csv(sample_csv)
            print(f"\nSample CSV file: {sample_csv.name}")
            print(f"Number of rows: {len(df)}")
            print("Columns:")
            for col in df.columns:
                print(f"  - {col}")
            
            # Display a few rows
            print("\nSample data (first 5 rows):")
            print(df.head())
            
            return df
        except Exception as e:
            print(f"Error reading CSV file: {e}")
    
    return None

def analyze_label_distribution(df):
    """Analyze the distribution of hemorrhage types."""
    if df is None:
        print("No dataframe provided for analysis.")
        return
    
    # Try to identify columns that might indicate hemorrhage type
    potential_type_columns = [col for col in df.columns if 'type' in col.lower() or 
                              'class' in col.lower() or 
                              'hemorrhage' in col.lower() or 
                              'classification' in col.lower()]
    
    if potential_type_columns:
        print("\nPotential hemorrhage type columns found:")
        for col in potential_type_columns:
            print(f"  - {col}")
            # Show distribution of values
            value_counts = df[col].value_counts()
            print(value_counts)
            
            # Plot the distribution
            plt.figure(figsize=(10, 6))
            value_counts.plot(kind='bar')
            plt.title(f'Distribution of {col}')
            plt.xlabel('Value')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(f'results/visualizations/{col}_distribution.png')
            plt.close()
    else:
        print("No potential hemorrhage type columns found. Please inspect the CSV manually.")

def explore_image_directories():
    """Explore the structure of image directories."""
    base_dir = Path("/content/hemorrhage_project/hemorrhage-project/renders/intraventricular").expanduser()
    if not base_dir.exists():
        raise FileNotFoundError(f"Image directory not found at {base_dir}")
    
    # List subdirectories (window types)
    window_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    print(f"\nFound {len(window_dirs)} window settings:")
    for window_dir in window_dirs:
        window_name = window_dir.name
        image_files = list(window_dir.glob('*.*'))  # Get all files
        extensions = {f.suffix for f in image_files}
        print(f"  - {window_name}: {len(image_files)} files with extensions {extensions}")
    
    return window_dirs

def sample_image_visualization(num_samples=2):
    """Visualize a few sample images from each window setting."""
    window_dirs = explore_image_directories()
    
    for window_dir in window_dirs:
        window_name = window_dir.name
        print(f"\nSampling images from {window_name}...")
        
        # Find image files (try common extensions)
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.dcm']:
            image_files.extend(list(window_dir.glob(f'*{ext}')))
        
        if not image_files:
            print(f"No image files found in {window_dir}")
            continue
        
        # Sample random images
        sampled_files = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
        
        # Create a figure to display the samples
        fig, axes = plt.subplots(1, len(sampled_files), figsize=(15, 5))
        if len(sampled_files) == 1:
            axes = [axes]  # Make sure axes is iterable
        
        for i, file_path in enumerate(sampled_files):
            try:
                img = cv2.imread(str(file_path))
                if img is None:
                    raise ValueError(f"Failed to load image {file_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                axes[i].imshow(img)
                axes[i].set_title(file_path.name, fontsize=8)
                axes[i].axis('off')
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                axes[i].text(0.5, 0.5, f"Error loading\n{file_path.name}", 
                             ha='center', va='center', color='red')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'results/visualizations/sample_{window_name}_images.png')
        plt.close()
        print(f"Visualized {len(sampled_files)} sample images from {window_name}")

def analyze_image_properties():
    """Analyze properties of the images across different window settings."""
    window_dirs = explore_image_directories()
    
    for window_dir in window_dirs:
        window_name = window_dir.name
        print(f"\nAnalyzing image properties from {window_name}...")
        
        # Find image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.dcm']:
            image_files.extend(list(window_dir.glob(f'*{ext}')))
        
        if not image_files:
            print(f"No image files found in {window_dir}")
            continue
        
        # Sample a subset of images to analyze
        sample_size = min(50, len(image_files))
        sampled_files = np.random.choice(image_files, sample_size, replace=False)
        
        # Collect image properties
        dimensions = []
        means = []
        stds = []
        
        for file_path in sampled_files:
            try:
                img = cv2.imread(str(file_path))
                if img is None:
                    continue
                
                height, width, channels = img.shape
                dimensions.append((height, width))
                means.append(img.mean())
                stds.append(img.std())
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
        
        # Plot histograms of image properties
        if dimensions:
            # Plot dimension distribution
            unique_dims = set(dimensions)
            print(f"Found {len(unique_dims)} unique dimensions: {unique_dims}")
            
            # Plot mean intensity distribution
            plt.figure(figsize=(10, 6))
            plt.hist(means, bins=20)
            plt.title(f'Mean Intensity Distribution - {window_name}')
            plt.xlabel('Mean Intensity')
            plt.ylabel('Frequency')
            plt.savefig(f'results/visualizations/{window_name}_mean_intensity.png')
            plt.close()
            
            # Plot standard deviation distribution
            plt.figure(figsize=(10, 6))
            plt.hist(stds, bins=20)
            plt.title(f'Intensity Standard Deviation - {window_name}')
            plt.xlabel('Standard Deviation')
            plt.ylabel('Frequency')
            plt.savefig(f'results/visualizations/{window_name}_std_intensity.png')
            plt.close()

def main():
    """Main function to run all exploration steps."""
    # Create visualizations directory if it doesn't exist
    os.makedirs('results/visualizations', exist_ok=True)
    
    try:
        # Explore label files
        print("Exploring label files...")
        df = load_hemorrhage_labels()
        analyze_label_distribution(df)
        
        # Explore image directories
        print("\nExploring image directories...")
        explore_image_directories()
        
        # Visualize sample images
        print("\nVisualizing sample images...")
        sample_image_visualization()
        
        # Analyze image properties
        print("\nAnalyzing image properties...")
        analyze_image_properties()
        
        print("\nData exploration completed successfully!")
    except Exception as e:
        print(f"Error during data exploration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()