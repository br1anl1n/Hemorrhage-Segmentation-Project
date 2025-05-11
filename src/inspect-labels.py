import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import json
import re

def inspect_label_format():
    """Inspect the format of the label files in detail."""
    labels_dir = Path("/content/hemorrhage_project/hemorrhage-project/labels").expanduser()
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found at {labels_dir}")
    
    # Get the first CSV file in Set1
    csv_files = list(labels_dir.glob("**/*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    # Inspect each CSV file to understand structure
    for csv_file in csv_files[:2]:  # Look at the first 2 files
        print(f"\nInspecting {csv_file.relative_to(labels_dir)}...")
        
        try:
            df = pd.read_csv(csv_file)
            print(f"Number of rows: {len(df)}")
            print(f"Columns: {', '.join(df.columns)}")
            
            # Check the first few entries in the 'All Annotations' column
            if 'All Annotations' in df.columns:
                print("\nSample annotation formats:")
                sample_annotations = df['All Annotations'].head(3).values
                
                for i, annotation in enumerate(sample_annotations):
                    print(f"\nSample {i+1}:")
                    
                    # Try to parse as JSON
                    try:
                        parsed = json.loads(annotation)
                        print(f"Structure: {type(parsed)}")
                        if isinstance(parsed, list):
                            print(f"Number of elements: {len(parsed)}")
                            if len(parsed) > 0:
                                print(f"First element type: {type(parsed[0])}")
                                print(f"First element content: {parsed[0]}")
                    except:
                        # If not JSON, print a snippet
                        print(f"Not JSON format. Sample content: {annotation[:200]}...")
            
            # Check if any rows contain useful file mapping information
            if 'Origin' in df.columns and 'Case ID' in df.columns:
                print("\nSample Origin-Case ID mapping:")
                mapping_sample = df[['Origin', 'Case ID']].head(5)
                print(mapping_sample)
            
            # Check the ROI column if it exists
            if 'ROI' in df.columns:
                print("\nSample ROI format:")
                roi_samples = df['ROI'].head(3).values
                
                for i, roi in enumerate(roi_samples):
                    print(f"\nROI Sample {i+1}:")
                    try:
                        parsed = json.loads(roi)
                        print(f"Structure: {type(parsed)}")
                        if isinstance(parsed, list):
                            print(f"Number of elements: {len(parsed)}")
                            if len(parsed) > 0:
                                print(f"First element type: {type(parsed[0])}")
                                print(f"First element content: {parsed[0]}")
                    except:
                        print(f"Not JSON format. Sample content: {roi[:200]}...")
                        
        except Exception as e:
            print(f"Error inspecting CSV file: {e}")

def check_image_label_mapping():
    """Check how images in the render directories map to labels."""
    # Get paths
    labels_dir = Path("/content/hemorrhage_project/hemorrhage-project/labels").expanduser()
    images_dir = Path("/content/hemorrhage_project/hemorrhage-project/renders/intraventricular/brain_window").expanduser()
    
    # Read the first CSV file
    csv_files = list(labels_dir.glob("**/*.csv"))
    if not csv_files:
        print("No CSV files found")
        return
    
    df = pd.read_csv(csv_files[0])
    
    # Extract unique image identifiers from Origin column
    if 'Origin' in df.columns:
        origins = df['Origin'].unique()
        print(f"Found {len(origins)} unique origin values in first CSV")
        print(f"Sample origins: {origins[:5]}")
        
        # Extract IDs from origins
        id_pattern = re.compile(r'ID_[a-f0-9]+')
        ids = [id_pattern.search(origin).group(0) if id_pattern.search(origin) else None 
               for origin in origins[:20]]
        print(f"Extracted IDs: {[id for id in ids if id]}")
        
        # Check if these IDs are in image filenames
        image_files = list(images_dir.glob('*.jpg'))
        image_filenames = [f.name for f in image_files[:20]]
        print(f"Sample image filenames: {image_filenames}")
        
        # Try to match IDs with image filenames
        matches = []
        for id in ids:
            if id is None:
                continue
            for img in image_filenames:
                if id in img:
                    matches.append((id, img))
                    break
        
        print(f"Found {len(matches)} matches between IDs and image filenames")
        for match in matches:
            print(f"ID: {match[0]} -> Image: {match[1]}")

if __name__ == "__main__":
    print("Inspecting label format...")
    inspect_label_format()
    
    print("\nChecking image-label mapping...")
    check_image_label_mapping()