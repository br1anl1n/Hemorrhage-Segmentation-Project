"""
Finds and organizes images: It searches through folders to locate brain CT scan 
images with different viewing settings (called "window types" like "brain_window").
Loads label information: It reads CSV files containing information about where 
the hemorrhages (bleeding) are located in each image.
Matches images with labels: It figures out which labels belong to which images 
by matching ID codes.
Converts annotations to masks: It takes the hemorrhage outlines (stored as series 
of points) and converts them into binary masks (white areas on black backgrounds) 
that mark exactly where the bleeding is.
Prepares data for training: It packages everything into a format that PyTorch 
(the deep learning framework) can use, with properly formatted image tensors and mask tensors.
Creates data loaders: It sets up efficient pipelines to feed batches of data into 
the neural network during training.
"""
import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import json
import ast
from torch.utils.data import Dataset, DataLoader
import torch
import re
from collections import defaultdict
import random

class HemorrhageDataset(Dataset):
    """Dataset for brain hemorrhage segmentation."""
    
    def __init__(self, base_dir, labels_dir, window_types=None, transform=None):
        """
        Initialize the dataset.
        
        Args:
            base_dir (str): Base directory with all the image directories
            labels_dir (str): Directory with all the segmentation labels
            window_types (list, optional): List of window types to include. Default: ['brain_window']
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.base_dir = Path(base_dir).expanduser()
        self.labels_dir = Path(labels_dir).expanduser()
        self.window_types = window_types if window_types else ['brain_window']
        self.transform = transform
        
        # Map image IDs to their file paths
        self.image_paths = {}
        self.load_image_paths()
        
        # Load label data
        self.labels = {}
        self.load_labels()
        
        # Create a list of valid image IDs (intersection of images and labels)
        self.valid_ids = list(set(self.image_paths.keys()).intersection(set(self.labels.keys())))
        print(f"Found {len(self.valid_ids)} valid images with corresponding labels")
    
    def load_image_paths(self):
        """Load image paths and organize by ID."""
        id_pattern = re.compile(r'ID_([a-f0-9]+)\.jpg')
        
        for window_type in self.window_types:
            window_dir = self.base_dir / window_type
            if not window_dir.exists():
                raise FileNotFoundError(f"Window directory not found: {window_dir}")
            
            for image_file in window_dir.glob('*.jpg'):
                match = id_pattern.search(image_file.name)
                if match:
                    image_id = match.group(1)
                    if image_id not in self.image_paths:
                        self.image_paths[image_id] = {}
                    self.image_paths[image_id][window_type] = str(image_file)
        
        # Keep only images that have all required window types
        complete_ids = [id for id, paths in self.image_paths.items() 
                        if all(wt in paths for wt in self.window_types)]
        
        self.image_paths = {id: self.image_paths[id] for id in complete_ids}
        print(f"Found {len(self.image_paths)} images with all required window types")
    
    def load_labels(self):
        """Load label data from CSV files."""
        id_pattern = re.compile(r'ID_([a-f0-9]+)\.jpg')
        csv_files = list(self.labels_dir.glob("**/*.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if 'ROI' in df.columns and 'Origin' in df.columns:
                    for _, row in df.iterrows():
                        origin = row['Origin']
                        match = id_pattern.search(origin) if isinstance(origin, str) else None
                        
                        if match and 'ROI' in row and pd.notna(row['ROI']):  # Check for NaN values
                            image_id = match.group(1)
                            roi_data = row['ROI']
                            
                            self.labels[image_id] = {
                                'roi': roi_data,
                                'case_id': row.get('Case ID', None),
                                'source': csv_file.name
                            }
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        print(f"Loaded {len(self.labels)} labels from CSV files")
    
    def roi_to_mask(self, roi_string, width=512, height=512):
        """Convert ROI string to binary mask."""
        # Check if roi_string is NaN or not a string
        if isinstance(roi_string, float) or not isinstance(roi_string, str):
            return np.zeros((height, width), dtype=np.uint8)
        
        # Parse the ROI string into a Python object
        try:
            # Try direct JSON parsing
            roi_data = json.loads(roi_string)
        except:
            # Try evaluating as a Python literal
            try:
                roi_data = ast.literal_eval(roi_string)
            except:
                print(f"Failed to parse ROI string: {roi_string[:50]}...")
                return np.zeros((height, width), dtype=np.uint8)
        
        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Handle different ROI data formats
        if isinstance(roi_data, list):
            # If it's a list of points (single polygon)
            if len(roi_data) > 0 and isinstance(roi_data[0], dict) and 'x' in roi_data[0]:
                points = []
                for point in roi_data:
                    try:
                        x = int(float(point['x']) * width)
                        y = int(float(point['y']) * height)
                        points.append([x, y])
                    except (ValueError, TypeError) as e:
                        print(f"Error processing point: {point} - {e}")
                        continue
                
                if len(points) > 2:  # Need at least 3 points for a polygon
                    points = np.array(points, dtype=np.int32)
                    points = points.reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [points], 255)
            
            # If it's a list of polygons
            elif len(roi_data) > 0 and isinstance(roi_data[0], list):
                for polygon in roi_data:
                    if len(polygon) > 0 and isinstance(polygon[0], dict) and 'x' in polygon[0]:
                        points = []
                        for point in polygon:
                            try:
                                x = int(float(point['x']) * width)
                                y = int(float(point['y']) * height)
                                points.append([x, y])
                            except (ValueError, TypeError) as e:
                                print(f"Error processing point in polygon: {point} - {e}")
                                continue
                        
                        if len(points) > 2:
                            points = np.array(points, dtype=np.int32)
                            points = points.reshape((-1, 1, 2))
                            cv2.fillPoly(mask, [points], 255)
        
        return mask
    
    def __len__(self):
        return len(self.valid_ids)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image ID
        image_id = self.valid_ids[idx]
        
        # Load images for each window type
        images = []
        for window_type in self.window_types:
            img_path = self.image_paths[image_id][window_type]
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        
        # Combine images from different windows into a multi-channel image
        if len(images) == 1:
            combined_image = images[0]
        else:
            # For multiple window types, use first 3 channels of each image
            # and stack into a single multi-channel image
            combined_image = np.concatenate([img[:,:,:3] for img in images], axis=2)
        
        # Generate mask from ROI
        label_data = self.labels[image_id]
        mask = self.roi_to_mask(label_data['roi'])
        
        # Convert to grayscale
        if combined_image.ndim == 3 and combined_image.shape[2] % 3 == 0:
            # Process each window type separately
            num_windows = combined_image.shape[2] // 3
            gray_images = []
            
            for i in range(num_windows):
                # Extract RGB channels for this window
                window = combined_image[:, :, i*3:(i+1)*3]
                # Convert to grayscale using standard formula
                gray = window[:, :, 0] * 0.299 + window[:, :, 1] * 0.587 + window[:, :, 2] * 0.114
                gray_images.append(gray)
            
            # Stack grayscale images into a multi-channel image
            combined_image = np.stack(gray_images, axis=2)
        
        # Convert to tensors
        if combined_image.ndim == 2:
            # If single grayscale, add channel dimension
            combined_image = combined_image[:, :, np.newaxis]
        
        # Normalize image to [0, 1] range
        combined_image = combined_image.astype(np.float32) / 255.0
        
        # Transpose from HWC to CHW format (height, width, channels) -> (channels, height, width)
        combined_image = combined_image.transpose((2, 0, 1))
        
        # Convert mask to float32 and normalize to [0, 1]
        mask = mask.astype(np.float32) / 255.0
        
        # Add channel dimension to mask: (height, width) -> (1, height, width)
        mask = mask[np.newaxis, :, :]
        
        sample = {
            'image': torch.from_numpy(combined_image),
            'mask': torch.from_numpy(mask),
            'id': image_id,
            'case_id': label_data.get('case_id', ''),
            'source': label_data.get('source', '')
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

def get_data_loaders(base_dir, labels_dir, window_types=None, batch_size=8, val_split=0.2, transform=None):
    """
    Create training and validation data loaders.
    
    Args:
        base_dir (str): Base directory with all the image directories
        labels_dir (str): Directory with all the segmentation labels
        window_types (list, optional): List of window types to include
        batch_size (int): Size of batches
        val_split (float): Fraction of data to use for validation
        transform (callable, optional): Optional transform to be applied on a sample
        
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    # Create the dataset
    dataset = HemorrhageDataset(base_dir, labels_dir, window_types, transform)
    
    # Split into train and validation sets
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    
    # Shuffle indices
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Create samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=4
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4
    )
    
    return train_loader, val_loader