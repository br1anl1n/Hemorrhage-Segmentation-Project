import os
from pathlib import Path
import pandas as pd
import re
from collections import defaultdict

def analyze_filename_patterns():
    """Analyze filename patterns in images across all window types and labels."""
    # Base directory for all window types
    base_dir = Path("/content/hemorrhage_project/hemorrhage-project/renders/intraventricular").expanduser()
    
    # Get all window type directories
    window_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    print(f"Found {len(window_dirs)} window types: {[d.name for d in window_dirs]}\n")
    
    # Dictionary to store IDs by window type
    window_ids = {}
    id_pattern = re.compile(r'ID_([a-f0-9]+)\.jpg')
    
    # Process each window type
    for window_dir in window_dirs:
        window_name = window_dir.name
        print(f"Processing {window_name}...")
        
        # Get all image files
        image_files = list(window_dir.glob('*.jpg'))
        
        # Extract IDs from filenames
        all_ids = [id_pattern.search(f.name).group(1) if id_pattern.search(f.name) else None 
                for f in image_files]
        unique_ids = set(filter(None, all_ids))
        
        # Store in dictionary
        window_ids[window_name] = unique_ids
        
        # Print info about this window type
        print(f"  Found {len(unique_ids)} unique IDs out of {len(image_files)} image files")
        print(f"  ID length: {len(next(iter(unique_ids))) if unique_ids else 'N/A'}")
        print(f"  Sample IDs: {list(unique_ids)[:3]}")
    
    # Compare IDs across window types
    all_unique_ids = set()
    for ids in window_ids.values():
        all_unique_ids.update(ids)
    
    print(f"\nTotal unique IDs across all window types: {len(all_unique_ids)}")
    
    # Check which IDs appear in all window types
    common_across_all = set.intersection(*window_ids.values()) if window_ids else set()
    print(f"IDs found in ALL window types: {len(common_across_all)}")
    
    # Check IDs unique to specific window types
    for window_name, ids in window_ids.items():
        other_ids = set()
        for other_name, other_window_ids in window_ids.items():
            if other_name != window_name:
                other_ids.update(other_window_ids)
        
        unique_to_window = ids - other_ids
        if unique_to_window:
            print(f"IDs unique to {window_name}: {len(unique_to_window)}")
            print(f"  Sample unique IDs: {list(unique_to_window)[:3]}")
    
    # Now process label files
    labels_dir = Path("/content/hemorrhage_project/hemorrhage-project/labels").expanduser()
    csv_files = list(labels_dir.glob("**/*.csv"))
    print(f"\nFound {len(csv_files)} CSV files in labels directory")
    
    # Extract origins from all CSV files
    all_origins = []
    csv_info = {}
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'Origin' in df.columns:
                origins = df['Origin'].tolist()
                all_origins.extend(origins)
                csv_info[csv_file.name] = len(origins)
                print(f"  {csv_file.name}: {len(origins)} entries")
        except Exception as e:
            print(f"  Error reading {csv_file}: {e}")
    
    # Extract IDs from origins
    origin_ids = []
    for origin in all_origins:
        if isinstance(origin, str):
            match = id_pattern.search(origin)
            if match:
                origin_ids.append(match.group(1))
    
    unique_origin_ids = set(origin_ids)
    print(f"\nFound {len(unique_origin_ids)} unique IDs in label origins")
    print(f"Sample origin IDs: {list(unique_origin_ids)[:5]}")
    
    # Compare with image IDs
    for window_name, ids in window_ids.items():
        common_ids = ids.intersection(unique_origin_ids)
        print(f"\nMatches between {window_name} and labels: {len(common_ids)} ({(len(common_ids)/len(ids))*100:.1f}%)")
        if common_ids:
            print(f"Sample common IDs: {list(common_ids)[:5]}")
    
    # Total matches across all window types
    all_matches = all_unique_ids.intersection(unique_origin_ids)
    print(f"\nTotal matches across all window types: {len(all_matches)} ({(len(all_matches)/len(all_unique_ids))*100:.1f}%)")
    
    # If few direct matches, try alternative matching strategies
    if len(all_matches) < 0.5 * len(all_unique_ids):
        print("\nTrying alternative matching strategies...")
        
        # 1. Case-insensitive matching
        lower_image_ids = {id.lower() for id in all_unique_ids}
        lower_origin_ids = {id.lower() for id in unique_origin_ids}
        case_insensitive_common = lower_image_ids.intersection(lower_origin_ids)
        print(f"Case-insensitive matches: {len(case_insensitive_common)}")
        
        # 2. Substring matching
        print("\nChecking for substring matches...")
        substring_matches = defaultdict(list)
        for img_id in list(all_unique_ids)[:100]:  # Check first 100 to save time
            for origin_id in origin_ids:
                if img_id in origin_id:
                    substring_matches[img_id].append(origin_id)
        
        print(f"Found {len(substring_matches)} image IDs that are substrings of origin IDs")
        if substring_matches:
            print("Sample substring matches:")
            for i, (img_id, matches) in enumerate(list(substring_matches.items())[:5]):
                print(f"  Image ID: {img_id} found in: {matches[:2]} {'...' if len(matches) > 2 else ''}")
        
        # 3. Suffix matching
        print("\nChecking for suffix pattern matches...")
        for suffix_len in [8, 6, 4]:
            image_id_suffixes = {id[-suffix_len:] for id in all_unique_ids if len(id) >= suffix_len}
            origin_id_suffixes = {id[-suffix_len:] for id in unique_origin_ids if len(id) >= suffix_len}
            suffix_common = image_id_suffixes.intersection(origin_id_suffixes)
            print(f"Common ID suffixes (last {suffix_len} chars): {len(suffix_common)}")
            
            if suffix_common and suffix_len == 8:
                print("Sample common suffixes:")
                for suffix in list(suffix_common)[:5]:
                    print(f"  {suffix}")

if __name__ == "__main__":
    analyze_filename_patterns()