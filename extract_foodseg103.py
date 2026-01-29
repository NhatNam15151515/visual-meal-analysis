
import os
import json
import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from pathlib import Path

def extract_dataset(data_dir, output_root):
    data_dir = Path(data_dir)
    output_root = Path(output_root)
    
    # Define paths
    images_root = output_root / 'images'
    masks_root = output_root / 'masks'
    
    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    try:
        dataset = load_dataset("parquet", data_files={
            'train': str(data_dir / 'train-*.parquet'),
            'validation': str(data_dir / 'validation-*.parquet')
        })
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Metadata storage
    image_classes_meta = {}
    
    # Validation counters
    stats = {
        'train': {'expected': dataset['train'].num_rows, 'extracted': 0, 'errors': 0},
        'validation': {'expected': dataset['validation'].num_rows, 'extracted': 0, 'errors': 0}
    }
    
    print("\nStarting extraction...")
    
    for split in ['train', 'validation']:
        split_images_dir = images_root / split
        split_masks_dir = masks_root / split
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_masks_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {split} split (Expected: {stats[split]['expected']})...")
        
        for item in tqdm(dataset[split]):
            try:
                # Get ID
                img_id = str(item['id']) # Ensure ID is string
                
                # Extract image
                image = item['image']
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Extract label (mask)
                mask = item['label']
                
                # Validate Mask Values
                mask_array = np.array(mask)
                unique_values = np.unique(mask_array)
                if np.any(unique_values > 103) or np.any(unique_values < 0):
                     print(f"Warning: Image {img_id} contains invalid class IDs: {unique_values[unique_values > 103]}")
                
                # Save files
                image_path = split_images_dir / f"{img_id}.jpg"
                mask_path = split_masks_dir / f"{img_id}.png"
                
                image.save(image_path)
                mask.save(mask_path)
                
                # Store metadata
                classes = item.get('classes_on_image', [])
                image_classes_meta[img_id] = classes
                
                stats[split]['extracted'] += 1
                
            except Exception as e:
                print(f"Error extracting item {item.get('id', 'unknown')}: {e}")
                stats[split]['errors'] += 1

    # Save metadata
    meta_path = output_root / 'image_classes.json'
    with open(meta_path, 'w') as f:
        json.dump(image_classes_meta, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")

    # Final Verification
    print("\n=== Verification Report ===")
    all_passed = True
    for split, data in stats.items():
        print(f"Split: {split}")
        print(f"  Expected: {data['expected']}")
        print(f"  Extracted: {data['extracted']}")
        print(f"  Errors: {data['errors']}")
        
        if data['expected'] != data['extracted']:
            print(f"  ❌ Mismatch in Count!")
            all_passed = False
        else:
            print(f"  ✅ Count Verified")
            
    if all_passed:
        print("\n✅ All validations passed successfully!")
    else:
        print("\n❌ Some validations failed. Check logs.")

if __name__ == "__main__":
    DATA_DIR = r"c:\Nhat Nam\do an chuyen nganh\visual-meal-analysis\data\FoodSeg103\data"
    OUTPUT_ROOT = r"c:\Nhat Nam\do an chuyen nganh\visual-meal-analysis\data\FoodSeg103"
    
    extract_dataset(DATA_DIR, OUTPUT_ROOT)
