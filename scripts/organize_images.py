"""
Image Organization Pipeline
Copies and organizes images from source to unified dataset directory
Handles PlantVillage and fruit images
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

class ImageOrganizer:
    def __init__(self, data_root: str, metadata_file: str):
        self.data_root = Path(data_root)
        self.unified_dir = self.data_root / "unified_dataset"
        
        if not Path(metadata_file).exists():
            print(f"✗ Metadata file not found: {metadata_file}")
            self.metadata_df = pd.DataFrame()
        else:
            self.metadata_df = pd.read_csv(metadata_file)
    
    def organize_images(self):
        """Copy images to unified directory based on split"""
        if len(self.metadata_df) == 0:
            print("✗ No metadata loaded. Skipping image organization.")
            return
        
        print("Organizing images into unified directory structure...")
        print(f"Total images to process: {len(self.metadata_df)}")
        
        # Create required directories
        for split in ['train', 'val', 'test']:
            (self.unified_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        
        # Copy images
        success = 0
        failed = 0
        
        for idx, row in tqdm(self.metadata_df.iterrows(), total=len(self.metadata_df)):
            try:
                if row['source'] == 'plantvillage':
                    # PlantVillage source path: raw/color/Disease/image.JPG
                    src_real = self.data_root / "PlantVillage-Dataset-master" / row['original_path']
                else:
                    # Fruit dataset source path
                    src_real = Path(row['original_path'])
                
                if not src_real.exists():
                    failed += 1
                    continue
                
                # Destination: unified_dataset/images/{split}/{disease}/{image_id}.jpg
                split = row['split']
                disease = row['disease'].replace(' ', '_').replace('/', '_')
                filename = f"{row['image_id']}.jpg"
                
                dest_dir = self.unified_dir / "images" / split / disease
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                dest = dest_dir / filename
                
                # Copy file
                shutil.copy2(src_real, dest)
                success += 1
                
            except Exception as e:
                failed += 1
        
        print(f"\n✓ Image organization complete:")
        print(f"  - Successfully copied: {success}")
        print(f"  - Failed/Missing: {failed}")
    
    def create_dataset_splits(self):
        """Create train/val/test split files"""
        if len(self.metadata_df) == 0:
            print("✗ No metadata loaded. Skipping split creation.")
            return
        
        print("\nCreating dataset splits...")
        
        for split in ['train', 'val', 'test']:
            split_df = self.metadata_df[self.metadata_df['split'] == split].copy()
            
            # Create file list with image paths
            split_df['image_path'] = split_df.apply(
                lambda x: f"images/{split}/{x['disease'].replace(' ', '_').replace('/', '_')}/{x['image_id']}.jpg",
                axis=1
            )
            
            # Save split file
            split_file = self.unified_dir / "metadata" / f"{split}_split.csv"
            split_file.parent.mkdir(parents=True, exist_ok=True)
            split_df[['image_id', 'image_path', 'disease', 'crop_type', 'source']].to_csv(split_file, index=False)
            
            print(f"  ✓ {split:5s} split: {len(split_df):5d} images")

def main():
    data_root = os.path.join(os.path.dirname(__file__), "..", "data")
    metadata_file = f"{data_root}/unified_dataset/metadata/combined_dataset_metadata.csv"
    
    organizer = ImageOrganizer(data_root, metadata_file)
    organizer.organize_images()
    organizer.create_dataset_splits()
    
    print("\n✓ Image organization complete!")

if __name__ == "__main__":
    main()
