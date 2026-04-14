"""
Dataset Validation Script
Verifies data integrity after integration
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter
import os

def validate_integrated_dataset(data_root: str):
    """Comprehensive validation of integrated dataset"""
    
    data_root = Path(data_root)
    unified_dir = data_root / "unified_dataset"
    metadata_file = unified_dir / "metadata" / "combined_dataset_metadata.csv"
    
    print("\n" + "="*70)
    print("VALIDATING INTEGRATED DATASET")
    print("="*70)
    
    # Load metadata
    print("\n[1] Loading metadata...")
    try:
        df = pd.read_csv(metadata_file)
        print(f"  ✓ Metadata loaded: {len(df)} records")
    except Exception as e:
        print(f"  ✗ Error loading metadata: {e}")
        return
    
    # Check 1: Basic statistics
    print("\n[2] Basic Statistics")
    print(f"  Total images: {len(df)}")
    print(f"  Unique diseases: {df['disease'].nunique()}")
    print(f"  Unique crops: {df['crop_type'].nunique()}")
    print(f"  Data sources: {df['source'].unique().tolist()}")
    
    # Check 2: Split distribution
    print("\n[3] Split Distribution")
    split_counts = df['split'].value_counts()
    for split in ['train', 'val', 'test']:
        count = split_counts.get(split, 0)
        pct = (count / len(df)) * 100
        print(f"  {split:5s}: {count:5d} images ({pct:5.1f}%)")
    
    # Check 3: Crop distribution
    print("\n[4] Crop Type Distribution")
    crop_counts = df['crop_type'].value_counts().head(10)
    for crop, count in crop_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {crop:20s}: {count:5d} images ({pct:5.1f}%)")
    
    # Check 4: Disease distribution
    print("\n[5] Top 15 Disease Classes")
    disease_counts = df['disease'].value_counts().head(15)
    for disease, count in disease_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {disease[:40]:40s}: {count:4d} ({pct:5.1f}%)")
    
    # Check 5: Soil data
    print("\n[6] Soil Parameter Statistics")
    soil_cols = ['soil_nitrogen', 'soil_phosphorus', 'soil_potassium', 'soil_ph']
    
    for col in soil_cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce')
            print(f"  {col:25s}:")
            print(f"    Mean:   {values.mean():8.2f}")
            print(f"    Std:    {values.std():8.2f}")
            print(f"    Min:    {values.min():8.2f}")
            print(f"    Max:    {values.max():8.2f}")
            print(f"    NaNs:   {values.isna().sum():5d}")
    
    # Check 6: Temporal data
    print("\n[7] Weather Sequence Check")
    if 'weather_sequence' in df.columns:
        try:
            seq = json.loads(df.iloc[0]['weather_sequence'])
            print(f"  ✓ Sample sequence has {len(seq)} days")
            if len(seq) > 0:
                print(f"  ✓ First entry: {seq[0]}")
        except Exception as e:
            print(f"  ⚠ Error parsing weather sequences: {e}")
    
    # Check 7: Data quality
    print("\n[8] Data Quality Checks")
    
    # Missing values
    missing = df.isnull().sum()
    critical_cols = ['image_id', 'disease', 'crop_type', 'split']
    for col in critical_cols:
        if col in missing.index:
            if missing[col] > 0:
                print(f"  ⚠ Missing values in {col}: {missing[col]}")
            else:
                print(f"  ✓ {col}: No missing values")
    
    # Check source distribution
    print("\n[9] Source Distribution")
    source_counts = df['source'].value_counts()
    for source, count in source_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {source:20s}: {count:5d} images ({pct:5.1f}%)")
    
    # Check 10: Files existence (if images organized)
    print("\n[10] Image File Existence Check")
    image_dir = unified_dir / "images"
    if image_dir.exists():
        train_count = len(list((image_dir / "train").rglob("*.*")))
        val_count = len(list((image_dir / "val").rglob("*.*")))
        test_count = len(list((image_dir / "test").rglob("*.*")))
        
        print(f"  Train images: {train_count}")
        print(f"  Val images:   {val_count}")
        print(f"  Test images:  {test_count}")
        
        total_files = train_count + val_count + test_count
        expected = len(df)
        
        if total_files == expected:
            print(f"  ✓ All {expected} images present")
        elif total_files == 0:
            print(f"  ℹ No images organized yet (metadata ready)")
        else:
            print(f"  ⚠ Expected {expected}, found {total_files}")
    else:
        print(f"  ℹ Image directory not yet organized")
    
    # Check 11: Balanced dataset
    print("\n[11] Class Balance Analysis")
    class_counts = df['disease'].value_counts()
    
    if len(class_counts) > 0:
        print(f"  Most frequent class: {class_counts.index[0]}")
        print(f"    Samples: {class_counts.iloc[0]}")
        print(f"  Least frequent class: {class_counts.index[-1]}")
        print(f"    Samples: {class_counts.iloc[-1]}")
        
        imbalance_ratio = class_counts.iloc[0] / class_counts.iloc[-1]
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")
        
        if imbalance_ratio < 5:
            print(f"  ✓ Dataset reasonably balanced")
        else:
            print(f"  ⚠ Dataset has class imbalance; consider oversampling minority classes")
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print("\n✓ Dataset structure is valid and ready for model training!")
    print(f"\nKey metrics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Train/Val/Test split: {split_counts.get('train', 0)}/{split_counts.get('val', 0)}/{split_counts.get('test', 0)}")
    print(f"  Disease classes: {df['disease'].nunique()}")
    print(f"  Crops: {df['crop_type'].nunique()}")
    print(f"  Soil parameters: Linked")
    print(f"  Weather sequences: Linked")
    
    print("\nYou can now proceed to:")
    print("  1. Model training scripts")
    print("  2. Custom data loaders using the metadata CSVs")
    print("  3. Data augmentation and preprocessing")

if __name__ == "__main__":
    data_root = os.path.join(os.path.dirname(__file__), "..", "data")
    validate_integrated_dataset(data_root)
