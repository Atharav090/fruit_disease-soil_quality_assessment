"""
Dataset Loader for SoilVisioNet
Loads unified dataset with images, soil parameters, and weather sequences
"""
import os
import pandas as pd
import numpy as np
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class UnifiedDiseaseDataset(Dataset):
    """
    Load unified dataset with multi-modal data:
    - Images (fruit/crop disease photos)
    - Soil parameters (N, P, K, pH)
    - Weather sequences (30-day temporal patterns)
    """
    
    def __init__(self, split='train', data_root='data/unified_dataset', 
                 image_size=224, augment=False):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        
        # Load metadata
        metadata_path = self.data_root / 'metadata' / 'combined_dataset_metadata.csv'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        full_metadata = pd.read_csv(metadata_path)
        print(f"\n[DataLoader] Loaded metadata: {len(full_metadata)} total records")
        print(f"[DataLoader] Columns: {full_metadata.columns.tolist()}")
        
        # Filter by split
        self.metadata = full_metadata[full_metadata['split'] == split].reset_index(drop=True)
        
        # Use 'disease' column for labels (disease_class is mostly NaN)
        print(f"[DataLoader] {split} split: {len(self.metadata)} samples")
        
        # Check for weather sequences
        if 'weather_sequence' in self.metadata.columns:
            non_null_weather = self.metadata['weather_sequence'].notna().sum()
            print(f"[DataLoader] Weather sequences available: {non_null_weather} / {len(self.metadata)}")
        else:
            print(f"[DataLoader] WARNING: 'weather_sequence' column NOT found!")
        
        # Create disease label mapping from 'disease' column (more complete than disease_class)
        unique_diseases = sorted(self.metadata['disease'].dropna().unique())
        self.disease_to_idx = {d: i for i, d in enumerate(unique_diseases)}
        self.num_classes = len(self.disease_to_idx)
        
        # Add label column
        self.metadata['label'] = self.metadata['disease'].map(self.disease_to_idx)
        
        print(f"[DataLoader] Number of disease classes: {self.num_classes}")
        
        # Property for disease names (for compatibility)
        self.diseases = list(self.disease_to_idx.keys())

        # Compute global weather normalization stats (mean/std per feature) from full metadata
        self.weather_mean = np.array([20.0, 5.0, 72.0], dtype=np.float32)
        self.weather_std = np.array([10.0, 10.0, 15.0], dtype=np.float32)
        if 'weather_sequence' in full_metadata.columns:
            vals = {0: [], 1: [], 2: []}
            for s in full_metadata['weather_sequence'].dropna():
                try:
                    seq = json.loads(s) if isinstance(s, str) else s
                    for day in seq[:30]:
                        vals[0].append(float(day.get('temp', 20.0)))
                        vals[1].append(float(day.get('rainfall', 5.0)))
                        vals[2].append(float(day.get('humidity', 72.0)))
                except Exception:
                    continue
            try:
                a0 = np.array(vals[0], dtype=np.float32)
                a1 = np.array(vals[1], dtype=np.float32)
                a2 = np.array(vals[2], dtype=np.float32)
                if a0.size > 0:
                    self.weather_mean = np.array([a0.mean(), a1.mean(), a2.mean()], dtype=np.float32)
                    self.weather_std = np.array([a0.std(ddof=0) or 1.0, a1.std(ddof=0) or 1.0, a2.std(ddof=0) or 1.0], dtype=np.float32)
            except Exception:
                pass
        
        # Compute soil normalization stats
        self.soil_mean = np.array([100.0, 40.0, 200.0, 6.5], dtype=np.float32)
        self.soil_std = np.array([50.0, 20.0, 100.0, 1.5], dtype=np.float32)
        try:
            soil_vals = self.metadata[['soil_nitrogen', 'soil_phosphorus', 'soil_potassium', 'soil_ph']].to_numpy(dtype=np.float32)
            valid_soil = soil_vals[~np.isnan(soil_vals).any(axis=1)]
            if len(valid_soil) > 0:
                self.soil_mean = valid_soil.mean(axis=0).astype(np.float32)
                self.soil_std = valid_soil.std(axis=0).astype(np.float32)
                self.soil_std[self.soil_std == 0] = 1.0
        except Exception:
            pass
        
        # Create disease risk labels from weather conditions
        # Risk = 1 if conditions favor disease spread, 0 otherwise
        self.metadata['disease_risk'] = 0
        if 'weather_sequence' in self.metadata.columns:
            risk_list = []
            for idx, row in self.metadata.iterrows():
                risk = 0
                try:
                    if pd.notna(row['weather_sequence']):
                        seq_str = row['weather_sequence']
                        seq = json.loads(seq_str) if isinstance(seq_str, str) else seq_str
                        # Calculate average weather conditions over 30 days
                        temps = [float(d.get('temp', 20.0)) for d in seq[:30]]
                        rainfalls = [float(d.get('rainfall', 5.0)) for d in seq[:30]]
                        humidities = [float(d.get('humidity', 72.0)) for d in seq[:30]]
                        
                        avg_temp = np.mean(temps)
                        avg_rainfall = np.mean(rainfalls)
                        avg_humidity = np.mean(humidities)
                        
                        # Disease risk = 1 if favorable conditions for pathogen spread
                        # High humidity (>70%), adequate rainfall (>5mm), moderate temp (15-30°C)
                        if avg_humidity > 70 and avg_rainfall > 5 and 15 <= avg_temp <= 30:
                            risk = 1
                except Exception:
                    pass
                risk_list.append(risk)
            self.metadata['disease_risk'] = risk_list
        
        risk_dist = self.metadata['disease_risk'].value_counts().to_dict()
        print(f"[DataLoader] Disease risk: 0={risk_dist.get(0, 0)}, 1={risk_dist.get(1, 0)}")
        
        # Image transformations
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Try to find image by ID (images are in disease-based subdirectories, not numeric paths)
        # The new_path stored in metadata may not match actual filesystem structure
        image_id = row['image_id']
        split = row['split']
        
        # Search for the image file by its ID in the split directory
        images_dir = self.data_root / 'images' / split
        image_files = list(images_dir.glob(f"**/{image_id}*"))  # Find files starting with image_id
        
        if image_files:
            image_path = image_files[0]
        else:
            # Fallback: try the metadata path
            image_path = self.data_root / 'images' / row['new_path']
        
        if not image_path.exists():
            # Suppress most warnings to avoid spam during training
            # print(f"WARNING: Image not found for {image_id}: {image_path}")
            image = Image.new('RGB', (self.image_size, self.image_size))
        else:
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                # print(f"ERROR loading image {image_path}: {e}")
                image = Image.new('RGB', (self.image_size, self.image_size))
        
        image = self.transform(image)
        
        # Load risk label (0=unfavorable, 1=favorable for disease spread)
        try:
            risk_label = int(self.metadata.iloc[idx]['disease_risk'])
        except (ValueError, TypeError):
            risk_label = 0
        risk_label = torch.tensor(risk_label, dtype=torch.long)

        # Load disease class label (if available)
        try:
            class_label = int(self.metadata.iloc[idx]['label'])
        except (ValueError, TypeError):
            class_label = -1
        class_label = torch.tensor(class_label, dtype=torch.long)
        
        # Load soil parameters and normalize
        soil = torch.zeros(4, dtype=torch.float32)
        try:
            soil_raw = np.array([
                float(row['soil_nitrogen']) if pd.notna(row['soil_nitrogen']) else 0.0,
                float(row['soil_phosphorus']) if pd.notna(row['soil_phosphorus']) else 0.0,
                float(row['soil_potassium']) if pd.notna(row['soil_potassium']) else 0.0,
                float(row['soil_ph']) if pd.notna(row['soil_ph']) else 0.0
            ], dtype=np.float32)
            # Normalize soil features
            soil_normalized = (soil_raw - self.soil_mean) / self.soil_std
            soil = torch.tensor(soil_normalized, dtype=torch.float32)
        except (ValueError, TypeError):
            soil = torch.zeros(4, dtype=torch.float32)
        
        # Load weather sequence (30 days × 3 features)
        weather = torch.zeros(30, 3, dtype=torch.float32)
        
        if 'weather_sequence' in row and pd.notna(row['weather_sequence']):
            try:
                seq_str = row['weather_sequence']
                # Handle both string and already-parsed formats
                if isinstance(seq_str, str):
                    seq_list = json.loads(seq_str)
                else:
                    seq_list = seq_str
                
                # Extract and normalize weather data
                for day_idx, day_data in enumerate(seq_list[:30]):  # Take first 30 days
                    if day_idx >= 30:
                        break
                    
                    # Extract features with fallback
                    temp = float(day_data.get('temp', 20.0))
                    rainfall = float(day_data.get('rainfall', 5.0))
                    humidity = float(day_data.get('humidity', 72.0))

                    # Normalize using dataset mean/std
                    temp_norm = (temp - float(self.weather_mean[0])) / float(self.weather_std[0])
                    rainfall_norm = (rainfall - float(self.weather_mean[1])) / float(self.weather_std[1])
                    humidity_norm = (humidity - float(self.weather_mean[2])) / float(self.weather_std[2])

                    # Clip to avoid extreme values
                    weather[day_idx, 0] = np.clip(temp_norm, -3, 3)
                    weather[day_idx, 1] = np.clip(rainfall_norm, -3, 3)
                    weather[day_idx, 2] = np.clip(humidity_norm, -3, 3)
            
            except Exception as e:
                print(f"ERROR parsing weather for sample {idx}: {e}")
                weather = torch.zeros(30, 3, dtype=torch.float32)
        
        return {
            'image': image,
            'risk': risk_label,
            'label': class_label,
            'soil': soil,
            'weather': weather
        }

def get_dataloaders(data_root='data/unified_dataset', batch_size=8, 
                    num_workers=0, augment=True, image_size=224):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_root: Path to unified_dataset
        batch_size: Batch size
        num_workers: Number of data loading workers (set to 0 for CPU)
        augment: Apply augmentation to training data
        image_size: Image resolution
    
    Returns:
        (train_loader, val_loader, test_loader, datasets)
    """
    
    print("\n" + "=" * 70)
    print("CREATING DATALOADERS")
    print("=" * 70)
    
    # Create datasets
    train_dataset = UnifiedDiseaseDataset(
        split='train', 
        data_root=data_root,
        image_size=image_size,
        augment=augment
    )
    
    val_dataset = UnifiedDiseaseDataset(
        split='val',
        data_root=data_root,
        image_size=image_size,
        augment=False
    )
    
    test_dataset = UnifiedDiseaseDataset(
        split='test',
        data_root=data_root,
        image_size=image_size,
        augment=False
    )
    
    # Create dataloaders
    # Use WeightedRandomSampler to balance classes during training
    try:
        # Ensure labels is a numpy integer array for numeric ops
        labels = train_dataset.metadata['label'].fillna(-1).to_numpy(dtype=np.int64)
        valid_idx = (labels >= 0)
        counts = np.bincount(labels[valid_idx].astype(np.int64), minlength=train_dataset.num_classes).astype(np.float32)
        counts[counts == 0] = 1.0
        class_weights = 1.0 / counts
        sample_weights = class_weights[labels]
        sample_weights[~valid_idx] = 0.0
        # Convert to plain Python list of floats for the sampler
        sample_weights_list = sample_weights.astype(float).tolist()
        sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights_list, num_samples=len(sample_weights_list), replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=False if num_workers == 0 else True
        )
    except Exception:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False if num_workers == 0 else True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False if num_workers == 0 else True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False if num_workers == 0 else True
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_loader)} batches × {batch_size} = {len(train_dataset)} samples")
    print(f"  Val:   {len(val_loader)} batches × {batch_size} = {len(val_dataset)} samples")
    print(f"  Test:  {len(test_loader)} batches × {batch_size} = {len(test_dataset)} samples")
    print(f"  Classes: {train_dataset.num_classes}")
    print("=" * 70)
    
    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    
    return train_loader, val_loader, test_loader, datasets

if __name__ == "__main__":
    train_loader, val_loader, test_loader, datasets = get_dataloaders(
        data_root='data/unified_dataset',
        batch_size=8,
        num_workers=0
    )
    
    # Test one batch
    print("\nTesting one batch...")
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Image shape: {batch['image'].shape}")
    print(f"Label shape: {batch['label'].shape}")
    print(f"Weather shape: {batch['weather'].shape}")
    print(f"Soil shape: {batch['soil'].shape}")
    print(f"Weather sample (first day): {batch['weather'][0, 0, :]}")