"""
Dataset Integration Pipeline for SoilVisioNet
Combines PlantVillage + Fruit Images + Synthetic Soil/Temporal Data

Pipeline:
1. Load and organize PlantVillage images
2. Load and organize Fruit disease images
3. Create unified disease taxonomy
4. Link soil parameters (synthetic) based on crop type
5. Generate temporal weather sequences
6. Create metadata CSV with all linkages
"""

import os
import json
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import uuid
from datetime import datetime, timedelta
from weather_integration import WeatherIntegrator, get_or_create_weather_cache

class DatasetIntegrator:
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.unified_dir = self.data_root / "unified_dataset"
        
        # Disease mapping: normalize all diseases to consistent taxonomy
        self.disease_taxonomy = {
            # Apple
            "apple_scab": "Apple___Apple_scab",
            "apple_healthy": "Apple___Healthy",
            "apple_blotch": "Apple___Blotch",
            "apple_rot": "Apple___Rot",
            "cedar_rust": "Apple___Cedar_apple_rust",
            
            # Guava
            "guava_anthracnose": "Guava___Anthracnose",
            "guava_fruitfly": "Guava___Fruitfly",
            "guava_healthy": "Guava___Healthy",
            
            # Mango
            "mango_anthracnose": "Mango___Anthracnose",
            "mango_alternaria": "Mango___Alternaria",
            "mango_black_mold": "Mango___Black_Mould_Rot",
            "mango_stem_rot": "Mango___Stem_Rot",
            "mango_healthy": "Mango___Healthy",
            
            # Pomegranate
            "pomegranate_anthracnose": "Pomegranate___Anthracnose",
            "pomegranate_alternaria": "Pomegranate___Alternaria",
            "pomegranate_bacterial": "Pomegranate___Bacterial_Blight",
            "pomegranate_cercospora": "Pomegranate___Cercospora",
            "pomegranate_healthy": "Pomegranate___Healthy",
            
            # From PlantVillage (additional crops for training)
            "tomato_bacterial_spot": "Tomato___Bacterial_spot",
            "tomato_early_blight": "Tomato___Early_blight",
            "tomato_late_blight": "Tomato___Late_blight",
            "tomato_healthy": "Tomato___Healthy",
            "potato_early_blight": "Potato___Early_blight",
            "potato_late_blight": "Potato___Late_blight",
            "potato_healthy": "Potato___Healthy",
            "grape_black_rot": "Grape___Black_rot",
            "grape_healthy": "Grape___Healthy",
            "peach_bacterial": "Peach___Bacterial_spot",
            "peach_healthy": "Peach___Healthy",
        }
        
        # Crop to state mapping (for soil data linkage)
        self.crop_state_mapping = {
            "apple": ["himachal_pradesh", "uttarakhand", "jammu_kashmir"],
            "guava": ["uttar_pradesh", "madhya_pradesh", "maharashtra"],
            "mango": ["maharashtra", "andhra_pradesh", "karnataka"],
            "pomegranate": ["maharashtra", "karnataka", "andhra_pradesh"],
            "tomato": ["karnataka", "maharashtra", "andhra_pradesh"],
            "potato": ["uttar_pradesh", "punjab", "bihar"],
            "grape": ["maharashtra", "karnataka"],
            "peach": ["himachal_pradesh", "punjab"],
        }
    
    def load_plantvillage_metadata(self) -> pd.DataFrame:
        """Extract PlantVillage image metadata from mapping file"""
        print("\n[1/6] Loading PlantVillage metadata...")
        
        pv_root = self.data_root / "PlantVillage-Dataset-master"
        mapping_file = pv_root / "data_distribution_for_SVM" / "train_mapping.txt"
        
        records = []
        if not mapping_file.exists():
            print(f"  ⚠ PlantVillage mapping file not found: {mapping_file}")
            return pd.DataFrame(records)
        
        with open(mapping_file, 'r') as f:
            for idx, line in enumerate(f):
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    src_path, dst_path = parts
                    # Extract disease label from source path
                    # Format: raw/color/Apple___Apple_scab/imagename.JPG
                    path_parts = src_path.split('/')
                    if len(path_parts) >= 3:
                        disease_name = path_parts[2]
                        class_id = dst_path.split('/')[-2]
                        
                        records.append({
                            'image_id': str(uuid.uuid4()),
                            'source': 'plantvillage',
                            'original_path': src_path,
                            'new_path': f"train/{class_id}/{Path(src_path).name}",
                            'disease': disease_name,
                            'class_id': class_id,
                            'crop_type': disease_name.split('___')[0].lower(),
                        })
        
        df = pd.DataFrame(records)
        print(f"  ✓ Loaded {len(df)} PlantVillage images")
        return df
    
    def load_fruit_images_metadata(self) -> pd.DataFrame:
        """Extract metadata from your fruit disease images"""
        print("\n[2/6] Loading fruit disease images metadata...")
        
        fruits_root = self.data_root / "fruits"
        records = []
        
        if not fruits_root.exists():
            print(f"  ⚠ Fruit images directory not found: {fruits_root}")
            return pd.DataFrame(records)
        
        fruit_types = ["APPLE", "GUAVA", "MANGO", "POMEGRANATE"]
        
        for fruit in fruit_types:
            fruit_path = fruits_root / fruit
            if not fruit_path.exists():
                continue
            
            for disease_dir in fruit_path.iterdir():
                if not disease_dir.is_dir():
                    continue
                
                disease_name = disease_dir.name.replace('_', ' ').lower()
                
                for img_file in disease_dir.glob("*.*"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        records.append({
                            'image_id': str(uuid.uuid4()),
                            'source': 'fruit_dataset',
                            'original_path': str(img_file),
                            'new_path': f"train/{fruit.lower()}/{disease_name}/{img_file.name}",
                            'disease': f"{fruit}___{disease_name.replace(' ', '_')}",
                            'crop_type': fruit.lower(),
                            'disease_class': disease_name,
                        })
        
        df = pd.DataFrame(records)
        print(f"  ✓ Loaded {len(df)} fruit disease images")
        return df
    
    def load_soil_parameters(self) -> pd.DataFrame:
        """Load soil parameters for crops"""
        print("\n[3/6] Loading soil parameters...")
        
        soil_file = self.data_root / "state_soil_data.csv"
        
        if not soil_file.exists():
            print(f"  ⚠ Soil data file not found: {soil_file}")
            # Return default values
            return pd.DataFrame({
                'state': ['maharashtra', 'karnataka', 'uttar_pradesh'],
                'N': [90.0, 85.0, 95.0],
                'P': [42.0, 40.0, 45.0],
                'K': [180.0, 170.0, 190.0],
                'pH': [6.8, 6.5, 7.0]
            })
        
        soil_df = pd.read_csv(soil_file)
        
        # Normalize state names
        soil_df['state'] = soil_df['state'].str.lower().str.replace(' ', '_')
        
        print(f"  ✓ Loaded soil data for {len(soil_df)} states")
        return soil_df
    
    def load_weather_data(self) -> pd.DataFrame:
        """Load real weather data from OpenWeatherMap or cached CSV"""
        print("\n[4/6] Loading weather data...")
        
        weather_file = self.data_root / "unified_dataset" / "temporal_data" / "openweathermap_cache.csv"
        
        # Try to load from cache first
        if weather_file.exists():
            print(f"  ✓ Loaded cached weather data from {weather_file}")
            weather_df = pd.read_csv(weather_file)
            weather_df['state'] = weather_df['state'].str.lower().str.replace(' ', '_')
            print(f"  ✓ Weather records: {len(weather_df)} rows across {weather_df['state'].nunique()} states")
            return weather_df
        
        # If no cache, try to fetch from OpenWeatherMap API
        api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
        if api_key:
            try:
                print(f"  → Attempting to fetch real weather data from OpenWeatherMap API...")
                weather_df = get_or_create_weather_cache(api_key, force_refresh=False)
                weather_df['state'] = weather_df['state'].str.lower().str.replace(' ', '_')
                print(f"  ✓ Fetched {len(weather_df)} weather records from API")
                return weather_df
            except Exception as e:
                print(f"  ⚠ Weather API fetch failed: {e}")
                print(f"  → Falling back to synthetic weather data")
        
        # Fallback: synthetic data
        print(f"  ⚠ No weather cache and OPENWEATHERMAP_API_KEY not set")
        print(f"  → Using synthetic weather data (set OPENWEATHERMAP_API_KEY env var to use real data)")
        
        return pd.DataFrame({
            'state': ['maharashtra'] * 5,
            'year': [2010, 2011, 2012, 2013, 2014],
            'avg_temp_c': [24.5, 25.0, 24.8, 25.2, 24.9],
            'total_rainfall_mm': [750.0, 800.0, 780.0, 820.0, 760.0],
            'avg_humidity_percent': [72.0, 73.0, 71.0, 74.0, 72.0]
        })
    
    def assign_soil_parameters(self, image_df: pd.DataFrame, soil_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign synthetic soil parameters to images based on crop type.
        This creates the linkage between images and soil data.
        """
        print("\n[5/6] Assigning soil parameters to images...")
        
        # For each image, assign soil data from a typical state for that crop
        def get_soil_for_crop(crop_type: str):
            states = self.crop_state_mapping.get(crop_type, ['maharashtra'])
            state = states[0]  # Take first state as representative
            
            soil_row = soil_df[soil_df['state'] == state]
            if len(soil_row) == 0:
                # Fallback: use averages
                return {
                    'soil_nitrogen': soil_df['N'].mean() if 'N' in soil_df.columns else 90.0,
                    'soil_phosphorus': soil_df['P'].mean() if 'P' in soil_df.columns else 42.0,
                    'soil_potassium': soil_df['K'].mean() if 'K' in soil_df.columns else 180.0,
                    'soil_ph': soil_df['pH'].mean() if 'pH' in soil_df.columns else 6.8,
                    'soil_state': 'unknown',
                }
            
            row = soil_row.iloc[0]
            return {
                'soil_nitrogen': float(row['N']) if 'N' in row.index else 90.0,
                'soil_phosphorus': float(row['P']) if 'P' in row.index else 42.0,
                'soil_potassium': float(row['K']) if 'K' in row.index else 180.0,
                'soil_ph': float(row['pH']) if 'pH' in row.index else 6.8,
                'soil_state': state,
            }
        
        soil_cols = ['soil_nitrogen', 'soil_phosphorus', 'soil_potassium', 'soil_ph', 'soil_state']
        
        for col in soil_cols:
            image_df[col] = None
        
        for idx, row in image_df.iterrows():
            crop_type = row['crop_type']
            soil_data = get_soil_for_crop(crop_type)
            for key, value in soil_data.items():
                image_df.at[idx, key] = value
        
        print(f"  ✓ Assigned soil parameters to {len(image_df)} images")
        return image_df
    
    def generate_temporal_sequences(self, image_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate 30-day temporal weather sequences for each image.
        Uses real weather data from OpenWeatherMap API via WeatherIntegrator.
        Falls back to synthetic data if weather_df is empty.
        """
        print("\n[6/6] Generating temporal weather sequences...")
        
        # Check if we have real weather data
        has_real_data = len(weather_df) > 1 and 'date' in weather_df.columns
        
        if has_real_data:
            print(f"  ✓ Using REAL OpenWeatherMap data ({len(weather_df)} records)")
            integrator = WeatherIntegrator(api_key="dummy")  # API key not needed for generation
            
            def create_weather_sequence(crop_type: str):
                """Create 30-day sequence from real weather data"""
                return integrator.generate_sequences(weather_df, crop_type, sequence_length=30)
        else:
            print(f"  ⚠ Using SYNTHETIC weather data")
            
            def create_weather_sequence(crop_type: str, weather_df: pd.DataFrame = weather_df):
                """Create synthetic 30-day sequence (original behavior)"""
                # Get average weather for crop's primary states
                crop_state_map = {
                    "apple": ["himachal_pradesh", "uttarakhand"],
                    "guava": ["uttar_pradesh", "madhya_pradesh", "maharashtra"],
                    "mango": ["maharashtra", "andhra_pradesh", "karnataka"],
                    "pomegranate": ["maharashtra", "karnataka", "andhra_pradesh"],
                    "tomato": ["karnataka", "maharashtra", "uttar_pradesh"],
                    "potato": ["uttar_pradesh", "punjab", "bihar"],
                    "grape": ["maharashtra", "karnataka"],
                    "peach": ["himachal_pradesh", "punjab"],
                }
                
                states = crop_state_map.get(crop_type, ['maharashtra'])
                state_weather = weather_df[weather_df['state'].isin(states)]
                if len(state_weather) == 0:
                    state_weather = weather_df
                
                # Create synthetic 30-day sequence
                sequence = []
                for day in range(30):
                    sample = state_weather.sample(1).iloc[0]
                    
                    temp_col = 'avg_temp_c' if 'avg_temp_c' in sample.index else 'temperature'
                    rain_col = 'total_rainfall_mm' if 'total_rainfall_mm' in sample.index else 'rainfall'
                    hum_col = 'avg_humidity_percent' if 'avg_humidity_percent' in sample.index else 'humidity'
                    
                    temp_val = float(sample[temp_col]) if temp_col in sample.index else 24.5
                    rain_val = float(sample[rain_col]) if rain_col in sample.index else 50.0
                    hum_val = float(sample[hum_col]) if hum_col in sample.index else 72.0
                    
                    sequence.append({
                        'temp': temp_val + np.random.normal(0, 2),
                        'rainfall': max(0, rain_val + np.random.normal(0, 5)),
                        'humidity': max(0, min(100, hum_val + np.random.normal(0, 5))),
                    })
                
                return json.dumps(sequence)
        
        image_df['weather_sequence'] = image_df['crop_type'].apply(create_weather_sequence)
        
        print(f"  ✓ Generated temporal sequences for {len(image_df)} images")
        return image_df
    
    def combine_datasets(self) -> pd.DataFrame:
        """Combine all datasets into unified metadata"""
        print("\n" + "="*60)
        print("COMBINING DATASETS")
        print("="*60)
        
        # Load all data
        pv_df = self.load_plantvillage_metadata()
        fruit_df = self.load_fruit_images_metadata()
        soil_df = self.load_soil_parameters()
        weather_df = self.load_weather_data()
        
        # Combine image sources
        combined_df = pd.concat([pv_df, fruit_df], ignore_index=True)
        
        if len(combined_df) == 0:
            print("✗ No images found! Check data directories.")
            return combined_df
        
        # Assign soil and weather data
        combined_df = self.assign_soil_parameters(combined_df, soil_df)
        combined_df = self.generate_temporal_sequences(combined_df, weather_df)
        
        # Create split assignments
        np.random.seed(42)
        combined_df['split'] = np.random.choice(['train', 'val', 'test'], 
                                                 size=len(combined_df), 
                                                 p=[0.7, 0.15, 0.15])
        
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Total images: {len(combined_df)}")
        print(f"  - PlantVillage: {len(pv_df)}")
        print(f"  - Fruit dataset: {len(fruit_df)}")
        print(f"\nSplit distribution:")
        print(combined_df['split'].value_counts())
        print(f"\nCrop types: {combined_df['crop_type'].nunique()}")
        print(f"Disease classes: {combined_df['disease'].nunique()}")
        print(f"\nTop crops:")
        print(combined_df['crop_type'].value_counts().head(10))
        
        return combined_df
    
    def save_metadata(self, combined_df: pd.DataFrame):
        """Save metadata CSV for reference"""
        metadata_file = self.unified_dir / "metadata" / "combined_dataset_metadata.csv"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        combined_df.to_csv(metadata_file, index=False)
        print(f"\n✓ Metadata saved: {metadata_file}")
        
        # Also save crop-disease summary
        summary = combined_df.groupby(['crop_type', 'disease']).size().reset_index(name='count')
        summary_file = self.unified_dir / "metadata" / "crop_disease_distribution.csv"
        summary.to_csv(summary_file, index=False)
        print(f"✓ Summary saved: {summary_file}")
        
        return combined_df

def main():
    data_root = os.path.join(os.path.dirname(__file__), "..", "data")
    
    integrator = DatasetIntegrator(data_root)
    combined_df = integrator.combine_datasets()
    
    if len(combined_df) > 0:
        integrator.save_metadata(combined_df)
        
        print("\n" + "="*60)
        print("✓ DATASET INTEGRATION COMPLETE")
        print("="*60)
        print("\nNext steps:")
        print("1. Review metadata CSV files in unified_dataset/metadata/")
        print("2. Run organize_images.py to copy/organize actual image files")
        print("3. Proceed to model training with organized datasets")
    else:
        print("\n✗ Integration failed - no images found")

if __name__ == "__main__":
    main()
