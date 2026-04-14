"""
Setup script for SoilVisioNet dataset integration
Creates directory structure and prepares data for hybrid model training
"""

import os
import shutil
from pathlib import Path

# Define project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Directories to create
DIRECTORIES = {
    "unified_dataset": {
        "images": ["train", "val", "test"],
        "metadata": [],
        "soil_data": [],
        "temporal_data": [],
    },
    "processed_data": {
        "combined_train": [],
        "combined_test": [],
        "augmented": [],
    },
    "models": {
        "vit": [],
        "lstm": [],
        "elm": [],
        "hybrid": [],
    },
    "outputs": {
        "logs": [],
        "predictions": [],
        "visualizations": [],
    },
}

def create_structure():
    """Create directory structure"""
    print("Creating project structure...")
    
    for category, subdirs in DIRECTORIES.items():
        cat_path = DATA_DIR / category
        cat_path.mkdir(parents=True, exist_ok=True)
        
        for subdir, nested in subdirs.items():
            sub_path = cat_path / subdir
            sub_path.mkdir(parents=True, exist_ok=True)
            
            for nested_dir in nested:
                nested_path = sub_path / nested_dir
                nested_path.mkdir(parents=True, exist_ok=True)
    
    print("✓ Directory structure created")

def check_data_sources():
    """Verify all required data sources exist"""
    print("\nChecking data sources...")
    
    sources = {
        "PlantVillage": DATA_DIR / "PlantVillage-Dataset-master" / "data_distribution_for_SVM",
        "Fruit Images": DATA_DIR / "fruits",
        "Soil Images": DATA_DIR / "Dataset",
        "Crop Recommendation": DATA_DIR / "Crop_recommendation.csv",
        "Crop Yield": DATA_DIR / "crop_yield.csv",
        "Soil Data": DATA_DIR / "data_core.csv",
        "State Soil": DATA_DIR / "state_soil_data.csv",
        "Weather Data": DATA_DIR / "state_weather_data_1997_2020.csv",
    }
    
    missing = []
    for name, path in sources.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")
        if not exists:
            missing.append(name)
    
    if missing:
        print(f"\n⚠ Missing: {', '.join(missing)}")
    else:
        print("\n✓ All data sources available")
    
    return len(missing) == 0

if __name__ == "__main__":
    print("="*60)
    print("SoilVisioNet Dataset Integration Setup")
    print("="*60)
    
    create_structure()
    check_data_sources()
    
    print("\n✓ Setup complete! Ready for data integration.")
