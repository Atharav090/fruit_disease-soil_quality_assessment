"""
QUICK START: SoilVisioNet Dataset Integration
Run this to execute all integration steps
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print status"""
    print(f"\n{'='*70}")
    print(f"▶ {description}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False)
        if result.returncode == 0:
            print(f"✓ {description} - SUCCESS")
            return True
        else:
            print(f"✗ {description} - FAILED")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║                  SOILVISIONET DATASET INTEGRATION QUICK START                 ║
╚════════════════════════════════════════════════════════════════════════════════╝

This script will:
  1. Setup directory structure
  2. Integrate PlantVillage + Fruit images + Soil + Weather data
  3. Organize images into unified directory
  4. Validate the complete dataset

Expected duration: 2-3 hours (mainly image copying)
Disk space needed: 15-20 GB

Press Enter to continue or Ctrl+C to cancel...
""")
    input()
    
    # Change to project directory
    project_root = os.path.dirname(__file__)
    os.chdir(project_root)
    
    steps = [
        ("python scripts/setup_environment.py",
         "Step 1/4: Setting up directory structure"),
        
        ("python scripts/integrate_dataset.py",
         "Step 2/4: Creating unified metadata"),
        
        ("python scripts/organize_images.py",
         "Step 3/4: Organizing image files"),
        
        ("python scripts/validate_integrated_dataset.py",
         "Step 4/4: Validating integrated dataset"),
    ]
    
    results = []
    for cmd, desc in steps:
        success = run_command(cmd, desc)
        results.append((desc, success))
        
        if not success:
            print(f"\n⚠ Stopping at {desc}")
            break
    
    # Summary
    print(f"\n{'='*70}")
    print("EXECUTION SUMMARY")
    print(f"{'='*70}")
    
    for desc, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {desc}")
    
    if all(s for _, s in results):
        print(f"\n{'='*70}")
        print("✓ DATASET INTEGRATION COMPLETE!")
        print(f"{'='*70}")
        print("""
Your integrated dataset is ready for model training!

📊 Dataset Statistics:
  - Total images: 15,288+
  - Train/Val/Test: 70% / 15% / 15%
  - Disease classes: 38+
  - Crops: 14+
  - Linked soil parameters: ✓
  - Linked weather sequences: ✓

📁 Key files:
  - unified_dataset/metadata/combined_dataset_metadata.csv (main metadata)
  - unified_dataset/images/train/ (training images)
  - unified_dataset/images/val/ (validation images)
  - unified_dataset/images/test/ (test images)

🚀 Next steps:
  1. Review metadata: data/unified_dataset/metadata/
  2. Load data using provided examples
  3. Train your ViT/LSTM/ELM models
  4. Build hybrid SoilVisioNet
  5. Deploy to mobile app

📖 For detailed information, see README.md

Good luck with your SoilVisioNet project! 🌾🍎
""")
    else:
        print(f"\n{'='*70}")
        print("✗ EXECUTION INCOMPLETE")
        print(f"{'='*70}")
        print("\nPlease check the error messages above and run again.")
        print("Ensure all data files exist in the data/ directory")

if __name__ == "__main__":
    main()
