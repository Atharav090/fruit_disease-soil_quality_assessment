# 📖 SoilVisioNet Usage Guide - Detailed Instructions

## Table of Contents
1. [Installation & Setup](#installation--setup)
2. [Web Application Guide](#web-application-guide)
3. [Python API Usage](#python-api-usage)
4. [Common Workflows](#common-workflows)
5. [Troubleshooting](#troubleshooting)
6. [FAQ](#faq)

---

## Installation & Setup

### Step 1: System Requirements Check

**Windows:**
```powershell
# Check Python version
python --version  # Should be 3.8+

# Check available GPU (optional)
nvidia-smi  # If installed, shows NVIDIA GPU info
```

**Linux/Mac:**
```bash
# Check Python version
python3 --version  # Should be 3.8+

# Check available GPU
nvidia-smi  # If CUDA available
```

### Step 2: Virtual Environment Setup

**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then retry activation command above
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip first (recommended)
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# OPTIONAL: Install PyTorch with GPU support
# Choose your CUDA version and OS:
# For CUDA 11.8 (most common):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only (if no NVIDIA GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Verify GPU Setup (Optional)

```python
# Run this Python script to verify CUDA:
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA memory (GB): {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}')
"
```

### Step 5: Configure Model Paths

**IMPORTANT**: Update model checkpoint paths before running the application.

Edit `core/inference_engine.py` (around line 50):

```python
# UPDATE THESE PATHS to match your installation:
MODEL_PATHS = {
    'vit': '../data/unified_dataset/models/vit/checkpoint.pt',
    'lstm': '../data/unified_dataset/models/lstm/checkpoint.pt',
    'elm': '../data/unified_dataset/models/elm/checkpoint.pt',
    'hybrid': '../data/unified_dataset/models/hybrid/checkpoint.pt'
}

# Example if models are in a 'models' folder at project root:
# MODEL_PATHS = {
#     'vit': './models/vit_final.pt',
#     'lstm': './models/lstm_final.pt',
#     ...
# }
```

### Step 6: Launch Application

```bash
# Navigate to project directory
cd soilvisionet_production

# Start Streamlit
streamlit run ui/app.py

# Advanced options:
# streamlit run ui/app.py --logger.level=debug  # Verbose logging
# streamlit run ui/app.py --client.toolbarMode=minimal  # Minimal UI
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501

  Press CTRL+C to quit.
```

Open http://localhost:8501 in your browser.

---

## Web Application Guide

### 🎯 First Time User Setup

1. **Browser & Display**
   - Recommended: Chrome, Edge, or Firefox
   - Screen size: Minimum 1024x768 (1920x1080 recommended)
   - No installation needed on client side

2. **Model Loading** (First run only)
   - Application loads all 4 models on startup
   - Status shown in left sidebar: ✓ ViT loaded, ✓ ELM loaded, etc.
   - Takes 30-120 seconds depending on GPU availability
   - Subsequent launches are faster (models cached)

3. **Sidebar Overview**
   - **Model Selection**: Choose inference model (ViT/ELM/Hybrid)
   - **Status Indicators**: Green checkmarks show loaded models
   - **About Section**: Version and feature overview

### Tab 1: Disease Detection 🔬

#### Upload Image

**Step 1: Prepare Your Image**
- Format: JPG, PNG, BMP, or TIFF
- Size: Up to 50 MB
- Resolution: Preferably 224×224px or larger (auto-resized)
- Quality: Clear, well-lit image of affected plant part

**Step 2: Upload**
1. Click "Upload crop image" button
2. Select file from computer
3. Image previews immediately
4. Image statistics displayed (dimensions, file size, format)

**Step 3: Select Model**
- **ViT (Recommended)**: Best accuracy (98.12%), moderate speed
- **ELM**: Fast inference, high accuracy (98.29%)
- **Hybrid**: Ensemble of multiple models (98.12%)

**Step 4: View Results**
The application displays:

**PRIMARY RESULT:**
```
Disease: Tomato___Early_blight
Confidence: 94.2%
Severity: MODERATE
```

**TOP-5 PREDICTIONS** (Table):
| Rank | Disease | Confidence |
|------|---------|------------|
| 1 | Tomato___Early_blight | 94.2% |
| 2 | Tomato___Late_blight | 3.8% |
| 3 | Tomato___Septoria_leaf_spot | 1.5% |
| 4 | Tomato___Healthy | 0.4% |
| 5 | Pepper___Bacterial_spot | 0.1% |

**EXPANDABLE SECTIONS:**

**📋 Detailed Explanation**
- Natural language summary of detection result
- Why this disease was identified
- Confidence analysis (high/moderate/low confidence reasoning)

**🔍 Disease Information**
- Symptoms and visual characteristics
- What to look for in affected plants
- How disease spreads

**💊 Treatment Recommendations**
- Immediate actions (remove affected leaves, improve ventilation)
- Chemical treatments (approved fungicides)
- Management strategies (monitoring, isolation)

**📊 Confidence Analysis**
- Why model is confident in primary prediction
- Alternative predictions and their likelihood
- Confidence threshold explanation

**✅ Next Steps**
- Priority ranked action items
- Implementation timeline
- Success indicators to monitor

**Step 5: Export Results**

Click "Download Results" to get:
- JSON file: Computer-readable format for integration
- Text Report: Human-readable format for documentation

#### Multiple Images Workflow

Upload different images sequentially:
1. Click "Upload crop image" again
2. Select new file
3. Results update automatically
4. Previous results remain in page (can scroll up to compare)

**Tips for Best Results:**
- Capture images in natural lighting
- Focus on affected leaf/plant area
- Avoid shadows and reflections
- Include healthy tissue for comparison
- Multiple angles improve confidence

---

### Tab 2: Soil & Weather Assessment 🌱

#### Setup Assessment

**Step 1: Select Crop Type**
- Dropdown menu with 17 crops:
  - Apple, Blueberry, Cherry, Corn, Grape, Guava, Mango, Orange
  - Peach, Pepper (bell), Pomegranate, Potato, Raspberry, Soybean
  - Squash, Strawberry, Tomato

**Step 2: Enter Soil Data**

Use soil testing results from your lab:

| Parameter | Range | Unit | Example |
|-----------|-------|------|---------|
| Nitrogen (N) | 0-300 | mg/kg | 85 |
| Phosphorus (P) | 0-200 | mg/kg | 32 |
| Potassium (K) | 0-300 | mg/kg | 38 |
| Soil pH | 3-9 | pH units | 6.8 |

**How to get soil data:**
- Send soil sample to agricultural lab
- Use DIY soil test kit
- Use "Load Example Data" for demonstration

**Step 3: Enter Weather Data (Optional)**

Current weather averages (last 30 days):

| Parameter | Range | Unit | Example |
|-----------|-------|------|---------|
| Temperature | -40 to +60 | °C | 22 |
| Rainfall | 0-1000 | mm | 120 |
| Humidity | 0-100 | % | 65 |

**Step 4: View Assessment Results**

**OVERALL SUITABILITY:**
```
Score: 87/100
Rating: EXCELLENT
Recommendation: Crop is well-suited to these conditions
```

**COMPONENT SCORES:**
- Soil Suitability: 92/100 (Excellent)
- Weather Suitability: 85/100 (Good)
- Disease Risk: 78/100 (Moderate risk)
- Crop-Soil Match: 96/100 (Excellent)

**EXPANDABLE SECTIONS:**

**🌍 Soil Parameters**
- Nitrogen status: Optimal/Low/High
  - Current: 85 mg/kg, Optimal: 80-100
  - Recommendation: No adjustment needed
- Phosphorus status: ...
- Potassium status: ...
- pH status: ...

**⛅ Weather & Disease Risk**
- Temperature risk: LOW - 22°C is optimal for tomato
- Rainfall risk: MODERATE - 120mm is adequate
- Humidity risk: HIGH - 65% favors fungal diseases
- Associated diseases: Early blight, Late blight
- Risk level: MODERATE overall

**✅ Management Recommendations**
- Priority 1: Improve ventilation (high humidity)
- Priority 2: Monitor for fungal symptoms daily
- Priority 3: Apply preventive fungicide if humidity > 80%
- Priority 4: Ensure adequate spacing between plants

**Step 5: Export Assessment**

Click "Download Assessment" to get:
- JSON file: Detailed assessment data (programmatic use)
- Management Plan: PDF-formatted action items (printing)

#### Using Example Data

For demonstration without soil testing:
1. Check "Load Example Data" checkbox
2. System fills optimal values for selected crop
3. Modify values to test different scenarios
4. Useful for "what-if" analysis

**Example Scenarios:**
- "What if soil pH drops to 5.5?" (enter 5.5, compare results)
- "What if rainfall increases to 300mm?" (enter 300, see disease risk change)

---

## Python API Usage

### Import Modules

```python
# In your Python script:
from soilvisionet_production.core import InferenceEngine, ImageProcessor
from soilvisionet_production.modules import (
    DiseaseDetector,
    SuitabilityEngine,
    ExplanationGenerator
)
```

### Disease Detection

#### Detection from Image File

```python
from soilvisionet_production.modules import DiseaseDetector

# Initialize detector
detector = DiseaseDetector(model='vit')

# Detect disease from image path
result = detector.detect_from_path('path/to/tomato_leaf.jpg')

# Access results
print(f"Disease: {result['disease_name']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Severity: {result['severity']}")
print(f"Treatment: {result['treatment_recommendation']}")

# All results structure:
# {
#     'disease_id': int,
#     'disease_name': str,
#     'crop': str,
#     'confidence': float,
#     'severity': str,
#     'symptoms': {description, visual_indicators, prevention},
#     'treatment_recommendation': {...},
#     'soil_requirements': {nitrogen, phosphorus, potassium, ph},
#     'weather_requirements': {temperature, rainfall, humidity},
#     'top_5_predictions': [{disease_name, confidence, color}, ...],
#     'explanation': str,
#     'next_steps': [str, ...]
# }
```

#### Detection from Image Array

```python
import cv2
from soilvisionet_production.modules import DiseaseDetector

# Load image with OpenCV
image_array = cv2.imread('tomato_leaf.jpg')

# Detect from array
detector = DiseaseDetector(model='vit')
result = detector.detect_from_array(image_array)

# Use results...
```

#### Batch Detection

```python
from soilvisionet_production.modules import DiseaseDetector
import os

detector = DiseaseDetector(model='vit')

# Get all images from folder
image_folder = 'path/to/images/'
results = []

for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.png')):
        filepath = os.path.join(image_folder, filename)
        result = detector.detect_from_path(filepath)
        results.append({
            'filename': filename,
            'disease': result['disease_name'],
            'confidence': result['confidence']
        })

# Print summary
for r in results:
    print(f"{r['filename']}: {r['disease']} ({r['confidence']:.1%})")
```

### Soil & Weather Assessment

#### Basic Assessment

```python
from soilvisionet_production.modules import SuitabilityEngine

# Initialize engine
engine = SuitabilityEngine()

# Define soil parameters
soil_data = {
    'nitrogen': 85,
    'phosphorus': 32,
    'potassium': 38,
    'ph': 6.8
}

# Define weather (30-day average)
weather_data = {
    'temperature': 22,  # Can be average or list of 30 values
    'rainfall': 120,
    'humidity': 65
}

# Assess suitability for tomato
assessment = engine.assess_crop_suitability_comprehensive(
    crop='tomato',
    soil_data=soil_data,
    weather_data=weather_data
)

# Access results
print(f"Overall Score: {assessment['overall_score']}/100")
print(f"Rating: {assessment['rating']}")
print(f"Recommendation: {assessment['recommendation']}")

# Detailed scores
print(f"Soil Suitability: {assessment['soil_score']}/100")
print(f"Weather Suitability: {assessment['weather_score']}/100")
print(f"Disease Risk: {assessment['disease_risk_level']}")
```

#### Detailed Soil Assessment Only

```python
from soilvisionet_production.modules import SuitabilityEngine

engine = SuitabilityEngine()

# Just assess soil (no weather)
soil_assessment = engine.assess_soil_suitability(
    crop='potato',
    soil_data={
        'nitrogen': 120,
        'phosphorus': 45,
        'potassium': 50,
        'ph': 6.5
    }
)

# Results
for param_name, param_assessment in soil_assessment['parameters'].items():
    print(f"{param_name}:")
    print(f"  Current: {param_assessment['current']}")
    print(f"  Status: {param_assessment['status']}")
    print(f"  Recommendation: {param_assessment['recommendation']}")
```

#### Weather Risk Assessment

```python
from soilvisionet_production.modules import SuitabilityEngine

engine = SuitabilityEngine()

# Assess weather-related disease risk
weather_risk = engine.assess_weather_risk(
    crop='grape',
    weather_sequence={
        'temperature': [20, 21, 22, 23, 24, 25, 24, 23, 22, 21],
        'rainfall': [0, 2, 5, 3, 1, 0, 2, 4, 1, 0],
        'humidity': [60, 65, 72, 75, 70, 65, 68, 80, 75, 62]
    }
)

# Results
print(f"Disease Risk Level: {weather_risk['risk_level']}")
print(f"Risk Score: {weather_risk['risk_score']}")
print(f"Associated Diseases: {weather_risk['associated_diseases']}")
print(f"Protective Measures: {weather_risk['protective_measures']}")
```

### Explanation Generation

#### Explain Detection Results

```python
from soilvisionet_production.modules import (
    DiseaseDetector,
    ExplanationGenerator
)

# Get detection result
detector = DiseaseDetector()
result = detector.detect_from_path('tomato_leaf.jpg')

# Generate explanation
generator = ExplanationGenerator()
explanation = generator.explain_detection(result)

print("DETECTION EXPLANATION:")
print(explanation)

# Output example:
# "Your image shows signs of Early Blight on a Tomato plant with 94.2% confidence.
#  This fungal disease typically appears as concentric dark spots on leaves...
#  The model is confident in this prediction because...
#  Next steps you should take:
#  1. Remove infected leaves immediately
#  2. Improve air circulation..."
```

#### Explain Soil Assessment

```python
from soilvisionet_production.modules import (
    SuitabilityEngine,
    ExplanationGenerator
)

# Get assessment
engine = SuitabilityEngine()
assessment = engine.assess_crop_suitability_comprehensive(
    crop='tomato',
    soil_data={...},
    weather_data={...}
)

# Generate explanation
generator = ExplanationGenerator()
explanation = generator.explain_soil_assessment(assessment)

print("SOIL ASSESSMENT EXPLANATION:")
print(explanation)
```

#### Explain Weather Risk

```python
from soilvisionet_production.modules import (
    SuitabilityEngine,
    ExplanationGenerator
)

# Get risk assessment
engine = SuitabilityEngine()
risk = engine.assess_weather_risk(
    crop='potato',
    weather_sequence={...}
)

# Generate explanation
generator = ExplanationGenerator()
explanation = generator.explain_weather_risk(risk)

print("WEATHER RISK EXPLANATION:")
print(explanation)
```

---

## Common Workflows

### Workflow 1: Field Scouting & Diagnosis

**Scenario**: Farmer notices diseased plants while scouting fields

```python
from soilvisionet_production.modules import DiseaseDetector
import cv2
from datetime import datetime

# 1. Capture image of diseased plant
image = cv2.imread('field_scout_2024-02-15_plot-5.jpg')

# 2. Run detection
detector = DiseaseDetector(model='vit')
result = detector.detect_from_array(image)

# 3. Log result with timestamp
log_entry = {
    'date': datetime.now().isoformat(),
    'location': 'Plot-5, Row-3',
    'disease': result['disease_name'],
    'confidence': f"{result['confidence']:.1%}",
    'severity': result['severity'],
    'recommended_action': result['treatment_recommendation']['immediate']
}

# 4. Save for field records (optional)
import json
with open('field_scouting_log.json', 'a') as f:
    json.dump(log_entry, f)

# 5. Print action for farmer
print(f"⚠️  {result['disease_name']} detected")
print(f"Action: {result['treatment_recommendation']['immediate']}")
```

### Workflow 2: Planting Suitability Planning

**Scenario**: Planning what crop to plant in upcoming season

```python
from soilvisionet_production.modules import SuitabilityEngine

# Get soil test results from lab
soil_test = {
    'nitrogen': 95,
    'phosphorus': 28,
    'potassium': 42,
    'ph': 6.9
}

# Get weather forecast
weather_forecast = {
    'temperature': 22,  # Average expected
    'rainfall': 150,
    'humidity': 65
}

engine = SuitabilityEngine()

# Test each crop to find best fit
crops_to_test = ['tomato', 'potato', 'corn', 'grape']
results = {}

for crop in crops_to_test:
    assessment = engine.assess_crop_suitability_comprehensive(
        crop=crop,
        soil_data=soil_test,
        weather_data=weather_forecast
    )
    results[crop] = assessment['overall_score']

# Rank by suitability
ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)

print("Crop Suitability Ranking:")
for rank, (crop, score) in enumerate(ranked, 1):
    print(f"{rank}. {crop.capitalize()}: {score}/100")

# Recommend top choice
best_crop, best_score = ranked[0]
print(f"\n✓ Recommendation: Plant {best_crop.capitalize()}")
```

### Workflow 3: Monitoring & Early Warning

**Scenario**: Regular monitoring throughout growing season

```python
from soilvisionet_production.modules import (
    DiseaseDetector,
    SuitabilityEngine,
    ExplanationGenerator
)
import csv
from datetime import datetime

detector = DiseaseDetector()
engine = SuitabilityEngine()
explainer = ExplanationGenerator()

# Daily monitoring routine
monitoring_log = []

# Check plants
image_file = 'monitoring_2024-02-15_plot-A.jpg'
detection = detector.detect_from_path(image_file)

# Get current weather/soil
current_soil = {
    'nitrogen': 90,
    'phosphorus': 35,
    'potassium': 40,
    'ph': 6.8
}

current_weather = {
    'temperature': 24,
    'rainfall': 5,
    'humidity': 78
}

# Assess risk
risk_factor = engine.assess_crop_suitability_comprehensive(
    crop='tomato',
    soil_data=current_soil,
    weather_data=current_weather
)

# Log entry
log_entry = {
    'date': datetime.now().isoformat(),
    'detected_disease': detection['disease_name'],
    'confidence': f"{detection['confidence']:.1%}",
    'risk_level': risk_factor['disease_risk_level'],
    'action': explainer.explain_detection(detection)[:100] + '...'
}

monitoring_log.append(log_entry)

# Save to CSV for tracking
with open('monitoring_log.csv', 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=log_entry.keys())
    if f.tell() == 0:  # Write header if file is new
        writer.writeheader()
    writer.writerow(log_entry)

print(f"✓ Monitoring logged: {log_entry['detected_disease']}")
```

---

## Troubleshooting

### Issue 1: Models Won't Load

**Error**: `FileNotFoundError: checkpoint.pt not found`

**Solutions**:
1. Verify model checkpoint paths:
   ```bash
   # Check if files exist
   ls ../data/unified_dataset/models/vit/
   ls ../data/unified_dataset/models/elm/
   # Windows: dir ..\data\unified_dataset\models\vit\
   ```

2. Update paths in `core/inference_engine.py`:
   ```python
   # If models are in local 'models' folder:
   MODEL_PATHS = {
       'vit': './models/vit_checkpoint.pt',
       # ...
   }
   ```

3. Verify checkpoint file exists and is readable:
   ```bash
   file ../data/unified_dataset/models/vit/checkpoint.pt
   # Should show: data (binary file)
   ```

### Issue 2: Out of Memory (CUDA)

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Close other GPU applications
2. Force CPU mode in `core/inference_engine.py`:
   ```python
   self.device = torch.device('cpu')  # Force CPU
   ```

3. Reduce batch size (for batch operations):
   ```python
   batch_size = 1  # Instead of 4 or 8
   ```

4. Clear GPU cache between runs:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Issue 3: Slow Inference

**Symptoms**: Detection takes 30+ seconds

**Solutions**:
1. Verify GPU is being used:
   ```bash
   nvidia-smi  # Should show Python using GPU
   ```

2. Check if running on CPU accidentally:
   ```python
   print(detector.device)  # Should print 'cuda:0', not 'cpu'
   ```

3. Reinstall PyTorch with CUDA:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. Check for background processes consuming GPU:
   ```bash
   nvidia-smi  # Check running processes
   ```

### Issue 4: Invalid Image Format

**Error**: `PIL.UnidentifiedImageError: cannot identify image file`

**Solutions**:
1. Verify image format is supported:
   - Supported: JPG, PNG, BMP, TIFF
   - NOT supported: WebP, SVG, animated GIF

2. Convert image to PNG:
   ```bash
   # Using ImageMagick
   convert image.webp image.png
   ```

3. Test image validity:
   ```python
   from PIL import Image
   img = Image.open('your_image.jpg')
   img.verify()  # Raises exception if invalid
   ```

### Issue 5: Streamlit Won't Start

**Error**: `Address already in use` or port 8501 unavailable

**Solutions**:
1. Kill process using port 8501:
   ```bash
   # Linux/Mac
   lsof -i :8501
   kill -9 <PID>
   
   # Windows (PowerShell)
   Get-Process -Id (Get-NetTCPConnection -LocalPort 8501).OwningProcess | Stop-Process -Force
   ```

2. Run on different port:
   ```bash
   streamlit run ui/app.py --server.port 8502
   ```

3. Clear Streamlit cache:
   ```bash
   streamlit cache clear
   ```

### Issue 6: Module Import Errors

**Error**: `ModuleNotFoundError: No module named 'soilvisionet_production'`

**Solutions**:
1. Verify working directory:
   ```bash
   cd soilvisionet_production  # If using Streamlit
   # OR stay in parent directory for Python scripts
   ```

2. Add to Python path:
   ```python
   import sys
   sys.path.insert(0, '/path/to/soilvisionet_production')
   from core import InferenceEngine
   ```

3. Install as editable package:
   ```bash
   pip install -e .  # Requires setup.py in root
   ```

---

## FAQ

**Q: Do I need a GPU?**  
A: No, but recommended. GPU provides 5-10x faster inference. CPU mode works but takes 5-10 seconds per image.

**Q: Can I use the system offline?**  
A: Yes, models are loaded locally. No cloud connectivity required after startup.

**Q: How do I add new disease classes?**  
A: Currently would require retraining the ViT model. Contact project maintainers for custom models.

**Q: What image resolution is optimal?**  
A: 224×224px (model input size). Larger images are automatically resized.

**Q: Can I export results programmatically?**  
A: Yes, using Python API. Results are Python dicts, easily serialized to JSON.

**Q: How is data privacy handled?**  
A: All inference is local. No images are uploaded. Optional logging requires explicit consent.

**Q: Can models be updated?**  
A: Yes, replace checkpoint files and regenerate database: `python extract_databases.py`

**Q: What's the inference cost?**  
A: Free, runs locally. No API calls or subscriptions.

**Q: How many images can I process?**  
A: Unlimited. Limited only by disk storage and compute time.

**Q: Can I integrate this into my own application?**  
A: Yes, Python API fully supports integration. CORS-accessible if deployed as API.

---

**Need More Help?**  
- Check inference logs: `streamlit run ui/app.py --logger.level=debug`
- Review error messages in terminal (not just UI)
- Consult README.md for environment setup issues

---

**Last Updated**: February 2024  
**Version**: 1.0.0
