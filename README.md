# fruit_disease-soil_quality_assessment

SoilVisioNet is a PhD-level research project that addresses the critical challenge of early and accurate detection of fruit crop diseases by integrating multi-modal data streams—fruit/leaf imagery, soil quality parameters (Nitrogen, Phosphorus, Potassium, pH), and temporal weather conditions (temperature, rainfall, humidity)—into a unified deep learning framework. Traditional plant disease detection systems rely solely on visual analysis of leaf or fruit images, ignoring the well-established agronomic relationship between soil health, environmental conditions, and disease susceptibility. This project bridges that gap by proposing a novel hybrid architecture, named SoilVisioNet, that fuses a Vision Transformer (ViT) for advanced image-based feature extraction, a Bidirectional Long Short-Term Memory (Bi-LSTM) network for analyzing 30-day temporal weather sequences, and an Extreme Learning Machine (ELM) for rapid soil-vision feature fusion—all unified through a late-fusion head to deliver a holistic disease diagnosis.
The system is built around a unified dataset comprising over 15,288 samples spanning 55 disease classes across 14+ crop types, assembled from PlantVillage image data, state-level soil survey data (state_soil_data), and real-time weather temporal data fetched from the OpenWeatherMap API. A production-grade Streamlit dashboard provides an interactive interface for uploading fruit images (currently focused on Pomegranate), receiving AI-powered disease identification, simulated soil sensor readings, fruit nutrient and vitamin impact assessments, and AI Expert Pathologist Reports generated via Llama 3.1 LLM. The project is designed with a future hardware deployment in mind: the current soil NPK readings are simulated deterministically from disease-soil profiles, but the architecture is hardware-ready so that once the IoT sensor hardware module is completed, actual soil sensor data will replace simulated readings seamlessly.

Key Terms Explained:
•	Vision Transformer (ViT): A deep learning architecture that treats an image as a sequence of fixed-size patches and processes them through transformer encoder layers. It captures global spatial relationships through self-attention, unlike CNNs which rely on local convolution filters.
•	Bi-LSTM (Bidirectional Long Short-Term Memory): A recurrent neural network variant that processes sequential data in both forward and backward directions, capturing long-range temporal dependencies in weather patterns.
•	Extreme Learning Machine (ELM): A single-hidden-layer feedforward network where input-to-hidden weights are randomly initialized and never trained; only the output layer is solved analytically via least-squares regression, enabling extremely fast training.
•	Late Fusion: A model integration strategy where individual sub-models produce their own predictions independently, and a small trainable fusion head learns to combine their outputs into a final prediction.
•	NPK: Nitrogen (N), Phosphorus (P), and Potassium (K)—the three primary macronutrients critical for plant growth and disease resistance.

3. Novelty:
1.	Multi-Modal Hybrid Architecture (SoilVisioNet): The primary novelty is the integration of three fundamentally different deep learning paradigms—Vision Transformer (spatial image analysis), Bidirectional LSTM (temporal weather pattern analysis), and Extreme Learning Machine (rapid soil-vision feature fusion)—into a single unified framework through late fusion. No prior work has combined ViT + Bi-LSTM + ELM in a single pipeline for agricultural disease detection.
2.	Soil-Disease Correlation Modeling: Unlike conventional plant disease detection systems that rely solely on image features, SoilVisioNet explicitly models the relationship between soil nutrient profiles (N, P, K, pH) and disease susceptibility, incorporating soil parameters as first-class input features alongside image embeddings.
3.	Temporal Weather Integration for Disease Risk Prediction: The system uses 30-day rolling weather sequences (temperature, rainfall, humidity) to predict disease risk, capturing the temporal dynamics of pathogen-favorable conditions that image-only models cannot detect.
4.	ELM-Based Rapid Fusion: The use of ELM for feature fusion is novel in this domain. Random hidden-layer weights with analytical least-squares output training enables sub-minute fusion training, making the system practical for resource-constrained environments.
5.	Hardware-Ready Simulated Sensor Architecture: The soil cause analyzer uses deterministic disease-soil profile mappings that simulate realistic sensor readings. This architecture is designed for seamless transition to real hardware sensors, making the system immediately deployable once IoT hardware is ready.
6.	AI-Augmented Pathological Reporting: Integration of Llama 3.1 LLM for generating structured, expert-level pathological reports with pesticide recommendations and cultural management advice—bringing generative AI into the agricultural diagnostics pipeline.
7.	Fruit Nutrient & Vitamin Impact Analysis: Disease-specific nutrient and vitamin impact assessments (e.g., Vitamin C degradation under anthracnose, potassium dependence for bacterial blight resistance) provide actionable agronomic insights beyond simple disease labeling.
________________________________________
4. Datasets & Usage:
Dataset	Source	Link / Location	Description	Usage
PlantVillage Dataset	PlantVillage (GitHub)	https://github.com/spMohanty/PlantVillage-Dataset
~54,000 images of healthy and diseased crop leaves across 38 classes, 14 crop species	Primary image dataset for ViT training (disease classification)
Fruit Disease Images	Custom Collection	Local directory: data/fruits/ (APPLE, GUAVA, MANGO, POMEGRANATE subdirectories)	Disease images for Apple (Scab, Blotch, Rot, Cedar Rust, Healthy), Guava (Anthracnose, Fruitfly, Healthy), Mango (Anthracnose, Alternaria, Black Mould Rot, Stem Rot, Healthy), Pomegranate (Anthracnose, Alternaria, Bacterial Blight, Cercospora, Healthy)	Supplementary training images for fruit-specific diseases
State Soil Data	Indian soil survey data	Local file: data/state_soil_data.csv	Soil nutrient parameters (N, P, K, pH) for major Indian agricultural states (Maharashtra, Karnataka, Uttar Pradesh, Himachal Pradesh, Punjab, Bihar, etc.)	Linked to images by crop-to-state mapping for soil-aware training
OpenWeatherMap Temporal Data	OpenWeatherMap API (Free Tier)	https://openweathermap.org/api
30-day weather data (temperature °C, rainfall mm, humidity %) for 10 major Indian agricultural cities	Used for LSTM temporal sequence training and weather-based disease risk prediction
Unified Dataset (Combined)	Internally generated	data/unified_dataset/metadata/combined_dataset_metadata.csv	15,288+ integrated samples with image paths, soil parameters, weather sequences, and train/val/test splits (70/15/15)	Final training dataset used by all model phases
Disease Database	Manually curated	config/disease_database.json	55 disease class entries with crop, condition, severity, symptoms, treatment, soil requirements, and weather requirements	Used by dashboard for disease information display and recommendations
Crop Database	Manually curated	config/crop_database.json	Optimal soil and weather parameters for each crop type	Used by SuitabilityEngine for soil suitability scoring
Disease-Soil Profiles	Manually curated with research backing	config/disease_soil_profiles.json	55 disease-soil profile mappings with healthy ranges, risk profiles, contributing parameters, evidence types	Used by SoilCauseAnalyzer for simulated sensor readings
Farm Soil Data (Excel)	Dynamically generated	data/farm_soil_data.xlsx	10-day historical soil trend data per Tree ID, generated dynamically during disease analysis	Used by dashboard for historical NPK/pH trend visualization
________________________________________
5. Algorithms Used & Where:
Algorithm / Technique	Purpose	File(s)
Vision Transformer (ViT-Base-Patch16-224)	Image-based disease classification (768-dim CLS token embeddings, 12 attention heads, 12 layers)	train_vit_phase1.py, soilvisionet_production/core/inference_engine.py
Bidirectional LSTM (2-layer, 256 hidden units)	Processing 30-day weather sequences (temp, rainfall, humidity) for binary disease risk prediction	train_lstm_phase2a.py
Extreme Learning Machine (ELM)	Rapid fusion of ViT features (768-dim) + Soil parameters (4-dim) using random hidden weights + least-squares output training	train_elm_phase2b.py
Late Fusion (FusionHead)	Trainable fusion of ViT logits + ELM logits + LSTM output via fully-connected layers with LayerNorm and dropout	train_hybrid_fusion.py, evaluate_hybrid_model.py
Ridge Regression	Analytical solution for ELM output layer (regularized least-squares)	train_elm_phase2b.py
AdamW Optimizer	ViT training with weight decay for regularization	train_vit_phase1.py
Adam Optimizer	LSTM and Fusion head training	train_lstm_phase2a.py, train_hybrid_fusion.py
Cosine Annealing LR Scheduler	Learning rate scheduling for ViT training	train_vit_phase1.py
Gradient Accumulation	Memory-efficient training for low-end systems (effective batch size = batch_size × accumulation_steps)	train_vit_phase1.py, train_lstm_phase2a.py
WeightedRandomSampler	Class imbalance handling during training	data_loader.py
BCEWithLogitsLoss	Binary cross-entropy for LSTM disease risk prediction	train_lstm_phase2a.py
CrossEntropyLoss	Multi-class classification loss for ViT and Fusion head	train_vit_phase1.py, train_hybrid_fusion.py
ImageNet Normalization	Standard image preprocessing (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])	data_loader.py, soilvisionet_production/core/image_processor.py
HSV Color Analysis	Crop-type detection from image color profiles	soilvisionet_production/modules/disease_detector.py
SHA-256 Deterministic Hashing	Generating stable, reproducible simulated soil sensor readings	soilvisionet_production/modules/soil_cause_analyzer.py
Llama 3.1 (8B-Instruct) LLM	AI expert pathological report generation via Hugging Face Inference API with LangChain	soilvisionet_production/modules/ai_explanation_generator.py
________________________________________
6. High-Level Architecture and Flow:
Phase 1: Data Collection & Integration
•	scripts/integrate_dataset.py – Master pipeline that loads PlantVillage metadata, fruit disease images, state soil data (CSV), and OpenWeatherMap weather data. Creates a unified metadata CSV with image paths, soil parameters (N, P, K, pH), and 30-day weather sequences (JSON) for each sample. Assigns train/val/test splits (70/15/15).
•	scripts/weather_integration.py – Fetches real weather data from OpenWeatherMap One Call API for 10 Indian cities (Maharashtra, Karnataka, UP, HP, Punjab, Bihar, etc.). Generates 30-day rolling sequences with temp/rainfall/humidity. Falls back to realistic synthetic data if API is unavailable.
•	scripts/organize_images.py – Copies and organizes raw images into the unified directory structure: unified_dataset/images/{train|val|test}/{disease_class}/{image_id}.jpg.
•	scripts/validate_integrated_dataset.py – Validates data integrity: split distribution, class balance, soil parameter statistics, weather sequence parsing, and image file existence.
Phase 2: Data Loading
•	data_loader.py – UnifiedDiseaseDataset PyTorch Dataset class that loads multi-modal data (images, soil, weather, labels) from the unified metadata CSV. Performs image transformations (resize, augment, normalize), soil normalization (z-score), weather normalization (z-score with clipping), and generates binary disease risk labels from weather conditions. get_dataloaders() creates train/val/test DataLoaders with WeightedRandomSampler for class balance.
Phase 3: Model Training (4-Phase Pipeline)
•	train_vit_phase1.py – Phase 1: ViT Training. Fine-tunes google/vit-base-patch16-224 (HuggingFace) on unified dataset images for 55-class disease classification. Uses AdamW optimizer, CosineAnnealing LR, gradient accumulation. Outputs: results/vit_phase1/best_model.pt.
•	train_lstm_phase2a.py – Phase 2A: LSTM Training. Trains a 2-layer Bidirectional LSTM on 30-day weather sequences (3 features: temp, rainfall, humidity) concatenated with soil features (4-dim) for binary disease risk prediction. Outputs: results/lstm_phase2a/best_lstm_model.pt.
•	train_elm_phase2b.py – Phase 2B: ELM Fusion. Extracts 768-dim CLS token features from the trained ViT, concatenates with 4-dim soil parameters (total 772-dim input), passes through a random 512-unit hidden layer, and trains the output layer analytically via Ridge regression. Outputs: results/elm_phase2b/elm_model.pt.
•	train_hybrid_fusion.py – Phase 3: Hybrid Fusion. Freezes all three pre-trained models (ViT, LSTM, ELM), concatenates their output logits, and trains a lightweight FusionHead (Linear → LayerNorm → ReLU → Dropout → Linear) for final 55-class classification. Outputs: results/hybrid/best_hybrid_fusion.pt.
Phase 4: Evaluation
•	evaluate_hybrid_model.py – Loads all four checkpoints, assembles the full SoilVisioNet pipeline, runs inference on the test set, and computes accuracy, F1, precision, recall, and inference time. Outputs: results/hybrid/soilvisionet_evaluation.json.
•	compare.ipynb – Jupyter notebook for comparative analysis against baseline models, generating high-resolution comparison charts.
Phase 5: Production Dashboard
•	soilvisionet_production/core/inference_engine.py – Unified model loader that automates checkpoint discovery across multiple paths, loads ViT/LSTM/ELM/Hybrid models, handles device selection (CUDA/CPU), and provides prediction interfaces.
•	soilvisionet_production/core/image_processor.py – Image handling pipeline: validation, loading, resizing (aspect-preserving or fixed), ImageNet normalization, tensor conversion, and augmentation.
•	soilvisionet_production/modules/disease_detector.py – Main detection interface. Preprocesses images, runs ViT inference, applies pomegranate-specific crop filtering/masking, maps class indices to disease names via class_mapping.json, and returns structured results with confidence, severity, symptoms, and treatment.
•	soilvisionet_production/modules/suitability_engine.py – Evaluates soil suitability and weather-based disease risk. Scores each NPK/pH parameter against crop-specific optimal ranges. Calculates per-disease weather risk scores. Generates combined recommendation scores.
•	soilvisionet_production/modules/soil_cause_analyzer.py – Generates deterministic simulated soil readings from disease-soil profiles using SHA-256 hashing for stability. Updates Excel historical data with 10-day trends. Provides fruit nutrient & vitamin impact assessments for pomegranate diseases.
•	soilvisionet_production/modules/explanation_generator.py – Generates human-readable reports for detections (severity, symptoms, treatment options, urgency levels) and soil/weather assessments.
•	soilvisionet_production/modules/ai_explanation_generator.py – Extends ExplanationGenerator with Llama 3.1 LLM integration via LangChain + Hugging Face for generating AI Expert Pathologist Reports with pesticide recommendations and cultural management advice.
•	soilvisionet_production/modules/disease_class_normalizer.py – Normalizes disease class names across data sources into canonical forms (crop :: disease).
•	soilvisionet_production/ui/app.py – Streamlit dashboard (928 lines) with: (a) Disease Detection tab: image upload, Tree ID selection, SoilVisioNet inference, confidence/severity display, soil factor analysis, historical NPK charts, standard + AI pathologist reports; (b) Soil & Weather Assessment tab: manual soil/weather input, crop suitability scoring, disease risk ranking, joint image analysis, exportable assessment reports.

Architecture Diagram (Text Format):
		                    ┌──────────────────────────────────────┐
                        │        UNIFIED DATASET               │
                        │  PlantVillage + Fruit Images +       │
                        │  State Soil Data + OpenWeatherMap    │
                        │  (15,288 samples, 55 classes)        │
                        └──────────┬───────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
           ┌───────────┐   ┌───────────┐   ┌───────────┐
           │  Images   │   │  Weather  │   │   Soil    │
           │ (224×224) │   │(30×3 seq) │   │ (N,P,K,pH)│
           └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
                 │               │               │
                 ▼               ▼               │
          ┌─────────────┐ ┌─────────────┐        │
          │  ViT-Base   │ │  Bi-LSTM    │        │
          │ (Phase 1)   │ │ (Phase 2A)  │        │
          │ 768-dim CLS │ │ Risk Score  │        │
          └──────┬──────┘ └──────┬──────┘        │
                 │               │               │
                 ├───────────────┼───────────────┤
                 │               │               │
                 ▼               │               ▼
          ┌─────────────┐        │        ┌─────────────┐
          │ ViT Logits  │        │        │    ELM      │
          │  (55-dim)   │        │        │ (Phase 2B)  │
          └──────┬──────┘        │        │ ViT+Soil →  │
                 │               │        │  55-dim     │
                 │               │        └──────┬──────┘
                 │               │               │
                 ▼               ▼               ▼
          ┌──────────────────────────────────────────┐
          │          FUSION HEAD (Phase 3)            │
          │    Concat → Linear → LayerNorm →         │
          │    ReLU → Dropout → Linear → 55 classes  │
          └──────────────────┬───────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  SoilVisioNet   │
                    │  Final Output   │
                    │  (55 classes)   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Streamlit     │
                    │   Dashboard     │
                    │  + AI Reports   │
                    │  (Llama 3.1)    │
                    └─────────────────┘
________________________________________
7. How It Works?
1.	Data Preparation: Images of fruit/leaf diseases (from PlantVillage and custom fruit datasets) are combined with soil nutrient data for Indian states and 30-day weather data from OpenWeatherMap. All data is linked together in a single metadata file and split into train (70%), validation (15%), and test (15%) sets.
2.	Model Training (4 Steps):
•	Step 1: A pre-trained Vision Transformer (ViT) is fine-tuned on the disease images to learn to recognize 55 different disease classes from visual patterns.
•	Step 2A: A Bidirectional LSTM network is trained on 30-day weather sequences to predict whether weather conditions are favorable for disease development (binary risk: yes/no).
•	Step 2B: An ELM takes the image features extracted by ViT and combines them with soil nutrient values (N, P, K, pH) to learn how soil conditions influence disease—trained in under one minute using a mathematical shortcut (least-squares).
•	Step 3: All three trained models are frozen, and a small Fusion Head is trained to combine their outputs into one final, more accurate prediction.
3.	Using the Dashboard:
•	User uploads a pomegranate fruit/leaf image and enters a Tree ID.
•	The system runs the trained SoilVisioNet model to identify the disease and confidence level.
•	Simulated soil sensor readings are generated based on the disease profile (this will be real hardware sensor data in future).
•	A 10-day historical soil trend is generated and displayed as charts.
•	Standard disease report (symptoms, treatment, urgency) is generated.
•	An AI Expert Report is auto-generated using Llama 3.1 with specific pesticide and management recommendations.
•	In the Soil & Weather tab, users can manually enter soil/weather values to get crop suitability scores and disease risk assessments.
4.	Future Hardware Integration: The existing dashboard is hardware-ready. Once the IoT sensor hardware module (NPK + pH sensor) is completed, the SoilCauseAnalyzer will read actual sensor data from the hardware instead of simulated values—no code changes required in the dashboard itself.
________________________________________
8. Where to Look in the Code (Quick Map):
What to Find	File Location
Dataset creation & integration pipeline	scripts/integrate_dataset.py
Weather data fetching (OpenWeatherMap)	scripts/weather_integration.py
Image organization into train/val/test	scripts/organize_images.py
Dataset validation	scripts/validate_integrated_dataset.py
Multi-modal data loading (images + soil + weather)	data_loader.py
ViT training (Phase 1)	train_vit_phase1.py
LSTM training for weather risk (Phase 2A)	train_lstm_phase2a.py
ELM soil-vision fusion (Phase 2B)	train_elm_phase2b.py
Hybrid fusion training (Phase 3)	train_hybrid_fusion.py
Final model evaluation (all metrics)	evaluate_hybrid_model.py
Comparative analysis notebook	compare.ipynb
Model inference engine (loads all checkpoints)	soilvisionet_production/core/inference_engine.py
Image preprocessing pipeline	soilvisionet_production/core/image_processor.py
Disease detection module	soilvisionet_production/modules/disease_detector.py
Soil suitability + weather risk engine	soilvisionet_production/modules/suitability_engine.py
Simulated soil sensor + nutrient analysis	soilvisionet_production/modules/soil_cause_analyzer.py
Explanation & report generation	soilvisionet_production/modules/explanation_generator.py
AI Pathologist reports (Llama 3.1)	soilvisionet_production/modules/ai_explanation_generator.py
Disease class normalization	soilvisionet_production/modules/disease_class_normalizer.py
Streamlit Dashboard (full UI)	soilvisionet_production/ui/app.py
Disease database (55 classes)	soilvisionet_production/config/disease_database.json
Crop database (optimal soil/weather)	soilvisionet_production/config/crop_database.json
Disease-soil profile mappings	soilvisionet_production/config/disease_soil_profiles.json
Class index → disease name mapping	soilvisionet_production/config/class_mapping.json
Result metrics (all phases)	result json/ directory
Training history plots	result json/training_history.png, result json/lstm_training_history.png
________________________________________
9. Results or Metrics Displayed:
A. ViT Phase 1 (Image Classification):
Metric	Value
Test Accuracy	98.12%
Precision (weighted)	98.19%
Recall (weighted)	98.12%
F1-Score (weighted)	98.13%
B. LSTM Phase 2A (Weather Risk Prediction):
Metric	Value
Test Accuracy	100.0% (binary disease risk classification)
C. ELM Phase 2B (Soil-Vision Fusion):
Metric	Value
Train Accuracy	99.61%
Validation Accuracy	98.41%
Test Accuracy	98.29%
Improvement over ViT alone	+6.29%
D. Hybrid Fusion (SoilVisioNet Final Model):
Metric	Value
Best Validation Accuracy	98.19%
Test Accuracy	98.12%
E. SoilVisioNet Comprehensive Evaluation:
Metric	Value
Test Set Size	2,283 samples
Number of Disease Classes	55
Accuracy	98.12%
F1-Score (weighted)	98.13%
Precision (weighted)	98.18%
Recall (weighted)	98.12%
Total Inference Time	1.43 seconds (full test set)
Per-Sample Inference Time	0.62 milliseconds
Batch Size	16
Device	CUDA (GPU)
F. Graphs & Visual Outputs:
•	Training & Validation Loss curves (ViT): result json/training_history.png
•	Training & Validation Loss curves (LSTM): result json/lstm_training_history.png
•	Comparative Analysis Chart: comparative_analysis_high_res.png, strict_comparative_analysis_1000dpi.png
•	Dashboard UI: Streamlit-based interactive dashboard with disease detection results, confidence meters, severity badges, NPK trend area/line/bar charts, soil parameter comparison tables, and AI-generated pathologist reports.
G. Dashboard Features Displayed:
•	Disease name, crop, condition, confidence percentage, severity level (Mild/Moderate/Severe)
•	Simulated sensor readings: Nitrogen (ppm), Phosphorus (ppm), Potassium (ppm), Soil pH
•	Healthy vs Current soil comparison table with status badges (🟢 Optimal / 🔴 Low / 🟠 High)
•	Fruit Nutrient & Vitamin Assessment (pomegranate-specific: Vitamin C, antioxidants, mineral impacts)
•	Historical NPK trend charts (10-day), pH stability chart, nutrient balance bar chart
•	Standard analysis report (symptoms, treatment, urgency)
•	AI Expert Pathologist Report (Llama 3.1: chemical treatments, biological alternatives, cultural management)
•	Soil & Weather suitability score (0-1 scale with Excellent/Good/Fair/Poor rating)
•	Weather-based disease risk ranking with risk level indicators
