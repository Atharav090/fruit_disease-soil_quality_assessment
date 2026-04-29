# 🌱 SoilVisioNet

## 📌 Project Overview
**SoilVisioNet** is a PhD-level research project focused on **early and accurate detection of fruit crop diseases** using a **multi-modal deep learning approach**.

Unlike traditional systems that rely only on image analysis, SoilVisioNet integrates:
- 🍃 Fruit/Leaf Images  
- 🌾 Soil Parameters (N, P, K, pH)  
- 🌦️ Weather Data (Temperature, Rainfall, Humidity)  

All inputs are processed through a **hybrid AI architecture** to provide a **holistic disease diagnosis**.

---

## 🧠 Core Architecture

The system combines three advanced models:

- **Vision Transformer (ViT)**  
  → Extracts deep image features from plant/fruit images  

- **Bidirectional LSTM (Bi-LSTM)**  
  → Analyzes 30-day weather patterns  

- **Extreme Learning Machine (ELM)**  
  → Performs fast fusion of soil and vision features  

- **Late Fusion Layer**  
  → Combines outputs of all models for final prediction  

---

## 📊 Dataset

- **15,288+ samples**
- **55 disease classes**
- **14+ crop types**

### Data Sources:
- PlantVillage Dataset (Images)
- State Soil Data
- OpenWeatherMap API (Weather Data)

---

## 🚀 Features

### 🔍 Disease Detection
- Upload fruit/leaf image (currently optimized for **Pomegranate**)
- AI-based disease classification

### 🌱 Soil & Weather Analysis
- Simulated soil sensor data (NPK, pH)
- Weather-based disease insights

### 🧾 AI Expert Reports
- Generated using **Llama 3.1 (LLM)**
- Provides detailed pathology insights

### 🍎 Nutrient Analysis
- Fruit-specific health insights
- Vitamin & mineral dependency tracking

### 📈 Advanced Functionalities
- 10-day soil trend analysis
- Excel logging with dynamic charts
- Crop suitability scoring
- Disease normalization system

---

## 🖥️ Dashboard

Built using **Streamlit**, includes:
- Disease Detection Tab
- Soil & Weather Assessment Tab

---

## 🔧 Additional Implementations

- Deterministic soil simulation engine
- Disease-soil profile mapping (55 classes)
- Hardware-ready architecture (IoT integration ready)
- Docker support for deployment

---

## 🧪 Novelty

### 🌟 Multi-Modal Hybrid AI
First system to combine:
- ViT (Image)
- Bi-LSTM (Time-series Weather)
- ELM (Feature Fusion)

### 🌾 Soil-Disease Intelligence
- Models correlation between **soil health and disease occurrence**
- Goes beyond traditional image-only systems

---

## 🛠️ Tech Stack

- Python
- Deep Learning (PyTorch / TensorFlow)
- Streamlit
- Hugging Face API
- OpenWeatherMap API
- Docker

---

## 🔮 Future Scope

- Integration with real IoT soil sensors  
- Multi-crop expansion  
- Mobile application deployment  
- Real-time farm monitoring system  

---

## 📦 Deployment

Supports:
- Docker
- Cloud platforms (via Procfile & app.yaml)

---

## 👨‍🔬 Use Case

- Farmers & Agronomists  
- Agricultural Research  
- Smart Farming Systems  

---

## 📜 License

This project is intended for research and academic purposes.

---

## 🙌 Acknowledgment

Inspired by the need to bridge the gap between:
**Soil Health + Weather Patterns + AI-based Disease Detection**
