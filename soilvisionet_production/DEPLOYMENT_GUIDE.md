# 🚀 Deployment Guide - SoilVisioNet Production System

Guide for deploying SoilVisioNet to production environments.

## Table of Contents
1. [Quick Deployment Options](#quick-deployment-options)
2. [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
3. [Heroku Deployment](#heroku-deployment)
4. [Docker Deployment](#docker-deployment)
5. [AWS Deployment](#aws-deployment)
6. [Google Cloud Deployment](#google-cloud-deployment)
7. [Production Best Practices](#production-best-practices)
8. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Quick Deployment Options

### Development (Local) - Already Complete
```bash
streamlit run ui/app.py
# Accessible at http://localhost:8501
```

### Production Quick Pick
| Platform | Difficulty | Cost | Speed | Best For |
|----------|-----------|------|-------|----------|
| **Streamlit Cloud** | Very Easy | Free | Minutes | Quick prototypes, demos |
| **Heroku** | Easy | $0/month (free tier) | 10-15 min | Small teams, single app |
| **Docker** | Medium | $0-50/month | 30 min setup | Any cloud, high control |
| **AWS EC2** | Medium-Hard | $5-20/month | 20 min | Scalability, reliability |
| **Google Cloud** | Medium-Hard | $5-20/month | 20 min | Integration with GCP tools |
| **Azure** | Medium-Hard | $5-20/month | 20 min | Enterprise integration |

---

## Streamlit Cloud Deployment

**Pros**: Easiest, free, automatic updates  
**Cons**: Limited to Streamlit hosting, less control

### Step 1: Prepare Your Repository

1. Ensure GitHub repo has proper structure:
```
soilvisionet_production/
├── ui/
│   └── app.py
├── core/
│   ├── inference_engine.py
│   └── image_processor.py
├── modules/
│   ├── disease_detector.py
│   ├── suitability_engine.py
│   └── explanation_generator.py
├── config/
│   ├── disease_database.json
│   └── crop_database.json
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
```

2. Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#4CAF50"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[logger]
level = "info"

[client]
showErrorDetails = true
toolbarMode = "auto"

[server]
maxUploadSize = 50
port = 8501
headless = true
```

### Step 2: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Sign in with GitHub account
3. Click "New App"
4. Select your repository
5. Select branch (usually `main`)
6. Enter path to app: `soilvisionet_production/ui/app.py`
7. Click "Deploy!"

**Important**: Make sure `requirements.txt` includes all dependencies.

### Step 3: Configure Secrets (for sensitive data)

Create `.streamlit/secrets.toml` in repository:
```toml
# Database credentials (if using remote DB)
db_user = "your_db_user"
db_password = "your_db_password"

# API keys (if integrating with external services)
openweathermap_key = "your_api_key"
```

Then in app.py:
```python
import streamlit as st
db_password = st.secrets["db_password"]
```

### Step 4: Monitor & Manage

- **App URL**: `https://share.streamlit.io/yourname/reponame`
- **Logs**: View in Streamlit Cloud dashboard
- **Auto-updates**: Redeploys when you push to GitHub

**Known Limitations**:
- No GPU (CPU inference only - 5-10 sec per image)
- 1 GB memory limit
- Cold start delays (first request takes longer)

---

## Heroku Deployment

**Pros**: Easy CI/CD, free tier available, good for small apps  
**Cons**: Can be slow on free tier, limited to 512 MB

### Step 1: Create Heroku Account
Go to https://www.heroku.com/ and create account.

### Step 2: Install Heroku CLI
```bash
# Windows
choco install heroku-cli

# Mac
brew tap heroku/brew && brew install heroku

# Linux
curl https://cli-assets.heroku.com/install.sh | sh
```

### Step 3: Prepare for Deployment

Create `Procfile` in project root:
```
web: streamlit run soilvisionet_production/ui/app.py --server.port=$PORT
```

Create `runtime.txt`:
```
python-3.10.0
```

Create `.gitignore`:
```
__pycache__/
*.pyc
.env
venv/
.streamlit/secrets.toml
cache/
logs/
```

### Step 4: Deploy

```bash
# Login to Heroku
heroku login

# Create app
heroku create your-soilvisionet-app

# Set buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# View logs
heroku logs --tail
```

**Cost**: Free tier (~$0), $5-25/month for production

---

## Docker Deployment

**Pros**: Works everywhere, easy scaling, production-ready  
**Cons**: Requires Docker installation, more setup

### Step 1: Create Dockerfile

```dockerfile
# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better layer caching)
COPY soilvisionet_production/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY soilvisionet_production/ .
COPY data/ ../data/

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 2: Create .dockerignore

```
__pycache__
*.pyc
.git
.gitignore
.streamlit/secrets.toml
.env
cache/
logs/
venv/
.pytest_cache
```

### Step 3: Build Image

```bash
docker build -t soilvisionet:latest .
```

### Step 4: Run Container Locally

```bash
# With GPU support (if available)
docker run --gpus all -p 8501:8501 soilvisionet:latest

# CPU only
docker run -p 8501:8501 soilvisionet:latest
```

### Step 5: Push to Docker Hub

```bash
# Tag image
docker tag soilvisionet:latest yourusername/soilvisionet:latest

# Login to Docker Hub
docker login

# Push
docker push yourusername/soilvisionet:latest
```

### Step 6: Deploy to Cloud

**AWS ECR (Elastic Container Registry)**:
```bash
# Create ECR repository
aws ecr create-repository --repository-name soilvisionet

# Push image
docker tag soilvisionet:latest <account_id>.dkr.ecr.<region>.amazonaws.com/soilvisionet:latest
docker push <account_id>.dkr.ecr.<region>.amazonaws.com/soilvisionet:latest
```

**Google Cloud Run**:
```bash
# Push to Google Container Registry
docker tag soilvisionet:latest gcr.io/your-project/soilvisionet:latest
docker push gcr.io/your-project/soilvisionet:latest

# Deploy
gcloud run deploy soilvisionet \
  --image gcr.io/your-project/soilvisionet:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8501 \
  --memory 2Gi \
  --timeout 3600
```

---

## AWS Deployment

### Option 1: EC2 Instance (Recommended for Control)

**Step 1: Launch EC2 Instance**
```bash
# Using AWS CLI
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t2.medium \
  --key-name your-key-pair \
  --security-groups streamlit-sg
```

**Step 2: SSH into Instance**
```bash
ssh -i "your-key.pem" ubuntu@your-ec2-ip
```

**Step 3: Install Dependencies**
```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python
sudo apt-get install python3-pip -y

# Install git
sudo apt-get install git -y

# Clone your repository
git clone https://github.com/yourusername/soilvisionet.git
cd soilvisionet
```

**Step 4: Setup Environment**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r soilvisionet_production/requirements.txt

# Copy model files (IMPORTANT!)
cp -r /path/to/trained/models data/unified_dataset/models/
```

**Step 5: Run Application**
```bash
# Option A: Direct run (for testing)
streamlit run soilvisionet_production/ui/app.py

# Option B: Use PM2 for persistent running
npm install -g pm2
pm2 start "streamlit run soilvisionet_production/ui/app.py" --name soilvisionet
pm2 save  # Save startup script
```

**Step 6: Set Up Reverse Proxy (Nginx)**
```bash
sudo apt-get install nginx -y
sudo nano /etc/nginx/sites-available/default
```

Add to nginx config:
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

Then restart nginx:
```bash
sudo systemctl restart nginx
```

**Cost**: ~$5-15/month for t2.medium instance

---

### Option 2: AWS Elastic Beanstalk (Easier Scaling)

```bash
# Install EB CLI
pip install awsebcli

# Initialize environment
eb init -p "Python 3.10 running on 64bit Amazon Linux 2" soilvisionet

# Create environment
eb create soilvisionet-env

# Deploy
eb deploy

# Open application
eb open
```

---

## Google Cloud Deployment

### Using Cloud Run (Recommended - Serverless)

**Step 1: Setup Google Cloud Project**
```bash
# Install Google Cloud SDK
# Download from: https://cloud.google.com/sdk/docs/install

# Login
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID
```

**Step 2: Prepare Application**

Create `app.yaml`:
```yaml
apiVersion: "serving.knative.dev/v1"
kind: "Service"
metadata:
  name: "soilvisionet"
spec:
  template:
    spec:
      containers:
      - image: "gcr.io/YOUR_PROJECT_ID/soilvisionet"
        ports:
        - containerPort: 8501
        resources:
          limits:
            memory: "2Gi"
            cpu: "2"
        env:
        - name: STREAMLIT_SERVER_PORT
          value: "8501"
        - name: STREAMLIT_SERVER_ADDRESS
          value: "0.0.0.0"
```

**Step 3: Build & Deploy**
```bash
# Build image using Cloud Build
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/soilvisionet

# Deploy to Cloud Run
gcloud run deploy soilvisionet \
  --image gcr.io/YOUR_PROJECT_ID/soilvisionet:latest \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --allow-unauthenticated \
  --timeout 3600
```

**Cost**: ~$0-20/month (pay-per-use, free for first 2M requests)

---

## Azure Deployment

### Using Azure Container Instances

**Step 1: Create Resource Group**
```bash
az group create \
  --name soilvisionet-rg \
  --location eastus
```

**Step 2: Create Container Registry**
```bash
az acr create \
  --resource-group soilvisionet-rg \
  --name soilvisionetacr \
  --sku Basic
```

**Step 3: Push Image**
```bash
# Build and push
az acr build \
  --registry soilvisionetacr \
  --image soilvisionet:latest .
```

**Step 4: Deploy Container**
```bash
az container create \
  --resource-group soilvisionet-rg \
  --name soilvisionet-app \
  --image soilvisionetacr.azurecr.io/soilvisionet:latest \
  --registry-login-server soilvisionetacr.azurecr.io \
  --ports 8501 \
  --environment-variables STREAMLIT_SERVER_PORT=8501 \
  --memory 2
```

**Cost**: ~$10-30/month

---

## Production Best Practices

### 1. Security

```python
# In ui/app.py, add authentication:
import streamlit as st
from streamlit_authenticator import Authenticate

# Configure authentication
config = {
    'usernames': {
        'user1': {'password': 'hashed_password1'},
        'user2': {'password': 'hashed_password2'}
    }
}

authenticator = Authenticate(
    config,
    'cookie-name',
    'signature-key',
    cookie_expiry_days=1
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
elif authentication_status:
    # App code here
    pass
```

### 2. Logging & Monitoring

Create `logging_config.py`:
```python
import logging
import os
from datetime import datetime

def setup_logging():
    log_dir = os.environ.get('LOG_OUTPUT_DIR', './logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(
        log_dir,
        f"soilvisionet_{datetime.now().strftime('%Y%m%d')}.log"
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('soilvisionet')

logger = setup_logging()
```

### 3. Error Handling

```python
import traceback
import streamlit as st

try:
    result = detector.detect_from_path(image_path)
except Exception as e:
    logger.error(f"Detection failed: {str(e)}\n{traceback.format_exc()}")
    st.error(f"Error: {str(e)}")
    st.info("Please check logs for details")
```

### 4. Resource Management

```python
import psutil
import torch

def check_resources():
    """Monitor system resources"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1e9  # GB
    
    return {
        'cpu': cpu_percent,
        'memory_gb': memory.used / 1e9,
        'memory_percent': memory.percent,
        'gpu_memory_gb': gpu_memory if torch.cuda.is_available() else None
    }
```

### 5. Configuration Management

Use environment variables (see `.env.example`):
```python
import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    'max_file_size': int(os.getenv('MAX_FILE_SIZE_MB', 50)),
    'device': os.getenv('COMPUTE_DEVICE', 'auto'),
    'log_level': os.getenv('LOG_LEVEL', 'info'),
    'cache_models': os.getenv('CACHE_MODELS', 'true').lower() == 'true',
}
```

---

##Monitoring & Maintenance

### Health Checks

Create `health_check.py`:
```python
import requests
import json
from datetime import datetime

def check_application_health(app_url):
    """Verify app is running and responsive"""
    try:
        response = requests.get(f"{app_url}/_stcore/health", timeout=5)
        return {
            'status': 'healthy' if response.status_code == 200 else 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'response_code': response.status_code
        }
    except Exception as e:
        return {
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }
```

### Automated Backups

```bash
#!/bin/bash
# backup.sh - Run daily via cron

BACKUP_DIR="/backups/soilvisionet"
APP_DIR="/app/soilvisionet_production"

mkdir -p $BACKUP_DIR

# Backup database and models
tar -czf "$BACKUP_DIR/backup_$(date +%Y%m%d).tar.gz" \
    "$APP_DIR/config/" \
    "$APP_DIR/logs/"

# Keep only 30 days of backups
find $BACKUP_DIR -name "backup_*.tar.gz" -mtime +30 -delete
```

Schedule via crontab:
```bash
# Run daily at 2 AM
0 2 * * * /app/backup.sh
```

### Log Rotation

Create `/etc/logrotate.d/soilvisionet`:
```
/app/soilvisionet_production/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    notifempty
    create 0640 www-data www-data
    sharedscripts
    postrotate
        systemctl reload soilvisionet
    endscript
}
```

### Performance Monitoring

```python
# Add to ui/app.py
import time
import psutil

start_time = time.time()
# ... model inference ...
inference_time = time.time() - start_time

st.sidebar.metric("Inference Time", f"{inference_time:.2f}s")
st.sidebar.metric("Memory Usage", f"{psutil.Process().memory_info().rss / 1e9:.2f} GB")
```

### Update Strategy

```bash
#!/bin/bash
# update.sh - Safe deployment updates

# Stop application
systemctl stop soilvisionet

# Backup current version
cp -r /app/soilvisionet_production /backups/soilvisionet_$(date +%Y%m%d_backup)

# Update code
cd /app && git pull origin main

# Install new dependencies
pip install -r soilvisionet_production/requirements.txt

# Regenerate databases if needed
python soilvisionet_production/extract_databases.py

# Start application
systemctl start soilvisionet

# Verify health
sleep 5
curl http://localhost:8501/_stcore/health
```

---

**Deployment Checklist**:
- [ ] Model checkpoints copied to correct location
- [ ] Environment variables configured
- [ ] SSL certificate installed (for HTTPS)
- [ ] Database backups configured
- [ ] Logging enabled
- [ ] Health monitoring setup
- [ ] User authentication configured
- [ ] Rate limiting enabled (if API)
- [ ] Resource limits set (CPU, memory)
- [ ] Disaster recovery plan documented

---

**Last Updated**: February 2024  
**Version**: 1.0.0
