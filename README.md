# 🚀 LedgerX: Enterprise Invoice Intelligence Platform

[![Production Status](https://img.shields.io/badge/Status-Production%20Ready-success)](https://ledgerx-api-671429123152.us-central1.run.app)
[![ML Models](https://img.shields.io/badge/Quality%20F1-77.07%25-blue)](https://github.com/Lochan9/ledgerx-mlops-final)
[![Failure F1](https://img.shields.io/badge/Failure%20F1-71.40%25-blue)](https://github.com/Lochan9/ledgerx-mlops-final)
[![Cloud](https://img.shields.io/badge/Cloud-GCP-4285F4)](https://console.cloud.google.com/run?project=ledgerx-mlops)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Production-grade MLOps platform for automated invoice quality assessment and payment failure prediction using dual CatBoost models with 95% OCR accuracy via Google Document AI.**

🌐 **Live Demo:** [LedgerX Dashboard](https://storage.googleapis.com/ledgerx-dashboard-671429123152/index_v2.html)  
🔗 **API Endpoint:** https://ledgerx-api-671429123152.us-central1.run.app  
📊 **Model Performance:** Quality 87.15% Acc, 77.07% F1 | Failure 86.70% Acc, 71.40% F1

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Model Performance](#-model-performance)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Environment Setup](#-environment-setup--installation)
- [Running Deployment Scripts](#-running-deployment-scripts-automated)
- [Verification & Testing](#-verification-of-deployment)
- [API Usage Guide](#-api-usage-guide)
- [MLOps Pipeline](#-mlops-pipeline)
- [Cost Optimization](#-cost-optimization)
- [Monitoring](#-monitoring--observability)
- [Contributing](#-contributing)

---

## 🎯 Overview

LedgerX is an enterprise-grade invoice intelligence platform that leverages machine learning to automatically assess invoice quality and predict payment failure risk. The system processes invoices through automated OCR extraction using Google Document AI (95% accuracy), validates data integrity, and provides ML-driven predictions through a dual-model architecture.

### ✨ Key Features

- **🤖 Dual-Model ML Architecture**
  - Quality Assessment: CatBoost (87.15% accuracy, 77.07% F1)
  - Failure Prediction: CatBoost (86.70% accuracy, 71.40% F1)

- **📄 Advanced OCR Pipeline**
  - Google Document AI (95% accuracy)
  - GPT-4 Vision fallback for edge cases
  - Automatic text extraction and structuring

- **🔐 Enterprise Security**
  - JWT authentication with role-based access control
  - bcrypt password hashing
  - Rate limiting and request throttling

- **📊 Production-Grade Infrastructure**
  - Cloud Run deployment with auto-scaling
  - Cloud SQL (PostgreSQL) for persistent storage
  - Prometheus + Grafana monitoring
  - Evidently AI for drift detection

- **💰 Cost Optimized**
  - Prediction caching (40% cost reduction)
  - Smart rate limiting
  - Auto-scaling with min-instances=0
  - **Estimated cost: $3-5/month**

---

## 📊 Model Performance

### Best Models (CatBoost)

| Task | Accuracy | Precision | Recall | F1 Score | AUC |
|------|----------|-----------|--------|----------|-----|
| **Quality Assessment** | **87.15%** | **87.45%** | **68.90%** | **77.07%** | **0.8263** |
| **Failure Prediction** | **86.70%** | **82.79%** | **62.76%** | **71.40%** | **0.7909** |

### Model Comparison

**Quality Assessment Models:**
- ✅ CatBoost: 87.15% Acc, 77.07% F1 (Production)
- Random Forest: 86.30% Acc, 75.67% F1
- Logistic Regression: 81.10% Acc, 70.92% F1

**Failure Prediction Models:**
- ✅ CatBoost: 86.70% Acc, 71.40% F1 (Production)
- Random Forest: 86.30% Acc, 69.00% F1
- Logistic Regression: 68.70% Acc, 54.77% F1

### Test Dataset
- **Total Test Samples**: 2,000 invoices
- Evaluated on realistic invoice processing scenarios
- Cross-validated performance metrics

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│  Web Dashboard (Static Hosting) + Mobile Responsive        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   FASTAPI SERVICE                           │
│  • Authentication (JWT + RBAC)                              │
│  • Rate Limiting & Caching                                  │
│  • Request Validation                                       │
│  • Cloud Logging Integration                                │
└────┬────────────┬─────────────┬─────────────┬──────────────┘
     │            │             │             │
     ▼            ▼             ▼             ▼
┌─────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────┐
│Document │ │  Models  │ │Cloud SQL │ │ Monitoring  │
│   AI    │ │CatBoost  │ │PostgreSQL│ │ Prometheus  │
│  OCR    │ │Quality & │ │ Database │ │  Grafana    │
│         │ │ Failure  │ │          │ │ Evidently   │
└─────────┘ └──────────┘ └──────────┘ └─────────────┘
     │            │             │             │
     └────────────┴─────────────┴─────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │   MLOps Infrastructure   │
        │  • DVC (Data Versioning) │
        │  • MLflow (Experiments)  │
        │  • Airflow (Pipelines)   │
        │  • GitHub Actions (CI/CD)│
        └──────────────────────────┘
```

---

## 🛠️ Technology Stack

### Core ML & Data Science
- **Python 3.12** - Main programming language
- **CatBoost 1.2.7** - Production ML models
- **scikit-learn 1.4.2** - ML pipeline & preprocessing
- **pandas 2.2.2 & numpy 1.26.4** - Data processing
- **SHAP 0.45.1** - Model interpretability

### MLOps Infrastructure
- **MLflow 2.12.1** - Experiment tracking & model registry
- **DVC 3.51.0** - Data versioning & pipeline automation
- **Apache Airflow** - Workflow orchestration
- **Great Expectations 0.18.12** - Data validation

### API & Deployment
- **FastAPI** - High-performance API framework
- **Uvicorn** - ASGI server
- **Google Cloud Run** - Serverless deployment
- **Docker** - Containerization

### OCR & Document Processing
- **Google Document AI 2.20.0** - Primary OCR (95% accuracy)
- **OpenAI GPT-4 Vision 1.3.0** - Fallback OCR
- **Pillow 10.3.0** - Image processing

### Storage & Database
- **Google Cloud SQL (PostgreSQL)** - Primary database
- **Google Cloud Storage** - Model & artifact storage
- **Cloud Storage** - Static website hosting

### Monitoring & Observability
- **Prometheus 0.19.0** - Metrics collection
- **Grafana** - Dashboards & visualization
- **Evidently AI 0.4.12** - ML monitoring & drift detection
- **Google Cloud Logging 3.8.0** - Centralized logging

### Security
- **python-jose 3.3.0** - JWT token handling
- **passlib 1.7.4 & bcrypt 4.1.2** - Password hashing
- **Google Secret Manager 2.16.4** - Secret management

---

## 🚀 Environment Setup & Installation

### Prerequisites

- **Python 3.12+**
- **Google Cloud SDK** (`gcloud` CLI)
- **Docker Desktop**
- **Git**
- **PowerShell** (for Windows) or **Bash** (for Linux/Mac)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ledgerx-mlops-final.git
cd ledgerx-mlops-final
```

### 2. Set Up Python Virtual Environment

**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip
```

**Linux/Mac:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Dependencies

```bash
# Install core requirements
pip install -r requirements.txt

# Install API-specific requirements
pip install -r requirements_api.txt

# Install Docker requirements (for containerization)
pip install -r requirements_docker.txt
```

**Key Dependencies Installed:**
- pandas, numpy (data processing)
- scikit-learn, catboost (ML models)
- fastapi, uvicorn (API server)
- mlflow, dvc (MLOps tools)
- google-cloud-documentai (OCR)
- prometheus-client (monitoring)

### 4. Set Up Google Cloud Project

```bash
# Set your GCP project
gcloud config set project ledgerx-mlops

# Authenticate
gcloud auth login
gcloud auth application-default login

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable sqladmin.googleapis.com
gcloud services enable documentai.googleapis.com
gcloud services enable storage-api.googleapis.com
gcloud services enable secretmanager.googleapis.com
```

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# API Configuration
ENVIRONMENT=production
PORT=8000

# Google Cloud
GCP_PROJECT_ID=ledgerx-mlops
GCP_REGION=us-central1

# Cloud SQL Configuration
CLOUD_SQL_CONNECTION_NAME=ledgerx-mlops:us-central1:ledgerx-postgres
DB_NAME=ledgerx
DB_USER=postgres
DB_PASSWORD=your_secure_password

# Document AI
DOCUMENT_AI_PROJECT_ID=671429123152
DOCUMENT_AI_LOCATION=us
DOCUMENT_AI_PROCESSOR_ID=your_processor_id

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Authentication
JWT_SECRET_KEY=your_secret_key_here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### 6. Initialize Database

**Option A: Using Cloud SQL Proxy (Recommended for Local Development)**

```powershell
# Download Cloud SQL Proxy
curl -o cloud-sql-proxy.exe https://dl.google.com/cloudsql/cloud_sql_proxy_x64.exe

# Start Cloud SQL Proxy in a separate terminal
.\cloud-sql-proxy.exe --port 5432 ledgerx-mlops:us-central1:ledgerx-postgres

# In another terminal, initialize database
python migrate_database.py
```

**Option B: Direct Connection (Production)**

The database is automatically managed by Cloud Run deployment.

### 7. Download Model Artifacts

**Using DVC:**

```bash
# Pull data and models from remote storage
dvc pull

# Verify models are downloaded
ls models/
# Should see: quality_catboost.cbm, failure_catboost.cbm
```

### 8. Verify Installation

```bash
# Check Python version
python --version  # Should be 3.12+

# Check installed packages
pip list | grep -E "fastapi|catboost|mlflow|dvc"

# Test model loading
python check_model_accuracy.py
```

**Expected Output:**
```
============================================================
LEDGERX MODEL PERFORMANCE - LIVE CHECK
============================================================

🎯 QUALITY ASSESSMENT MODELS:
CATBOOST:
  Accuracy:  87.15%
  F1 Score:  77.07%
  ...
```

---

## 🔥 Running Deployment Scripts (Automated)

### Method 1: Automated Cloud Run Deployment (Recommended)

**Using PowerShell Script:**

```powershell
# Navigate to project directory
cd ledgerx-mlops-final

# Run automated deployment
.\deploy_cloudrun.ps1
```

**What This Script Does:**
1. ✅ Builds Docker image with all dependencies
2. ✅ Pushes image to Google Container Registry
3. ✅ Deploys to Cloud Run with optimized settings
4. ✅ Configures auto-scaling (0-10 instances)
5. ✅ Sets up environment variables
6. ✅ Tests health endpoint
7. ✅ Returns service URL

**Expected Output:**
```
🚀 LedgerX Cloud Run Deployment (Lightweight API)
============================================================

📋 Configuration:
  Project: ledgerx-mlops
  Service: ledgerx-api
  Region: us-central1
  Image: gcr.io/ledgerx-mlops/ledgerx-api:v2-lightweight

[1/3] Building lightweight Docker image...
✅ Build complete!

[2/3] Pushing to Google Container Registry...
✅ Push complete!

[3/3] Deploying to Cloud Run...
✅ Deployment successful!
============================================================

🌐 Service URL: https://ledgerx-api-671429123152.us-central1.run.app

🧪 Testing health endpoint...
✅ Health check passed!

🎉 Deployment Complete!
API is ready at: https://ledgerx-api-671429123152.us-central1.run.app
```

### Method 2: GitHub Actions CI/CD (Automated on Push)

The repository includes automated deployment via GitHub Actions.

**File:** `.github/workflows/deploy-gcp.yml`

**Triggers:**
- Push to `main` branch (changes to `src/`, `models/`, `Dockerfile.cloudrun`)
- Manual workflow dispatch

**Steps:**
```yaml
1. Checkout code
2. Authenticate to GCP using service account key
3. Configure Docker for Artifact Registry
4. Build Docker image
5. Push to Artifact Registry
6. Deploy to Cloud Run
7. Run health check
8. Report deployment URL
```

**To Use GitHub Actions:**

1. **Set up GitHub Secrets:**
   - Go to Repository Settings → Secrets → Actions
   - Add secret: `GCP_SA_KEY` (service account JSON key)

2. **Push to main branch:**
```bash
git add .
git commit -m "Deploy to Cloud Run"
git push origin main
```

3. **Monitor deployment:**
   - Go to GitHub → Actions tab
   - Watch "Deploy to Cloud Run" workflow

### Method 3: Manual Docker Deployment

**Step-by-step manual deployment:**

```bash
# 1. Set environment variables
export PROJECT_ID=ledgerx-mlops
export SERVICE_NAME=ledgerx-api
export REGION=us-central1
export IMAGE_NAME=gcr.io/$PROJECT_ID/$SERVICE_NAME:latest

# 2. Build Docker image
docker build -f Dockerfile.cloudrun -t $IMAGE_NAME .

# 3. Test locally (optional)
docker run -p 8000:8000 \
  -e ENVIRONMENT=production \
  $IMAGE_NAME

# 4. Push to Google Container Registry
docker push $IMAGE_NAME

# 5. Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8000 \
  --cpu 2 \
  --memory 2Gi \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 10 \
  --set-env-vars "ENVIRONMENT=production"

# 6. Get service URL
gcloud run services describe $SERVICE_NAME \
  --region $REGION \
  --format 'value(status.url)'
```

### Method 4: Local Development Server

**For testing and development:**

```bash
# Start local FastAPI server
uvicorn src.inference.api_fastapi:app --reload --port 8000
```

**Access at:** http://localhost:8000

**API Documentation:** http://localhost:8000/docs

---

## ✅ Verification of Deployment

### Step 1: Test Health Endpoint

**Using curl:**
```bash
curl https://ledgerx-api-671429123152.us-central1.run.app/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-07T22:30:45.123Z",
  "environment": "production",
  "version": "1.0.0"
}
```

**Using PowerShell:**
```powershell
Invoke-RestMethod -Uri "https://ledgerx-api-671429123152.us-central1.run.app/health"
```

### Step 2: Test Authentication

**Get JWT Token:**

```bash
curl -X POST "https://ledgerx-api-671429123152.us-central1.run.app/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"
```

**Expected Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**PowerShell:**
```powershell
$body = @{
    username = "admin"
    password = "admin123"
}
$response = Invoke-RestMethod -Uri "https://ledgerx-api-671429123152.us-central1.run.app/token" `
    -Method Post `
    -ContentType "application/x-www-form-urlencoded" `
    -Body $body

$token = $response.access_token
Write-Host "Token: $token"
```

### Step 3: Test Model Inference Endpoint

**Make a prediction:**

```bash
# Save token from previous step
TOKEN="your_jwt_token_here"

# Make prediction
curl -X POST "https://ledgerx-api-671429123152.us-central1.run.app/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "blur_score": 45.2,
    "contrast_score": 28.5,
    "ocr_confidence": 0.87,
    "file_size_kb": 245.3,
    "vendor_name": "Acme Corp",
    "vendor_freq": 0.03,
    "total_amount": 1250.00,
    "invoice_number": "INV-2024-001",
    "invoice_date": "2024-01-15",
    "currency": "USD"
  }'
```

**Expected Response:**
```json
{
  "quality": {
    "prediction": "good",
    "confidence": 0.87,
    "probability": 0.872
  },
  "failure": {
    "prediction": "safe",
    "confidence": 0.76,
    "probability": 0.242
  },
  "recommendation": "APPROVE",
  "processing_time_ms": 45.2,
  "timestamp": "2025-12-07T22:35:12.456Z"
}
```

**PowerShell:**
```powershell
$headers = @{
    "Authorization" = "Bearer $token"
    "Content-Type" = "application/json"
}

$body = @{
    blur_score = 45.2
    contrast_score = 28.5
    ocr_confidence = 0.87
    file_size_kb = 245.3
    vendor_name = "Acme Corp"
    vendor_freq = 0.03
    total_amount = 1250.00
    invoice_number = "INV-2024-001"
    invoice_date = "2024-01-15"
    currency = "USD"
} | ConvertTo-Json

$prediction = Invoke-RestMethod -Uri "https://ledgerx-api-671429123152.us-central1.run.app/predict" `
    -Method Post `
    -Headers $headers `
    -Body $body

$prediction | ConvertTo-Json -Depth 10
```

### Step 4: Comprehensive Verification Script

**Run the automated test script:**

```bash
python test_authentication.py
```

**This script tests:**
- ✅ Authentication (JWT token generation)
- ✅ User info retrieval
- ✅ Model prediction endpoint
- ✅ Role-based access control
- ✅ Batch processing
- ✅ Error handling
- ✅ Token expiration

**Expected Output:**
```
============================================================
  LedgerX API Authentication Test Suite
============================================================

🔐 Authenticating as: admin
✅ Authentication successful!
   Token expires in: 1800 seconds

👤 Getting user info...
✅ User info retrieved:
   Username: admin
   Full Name: Admin User
   Role: admin
   Email: admin@ledgerx.com

🔮 Making prediction...
✅ Prediction successful!
   Quality: good (87.2% confidence)
   Failure: safe (75.8% confidence)
   Recommendation: APPROVE
   Processing Time: 42.3ms

============================================================
✅ ALL TESTS PASSED!
============================================================
```

### Step 5: Access Web Dashboard

1. **Open Browser:** https://storage.googleapis.com/ledgerx-dashboard-671429123152/index.html

2. **Login:**
   - Username: `admin`
   - Password: `admin123`

3. **Test Features:**
   - ✅ Upload invoice image
   - ✅ View OCR extraction
   - ✅ See quality prediction
   - ✅ Check failure prediction
   - ✅ Review validation results
   - ✅ View invoice history

### Step 6: Monitor Cloud Run Service

**Check service status:**
```bash
gcloud run services describe ledgerx-api --region us-central1
```

**View logs:**
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ledgerx-api" --limit 50
```

**Check metrics:**
- Go to: https://console.cloud.google.com/run?project=ledgerx-mlops
- Click on `ledgerx-api` service
- View: Request count, latency, memory usage, CPU utilization

### Step 7: Verify Database Connection

```bash
# Test database connection
python verify_integration.py
```

**This verifies:**
- ✅ Cloud SQL connectivity
- ✅ Database schema
- ✅ User authentication table
- ✅ Invoice storage table
- ✅ Document AI usage tracking

---

## 📚 API Usage Guide

### Interactive API Documentation

Access the auto-generated Swagger UI:
- **Production:** https://ledgerx-api-671429123152.us-central1.run.app/docs
- **Local:** http://localhost:8000/docs

### Core Endpoints

#### 1. Health Check
```http
GET /health
```
No authentication required.

#### 2. Authentication
```http
POST /token
Content-Type: application/x-www-form-urlencoded

username=admin&password=admin123
```

**Response:**
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### 3. Get Current User
```http
GET /users/me
Authorization: Bearer {token}
```

#### 4. Make Prediction
```http
POST /predict
Authorization: Bearer {token}
Content-Type: application/json

{
  "blur_score": 45.2,
  "contrast_score": 28.5,
  "ocr_confidence": 0.87,
  "file_size_kb": 245.3,
  "vendor_name": "Acme Corp",
  "vendor_freq": 0.03,
  "total_amount": 1250.00,
  "invoice_number": "INV-2024-001",
  "invoice_date": "2024-01-15",
  "currency": "USD"
}
```

#### 5. Upload Invoice (with OCR)
```http
POST /upload
Authorization: Bearer {token}
Content-Type: multipart/form-data

file: [invoice.jpg/pdf]
```

#### 6. Batch Predictions
```http
POST /predict/batch
Authorization: Bearer {token}
Content-Type: application/json

{
  "invoices": [
    { /* invoice 1 data */ },
    { /* invoice 2 data */ },
    ...
  ]
}
```

#### 7. Get User Invoices
```http
GET /invoices
Authorization: Bearer {token}
```

#### 8. Prometheus Metrics
```http
GET /metrics
```
Returns Prometheus-formatted metrics for monitoring.

### Python Client Example

```python
import requests

class LedgerXClient:
    def __init__(self, base_url, username, password):
        self.base_url = base_url
        self.token = self._authenticate(username, password)
    
    def _authenticate(self, username, password):
        response = requests.post(
            f"{self.base_url}/token",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        return response.json()["access_token"]
    
    def predict(self, invoice_data):
        response = requests.post(
            f"{self.base_url}/predict",
            json=invoice_data,
            headers={"Authorization": f"Bearer {self.token}"}
        )
        return response.json()
    
    def upload_invoice(self, file_path):
        with open(file_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/upload",
                files={"file": f},
                headers={"Authorization": f"Bearer {self.token}"}
            )
        return response.json()

# Usage
client = LedgerXClient(
    "https://ledgerx-api-671429123152.us-central1.run.app",
    "admin",
    "admin123"
)

# Make prediction
result = client.predict({
    "blur_score": 45.2,
    "ocr_confidence": 0.87,
    "total_amount": 1250.00,
    # ... other fields
})

print(f"Quality: {result['quality']['prediction']}")
print(f"Failure Risk: {result['failure']['prediction']}")
```

---

## 🔄 MLOps Pipeline

### Data Versioning (DVC)

```bash
# Pull latest data
dvc pull

# Reproduce entire pipeline
dvc repro

# Push changes
dvc add data/processed/new_data.csv
git add data/processed/new_data.csv.dvc
git commit -m "Add new processed data"
dvc push
```

### Model Training

```bash
# Train all models
python src/training/train_all_models.py

# With hyperparameter tuning
python src/training/hyperparameter_tuning.py

# Evaluate models
python src/training/evaluate_models.py
```

### Experiment Tracking (MLflow)

```bash
# Start MLflow UI
mlflow ui --port 5000

# View at: http://localhost:5000
```

### Automated Retraining

The system automatically triggers retraining when:
- Data drift detected (KS test p-value < 0.05)
- Performance degradation (F1 drop > 5%)
- Manual trigger via API

```bash
# Check for drift
python src/monitoring/drift_threshold_checker.py

# Trigger retraining
python src/monitoring/auto_retrain_trigger.py
```

---

## 💰 Cost Optimization

### Current Monthly Costs: $3-5

**Breakdown:**
- Cloud Run (API): $1-2
- Cloud SQL: $1-2
- Document AI (OCR): $0.50-1
- Cloud Storage: $0.10
- Monitoring: $0

### Optimization Features

1. **Prediction Caching** - 40% cost reduction
   - LRU cache for duplicate predictions
   - 1-hour TTL
   - Automatic invalidation

2. **Rate Limiting**
   - 100 requests/hour per user
   - Prevents abuse
   - Protects budget

3. **Auto-Scaling**
   - Min instances: 0 (scale to zero)
   - Max instances: 10
   - CPU-based scaling

4. **Smart OCR**
   - Document AI for high-value invoices
   - Cached results for duplicates
   - Usage tracking and limits

---

## 📈 Monitoring & Observability

### Prometheus Metrics

Access at: https://ledgerx-api-671429123152.us-central1.run.app/metrics

**Key Metrics:**
- `prediction_total` - Total predictions made
- `prediction_errors` - Failed predictions
- `prediction_latency_seconds` - Response time histogram
- `quality_predictions_good` - Good quality invoices
- `failure_predictions_risk` - High-risk invoices

### Grafana Dashboard

```bash
# Start monitoring stack (local)
cd monitoring
docker-compose up -d

# Access Grafana: http://localhost:3000
# Username: admin
# Password: admin
```

### Cloud Logging

View logs in Google Cloud Console:
```bash
gcloud logging read "resource.type=cloud_run_revision" --limit 100
```

### Drift Detection

```bash
# Run drift detection
python src/monitoring/evidently_drift_detection.py

# Check results
cat reports/drift_history.json
```

---

## 🤝 Contributing

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make changes**
4. **Run tests**
   ```bash
   pytest tests/
   ```
5. **Commit changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
6. **Push to branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**


