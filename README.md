# 🧾 LedgerX – Failure-Aware Invoice MLOps Pipeline

This repository contains the data pipeline for the LedgerX Invoice Intelligence Platform built for IE 7305 (MLOps) at Northeastern University.  
It includes OCR extraction, preprocessing, schema validation, testing, DVC versioning, and Airflow orchestration.

---

# ⚙️ Environment Setup

## 1️⃣ Clone the Repository
git clone https://github.com/Lochan9/ledgerx-mlops-final.git  
cd ledgerx-mlops-final

---

## 2️⃣ Create & Activate Virtual Environment

### Windows
python -m venv .venv  
.\.venv\Scripts\activate  

### macOS / Linux
python -m venv .venv  
source .venv/bin/activate  

---

## 3️⃣ Install Dependencies
pip install -r requirements.txt

---

# 📦 Data Setup

## 4️⃣ Add Invoice Images
Place invoice JPGs into:
data/raw/FATURA/invoices_dataset_final/images/

Verify images exist:
ls data/raw/FATURA/invoices_dataset_final -Recurse | select -First 5

---

# 🧠 OCR Processing

## 5️⃣ Run OCR Extraction
python src/preprocess_fatura.py

This generates:
data/processed/fatura_ocr.csv

Verify output:
python -c "import pandas as pd; df=pd.read_csv('data/processed/fatura_ocr.csv'); print(df.head())"

---

# 🐳 Run Airflow

## 6️⃣ Start Airflow (Docker)
docker compose up --build -d

## 7️⃣ Check Containers
docker ps

You should see:
- webserver  
- scheduler  
- postgres  

## 8️⃣ Open Airflow UI
http://localhost:8081

Login:
admin / admin

---

# 📊 Running the LedgerX Pipeline

## 9️⃣ Trigger DAG in Airflow
In the UI:
1. Open ledgerx_fatura_pipeline  
2. Click Trigger DAG  

Pipeline tasks:
- extract_ocr  
- validate_schema  
- bias_detection  
- unit_tests  
- dvc_push  

Expected logs:
File found: data/processed/fatura_ocr.csv  
Schema validation passed  
DVC push complete  

---

# 📁 Repository Structure
ledgerx-mlops-final/  
├── dags/  
├── src/  
├── data/  
├── tests/  
├── reports/  
├── Dockerfile  
├── docker-compose.yml  
└── start_ledgerx.sh  

---

# 📚 DVC Version Control (Optional)

## 1️⃣0️⃣ Initialize DVC
dvc init  
dvc remote add -d gdrive gdrive://<YOUR_DRIVE_ID>

## 1️⃣1️⃣ Track Processed Data
dvc add data/processed/fatura_ocr.csv  
git add data/processed/fatura_ocr.csv.dvc  
git commit -m "Track OCR dataset"  
dvc push  

---

# 🧪 Testing

## 1️⃣2️⃣ Run All Unit Tests
pytest -v --disable-warnings

---

# 🛠️ Troubleshooting

Issue: File not found (data/processed)  
Fix: Ensure ./data:/opt/airflow/data is mounted in docker-compose.yml  

Issue: Airflow webserver restarting  
Fix: Add "git" to apt-get install list in Dockerfile  

Issue: DAG not visible  
Fix: Mount ./dags:/opt/airflow/dags  

Issue: Permissions  
Fix: chmod -R 777 data/ logs/ reports/  

---

# 🎯 Metrics Targets
OCR F1: ≥ 90%  
Validation Pass Rate: ≥ 92%  
Failure AUC: ≥ 0.85  
GL Mapping Accuracy: ≥ 88%  
Latency p95: ≤ 5 seconds  

---

# 🧭 Roadmap
[x] OCR Pipeline  
[x] Great Expectations Validation  
[x] Airflow DAG  
[x] DVC Integration  
[ ] Failure Prediction Model  
[ ] Vendor Graph  
[ ] CI/CD (GitHub Actions + MLflow)  

---

# 📄 License
For academic use – Northeastern University IE 7305 (MLOps).
