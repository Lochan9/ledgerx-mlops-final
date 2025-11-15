# 🧾 LedgerX – Fatura MLOps Data Pipeline  
#### **AI-powered invoice (Fatura) ingestion, validation, testing, versioning & automation using Apache Airflow + DVC**

---

## 📌 Overview
LedgerX (Fatura Edition) is a production-grade **MLOps pipeline** for invoice OCR data processing.  
The system automates:

- **Data acquisition** (Fatura images → OCR output)
- **Preprocessing / cleaning**
- **Schema validation**
- **Unit testing for reliability**
- **Bias checking**
- **Data versioning with DVC**
- **Report generation**
- **End-to-end orchestration via Airflow**

This repository implements the **IE7305 – MLOps Data Pipeline Submission** requirements and serves as the foundation of the larger LedgerX invoice intelligence platform.

---

## 🚀 Key Features (Aligned to MLOps Guidelines)
- **Automated OCR ingestion** for Fatura datasets  
- **Preprocessing**: normalization, cleaning, text extraction  
- **Data validation** using Great Expectations / schema checks  
- **Unit tests** (pytest) for transformations & workflows  
- **Bias detection** using slice-based analysis  
- **DVC tracking** (`raw`, `processed`)  
- **Airflow DAG orchestration** for full workflow automation  
- **Logs + reports** stored locally for reproducibility  
- **Containerized environment** using Docker Compose (Airflow ready)

---

## 📁 Repository Structure
```
ledgerx-mlops-final/
│
├── dags/                     # Airflow DAGs
│   └── ledgerx_fatura_pipeline.py
│
├── data/
│   ├── raw/                  # Raw OCR inputs
│   ├── processed/            # Processed CSV (OCR normalized)
│   └── reports/              # Schema, tests, bias reports
│
├── src/
│   ├── acquire_fatura_data.py
│   ├── preprocess_fatura.py
│   ├── validate_schema.py
│   ├── test_runner.py
│   ├── generate_report.py
│   └── bias_check.py
│
├── tests/
│   ├── test_preprocess.py
│   └── test_schema.py
│
├── dvc.yaml                  # DVC pipeline definition
├── Dockerfile                # Airflow custom image
├── docker-compose.yml        # Airflow environment
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Environment Setup

### **1️⃣ Clone the repository**
```
git clone https://github.com/Lochan9/ledgerx-mlops-final.git
cd ledgerx-mlops-final
```

### **2️⃣ Create virtual environment**
```
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
```

### **3️⃣ Install DVC**
```
pip install dvc
```

### **4️⃣ Initialize DVC**
```
dvc init
```

---

## 🐳 Running Airflow (Docker)
This project includes a **ready-to-run** Airflow Docker environment.

### **1️⃣ Start Airflow**
```
docker compose up --build
```

### **2️⃣ Access Airflow UI**  
Open browser → **http://localhost:8081**

### **3️⃣ Locate your DAG**  
Search for: **ledgerx_fatura_pipeline**

Enable → Trigger DAG

---

## 🔁 Pipeline Flow (Airflow DAG)
The pipeline follows the required academic MLOps flow:

1. **Acquire Data**  
   `acquire_fatura_data.py` loads the OCR dataset.

2. **Skip Preprocessing (OCR already processed)**  
   For Fatura, preprocessing is minimal.

3. **Schema Validation**  
   Checks formatting, columns, missing values.

4. **Unit Tests**  
   Pytest runs all tests under `/tests`.

5. **DVC Versioning**  
   Creates `.dvc` files for processed data.

6. **Bias Detection**  
   Splits by slices and evaluates fairness.

7. **Generate Reports**  
   Output stored in `/reports`:
   - `schema_check.txt`
   - `test_report.txt`
   - `bias_check_summary.txt`
   - `summary_report.txt`

---

## 📊 Running Each Component Manually (Optional)

### **1️⃣ Acquire**
```
python src/acquire_fatura_data.py
```

### **2️⃣ Preprocess**
```
python src/preprocess_fatura.py
```

### **3️⃣ Schema Validation**
```
python src/validate_schema.py
```

### **4️⃣ Unit Tests**
```
pytest -q
```

### **5️⃣ Bias Check**
```
python src/bias_check.py
```

### **6️⃣ Report Generation**
```
python src/generate_report.py
```

### **7️⃣ Track data with DVC**
```
dvc add data/processed/fatura_ocr.csv
git add data/processed/fatura_ocr.csv.dvc
git commit -m "Versioned processed data"
```

---

## 📘 Logs & Where to Find Them
Logs are automatically generated inside the container:

```
/opt/airflow/logs/<dag_id>/<task_id>/
```

To view logs locally:
```
docker logs ledgerx-airflow
```

Reports are saved locally:
```
data/reports/
```

Expected files:
- `schema_check.txt`
- `test_report.txt`
- `bias_check_summary.txt`
- `summary_report.txt`

---

## 📑 Deliverables Covered (Matches IE7305 Guidelines)
✔ Data acquisition  
✔ Preprocessing  
✔ Schema validation  
✔ Unit tests  
✔ Bias detection (slice-based)  
✔ Airflow DAG orchestration  
✔ DVC versioning  
✔ Logging + report generation  
✔ Clean professional documentation  

---

## 🧪 Testing
Run all tests:
```
pytest
```

Specific test:
```
pytest tests/test_preprocess.py
```

---

## 📦 DVC Workflow
Check status:
```
dvc status
```

Push to remote (if configured):
```
dvc push
```

Pull versions:
```
dvc pull
```

---

## 🧠 Project Summary
This repository implements a **full MLOps data pipeline** tailored for the **Fatura invoice OCR dataset** and satisfies the complete academic submission requirements:

- Reproducible  
- Automated  
- Versioned  
- Validated  
- Tested  
- Orchestrated  

This will serve as the foundation for the **Stage-3 Model Pipeline** and **Stage-4 Deployment** phases.

---

## 👥 Contributors
- Lochan Enugula  
- Team LedgerX  

---

## 📄 License
This project is for academic use under the IE7305 MLOps course guidelines.

