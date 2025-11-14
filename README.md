# LedgerX – Fatura OCR Data Pipeline (Pre–Option 3 Version)

This repository contains the complete MLOps data pipeline for processing Fatura invoices. It includes preprocessing, schema validation, unit testing, bias detection, report generation, DVC tracking, and full Airflow orchestration.

This README includes everything completed BEFORE selecting Option 3. MinIO and production storage setup are NOT included here.

---

## Project Overview

The LedgerX Fatura Pipeline automates the complete OCR processing lifecycle:

1. Load raw Fatura invoice images
2. Extract OCR text and preprocess fields
3. Validate schema integrity using Great Expectations
4. Execute unit tests for preprocessing functions
5. Detect dataset bias via slice-based analysis
6. Generate reports at each stage
7. Version processed data through DVC
8. Run everything automatically using Airflow DAGs

This system is reproducible, modular, and aligned with MLOps course expectations.

---

## Pipeline Stages

### 1. OCR Preprocessing
- Extracts text from invoice images
- Normalizes numeric and string fields
- Detects blur and incomplete scans
- Outputs fatura_ocr.csv

### 2. Schema Validation
- Enforces required column presence
- Checks data types and numeric validity
- Detects missing values 
- Outputs schema_check.txt

### 3. Unit Testing
- Tests OCR preprocessing logic
- Ensures CSV format correctness
- Prevents silent data corruption
- Outputs test_report.txt

### 4. Bias Detection
- Performs dataset slicing
- Evaluates representation imbalance
- Detects bias across categories
- Outputs bias_check_summary.txt

### 5. Reporting
- summary_report.txt aggregates:
  - OCR status
  - Schema validation pass/fail
  - Unit test results
  - Bias analysis notes

### 6. DVC Tracking
- Versions processed output
- Ensures reproducible pipeline runs
- Tracks added/modified processed files

### 7. Airflow Orchestration
Airflow DAG executes all tasks in order:

1. Check OCR file presence  
2. Preprocess invoices  
3. Run schema validation  
4. Execute unit tests  
5. Run bias detection  
6. Generate reports  
7. Track data using DVC  

---

## Repository Structure
```
ledgerx-mlops-final/
├── airflow/
│   ├── dags/
│   │   └── ledgerx_fatura_pipeline.py
│   ├── reports/
│   │   ├── summary_report.txt
│   │   ├── schema_check.txt
│   │   ├── test_report.txt
│   │   └── bias_check_summary.txt
│   └── data/
│       └── processed/
│           ├── fatura_ocr.csv
│           └── fatura_ocr_cache.csv
├── src/
│   ├── preprocess_fatura.py
│   ├── schema_validation.py
│   ├── bias_detection.py
│   └── tests/
│       └── test_preprocess.py
├── dvc.yaml
├── requirements.txt
└── README.md
```
---

## How to Use This Repository

Follow these steps to run the pipeline locally.

### 1. Clone the repository

git clone https://github.com/Lochan9/ledgerx-mlops-final.git
cd ledgerx-mlops-final

### 2. Create and activate a virtual environment

python -m venv .venv
.venv\Scripts\activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Launch Airflow using Docker

docker compose up --build

After startup, open Airflow UI:
http://localhost:8081

### 5. Trigger the Pipeline

1. Open Airflow web UI
2. Enable the DAG: ledgerx_fatura_pipeline
3. Click “Trigger DAG”
4. Pipeline will run automatically through all stages

### 6. View Output Files

Processed data:
airflow/data/processed/fatura_ocr.csv

Reports:
airflow/reports/
- summary_report.txt
- schema_check.txt
- test_report.txt
- bias_check_summary.txt

### 7. Track Data Using DVC (Local)

dvc add airflow/data/processed/fatura_ocr.csv
dvc commit
dvc push  (local default cache)

---

## Completion Status (Before Option 3)

All these tasks are completed and included:
✓ OCR preprocessing  
✓ Schema validation  
✓ Unit testing  
✓ Bias detection  
✓ Report generation  
✓ Airflow DAG  
✓ DVC local tracking  
✓ Fully working end-to-end pipeline  

