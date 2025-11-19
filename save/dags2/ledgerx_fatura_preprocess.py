"""
DAG: ledgerx_fatura_preprocess
Purpose: Run OCR preprocessing for FATURA dataset (skip if already processed)
"""

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from pathlib import Path
import subprocess
import os

# --- DAG Metadata ---
default_args = {
    'owner': 'ledgerx',
    'retries': 1,
}

dag = DAG(
    dag_id='ledgerx_fatura_preprocess',
    default_args=default_args,
    description='Preprocess FATURA data (OCR + skip if cached)',
    schedule_interval=None,
    start_date=datetime(2025, 11, 1),
    catchup=False,
)

# --- Python callable to skip OCR if CSV exists ---
def run_preprocess_if_needed():
    processed_csv = Path("/opt/airflow/data/processed/fatura_ocr.csv")

    if processed_csv.exists():
        print("✅ OCR already done. Skipping preprocessing stage.")
    else:
        print("⚙️ OCR not found — running preprocess_fatura.py ...")
        cmd = ["python", "src/stages/preprocess_fatura.py"]
        result = subprocess.run(cmd, cwd="/opt/airflow", check=False)
        if result.returncode == 0:
            print("✅ OCR completed successfully.")
        else:
            raise RuntimeError("❌ OCR script failed.")

# --- Airflow Tasks ---
check_or_run_preprocess = PythonOperator(
    task_id='run_preprocess_fatura',
    python_callable=run_preprocess_if_needed,
    dag=dag,
)

# optional cleanup or validation after OCR
validate_output = BashOperator(
    task_id='validate_ocr_output',
    bash_command="test -f /opt/airflow/data/processed/fatura_ocr.csv && echo '✅ OCR file verified.'",
    dag=dag,
)

# DAG flow
check_or_run_preprocess >> validate_output
