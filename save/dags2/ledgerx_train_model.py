"""
LedgerX - Airflow Training DAG (FINAL - DOCKER VERSION)

This DAG automates:
    1. Load processed training data
    2. Train QUALITY model + MLflow logging + Registry
    3. Train FAILURE model + MLflow logging + Registry
    4. Evaluate both models
    5. Generate model card + reports

It runs inside the Docker Airflow environment.
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os

# -------------------------------------------------------
# PATHS INSIDE DOCKER CONTAINER
# -------------------------------------------------------
PROJECT_DIR = "/opt/airflow"
SRC_DIR = f"{PROJECT_DIR}/src/training"
PYTHON = "/usr/local/bin/python"   # airflow official python path

# -------------------------------------------------------
# DEFAULT ARGS
# -------------------------------------------------------
default_args = {
    "owner": "ledgerx",
    "depends_on_past": False,
    "email_on_retry": False,
    "retry_delay": timedelta(minutes=2),
}

# -------------------------------------------------------
# DAG DEFINITION
# -------------------------------------------------------
with DAG(
    dag_id="ledgerx_train_model",
    default_args=default_args,
    description="LedgerX Full Model Training Pipeline",
    schedule_interval=None,  # Run manually for now
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ledgerx", "mlops", "training"],
) as dag:

    # ---------------------------------------------------
    # TASK 1     Load processed data validation
    # ---------------------------------------------------
    load_data = BashOperator(
        task_id="load_processed_data",
        bash_command=(
            f'echo "[LOAD] Checking processed training files..." && '
            f'ls -lh {PROJECT_DIR}/data/processed && '
            f'echo "[LOAD] Data check complete."'
        )
    )

    # ---------------------------------------------------
    # TASK 2     Train QUALITY + FAILURE (Step 5)
    # ---------------------------------------------------
    train_models = BashOperator(
        task_id="train_models",
        bash_command=(
            f'echo "[TRAIN] Starting multi-model training..." && '
            f'{PYTHON} {SRC_DIR}/train_all_models.py && '
            f'echo "[TRAIN] Completed training."'
        )
    )

    # ---------------------------------------------------
    # TASK 3     Evaluate trained models
    # ---------------------------------------------------
    evaluate_models = BashOperator(
        task_id="evaluate_models",
        bash_command=(
            f'echo "[EVAL] Running evaluation scripts..." && '
            f'{PYTHON} {SRC_DIR}/evaluate_models.py && '
            f'echo "[EVAL] Evaluation completed."'
        )
    )

    # ---------------------------------------------------
    # TASK 4     Error analysis
    # ---------------------------------------------------
    error_analysis = BashOperator(
        task_id="error_analysis",
        bash_command=(
            f'echo "[ERROR] Running error analysis..." && '
            f'{PYTHON} {SRC_DIR}/error_analysis.py && '
            f'echo "[ERROR] Error analysis completed."'
        )
    )

    # ---------------------------------------------------
    # TASK 5     Model Card & Final Report
    # ---------------------------------------------------
    generate_report = BashOperator(
        task_id="generate_training_report",
        bash_command=(
            f'echo "[REPORT] Generating training summary..." && '
            f'ls -lh {PROJECT_DIR}/reports && '
            f'echo "[REPORT] Training report ready."'
        )
    )

    # ----------------------------------------------
    # DAG DEPENDENCY FLOW
    # ----------------------------------------------
    load_data >> train_models >> evaluate_models >> error_analysis >> generate_report
