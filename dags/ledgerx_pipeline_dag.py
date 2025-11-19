"""
===============================================================
 LEDGERX ENTERPRISE AIRFLOW PIPELINE (v3.0)
===============================================================

Pipeline Stages:
  1. acquire_data
  2. preprocess_fatura_enterprise   <-- NEW ENTERPRISE MODULE
  3. prepare_training_data
  4. train_all_models
  5. evaluate_models
  6. error_analysis
  7. generate_summary_report

This is the full enterprise-level DAG for the entire LedgerX
platform, used for:
   • Automated retraining
   • Model registry updates (MLflow)
   • Reporting (ROC, SHAP, Permutation)
===============================================================
"""

from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "ledgerx",
    "start_date": datetime(2023, 1, 1),
    "retries": 1
}

with DAG(
    dag_id="ledgerx_fatura_pipeline_enterprise",
    default_args=default_args,
    schedule_interval=None,
    catchup=False
):

    # ---------------------------------------------------------
    # 1) ACQUIRE RAW DATA (OCR + STRUCTURED)
    # ---------------------------------------------------------
    acquire_data = BashOperator(
        task_id="acquire_data",
        bash_command="python /opt/airflow/src/stages/data_acquisition_fatura.py"
    )

    # ---------------------------------------------------------
    # 2) ENTERPRISE PREPROCESSING
    # ---------------------------------------------------------
    preprocess_enterprise = BashOperator(
        task_id="preprocess_enterprise",
        bash_command="python /opt/airflow/src/stages/preprocess_fatura_enterprise.py"
    )

    # ---------------------------------------------------------
    # 3) PREPARE TRAINING DATA (reads enterprise preprocessed data)
    # ---------------------------------------------------------
    prepare_training = BashOperator(
        task_id="prepare_training_data",
        bash_command="python /opt/airflow/src/training/prepare_training_data.py"
    )

    # ---------------------------------------------------------
    # 4) TRAIN MODELS (quality + failure)
    # ---------------------------------------------------------
    train_models = BashOperator(
        task_id="train_all_models",
        bash_command="python /opt/airflow/src/training/train_all_models.py"
    )

    # ---------------------------------------------------------
    # 5) MODEL EVALUATION (ROC, SHAP, permutation)
    # ---------------------------------------------------------
    evaluate_models = BashOperator(
        task_id="evaluate_models",
        bash_command="python /opt/airflow/src/training/evaluate_models.py"
    )

    # ---------------------------------------------------------
    # 6) ERROR ANALYSIS (FP/FN, slice analysis)
    # ---------------------------------------------------------
    error_analysis = BashOperator(
        task_id="error_analysis",
        bash_command="python /opt/airflow/src/training/error_analysis.py"
    )

    # ---------------------------------------------------------
    # 7) SUMMARY REPORT (text file to /reports)
    # ---------------------------------------------------------
    summary_report = BashOperator(
        task_id="generate_summary_report",
        bash_command="python /opt/airflow/src/reporting/generate_summary_report.py"
    )

    # ---------------------------------------------------------
    # PIPELINE ORDER
    # ---------------------------------------------------------
    acquire_data >> preprocess_enterprise >> prepare_training >> train_models >> evaluate_models >> error_analysis >> summary_report
