"""
LedgerX - Unit Tests for Training Pipeline (Rubric Required)
============================================================

Tests included:
  1. test_train_quality_model_runs
  2. test_train_failure_model_runs
  3. test_quality_model_accuracy_threshold
  4. test_failure_model_accuracy_threshold
  5. test_feature_columns_exist_after_prepare_data

These tests validate:
  ✔ Model pipeline runs without errors
  ✔ F1 scores meet minimum threshold
  ✔ Data schema after preprocessing matches expected features
"""

import os
import joblib
import pandas as pd
from pathlib import Path

from src.training.train_all_models import train_quality, train_failure
from src.training.prepare_training_data import main as prepare_data


# --- PATHS ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

QUALITY_MODEL = MODELS_DIR / "quality_model.pkl"
FAILURE_MODEL = MODELS_DIR / "failure_model.pkl"

QUALITY_CSV = DATA_PROCESSED / "quality_training.csv"
FAILURE_CSV = DATA_PROCESSED / "failure_training.csv"


# ------------------------------------------------------------------
#  TEST 1: Ensure quality model training runs (no exceptions)
# ------------------------------------------------------------------
def test_train_quality_model_runs():
    results, best = train_quality()
    assert len(results) == 3, "Should train 3 quality models"
    assert QUALITY_MODEL.exists(), "Best quality model must be saved"


# ------------------------------------------------------------------
#  TEST 2: Ensure failure model training runs (no exceptions)
# ------------------------------------------------------------------
def test_train_failure_model_runs():
    results, best = train_failure()
    assert len(results) == 3, "Should train 3 failure models"
    assert FAILURE_MODEL.exists(), "Best failure model must be saved"


# ------------------------------------------------------------------
#  TEST 3: QUALITY model must exceed F1 threshold
# ------------------------------------------------------------------
def test_quality_model_accuracy_threshold():
    results, best = train_quality()

    assert (
        best["f1"] > 0.95
    ), f"QUALITY model F1 is too low: {best['f1']}"


# ------------------------------------------------------------------
#  TEST 4: FAILURE model must exceed F1 threshold
# ------------------------------------------------------------------
def test_failure_model_accuracy_threshold():
    results, best = train_failure()

    assert (
        best["f1"] > 0.95
    ), f"FAILURE model F1 is too low: {best['f1']}"


# ------------------------------------------------------------------
#  TEST 5: Confirm feature columns exist after preparation
# ------------------------------------------------------------------
def test_feature_columns_exist_after_prepare_data():
    """
    Ensures prepare_training_data.py always outputs consistent schema.
    """

    # Run data preparation
    prepare_data()

    # Load processed CSVs
    df_q = pd.read_csv(QUALITY_CSV)
    df_f = pd.read_csv(FAILURE_CSV)

    # Required core features (QUALITY)
    required_quality_cols = [
        "blur_score",
        "contrast_score",
        "ocr_confidence",
        "num_missing_fields",
        "has_critical_missing",
        "num_pages",
        "file_size_kb",
        "vendor_freq",
        "label_quality_bad",
    ]

    for col in required_quality_cols:
        assert col in df_q.columns, f"Missing column in quality CSV: {col}"

    # Required core features (FAILURE)
    required_failure_cols = [
        "blur_score",
        "contrast_score",
        "ocr_confidence",
        "num_missing_fields",
        "has_critical_missing",
        "num_pages",
        "file_size_kb",
        "vendor_freq",
        "total_amount",
        "vendor_name_length",
        "invoice_number_present",
        "amount_bucket",
        "label_failure",
    ]

    for col in required_failure_cols:
        assert col in df_f.columns, f"Missing column in failure CSV: {col}"

