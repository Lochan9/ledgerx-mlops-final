"""
LedgerX – Inference Service (Core Engine)
Corrected version using DataFrames for sklearn pipelines.
"""

import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("ledgerx_inference")

# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

QUALITY_MODEL_PATH = MODELS_DIR / "quality_model.pkl"
FAILURE_MODEL_PATH = MODELS_DIR / "failure_model.pkl"

# -------------------------------------------------------------------
# LOAD MODELS
# -------------------------------------------------------------------
logger.info("[LOAD] Loading trained models...")

quality_model = joblib.load(QUALITY_MODEL_PATH)
failure_model = joblib.load(FAILURE_MODEL_PATH)

logger.info("[LOAD] Quality model loaded → OK")
logger.info("[LOAD] Failure model loaded → OK")

# -------------------------------------------------------------------
# Feature Engineering Helpers
# -------------------------------------------------------------------
def amount_bucket(value):
    try:
        v = float(value)
    except:
        return "unknown"
    if v < 50:
        return "low"
    elif v < 200:
        return "medium"
    return "high"


def compute_missing_fields(f):
    mandatory = ["invoice_number", "invoice_date", "total_amount", "vendor_name"]
    missing = 0

    for col in mandatory:
        v = f.get(col, None)
        if v is None or str(v).strip() == "":
            missing += 1
            continue
        if col == "total_amount":
            try:
                if float(v) <= 0:
                    missing += 1
            except:
                missing += 1

    return missing


def compute_has_critical_missing(f):
    critical = ["invoice_number", "invoice_date", "vendor_name", "total_amount", "currency"]
    missing = 0

    for col in critical:
        v = f.get(col, None)
        if v is None or str(v).strip() == "":
            missing += 1
            continue
        if col == "total_amount":
            try:
                if float(v) <= 0:
                    missing += 1
            except:
                missing += 1

    return 1 if missing > 0 else 0


# -------------------------------------------------------------------
# MAIN INFERENCE FUNCTION
# -------------------------------------------------------------------
def predict_invoice(features: dict):

    logger.info("=======================================================")
    logger.info("           LEDGERX INFERENCE REQUEST RECEIVED          ")
    logger.info("=======================================================")

    logger.info("[STEP 1] Raw input features:")
    for k, v in features.items():
        logger.info(f"    - {k}: {v}")

    required = [
        "blur_score", "contrast_score", "ocr_confidence",
        "file_size_kb", "vendor_name", "vendor_freq",
        "total_amount", "invoice_number", "invoice_date", "currency"
    ]

    for f in required:
        if f not in features:
            raise ValueError(f"Missing required field: {f}")

    # -------------------------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------------------------
    logger.info("[STEP 2] Computing engineered features...")

    num_missing = compute_missing_fields(features)
    crit_missing = compute_has_critical_missing(features)

    inv_num_present = 1 if str(features["invoice_number"]).strip() != "" else 0
    vendor_len = len(str(features["vendor_name"]))
    amt_bucket = amount_bucket(features["total_amount"])
    num_pages = 1  # default assumption

    quality_row = {
        "blur_score": float(features["blur_score"]),
        "contrast_score": float(features["contrast_score"]),
        "ocr_confidence": float(features["ocr_confidence"]),
        "num_missing_fields": num_missing,
        "has_critical_missing": crit_missing,
        "num_pages": num_pages,
        "file_size_kb": float(features["file_size_kb"]),
        "vendor_freq": float(features["vendor_freq"]),
    }

    failure_row = {
        **quality_row,
        "total_amount": float(features["total_amount"]),
        "invoice_number_present": inv_num_present,
        "vendor_name_length": vendor_len,
        "amount_bucket": amt_bucket,
    }

    logger.info("[STEP 2] Engineered features:")
    for k, v in failure_row.items():
        logger.info(f"    - {k}: {v}")

    # -------------------------------------------------------------------
    # DataFrame Conversion (IMPORTANT)
    # -------------------------------------------------------------------
    quality_df = pd.DataFrame([quality_row])
    failure_df = pd.DataFrame([failure_row])

    # -------------------------------------------------------------------
    # PREDICT QUALITY
    # -------------------------------------------------------------------
    logger.info("[STEP 3] Running Quality Model prediction...")

    yq_pred = int(quality_model.predict(quality_df)[0])
    yq_proba = float(quality_model.predict_proba(quality_df)[0][1])

    logger.info(f"[QUALITY] Prediction = {yq_pred}, Probability = {yq_proba:.4f}")

    # -------------------------------------------------------------------
    # PREDICT FAILURE
    # -------------------------------------------------------------------
    logger.info("[STEP 4] Running Failure Model prediction...")

    yf_pred = int(failure_model.predict(failure_df)[0])
    yf_proba = float(failure_model.predict_proba(failure_df)[0][1])

    logger.info(f"[FAILURE] Prediction = {yf_pred}, Probability = {yf_proba:.4f}")

    # -------------------------------------------------------------------
    # RULE-BASED WARNINGS
    # -------------------------------------------------------------------
    warnings = []
    if crit_missing:
        warnings.append("Missing critical financial fields")
    if num_missing > 1:
        warnings.append("Multiple important invoice fields missing")
    if float(features["ocr_confidence"]) < 0.65:
        warnings.append("Low OCR confidence")
    if float(features["blur_score"]) < 40:
        warnings.append("Image is too blurry")

    logger.info(f"[WARNINGS] {warnings}")

    # -------------------------------------------------------------------
    # FINAL RESPONSE
    # -------------------------------------------------------------------
    response = {
        "quality_bad": yq_pred,
        "failure_risk": yf_pred,
        "quality_probability": yq_proba,
        "failure_probability": yf_proba,
        "engineered_features": failure_row,
        "warnings": warnings
    }

    logger.info("[DONE] Inference completed.")
    logger.info("=======================================================")

    return response
