"""
LedgerX - Error Analysis Script (FINAL, WINDOWS/DOCKER SAFE)
============================================================

This script:

  ✔ Loads processed training data (quality & failure)
  ✔ Recreates the same train/test split as training
  ✔ Loads the best trained models (quality & failure)
  ✔ Computes confusion matrices
  ✔ Saves false positives / false negatives into CSVs
  ✔ Performs slice analysis by:
        - blur quality (blur_score)
        - OCR quality (ocr_confidence)
        - vendor frequency (vendor_freq)
  ✔ Writes a human-readable text summary

Outputs:
  reports/error_analysis/
    - quality_false_positives.csv
    - quality_false_negatives.csv
    - failure_false_positives.csv
    - failure_false_negatives.csv
    - quality_slice_blur.csv
    - quality_slice_ocr.csv
    - quality_slice_vendor.csv
    - failure_slice_blur.csv
    - failure_slice_ocr.csv
    - failure_slice_vendor.csv
    - error_analysis_summary.txt
"""

import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Use headless backend for safety (no Tk / GUI)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401 (kept if you later add plots)

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# -------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ledgerx_error_analysis")


# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
ERROR_DIR = REPORTS_DIR / "error_analysis"

QUALITY_DATA_PATH = DATA_PROCESSED / "quality_training.csv"
FAILURE_DATA_PATH = DATA_PROCESSED / "failure_training.csv"

QUALITY_MODEL_PATH = MODELS_DIR / "quality_model.pkl"
FAILURE_MODEL_PATH = MODELS_DIR / "failure_model.pkl"

ERROR_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# SLICE HELPERS
# -------------------------------------------------------------------
def add_slices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add slice columns:
      - blur_slice  (from blur_score)
      - ocr_slice   (from ocr_confidence)
      - vendor_slice (from vendor_freq)
    """
    df = df.copy()

    # Blur: low / medium / high
    if "blur_score" in df.columns:
        df["blur_slice"] = pd.cut(
            df["blur_score"],
            bins=[-np.inf, 30, 60, np.inf],
            labels=["low_blur", "medium_blur", "high_blur"],
        )
    else:
        df["blur_slice"] = "unknown"

    # OCR: low / medium / high
    if "ocr_confidence" in df.columns:
        df["ocr_slice"] = pd.cut(
            df["ocr_confidence"],
            bins=[-np.inf, 0.6, 0.8, np.inf],
            labels=["low_ocr", "medium_ocr", "high_ocr"],
        )
    else:
        df["ocr_slice"] = "unknown"

    # Vendor: low / medium / high frequency
    if "vendor_freq" in df.columns:
        df["vendor_slice"] = pd.cut(
            df["vendor_freq"],
            bins=[-np.inf, 0.001, 0.01, np.inf],
            labels=["rare_vendor", "medium_vendor", "frequent_vendor"],
        )
    else:
        df["vendor_slice"] = "unknown"

    return df


def save_slice_stats(df: pd.DataFrame, slice_col: str, out_path: Path, task_name: str):
    """
    Compute accuracy per slice and write to CSV.
    df must contain:
      - 'correct' (1 if prediction == label, else 0)
      - slice_col column (categorical)
    """
    logger.info(f"[SLICE] Computing slice stats by {slice_col} for {task_name}...")

    # observed=False to silence the FutureWarning in pandas
    grouped = (
        df.groupby(slice_col, observed=False)["correct"]
        .agg(count="count", accuracy="mean")
        .reset_index()
    )

    grouped.to_csv(out_path, index=False)
    logger.info(f"[SLICE] Saved slice analysis by {slice_col} → {out_path}")


# -------------------------------------------------------------------
# CORE ANALYSIS FUNCTION
# -------------------------------------------------------------------
def analyze_task(
    task_name: str,
    df: pd.DataFrame,
    label_col: str,
    model_path: Path,
):
    """
    Generic error-analysis for a binary classifier.

    - task_name: "quality" or "failure"
    - df: full dataset (processed training CSV)
    - label_col: column name of label
    - model_path: path to .pkl model
    """
    logger.info(f"[TASK] Starting error analysis for: {task_name.upper()}")

    # ----------------------------------------
    # Split (must match training logic)
    # ----------------------------------------
    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col, "file_name"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    logger.info(f"[SPLIT {task_name}] Train rows: {len(X_train)}, Test rows: {len(X_test)}")

    # ----------------------------------------
    # Load model
    # ----------------------------------------
    logger.info(f"[LOAD {task_name}] Loading model from: {model_path}")
    model = joblib.load(model_path)
    logger.info(f"[LOAD {task_name}] Model loaded.")

    # ----------------------------------------
    # Predict on test
    # ----------------------------------------
    logger.info(f"[PRED {task_name}] Predicting on test set...")
    y_pred = model.predict(X_test)

    # ----------------------------------------
    # Confusion matrix
    # ----------------------------------------
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    logger.info(f"[CM {task_name}] Confusion matrix:\n{cm}")

    # ----------------------------------------
    # False Positives / False Negatives
    # ----------------------------------------
    logger.info(f"[FP/FN {task_name}] Extracting FP/FN rows...")

    test_df = X_test.copy()
    test_df["true_label"] = y_test
    test_df["pred_label"] = y_pred
    test_df["correct"] = (test_df["true_label"] == test_df["pred_label"]).astype(int)

    fp = test_df[(test_df["true_label"] == 0) & (test_df["pred_label"] == 1)]
    fn = test_df[(test_df["true_label"] == 1) & (test_df["pred_label"] == 0)]

    fp_out = ERROR_DIR / f"{task_name}_false_positives.csv"
    fn_out = ERROR_DIR / f"{task_name}_false_negatives.csv"

    fp.to_csv(fp_out, index=False)
    fn.to_csv(fn_out, index=False)

    logger.info(
        f"[FP/FN {task_name}] Saved FP/FN → {fp_out.name}, {fn_out.name} "
        f"(FP={len(fp)}, FN={len(fn)})"
    )

    # ----------------------------------------
    # Slice analysis
    # ----------------------------------------
    logger.info(f"[SLICE {task_name}] Building slice-level analysis DataFrame...")
    sliced_df = add_slices(test_df)

    # blur_slice
    save_slice_stats(
        sliced_df,
        "blur_slice",
        ERROR_DIR / f"{task_name}_slice_blur.csv",
        task_name,
    )

    # ocr_slice
    save_slice_stats(
        sliced_df,
        "ocr_slice",
        ERROR_DIR / f"{task_name}_slice_ocr.csv",
        task_name,
    )

    # vendor_slice
    save_slice_stats(
        sliced_df,
        "vendor_slice",
        ERROR_DIR / f"{task_name}_slice_vendor.csv",
        task_name,
    )

    return {
        "task": task_name,
        "cm": cm,
        "fp_count": int(len(fp)),
        "fn_count": int(len(fn)),
    }


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    logger.info("===================================================")
    logger.info("         LEDGERX ERROR ANALYSIS - START")
    logger.info("===================================================")

    start_time = time.time()

    # ----------------------------------------
    # Load Data
    # ----------------------------------------
    logger.info("[LOAD] Loading processed CSVs...")
    df_quality = pd.read_csv(QUALITY_DATA_PATH)
    df_failure = pd.read_csv(FAILURE_DATA_PATH)
    logger.info("[LOAD] Data loaded successfully.")

    # ----------------------------------------
    # ANALYZE QUALITY
    # ----------------------------------------
    logger.info("[QUALITY] Starting analysis...")
    quality_summary = analyze_task(
        task_name="quality",
        df=df_quality,
        label_col="label_quality_bad",
        model_path=QUALITY_MODEL_PATH,
    )

    # ----------------------------------------
    # ANALYZE FAILURE
    # ----------------------------------------
    logger.info("[FAILURE] Starting analysis...")
    failure_summary = analyze_task(
        task_name="failure",
        df=df_failure,
        label_col="label_failure",
        model_path=FAILURE_MODEL_PATH,
    )

    # ----------------------------------------
    # WRITE NARRATIVE SUMMARY
    # ----------------------------------------
    summary_path = ERROR_DIR / "error_analysis_summary.txt"
    logger.info("[WRITE] Generating narrative summary...")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("LedgerX Error Analysis Summary\n")
        f.write("=================================\n\n")

        def write_block(name: str, s: dict):
            f.write(f"=== {name.upper()} MODEL ===\n")
            f.write("Confusion Matrix [rows=true, cols=pred]:\n")
            cm = s["cm"]
            f.write(f"TN={cm[0,0]}  FP={cm[0,1]}\n")
            f.write(f"FN={cm[1,0]}  TP={cm[1,1]}\n\n")
            f.write(f"False Positives: {s['fp_count']}\n")
            f.write(f"False Negatives: {s['fn_count']}\n")
            f.write("\n\n")

        write_block("quality", quality_summary)
        write_block("failure", failure_summary)

        f.write("Slice analysis files (per-task):\n")
        f.write("  - *_slice_blur.csv (blur_score buckets)\n")
        f.write("  - *_slice_ocr.csv (ocr_confidence buckets)\n")
        f.write("  - *_slice_vendor.csv (vendor frequency buckets)\n")

    logger.info(f"[WRITE] Summary saved → {summary_path}")
    logger.info(f"[DONE] Error analysis finished in {time.time() - start_time:.2f}s")
    logger.info("===================================================")
    logger.info("         LEDGERX ERROR ANALYSIS - END")
    logger.info("===================================================")


if __name__ == "__main__":
    main()
