"""
LedgerX - Model Evaluation & Interpretability (FINAL VERSION, WINDOWS & DOCKER SAFE)
====================================================================================

This script performs:

  ✔ ROC curves for best selected models
  ✔ Permutation importance (pipeline-safe)
  ✔ SHAP explainability (optional)
  ✔ Model card generation

Outputs saved to: /reports/
"""

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Headless backend for Airflow, Docker, pytest, Windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# Try using SHAP
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# -------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ledgerx_evaluate_models")


# -------------------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

QUALITY_DATA_PATH = DATA_PROCESSED / "quality_training.csv"
FAILURE_DATA_PATH = DATA_PROCESSED / "failure_training.csv"

QUALITY_MODEL_PATH = MODELS_DIR / "quality_model.pkl"
FAILURE_MODEL_PATH = MODELS_DIR / "failure_model.pkl"

LEADERBOARD_PATH = REPORTS_DIR / "model_leaderboard.json"
MODEL_CARD_PATH = REPORTS_DIR / "model_card.md"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# ROC CURVE
# -------------------------------------------------------------------
def plot_roc(y_true, y_proba, title, out_path):
    logger.info(f"[ROC] Starting ROC computation → {title}")

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    logger.info(f"[ROC] FPR/TPR computed. AUC={auc:.4f}")

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    logger.info(f"[ROC] Saved ROC curve → {out_path}")
    return auc


# -------------------------------------------------------------------
# PERMUTATION IMPORTANCE (pipeline-safe)
# -------------------------------------------------------------------
def compute_perm_importance(
    pipeline,
    X_test,
    y_test,
    feature_names,
    task_name,
    out_png,
    out_txt,
):
    logger.info(f"[PERM] Starting permutation importance for {task_name}...")
    logger.info(f"[PERM] Test set size: {len(X_test)} rows")

    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=12,
        random_state=42,
        n_jobs=-1,
    )

    means = result.importances_mean
    stds = result.importances_std
    sorted_idx = np.argsort(means)[::-1]

    # Save text summary
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Permutation Importance — {task_name}\n")
        f.write("=" * 70 + "\n\n")

        for idx in sorted_idx:
            f.write(
                f"{feature_names[idx]:35s} | mean={means[idx]:.6f} | std={stds[idx]:.6f}\n"
            )

    logger.info(f"[PERM] Text summary saved → {out_txt}")

    # Plot top-K
    top_k = min(15, len(sorted_idx))
    plt.figure(figsize=(8, 6))
    plt.barh(
        [feature_names[i] for i in sorted_idx[:top_k]][::-1],
        means[sorted_idx[:top_k]][::-1],
    )
    plt.title(f"Permutation Importance (Top {top_k}) — {task_name}")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

    logger.info(f"[PERM] Plot saved → {out_png}")


# -------------------------------------------------------------------
# SHAP (quality model only)
# -------------------------------------------------------------------
def run_shap(quality_pipeline, X_train, X_test, feature_names):
    if not HAS_SHAP:
        logger.warning("[SHAP] SHAP is not installed. Skipping.")
        return

    logger.info("[SHAP] Running SHAP explainability for QUALITY model...")

    # Extract transformer + classifier
    pre = quality_pipeline.named_steps["pre"]
    clf = quality_pipeline.named_steps["clf"]

    logger.info("[SHAP] Sampling up to 500 rows...")
    sample_n = min(500, len(X_test))
    X_sample = X_test.sample(n=sample_n, random_state=42)

    logger.info("[SHAP] Transforming sample features...")
    X_trans = pre.transform(X_sample)

    logger.info("[SHAP] Creating TreeExplainer...")
    explainer = shap.TreeExplainer(clf)

    logger.info("[SHAP] Computing SHAP values...")
    shap_vals = explainer.shap_values(X_trans)

    # CatBoost returns list for multi-class, we need index 1
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    out_path = REPORTS_DIR / "quality_shap_summary.png"
    shap.summary_plot(shap_vals, X_trans, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    logger.info(f"[SHAP] Saved SHAP summary → {out_path}")


# -------------------------------------------------------------------
# MODEL CARD GENERATOR
# -------------------------------------------------------------------
def generate_model_card(leaderboard):
    logger.info("[CARD] Generating model card...")

    qbest = leaderboard["best_models"]["quality"]
    fbest = leaderboard["best_models"]["failure"]

    with open(MODEL_CARD_PATH, "w", encoding="utf-8") as f:
        f.write("# LedgerX Model Card\n")
        f.write("_Automatically generated by evaluate_models.py_\n\n")

        f.write("## Best Models\n")
        f.write(f"- **Quality Model** → `{qbest}`\n")
        f.write(f"- **Failure Model** → `{fbest}`\n\n")

        f.write("## Evaluation Artifacts\n")
        f.write("- quality_best_roc.png\n")
        f.write("- failure_best_roc.png\n")
        f.write("- quality_perm_importance.png\n")
        f.write("- failure_perm_importance.png\n")
        f.write("- quality_shap_summary.png (if SHAP installed)\n\n")

    logger.info(f"[CARD] Model card saved → {MODEL_CARD_PATH}")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    logger.info("===================================================")
    logger.info("       LEDGERX MODEL EVALUATION — START")
    logger.info("===================================================")

    start = time.time()

    # -----------------------------
    # Load Data
    # -----------------------------
    logger.info("[LOAD] Loading processed CSVs...")
    df_q = pd.read_csv(QUALITY_DATA_PATH)
    df_f = pd.read_csv(FAILURE_DATA_PATH)

    # QUALITY split
    logger.info("[SPLIT] Splitting QUALITY data...")
    yq = df_q["label_quality_bad"].astype(int)
    Xq = df_q.drop(columns=["label_quality_bad", "file_name"])
    q_features = list(Xq.columns)

    Xq_train, Xq_test, yq_train, yq_test = train_test_split(
        Xq, yq, test_size=0.2, stratify=yq, random_state=42
    )

    # FAILURE split
    logger.info("[SPLIT] Splitting FAILURE data...")
    yf = df_f["label_failure"].astype(int)
    Xf = df_f.drop(columns=["label_failure", "file_name"])
    f_features = list(Xf.columns)

    Xf_train, Xf_test, yf_train, yf_test = train_test_split(
        Xf, yf, test_size=0.2, stratify=yf, random_state=42
    )

    # -----------------------------
    # Load Models
    # -----------------------------
    logger.info("[LOAD] Loading trained models...")
    q_model = joblib.load(QUALITY_MODEL_PATH)
    f_model = joblib.load(FAILURE_MODEL_PATH)
    logger.info("[LOAD] Models loaded successfully.")

    # -----------------------------
    # ROC Curves
    # -----------------------------
    logger.info("[EVAL] Computing ROC curves...")

    yq_proba = q_model.predict_proba(Xq_test)[:, 1]
    plot_roc(
        yq_test,
        yq_proba,
        "QUALITY — Best Model ROC",
        REPORTS_DIR / "quality_best_roc.png",
    )

    yf_proba = f_model.predict_proba(Xf_test)[:, 1]
    plot_roc(
        yf_test,
        yf_proba,
        "FAILURE — Best Model ROC",
        REPORTS_DIR / "failure_best_roc.png",
    )

    # -----------------------------
    # Permutation Importance
    # -----------------------------
    logger.info("[EVAL] Starting permutation importance...")

    compute_perm_importance(
        q_model,
        Xq_test,
        yq_test,
        q_features,
        "QUALITY",
        REPORTS_DIR / "quality_perm_importance.png",
        REPORTS_DIR / "quality_perm_importance.txt",
    )

    compute_perm_importance(
        f_model,
        Xf_test,
        yf_test,
        f_features,
        "FAILURE",
        REPORTS_DIR / "failure_perm_importance.png",
        REPORTS_DIR / "failure_perm_importance.txt",
    )

    # -----------------------------
    # SHAP Explainability
    # -----------------------------
    logger.info("[EVAL] Running SHAP explainability...")
    run_shap(q_model, Xq_train, Xq_test, q_features)

    # -----------------------------
    # MODEL CARD
    # -----------------------------
    logger.info("[CARD] Loading leaderboard & generating model card...")
    leaderboard = json.load(open(LEADERBOARD_PATH, "r", encoding="utf-8"))
    generate_model_card(leaderboard)

    # DONE
    logger.info(f"[DONE] Evaluation completed in {time.time() - start:.2f}s")
    logger.info("===================================================")
    logger.info("       LEDGERX MODEL EVALUATION — END")
    logger.info("===================================================")


if __name__ == "__main__":
    main()
