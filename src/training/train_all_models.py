"""
LedgerX - FULL Multi-Model Training & Model Selection (FINAL VERSION, WINDOWS & CI SAFE)

Trains SIX models total:

QUALITY MODELS (label_quality_bad):
    - Logistic Regression
    - Random Forest
    - CatBoostClassifier

FAILURE MODELS (label_failure):
    - Logistic Regression
    - Random Forest
    - CatBoostClassifier

Outputs:
    - models/quality_model.pkl
    - models/failure_model.pkl
    - reports/model_leaderboard.json
    - reports/model_report.txt
    - Confusion matrix PNGs + classification reports (per model)
    - MLflow runs + local Model Registry entries:
        - ledgerx_quality_model
        - ledgerx_failure_model
"""

import json
import time
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# IMPORTANT: headless backend for tests / Docker / Airflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Logging Setup
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ledgerx_train_all_models")


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

QUALITY_DATA_PATH = DATA_PROCESSED / "quality_training.csv"
FAILURE_DATA_PATH = DATA_PROCESSED / "failure_training.csv"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# MLflow Configuration (local file-based tracking + registry)
# -------------------------------------------------------------------
MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"
mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_DIR}")
mlflow.set_experiment("ledgerx_multi_model_training")


# -------------------------------------------------------------------
# Confusion Matrix Plot Helper (headless-safe)
# -------------------------------------------------------------------
def plot_confusion_matrix(cm, classes, title, out_path: Path):
    """Create and save a confusion matrix plot (no GUI backend)."""
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------
# Generic single model trainer (used for both QUALITY & FAILURE)
# -------------------------------------------------------------------
def train_one_model(
    X_train,
    X_test,
    y_train,
    y_test,
    model_name: str,
    pipeline: Pipeline,
    task_name: str,
):
    """
    Train a single model with MLflow logging, confusion matrix, and
    classification report.
    """
    start = time.time()
    logger.info("--------------------------------------------------")
    logger.info(f"[{task_name.upper()}] TRAINING MODEL: {model_name}")
    logger.info("--------------------------------------------------")

    with mlflow.start_run(run_name=f"{task_name}_{model_name}"):
        # Basic run metadata
        mlflow.log_param("task", task_name)
        mlflow.log_param("model", model_name)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("num_features", X_train.shape[1])

        logger.info(f"[{task_name}/{model_name}] Fitting pipeline...")
        pipeline.fit(X_train, y_train)

        logger.info(f"[{task_name}/{model_name}] Predicting...")
        y_pred = pipeline.predict(X_test)

        y_proba = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_proba = pipeline.predict_proba(X_test)[:, 1]
            except Exception:
                y_proba = None

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        auc = None
        if y_proba is not None:
            try:
                auc = roc_auc_score(y_test, y_proba)
            except Exception:
                auc = None

        # MLflow metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        if auc is not None:
            mlflow.log_metric("roc_auc", auc)

        logger.info(
            f"[RESULT {task_name}/{model_name}] "
            f"Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} AUC={auc}"
        )

        # Confusion matrix (saved as PNG so tests / CI are safe)
        cm = confusion_matrix(y_test, y_pred)
        cm_path = REPORTS_DIR / f"{task_name}_{model_name}_cm.png"
        plot_confusion_matrix(
            cm,
            classes=["0", "1"],
            title=f"{task_name.upper()} - {model_name}",
            out_path=cm_path,
        )
        mlflow.log_artifact(str(cm_path))

        # Classification report
        clf_report = classification_report(y_test, y_pred, digits=4)
        report_path = REPORTS_DIR / f"{task_name}_{model_name}_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(clf_report)
        mlflow.log_artifact(str(report_path))

        # Log full pipeline as artifact
        mlflow.sklearn.log_model(pipeline, artifact_path=f"{task_name}_{model_name}")

    logger.info(
        f"[{task_name}/{model_name}] Training completed in {time.time() - start:.2f}s"
    )

    return {
        "task": task_name,
        "model_name": model_name,
        "pipeline": pipeline,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
    }


# -------------------------------------------------------------------
# QUALITY MODEL TRAINING (label_quality_bad)
# -------------------------------------------------------------------
def train_quality():
    logger.info("===================================================")
    logger.info("   TRAINING QUALITY MODEL  (label_quality_bad)")
    logger.info("===================================================")

    df = pd.read_csv(QUALITY_DATA_PATH)

    y = df["label_quality_bad"].astype(int)
    X = df.drop(columns=["label_quality_bad", "file_name"])

    numeric_features = list(X.columns)

    logger.info(f"[QUALITY] Features: {numeric_features}")
    logger.info(f"[QUALITY] Target Distribution: {y.value_counts().to_dict()}")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
        ]
    )

    models = {
        "logreg": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
            solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "catboost": CatBoostClassifier(
            depth=6,
            learning_rate=0.08,
            iterations=300,
            loss_function="Logloss",
            eval_metric="F1",
            verbose=False,
            random_seed=42,
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = []
    for name, model in models.items():
        pipeline = Pipeline(steps=[("pre", preprocessor), ("clf", model)])
        r = train_one_model(X_train, X_test, y_train, y_test, name, pipeline, "quality")
        results.append(r)

    best = max(results, key=lambda r: r["f1"])
    best_pipeline = best["pipeline"]
    best_name = best["model_name"]

    best_path = MODELS_DIR / "quality_model.pkl"
    joblib.dump(best_pipeline, best_path)
    logger.info(
        f"[QUALITY] Best Model: {best_name} saved to {best_path}"
    )

    # MLflow Model Registry registration (local file-based)
    logger.info("[MLFLOW] Registering QUALITY model...")
    with mlflow.start_run(run_name="register_quality_best_model"):
        mlflow.log_artifact(str(best_path), artifact_path="artifacts")
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="model",
            registered_model_name="ledgerx_quality_model",
        )
    logger.info("[MLFLOW] QUALITY model registered.")

    return results, best


# -------------------------------------------------------------------
# FAILURE MODEL TRAINING (label_failure)
# -------------------------------------------------------------------
def train_failure():
    logger.info("===================================================")
    logger.info("   TRAINING FAILURE MODEL  (label_failure)")
    logger.info("===================================================")

    df = pd.read_csv(FAILURE_DATA_PATH)

    y = df["label_failure"].astype(int)
    X = df.drop(columns=["label_failure", "file_name"])

    all_cols = list(X.columns)

    numeric_quality = [
        c
        for c in [
            "blur_score",
            "contrast_score",
            "ocr_confidence",
            "num_missing_fields",
            "has_critical_missing",
            "num_pages",
            "file_size_kb",
            "vendor_freq",
        ]
        if c in all_cols
    ]

    numeric_financial = [
        c
        for c in [
            "total_amount",
            "invoice_number_present",
            "vendor_name_length",
        ]
        if c in all_cols
    ]

    categorical = [c for c in ["amount_bucket"] if c in all_cols]

    logger.info("[FAILURE] Feature Groups:")
    logger.info(f"   Numeric Quality   : {numeric_quality}")
    logger.info(f"   Numeric Financial : {numeric_financial}")
    logger.info(f"   Categorical       : {categorical}")
    logger.info(f"   Target Dist       : {y.value_counts().to_dict()}")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num_quality",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_quality,
            ),
            (
                "num_financial",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value=0.0,
                            ),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_financial,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="unknown",
                            ),
                        ),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )

    models = {
        "logreg": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
            solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "catboost": CatBoostClassifier(
            depth=7,
            learning_rate=0.06,
            iterations=400,
            loss_function="Logloss",
            eval_metric="F1",
            verbose=False,
            random_seed=42,
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = []
    for name, model in models.items():
        pipeline = Pipeline(steps=[("pre", preprocessor), ("clf", model)])
        r = train_one_model(X_train, X_test, y_train, y_test, name, pipeline, "failure")
        results.append(r)

    best = max(results, key=lambda r: r["f1"])
    best_pipeline = best["pipeline"]
    best_name = best["model_name"]
    best_path = MODELS_DIR / "failure_model.pkl"

    joblib.dump(best_pipeline, best_path)
    logger.info(
        f"[FAILURE] Best Model: {best_name} saved to {best_path}"
    )

    logger.info("[MLFLOW] Registering FAILURE model...")
    with mlflow.start_run(run_name="register_failure_best_model"):
        mlflow.log_artifact(str(best_path), artifact_path="artifacts")
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="model",
            registered_model_name="ledgerx_failure_model",
        )
    logger.info("[MLFLOW] FAILURE model registered.")

    return results, best


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    logger.info("===================================================")
    logger.info("   LEDGERX MULTI-MODEL TRAINING - START")
    logger.info("===================================================")

    start_time = time.time()

    quality_results, quality_best = train_quality()
    failure_results, failure_best = train_failure()

    # Build leaderboard JSON
    leaderboard = {
        "quality": [
            {
                "model": r["model_name"],
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "auc": r["auc"],
            }
            for r in sorted(quality_results, key=lambda x: x["f1"], reverse=True)
        ],
        "failure": [
            {
                "model": r["model_name"],
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "auc": r["auc"],
            }
            for r in sorted(failure_results, key=lambda x: x["f1"], reverse=True)
        ],
        "best_models": {
            "quality": quality_best["model_name"],
            "failure": failure_best["model_name"],
        },
    }

    leaderboard_path = REPORTS_DIR / "model_leaderboard.json"
    with open(leaderboard_path, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)
    logger.info(f"[GLOBAL] Leaderboard saved to {leaderboard_path}")

    # Text summary report
    report_path = REPORTS_DIR / "model_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("LedgerX Multi-Model Training Summary\n")
        f.write("=====================================\n\n")

        def write_section(title, results, best):
            f.write(f"{title}\n")
            for r in results:
                f.write(
                    f"  - {r['model_name']:>12} | "
                    f"Acc={r['accuracy']:.4f} | "
                    f"Prec={r['precision']:.4f} | "
                    f"Rec={r['recall']:.4f} | "
                    f"F1={r['f1']:.4f} | "
                    f"AUC={r['auc']}\n"
                )
            f.write(
                f"  BEST_MODEL: {best['model_name']} (F1={best['f1']:.4f})\n\n"
            )

        write_section("QUALITY MODEL RESULTS:", quality_results, quality_best)
        write_section("FAILURE MODEL RESULTS:", failure_results, failure_best)

    logger.info(f"[GLOBAL] Report saved to {report_path}")
    logger.info(
        f"[GLOBAL] Training completed in {time.time() - start_time:.2f} seconds."
    )
    logger.info("===================================================")
    logger.info("   LEDGERX MULTI-MODEL TRAINING - END")
    logger.info("===================================================")


if __name__ == "__main__":
    main()
