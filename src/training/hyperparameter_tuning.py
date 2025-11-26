"""
LedgerX - Hyperparameter Tuning with Optuna
============================================

Systematic hyperparameter optimization for:
- Quality Model (CatBoost)
- Failure Model (Random Forest & Logistic Regression)

Features:
- Automated search space exploration
- MLflow integration for tracking
- Best parameters saved
- Visualization of optimization history
"""

import logging
import json
from pathlib import Path
from datetime import datetime

import optuna
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("ledgerx_hyperparameter_tuning")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
TUNING_DIR = REPORTS_DIR / "hyperparameter_tuning"

TUNING_DIR.mkdir(parents=True, exist_ok=True)

QUALITY_DATA = DATA_PROCESSED / "quality_training.csv"
FAILURE_DATA = DATA_PROCESSED / "failure_training.csv"

# MLflow setup
MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"
mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_DIR}")


# ============================================================================
# QUALITY MODEL TUNING (CATBOOST)
# ============================================================================

def tune_quality_catboost(n_trials=50):
    """
    Tune CatBoost hyperparameters for quality model
    
    Args:
        n_trials: Number of optimization trials
        
    Returns:
        Dictionary with best parameters and metrics
    """
    logger.info("="*60)
    logger.info("TUNING QUALITY MODEL (CatBoost)")
    logger.info("="*60)
    
    # Load data
    logger.info("[DATA] Loading quality training data...")
    df = pd.read_csv(QUALITY_DATA)
    
    # Features and target
    feature_cols = [
        'blur_score', 'contrast_score', 'ocr_confidence',
        'num_missing_fields', 'has_critical_missing', 'num_pages',
        'file_size_kb', 'vendor_freq'
    ]
    
    X = df[feature_cols]
    y = df['label_quality_bad']
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    logger.info(f"[DATA] Train: {len(X_train)}, Validation: {len(X_val)}")
    
    # Define objective function
    def objective(trial):
        """Optuna objective function for CatBoost"""
        
        # Suggest hyperparameters
        params = {
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'iterations': trial.suggest_int('iterations', 100, 500),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_strength': trial.suggest_float('random_strength', 0, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        }
        
        # Train model
        model = CatBoostClassifier(
            **params,
            loss_function='Logloss',
            eval_metric='F1',
            verbose=False,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("trial_number", trial.number)
        
        return f1
    
    # Create study
    study_name = f"quality_catboost_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=study_name):
        logger.info(f"[OPTUNA] Starting optimization with {n_trials} trials...")
        
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Log best results
        best_params = study.best_params
        best_value = study.best_value
        
        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score", best_value)
        
        logger.info("="*60)
        logger.info(f"[OPTUNA] Optimization complete!")
        logger.info(f"[BEST] F1 Score: {best_value:.4f}")
        logger.info(f"[BEST] Parameters: {best_params}")
        logger.info("="*60)
        
        # Save results
        results = {
            "model": "quality_catboost",
            "best_params": best_params,
            "best_f1_score": best_value,
            "n_trials": n_trials,
            "timestamp": datetime.now().isoformat(),
            "all_trials": [
                {
                    "number": t.number,
                    "params": t.params,
                    "value": t.value
                }
                for t in study.trials
            ]
        }
        
        results_file = TUNING_DIR / "quality_catboost_tuning.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"[SAVE] Results saved to: {results_file}")
        
        return results


# ============================================================================
# FAILURE MODEL TUNING (RANDOM FOREST)
# ============================================================================

def tune_failure_random_forest(n_trials=50):
    """
    Tune Random Forest hyperparameters for failure model
    
    Args:
        n_trials: Number of optimization trials
        
    Returns:
        Dictionary with best parameters and metrics
    """
    logger.info("="*60)
    logger.info("TUNING FAILURE MODEL (Random Forest)")
    logger.info("="*60)
    
    # Load data
    logger.info("[DATA] Loading failure training data...")
    df = pd.read_csv(FAILURE_DATA)
    
    # Features and target
    feature_cols = [
        'blur_score', 'contrast_score', 'ocr_confidence',
        'num_missing_fields', 'has_critical_missing', 'num_pages',
        'file_size_kb', 'vendor_freq', 'total_amount',
        'invoice_number_present', 'vendor_name_length'
    ]
    
    categorical_cols = ['amount_bucket']
    
    X_numeric = df[feature_cols]
    X_categorical = df[categorical_cols]
    y = df['label_failure']
    
    # Split
    X_train_num, X_val_num, X_train_cat, X_val_cat, y_train, y_val = train_test_split(
        X_numeric, X_categorical, y, test_size=0.2, stratify=y, random_state=42
    )
    
    logger.info(f"[DATA] Train: {len(X_train_num)}, Validation: {len(X_val_num)}")
    
    # Define objective
    def objective(trial):
        """Optuna objective for Random Forest"""
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), list(range(len(feature_cols)))),
            ('cat', OneHotEncoder(handle_unknown='ignore'), [len(feature_cols)])
        ])
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(**params))
        ])
        
        # Combine features
        X_train_combined = pd.concat([X_train_num.reset_index(drop=True), 
                                      X_train_cat.reset_index(drop=True)], axis=1)
        X_val_combined = pd.concat([X_val_num.reset_index(drop=True), 
                                    X_val_cat.reset_index(drop=True)], axis=1)
        
        # Train
        pipeline.fit(X_train_combined, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_val_combined)
        f1 = f1_score(y_val, y_pred)
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("f1_score", f1)
        
        return f1
    
    # Create study
    study_name = f"failure_rf_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=study_name):
        logger.info(f"[OPTUNA] Starting optimization with {n_trials} trials...")
        
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_value = study.best_value
        
        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score", best_value)
        
        logger.info("="*60)
        logger.info(f"[OPTUNA] Optimization complete!")
        logger.info(f"[BEST] F1 Score: {best_value:.4f}")
        logger.info(f"[BEST] Parameters: {best_params}")
        logger.info("="*60)
        
        # Save results
        results = {
            "model": "failure_random_forest",
            "best_params": best_params,
            "best_f1_score": best_value,
            "n_trials": n_trials,
            "timestamp": datetime.now().isoformat(),
            "all_trials": [
                {"number": t.number, "params": t.params, "value": t.value}
                for t in study.trials
            ]
        }
        
        results_file = TUNING_DIR / "failure_rf_tuning.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"[SAVE] Results saved to: {results_file}")
        
        return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_all_tuning(n_trials_per_model=50):
    """
    Run hyperparameter tuning for all models
    
    Args:
        n_trials_per_model: Number of trials for each model
    """
    logger.info("="*70)
    logger.info("LEDGERX HYPERPARAMETER TUNING - STARTING")
    logger.info("="*70)
    
    mlflow.set_experiment("ledgerx_hyperparameter_tuning")
    
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "n_trials_per_model": n_trials_per_model,
        "models": {}
    }
    
    # Tune Quality Model (CatBoost)
    logger.info("\n" + "="*70)
    logger.info("1/2: TUNING QUALITY MODEL")
    logger.info("="*70 + "\n")
    quality_results = tune_quality_catboost(n_trials=n_trials_per_model)
    results_summary["models"]["quality"] = quality_results
    
    # Tune Failure Model (Random Forest)
    logger.info("\n" + "="*70)
    logger.info("2/2: TUNING FAILURE MODEL")
    logger.info("="*70 + "\n")
    failure_results = tune_failure_random_forest(n_trials=n_trials_per_model)
    results_summary["models"]["failure"] = failure_results
    
    # Save combined summary
    summary_file = TUNING_DIR / "tuning_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info("HYPERPARAMETER TUNING COMPLETE!")
    logger.info("="*70)
    logger.info(f"Quality Model - Best F1: {quality_results['best_f1_score']:.4f}")
    logger.info(f"Failure Model - Best F1: {failure_results['best_f1_score']:.4f}")
    logger.info(f"\nResults saved to: {TUNING_DIR}")
    logger.info("="*70 + "\n")
    
    # Print best parameters
    logger.info("BEST PARAMETERS FOUND:")
    logger.info("-"*70)
    logger.info("\nQuality Model (CatBoost):")
    for param, value in quality_results['best_params'].items():
        logger.info(f"  {param}: {value}")
    
    logger.info("\nFailure Model (Random Forest):")
    for param, value in failure_results['best_params'].items():
        logger.info(f"  {param}: {value}")
    logger.info("-"*70)
    
    return results_summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LedgerX Hyperparameter Tuning")
    parser.add_argument(
        '--trials',
        type=int,
        default=50,
        help='Number of trials per model (default: 50)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: only 10 trials per model'
    )
    
    args = parser.parse_args()
    
    n_trials = 10 if args.quick else args.trials
    
    logger.info(f"Starting hyperparameter tuning with {n_trials} trials per model...")
    logger.info("This may take 10-30 minutes depending on your hardware.\n")
    
    results = run_all_tuning(n_trials_per_model=n_trials)
    
    logger.info("\n✅ HYPERPARAMETER TUNING COMPLETE!")
    logger.info(f"Check results in: {TUNING_DIR}")
    logger.info("\nNext steps:")
    logger.info("1. Review best parameters in tuning_summary.json")
    logger.info("2. Update train_all_models.py with best parameters")
    logger.info("3. Retrain models with optimized hyperparameters")