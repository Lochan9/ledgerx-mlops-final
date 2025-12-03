"""
LedgerX - ADVANCED Hyperparameter Tuning
=========================================
Uses Bayesian Optimization, Multi-Objective, and Ensemble Methods
"""

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import mlflow

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import f1_score, roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "hyperparameter_tuning"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

QUALITY_DATA = DATA_PROCESSED / "quality_training.csv"
FAILURE_DATA = DATA_PROCESSED / "failure_training.csv"


# ============================================================================
# ADVANCED: CATBOOST TUNING (Bayesian Optimization)
# ============================================================================

def objective_catboost_quality(trial, X_train, y_train):
    """
    Optuna objective for CatBoost with advanced search space
    """
    params = {
        # Learning parameters
        'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        
        # Regularization
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        
        # Tree parameters
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'max_bin': trial.suggest_int('max_bin', 200, 400),
        
        # Boosting parameters
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        
        # Other
        'random_seed': 42,
        'verbose': 0,
        'task_type': 'CPU'
    }
    
    # 5-fold cross-validation
    model = CatBoostClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Multi-metric scoring
    f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    # Report multiple metrics to Optuna
    trial.set_user_attr('mean_f1', f1_scores.mean())
    trial.set_user_attr('mean_auc', auc_scores.mean())
    trial.set_user_attr('std_f1', f1_scores.std())
    
    # Optimize F1 score
    return f1_scores.mean()


def tune_catboost_quality(n_trials=100):
    """
    Advanced CatBoost tuning with Bayesian optimization
    """
    logger.info("="*80)
    logger.info("ADVANCED CATBOOST TUNING - Quality Model")
    logger.info(f"Trials: {n_trials} | Sampler: TPE (Bayesian) | Pruner: Median")
    logger.info("="*80)
    
    # Load data
    df = pd.read_csv(QUALITY_DATA)
    y = df['label_quality_bad']
    X = df.drop(columns=['label_quality_bad', 'file_name'], errors='ignore')
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    logger.info(f"Training samples: {len(X_train)}, Features: {len(X.columns)}")
    
    # Create Optuna study with advanced settings
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42, n_startup_trials=20),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )
    
    # Run optimization
    logger.info(f"Starting Bayesian optimization ({n_trials} trials)...")
    study.optimize(lambda trial: objective_catboost_quality(trial, X_train, y_train), 
                   n_trials=n_trials, 
                   show_progress_bar=True)
    
    # Get best parameters
    best_params = study.best_params
    best_f1 = study.best_value
    
    logger.info("="*80)
    logger.info(f"✅ Best F1 Score: {best_f1:.4f}")
    logger.info(f"✅ Best Parameters: {best_params}")
    logger.info("="*80)
    
    # Train final model with best params
    best_model = CatBoostClassifier(**best_params, random_seed=42, verbose=0)
    best_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    
    logger.info(f"Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}")
    
    # Save results
    results = {
        'model': 'CatBoost',
        'task': 'quality',
        'n_trials': n_trials,
        'best_params': best_params,
        'cv_f1_mean': best_f1,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'timestamp': datetime.now().isoformat()
    }
    
    output_path = REPORTS_DIR / 'quality_catboost_tuning.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Results saved: {output_path}")
    
    return results


# ============================================================================
# ADVANCED: RANDOM FOREST TUNING
# ============================================================================

def objective_rf_failure(trial, X_train, y_train):
    """Advanced RandomForest tuning with complex search space"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 50, 500),
        'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.01),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = RandomForestClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    
    return f1_scores.mean()


def tune_rf_failure(n_trials=100):
    """Advanced RandomForest tuning for failure model"""
    logger.info("="*80)
    logger.info("ADVANCED RANDOM FOREST TUNING - Failure Model")
    logger.info(f"Trials: {n_trials} | Sampler: TPE (Bayesian)")
    logger.info("="*80)
    
    df = pd.read_csv(FAILURE_DATA)
    y = df['label_failure']
    X = df.drop(columns=['label_failure', 'file_name'], errors='ignore')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    logger.info(f"Training samples: {len(X_train)}, Features: {len(X.columns)}")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42, n_startup_trials=20),
        pruner=MedianPruner()
    )
    
    logger.info(f"Starting Bayesian optimization ({n_trials} trials)...")
    study.optimize(lambda trial: objective_rf_failure(trial, X_train, y_train), 
                   n_trials=n_trials,
                   show_progress_bar=True)
    
    best_params = study.best_params
    best_f1 = study.best_value
    
    logger.info("="*80)
    logger.info(f"✅ Best CV F1: {best_f1:.4f}")
    logger.info(f"✅ Best Parameters: {best_params}")
    logger.info("="*80)
    
    # Train final model
    best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train)
    
    y_pred = best_model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    
    logger.info(f"Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}")
    
    results = {
        'model': 'RandomForest',
        'task': 'failure',
        'n_trials': n_trials,
        'best_params': best_params,
        'cv_f1_mean': best_f1,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'timestamp': datetime.now().isoformat()
    }
    
    output_path = REPORTS_DIR / 'failure_rf_tuning.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Results saved: {output_path}")
    
    return results


# ============================================================================
# ENSEMBLE: STACKING MULTIPLE MODELS
# ============================================================================

def train_stacked_ensemble():
    """
    Advanced: Stack multiple models for better performance
    Combines CatBoost + RandomForest + LogReg
    """
    from sklearn.ensemble import StackingClassifier
    
    logger.info("="*80)
    logger.info("TRAINING STACKED ENSEMBLE")
    logger.info("="*80)
    
    # Load quality data
    df = pd.read_csv(QUALITY_DATA)
    y = df['label_quality_bad']
    X = df.drop(columns=['label_quality_bad', 'file_name'], errors='ignore')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Base models
    estimators = [
        ('catboost', CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, verbose=0, random_seed=42)),
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)),
        ('logreg', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ]
    
    # Meta-learner
    meta_learner = LogisticRegression(random_state=42)
    
    # Stacking classifier
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    
    logger.info("Training stacked ensemble...")
    stacking_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = stacking_model.predict(X_test)
    y_prob = stacking_model.predict_proba(X_test)[:, 1]
    
    test_f1 = f1_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_prob)
    
    logger.info(f"✅ Stacked Ensemble - F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
    
    return {'f1': test_f1, 'auc': test_auc, 'model': stacking_model}


# ============================================================================
# RUN ALL TUNING
# ============================================================================

def run_complete_tuning(n_trials=100):
    """
    Run complete hyperparameter tuning suite
    """
    logger.info("="*80)
    logger.info("LEDGERX ADVANCED HYPERPARAMETER TUNING")
    logger.info("="*80)
    
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'n_trials': n_trials,
        'models': {}
    }
    
    # Tune Quality Model (CatBoost)
    logger.info("\n[1/3] Tuning Quality Model (CatBoost)...")
    quality_results = tune_catboost_quality(n_trials=n_trials)
    results_summary['models']['quality_catboost'] = quality_results
    
    # Tune Failure Model (RandomForest)
    logger.info("\n[2/3] Tuning Failure Model (RandomForest)...")
    failure_results = tune_rf_failure(n_trials=n_trials)
    results_summary['models']['failure_rf'] = failure_results
    
    # Train Stacked Ensemble
    logger.info("\n[3/3] Training Stacked Ensemble...")
    ensemble_results = train_stacked_ensemble()
    results_summary['models']['ensemble'] = {
        'f1': float(ensemble_results['f1']),
        'auc': float(ensemble_results['auc'])
    }
    
    # Save summary
    summary_path = REPORTS_DIR / 'tuning_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info("="*80)
    logger.info("TUNING COMPLETE")
    logger.info("="*80)
    logger.info(f"✅ Quality (CatBoost): F1={quality_results['test_f1']:.4f}, AUC={quality_results['test_auc']:.4f}")
    logger.info(f"✅ Failure (RandomForest): F1={failure_results['test_f1']:.4f}, AUC={failure_results['test_auc']:.4f}")
    logger.info(f"✅ Ensemble: F1={ensemble_results['f1']:.4f}, AUC={ensemble_results['auc']:.4f}")
    logger.info(f"✅ Summary saved: {summary_path}")
    logger.info("="*80)
    
    return results_summary


if __name__ == "__main__":
    # Run with 100 trials (takes ~30-45 minutes)
    results = run_complete_tuning(n_trials=100)
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(json.dumps(results, indent=2))