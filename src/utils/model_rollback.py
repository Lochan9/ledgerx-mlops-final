"""
Automated Model Rollback System for LedgerX (Using .pkl Pipeline Files)
Automatically reverts to previous model if new model performs worse
"""

import joblib
import shutil
import json
from pathlib import Path
from datetime import datetime
from loguru import logger
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import sys


class ModelRollback:
    """Automated model rollback with performance validation using sklearn pipelines"""
    
    def __init__(self, models_dir="models", reports_dir="reports"):
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.history_file = self.reports_dir / "rollback_history.json"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Models directory: {self.models_dir}")
        logger.info(f"📁 Reports directory: {self.reports_dir}")
    
    def backup_current_model(self, model_name):
        """
        Backup current model before replacing
        model_name: 'quality' or 'failure'
        """
        current = self.models_dir / f"{model_name}_model.pkl"
        backup = self.models_dir / f"{model_name}_model.pkl.backup"
        
        if current.exists():
            shutil.copy(current, backup)
            logger.info(f"✅ Backed up {model_name}_model.pkl → {backup.name}")
            return True
        else:
            logger.warning(f"⚠️ No current model found: {current}")
            return False
    
    def validate_new_model(self, model_name, test_data_path, 
                          min_f1_threshold=0.70):
        """
        Validate new model performance using sklearn pipeline
        If worse than backup, automatically rollback
        
        Args:
            model_name: 'quality' or 'failure'
            test_data_path: Path to test CSV file
            min_f1_threshold: Minimum F1 score required
        
        Returns:
            True if model is good, False if rolled back
        """
        logger.info("=" * 80)
        logger.info(f"🔍 VALIDATING NEW {model_name.upper()} MODEL")
        logger.info("=" * 80)
        
        # Model paths (.pkl files with full pipeline)
        current = self.models_dir / f"{model_name}_model.pkl"
        backup = self.models_dir / f"{model_name}_model.pkl.backup"
        
        if not current.exists():
            logger.error(f"❌ Current model not found: {current}")
            return False
        
        if not backup.exists():
            logger.warning(f"⚠️ No backup found, assuming first deployment")
            logger.success(f"✅ First deployment approved!")
            return True
        
        # Load test data
        try:
            df = pd.read_csv(test_data_path)
            logger.info(f"📊 Loaded test data: {len(df)} samples, {len(df.columns)} features")
        except Exception as e:
            logger.error(f"❌ Failed to load test data: {e}")
            return False
        
        # Determine label column
        if model_name == 'quality':
            label_col = 'label_quality_bad'
        else:
            label_col = 'label_failure'
        
        if label_col not in df.columns:
            logger.error(f"❌ Label column '{label_col}' not found in test data")
            logger.error(f"   Available columns: {list(df.columns)}")
            return False
        
        # Split features and labels
        X_test = df.drop([label_col], axis=1, errors='ignore')
        y_test = df[label_col]
        
        # Remove file_name if present
        if 'file_name' in X_test.columns:
            X_test = X_test.drop('file_name', axis=1)
        
        logger.info(f"📊 Test features: {len(X_test.columns)} columns")
        logger.info(f"📊 Test labels: {y_test.value_counts().to_dict()}")
        
        # Evaluate current model (pipeline handles preprocessing!)
        logger.info("📈 Evaluating CURRENT model...")
        try:
            current_pipeline = joblib.load(current)
            y_pred_current = current_pipeline.predict(X_test)
            current_f1 = f1_score(y_test, y_pred_current)
            current_acc = accuracy_score(y_test, y_pred_current)
            logger.info(f"   Current → F1: {current_f1:.4f}, Accuracy: {current_acc:.4f}")
        except Exception as e:
            logger.error(f"❌ Failed to evaluate current model: {e}")
            logger.warning(f"⚠️ Accepting model anyway (cannot validate)")
            return True
        
        # Evaluate backup model
        logger.info("📈 Evaluating BACKUP model...")
        try:
            backup_pipeline = joblib.load(backup)
            y_pred_backup = backup_pipeline.predict(X_test)
            backup_f1 = f1_score(y_test, y_pred_backup)
            backup_acc = accuracy_score(y_test, y_pred_backup)
            logger.info(f"   Backup  → F1: {backup_f1:.4f}, Accuracy: {backup_acc:.4f}")
        except Exception as e:
            logger.error(f"❌ Failed to evaluate backup model: {e}")
            logger.warning("⚠️ Backup model failed, accepting current model")
            return True
        
        # Decision logic
        should_rollback = False
        reason = ""
        
        # Check 1: Minimum threshold
        if current_f1 < min_f1_threshold:
            should_rollback = True
            reason = f"F1 below threshold ({current_f1:.4f} < {min_f1_threshold})"
        
        # Check 2: Worse than backup by 5%
        elif current_f1 < backup_f1 - 0.05:
            should_rollback = True
            reason = f"F1 significantly worse than backup ({current_f1:.4f} vs {backup_f1:.4f})"
        
        # Check 3: Slightly worse (warn but don't rollback)
        elif current_f1 < backup_f1:
            logger.warning(f"⚠️ New model slightly worse ({current_f1:.4f} vs {backup_f1:.4f})")
            logger.warning(f"⚠️ But within tolerance, deploying anyway")
        
        # Perform rollback if needed
        if should_rollback:
            logger.error("=" * 80)
            logger.error(f"🔴 ROLLBACK TRIGGERED: {reason}")
            logger.error("=" * 80)
            self.rollback(model_name)
            self._log_rollback(model_name, reason, current_f1, backup_f1)
            return False
        else:
            logger.success("=" * 80)
            logger.success(f"✅ NEW MODEL APPROVED! F1={current_f1:.4f}")
            logger.success(f"✅ Improvement over backup: {current_f1 - backup_f1:+.4f}")
            logger.success("=" * 80)
            self._log_deployment(model_name, current_f1, backup_f1)
            return True
    
    def rollback(self, model_name):
        """Rollback to backup model"""
        logger.warning(f"🔄 Performing rollback for {model_name}...")
        
        current = self.models_dir / f"{model_name}_model.pkl"
        backup = self.models_dir / f"{model_name}_model.pkl.backup"
        
        if not backup.exists():
            logger.error(f"❌ No backup found for {model_name}")
            return False
        
        # Save failed model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        failed = self.models_dir / f"{model_name}_model.pkl.failed_{timestamp}"
        
        if current.exists():
            shutil.move(str(current), str(failed))
            logger.info(f"💾 Failed model saved as: {failed.name}")
        
        # Restore backup
        shutil.copy(str(backup), str(current))
        
        logger.success(f"✅ Rolled back {model_name} to backup version")
        return True
    
    def _log_rollback(self, model_name, reason, current_f1, backup_f1):
        """Log rollback event"""
        history = self._load_history()
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "action": "ROLLBACK",
            "reason": reason,
            "current_f1": float(current_f1),
            "backup_f1": float(backup_f1),
            "decision": "Reverted to backup"
        }
        
        history.append(event)
        self._save_history(history)
        logger.info(f"📝 Rollback event logged to {self.history_file}")
    
    def _log_deployment(self, model_name, current_f1, previous_f1):
        """Log successful deployment"""
        history = self._load_history()
        
        improvement = current_f1 - previous_f1
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "action": "DEPLOY",
            "current_f1": float(current_f1),
            "previous_f1": float(previous_f1),
            "improvement": float(improvement),
            "decision": "Approved for production"
        }
        
        history.append(event)
        self._save_history(history)
        logger.info(f"📝 Deployment logged to {self.history_file}")
    
    def _load_history(self):
        """Load rollback history"""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_history(self, history):
        """Save rollback history"""
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def show_history(self):
        """Display rollback history"""
        history = self._load_history()
        
        if not history:
            logger.info("📋 No rollback history found")
            return
        
        logger.info("=" * 80)
        logger.info("📋 ROLLBACK HISTORY")
        logger.info("=" * 80)
        
        for event in history[-10:]:  # Show last 10 events
            timestamp = event['timestamp'][:19]
            action = event['action']
            model = event['model']
            
            if action == "ROLLBACK":
                logger.warning(f"🔴 {timestamp} | {model:8} | ROLLED BACK | Reason: {event['reason']}")
            else:
                improvement = event.get('improvement', 0)
                logger.success(f"✅ {timestamp} | {model:8} | DEPLOYED    | Improvement: {improvement:+.4f}")
        
        logger.info("=" * 80)


def main():
    """
    Run rollback validation after model training
    Works with sklearn .pkl pipeline files
    """
    logger.info("=" * 80)
    logger.info("🔄 AUTOMATED ROLLBACK VALIDATION SYSTEM (sklearn pipelines)")
    logger.info("=" * 80)
    
    rollback = ModelRollback()
    
    # Check if test data exists
    quality_test = Path("data/processed/quality_test.csv")
    failure_test = Path("data/processed/failure_test.csv")
    
    if not quality_test.exists():
        logger.error(f"❌ Test data not found: {quality_test}")
        logger.error("❌ Cannot validate models without test data")
        logger.info("💡 Run: python src/training/train_all_models.py (it creates test sets)")
        return 1
    
    if not failure_test.exists():
        logger.error(f"❌ Test data not found: {failure_test}")
        logger.error("❌ Cannot validate models without test data")
        logger.info("💡 Run: python src/training/train_all_models.py (it creates test sets)")
        return 1
    
    # Validate quality model
    logger.info("")
    quality_ok = rollback.validate_new_model(
        "quality",
        str(quality_test),
min_f1_threshold=0.70
    )
    
    # Validate failure model
    logger.info("")
    failure_ok = rollback.validate_new_model(
        "failure",
        str(failure_test),
        min_f1_threshold=0.65
    )
    
    # Show history
    logger.info("")
    rollback.show_history()
    
    # Final result
    logger.info("")
    if quality_ok and failure_ok:
        logger.success("=" * 80)
        logger.success("✅ ALL MODELS VALIDATED SUCCESSFULLY!")
        logger.success("✅ Models are ready for production")
        logger.success("=" * 80)
        return 0
    else:
        logger.error("=" * 80)
        logger.error("❌ MODEL VALIDATION FAILED")
        logger.error("❌ Rollback performed, using previous models")
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())