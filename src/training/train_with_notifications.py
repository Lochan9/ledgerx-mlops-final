"""
LedgerX - Training Pipeline with Notifications (Simple)
========================================================

Sends notifications before and after training.
"""

import logging
import time
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.alerts import send_alert, get_alert_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("training_notifier")

def run_training_with_notifications():
    """Run training pipeline with Slack notifications"""
    
    logger.info("="*70)
    logger.info("LEDGERX TRAINING PIPELINE - WITH NOTIFICATIONS")
    logger.info("="*70)
    
    # Check alerts
    status = get_alert_status()
    logger.info(f"Slack: {status['slack_enabled']}, Email: {status['email_enabled']}")
    
    # Start notification
    send_alert(
        message=f"🚀 LedgerX Training Started\n\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        severity="info"
    )
    
    start_time = time.time()
    
    try:
        # Run hyperparameter tuning
        logger.info("\n[1/4] Hyperparameter tuning...")
        result = subprocess.run(
            [sys.executable, "src/training/hyperparameter_tuning.py", "--quick"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("✅ Tuning complete")
        
        # Load tuning results
        import json
        tuning_path = Path("reports/hyperparameter_tuning/tuning_summary.json")
        if tuning_path.exists():
            with open(tuning_path) as f:
                tuning_data = json.load(f)
            quality_f1 = tuning_data['models']['quality']['best_f1_score']
            failure_f1 = tuning_data['models']['failure']['best_f1_score']
            
            send_alert(
                message=f"✅ Hyperparameter Tuning Complete\n\nQuality: {quality_f1:.4f}\nFailure: {failure_f1:.4f}",
                severity="info"
            )
        
        # Run model registration
        logger.info("\n[2/4] Registering models...")
        result = subprocess.run(
            [sys.executable, "src/training/register_models.py"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("✅ Models registered")
        
        total_time = time.time() - start_time
        
        # Final notification
        send_alert(
            message=f"""🎉 Training Pipeline Complete!

⏱️ Time: {total_time:.2f}s
📊 Quality F1: {quality_f1:.4f}
📊 Failure F1: {failure_f1:.4f}

✅ Models registered and ready!
""",
            severity="info"
        )
        
        logger.info("="*70)
        logger.info("✅ PIPELINE COMPLETE!")
        logger.info("="*70)
        
        return {"status": "success", "time": total_time}
        
    except Exception as e:
        logger.exception("Pipeline failed!")
        
        send_alert(
            message=f"🚨 Training Failed!\n\nError: {str(e)}\n\nCheck logs for details.",
            severity="critical"
        )
        
        return {"status": "failed", "error": str(e)}


if __name__ == "__main__":
    logger.info("Starting training with notifications...\n")
    result = run_training_with_notifications()
    
    if result['status'] == 'success':
        logger.info(f"\n✅ SUCCESS in {result['time']:.2f}s")
    else:
        logger.error(f"\n❌ FAILED: {result.get('error')}")
        sys.exit(1)