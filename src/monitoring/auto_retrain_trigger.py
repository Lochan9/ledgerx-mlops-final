"""
Auto-Retrain Trigger - Main Orchestrator
Checks performance and drift, triggers retraining when thresholds breached
"""

import logging
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from .performance_tracker import PerformanceTracker
from .drift_threshold_checker import DriftThresholdChecker

logger = logging.getLogger(__name__)

class AutoRetrainTrigger:
    """Main orchestrator for automatic model retraining"""
    
    def __init__(self):
        self.performance_tracker = PerformanceTracker()
        self.drift_checker = DriftThresholdChecker()
        self.retraining_log = Path("reports/retraining_log.json")
        self.retraining_log.parent.mkdir(parents=True, exist_ok=True)
    
    def check_and_trigger(self, 
                         current_data: Optional[pd.DataFrame] = None,
                         quality_f1: Optional[float] = None,
                         failure_f1: Optional[float] = None) -> Dict:
        """
        Check all triggers and initiate retraining if needed
        
        Args:
            current_data: Recent production data for drift detection
            quality_f1: Current quality model F1 score
            failure_f1: Current failure model F1 score
            
        Returns:
            {
                'retraining_triggered': bool,
                'trigger_reason': str,
                'timestamp': str,
                'details': dict
            }
        """
        timestamp = datetime.now().isoformat()
        triggers = []
        details = {}
        
        # Check 1: Performance degradation
        if quality_f1 is not None and failure_f1 is not None:
            perf_record = self.performance_tracker.record_performance(quality_f1, failure_f1)
            degradation = self.performance_tracker.check_degradation()
            
            details['performance'] = degradation
            
            if degradation['should_retrain']:
                triggers.append('PERFORMANCE_DEGRADATION')
                logger.warning(f"⚠️ Performance degradation detected: {degradation}")
        
        # Check 2: Data drift
        if current_data is not None:
            drift_result = self.drift_checker.detect_drift(current_data)
            details['drift'] = drift_result
            
            if drift_result['should_retrain']:
                triggers.append('DATA_DRIFT')
                logger.warning(f"⚠️ Data drift detected: {drift_result['drift_score']:.2%}")
        
        # Decision: Should we retrain?
        should_retrain = len(triggers) > 0
        
        result = {
            'retraining_triggered': should_retrain,
            'trigger_reasons': triggers,
            'timestamp': timestamp,
            'details': details
        }
        
        # Log decision
        self._log_decision(result)
        
        # Trigger retraining if needed
        if should_retrain:
            logger.info(f"🚀 Triggering retraining due to: {', '.join(triggers)}")
            retraining_result = self._trigger_retraining()
            result['retraining_result'] = retraining_result
            
            # Send notification
            self._send_notification(result)
        else:
            logger.info("✅ All checks passed. No retraining needed.")
        
        return result
    
    def _trigger_retraining(self) -> Dict:
        """
        Trigger the model retraining pipeline
        
        Returns:
            {
                'success': bool,
                'method': str,
                'message': str
            }
        """
        logger.info("Starting model retraining pipeline...")
        
        try:
            # Method 1: Trigger via DVC pipeline
            result = subprocess.run(
                ['dvc', 'repro'],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ DVC pipeline completed successfully")
                return {
                    'success': True,
                    'method': 'dvc_repro',
                    'message': 'Model retraining completed via DVC',
                    'output': result.stdout
                }
            else:
                logger.error(f"DVC pipeline failed: {result.stderr}")
                return {
                    'success': False,
                    'method': 'dvc_repro',
                    'message': f'DVC pipeline failed: {result.stderr}'
                }
        
        except subprocess.TimeoutExpired:
            logger.error("Retraining timed out after 1 hour")
            return {
                'success': False,
                'method': 'dvc_repro',
                'message': 'Retraining timed out'
            }
        
        except FileNotFoundError:
            # Fallback: Trigger via Python script
            logger.info("DVC not available, using Python training script")
            try:
                result = subprocess.run(
                    ['python', 'src/training/train_all_models.py'],
                    capture_output=True,
                    text=True,
                    timeout=3600
                )
                
                return {
                    'success': result.returncode == 0,
                    'method': 'python_script',
                    'message': 'Retraining via Python script',
                    'output': result.stdout if result.returncode == 0 else result.stderr
                }
            except Exception as e:
                logger.error(f"Python retraining failed: {e}")
                return {
                    'success': False,
                    'method': 'python_script',
                    'message': str(e)
                }
        
        except Exception as e:
            logger.error(f"Retraining trigger failed: {e}")
            return {
                'success': False,
                'method': 'unknown',
                'message': str(e)
            }
    
    def _send_notification(self, result: Dict):
        """Send notification about retraining trigger"""
        try:
            from ..utils.notifications import NotificationManager
            
            notifier = NotificationManager()
            
            message = f"""
🚨 **Automated Retraining Triggered**

**Reason:** {', '.join(result['trigger_reasons'])}
**Timestamp:** {result['timestamp']}

**Details:**
{json.dumps(result['details'], indent=2)}

**Action:** Model retraining pipeline has been initiated.
            """
            
            notifier.send_alert(
                title="LedgerX: Automated Retraining Triggered",
                message=message,
                severity='warning'
            )
            
            logger.info("Notification sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def _log_decision(self, result: Dict):
        """Log retraining decision to file"""
        try:
            # Load existing log
            if self.retraining_log.exists():
                with open(self.retraining_log, 'r') as f:
                    log = json.load(f)
            else:
                log = []
            
            # Append new entry
            log.append(result)
            
            # Save log
            with open(self.retraining_log, 'w') as f:
                json.dump(log, f, indent=2)
            
            logger.info(f"Logged decision to {self.retraining_log}")
            
        except Exception as e:
            logger.error(f"Failed to log decision: {e}")
    
    def get_retraining_history(self) -> list:
        """Get history of retraining decisions"""
        if self.retraining_log.exists():
            try:
                with open(self.retraining_log, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load retraining log: {e}")
                return []
        return []
    
    def get_summary(self) -> Dict:
        """Get summary of all monitoring metrics"""
        return {
            'performance': self.performance_tracker.get_performance_summary(),
            'drift': self.drift_checker.get_drift_summary(),
            'retraining_events': len([
                r for r in self.get_retraining_history() 
                if r.get('retraining_triggered', False)
            ])
        }


def main():
    """Main function for scheduled execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    logger.info("="*80)
    logger.info("LedgerX Auto-Retrain Trigger - Starting Check")
    logger.info("="*80)
    
    trigger = AutoRetrainTrigger()
    
    # Example: Load recent production data
    try:
        # In production, this would be recent inference data
        recent_data_path = Path("data/production/recent_features.csv")
        if recent_data_path.exists():
            current_data = pd.read_csv(recent_data_path).tail(500)  # Last 500 rows
        else:
            logger.warning("No recent data available for drift detection")
            current_data = None
    except Exception as e:
        logger.error(f"Failed to load current data: {e}")
        current_data = None
    
    # Example: Get current model performance
    # In production, these would come from actual model evaluation
    try:
        # Simulate performance check
        from .performance_tracker import simulate_performance_check
        quality_f1, failure_f1 = simulate_performance_check()
    except:
        logger.warning("Could not get current performance metrics")
        quality_f1, failure_f1 = None, None
    
    # Run check and trigger if needed
    result = trigger.check_and_trigger(
        current_data=current_data,
        quality_f1=quality_f1,
        failure_f1=failure_f1
    )
    
    print("\n" + "="*80)
    print("📊 AUTO-RETRAIN CHECK RESULT")
    print("="*80)
    print(json.dumps(result, indent=2))
    print("="*80)
    
    # Show summary
    summary = trigger.get_summary()
    print("\n📈 MONITORING SUMMARY:")
    print(json.dumps(summary, indent=2))
    
    if result['retraining_triggered']:
        print("\n🚨 RETRAINING TRIGGERED!")
        print(f"Reason: {', '.join(result['trigger_reasons'])}")
    else:
        print("\n✅ ALL CHECKS PASSED - No retraining needed")


if __name__ == "__main__":
    main()